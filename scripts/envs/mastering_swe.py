"""SFT data-prep for the `swe_infinite` env: MiniSWE **tool-calling** bug-fix trajectories.

DIUKUR 15 Agu di VM eval (image `gradientsio/swe-infinite:v2`, mini-swe-agent 1.17.5):
harness memilih blok `tool_agent` dari `/app/agents/config.yaml` (`miniswe.py:482-490`),
membuang parsing blok bash (`miniswe.py:214`), dan melempar `FormatError` begitu
`response["tool_calls"]` kosong (`:253-273`) — TANPA jalur cadangan. Konsekuensinya
terukur pada model nyata:

    base polos Qwen2.5-3B-Instruct  -> finish=tool_calls, execute_bash(...)  VALID
    boss 5GU4Xkd3 (bobot penuh)     -> tool_calls null, banjir <|im_start|>   0,0
    boss 5HKEAZxF (LoRA di atasnya) -> tool_calls null, prosa sopan           0,0

Jadi kontrak yang benar adalah TOOL-CALLING. Versi lama berkas ini mencetak
'THOUGHT: ...```bash ...```' — format yang harness tidak lagi baca sama sekali —
sehingga model kita akan gagal dengan mekanisme yang sama persis seperti boss.

SWE-ZERO sendiri masih memakai kontrak lama (60.000 baris, NOL `tool_calls`), jadi
berkas ini mengonversinya. Konversi itu aman karena bentuk sumbernya seragam:

    giliran asisten dengan TEPAT SATU blok bash   97%   -> execute_bash({"command": ...})
    giliran user diawali 'Observation:'           91%   -> role='tool' + tool_call_id
    perintah terakhir 'echo <SENTINEL> ...'    dominan  -> submit()

Ca32 MENAMBAH SATU mekanisme di atas itu (lihat _subsample_exploration): baris korpus
DIUKUR 7 Sep menunjukkan 84-90%% giliran asisten yang selamat dari saw_edit tak pernah
menyunting berkas (kepala find/grep/cat/sed(-n)/ls — reconnaissance, bukan perbaikan).
Baris yang TAK PERNAH menyunting sudah dibuang lewat gerbang saw_edit lama; ini menyubsampel
DI DALAM baris yang selamat, membatasi giliran eksplorasi non-edit ke maksimal K=6 per
trajektori (deterministik lewat hash isi baris, urutan + giliran eksplorasi PERTAMA selalu
disimpan) sehingga target latih terkonsentrasi pada keputusan yang mengubah repo, bukan
tersebar pada rekonaisans yang toh diderivasi ulang harness sendiri dalam anggaran 25
giliran. Berbeda dari klip observasi Fpds (yang memotong ISI satu giliran); ini memotong
JUMLAH giliran eksplorasi yang disimpan.
"""
import argparse
import hashlib
import json
import random
import re

VALIDATION_RATIO = 0.01

# Cap on retained non-edit exploration turns per trajectory (see _subsample_exploration).
_MAX_EXPLORE_TURNS = 6

# Sentinel penanda giliran submit. SWE-ZERO memuat DUA dialek:
#   mini-swe-agent-1     -> COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   (50.000 baris, 'Submitted')
#   mini-swe-agent-plus  -> MINI_SWE_AGENT_FINAL_OUTPUT             (10.000 baris, 'Accepted')
# Keduanya dipetakan ke tool `submit`, jadi teks sentinelnya sendiri jadi tak relevan.
_SENTINELS = ('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 'MINI_SWE_AGENT_FINAL_OUTPUT')
_ACCEPTED_FORMATS = frozenset({'mini-swe-agent-1', 'mini-swe-agent-plus'})

# Trajektori hanya mengajarkan PERBAIKAN kalau ada giliran yang benar-benar mengubah
# berkas. Tanpa syarat ini kita menyuling "jelajahi lalu submit tanpa menyunting" — dan
# itu persis kegagalan yang TERUKUR pada base polos di eval 15 Agu:
#     [SWE-INFINITE] Verifying fix... / [SWE-INFINITE] No patch generated  -> skor 0,0
# (`env.py`: `if not patch: return 0.0, {"error": "no patch"}`).
_EDIT_RE = re.compile(
    r'sed -i|>\s*[^ >|&]|>>\s*\S|\btee\b|cat\s*<<|cat\s*>|\bpatch\b|apply_patch'
    r'|python[0-9]?\s+-c|printf[^|]*>',
    re.IGNORECASE)

# Turn-level edit classifier for _subsample_exploration. DELIBERATELY SEPARATE from _EDIT_RE
# above: _EDIT_RE gates whole ROWS (saw_edit: does this trajectory edit ANYWHERE, checked
# against the harness's own 'no patch' failure mode) and must stay exactly as measured
# 15 Agu. This one classifies each SURVIVING turn's `command` argument so exploration turns
# can be thinned without disturbing that row-level gate. Measured 7 Sep on the real corpus
# (scratch measure.py, n=50000 rows / 737k exec turns): only 10.0%% of turns edit; the rest
# are find/grep/cat/sed(-n)/ls heads (top_heads). List matches the task brief verbatim:
# sed -i, cat >/>>, echo >/>>, tee, apply_patch, git apply, python - <<, mv/cp into repo, patch.
_EXPLORE_TURN_EDIT_RE = re.compile(
    r'sed\s+-i|cat\s*(?:<<[^\n]*)?>|echo[^|\n]*>|printf[^|\n]*>|apply_patch|git\s+apply'
    r'|python[0-9]?\s+-\s*<<|\btee\b|\bpatch\b|^\s*(?:mv|cp)\s+\S+\s+\S|&&\s*(?:mv|cp)\s+\S+\s+\S',
    re.IGNORECASE | re.MULTILINE)

_BASH_BLOCK = re.compile(r'```(?:bash|sh)?\s*\n?(.*?)```', re.DOTALL)
_THOUGHT_HEAD = re.compile(r'^\s*THOUGHT\s*:\s*', re.IGNORECASE)
_OBS_HEAD = re.compile(r'^\s*Observation\s*:\s*')
# 'Please solve this issue in the repository foo/bar.\n\n## Issue\n\n<isi>'
_TASK_PREAMBLE = re.compile(r'^\s*Please solve[^\n]*\n+(?:#+\s*Issue\s*\n+)?', re.IGNORECASE)

# Skema tool DISALIN VERBATIM dari harness (`/app/agents/miniswe.py` AGENT_TOOLS).
# Model melihat definisi ini saat eval; melatih dengan definisi yang berbeda berarti
# melatih kondisional yang salah.
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'execute_bash',
        'description': ('Execute one bash command (or commands connected with && or ||) in the '
                        'repository environment and return its output. Each call runs in a fresh '
                        'subshell in /app: directory changes and environment variables do not '
                        'persist between calls.'),
        'parameters': {'type': 'object',
                       'properties': {'command': {'type': 'string',
                                                  'description': 'The bash command to execute.'}},
                       'required': ['command'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'submit',
        'description': ('Stage all changes and submit them as your final answer, ending the task. '
                        'Call this once you have verified your fix. You cannot continue working '
                        'after submitting.'),
        'parameters': {'type': 'object', 'properties': {}, 'required': [],
                       'additionalProperties': False}}},
]

# system_template harness, verbatim.
_SYSTEM = (
    'You are a software engineering agent working in a Linux shell environment to solve a real '
    'GitHub issue or PR task.\n\nYou interact with the environment exclusively through tool calls. '
    'Every turn you must call exactly ONE tool:\n'
    '1. execute_bash — execute one bash command (or commands connected with && or ||) and '
    'observe its output\n'
    '2. submit — stage all changes and submit your work, ending the task\n\n'
    'You may reason briefly in normal assistant text before the tool call, but never write '
    'commands as plain text: only commands issued through the execute_bash tool are executed.'
)

# instance_template harness, verbatim kecuali {{task}} -> {task}. Dipertahankan penuh supaya
# kondisi yang dilatih sama dengan kondisi yang dinilai.
_INSTANCE = """<pr_description>
Consider the following issue or PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer through tool calls.
You'll be helping implement necessary changes to meet requirements described above.
Your task is to make changes to source files in the current directory to resolve the described issue in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think, call ONE tool, see its result, then think and call your next tool.

For each response:
1. You may include a short thought explaining your reasoning and what you're trying to accomplish
2. Call exactly ONE tool: execute_bash (with a "command" argument) or submit (no arguments)

## Important Boundaries
- MODIFY: Regular source code files in /app (this is the working directory for all your commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)
- NEVER add or modify unit tests. Your job is ONLY to implement or fix the source code.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a simple script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Call the submit tool to finish your work.
   <important>After calling submit, you cannot continue working on this task.</important>

## Command Execution Rules
You are operating in an environment where
1. You call the execute_bash tool with a single command
2. The system executes that command in a subshell
3. You see the result
4. You call your next tool

**CRITICAL REQUIREMENTS:**
- Every response MUST include exactly ONE tool call
- execute_bash runs EXACTLY ONE command (or a set of commands connected with && or ||)
- Plain-text commands (e.g. in code blocks) are NOT executed — always use the execute_bash tool
- Directory or environment variable changes are not persistent. Every command is executed in a new subshell.
- However, you can prefix any command with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- If you need to run multiple independent commands, either combine them with && or ||, or issue them one per turn across successive responses

## Environment Details
- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- If a command isn't available, you can install it

## Submission
When you've completed your work (reading, editing, testing) and cannot make further progress,
call the submit tool. It stages your changes (git add -A) and submits the resulting diff as your final answer.
You cannot continue working (reading, editing, testing) in any way on this task after submitting.
</instructions>"""


def _blocks(content):
    return [m.strip() for m in _BASH_BLOCK.findall(content or '') if m.strip()]


def _is_submit_cmd(cmd):
    return any(s in cmd for s in _SENTINELS)


def _call(idx, name, arguments):
    return {'id': 'call_' + str(idx), 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _task_text(first_user):
    """Kupas pembungkus SWE-ZERO agar isi issue bisa dibungkus ulang dengan template harness."""
    body = _TASK_PREAMBLE.sub('', first_user or '', count=1).strip()
    return body or (first_user or '').strip()


def _row_hash(msgs):
    """Deterministic seed for _subsample_exploration, keyed on the trajectory's own task text
    (the second message: the wrapped {task} instance) rather than a shared counter — so the
    same corpus + seed always keeps the same turns for a given row, independent of scan order,
    without any cross-row state to thread through _filter."""
    key = (msgs[1].get('content') or '') if len(msgs) > 1 else ''
    return int(hashlib.sha256(key.encode('utf-8', 'surrogatepass')).hexdigest(), 16)


def _subsample_exploration(msgs, k=_MAX_EXPLORE_TURNS):
    """Keep every edit/submit turn; subsample non-edit exploration turns to at most `k`.

    Measured 7 Sep: 84-90%% of assistant turns never touch a file (find/grep/cat/sed -n/ls —
    orienting in the repo, not fixing it). Rows that never edit AT ALL are already dropped by
    the saw_edit gate in _convert; this trims the turns WITHIN a surviving row so the trained
    target concentrates loss on the decisions that change the repo, instead of spending most
    of every trajectory's tokens on read-only reconnaissance the harness re-derives itself at
    eval time. Order is preserved and the FIRST exploration turn is always kept (every real
    trajectory starts by looking around before it edits, so that opening step stays legible).
    """
    head = msgs[:2]          # [system, user(instance)] -- constant, never touched
    turns = []                # (kind, [assistant] or [assistant, tool])
    i = 2
    while i < len(msgs):
        a = msgs[i]
        pair = [a]
        if i + 1 < len(msgs) and msgs[i + 1].get('role') == 'tool':
            pair.append(msgs[i + 1])
        call = (a.get('tool_calls') or [{}])[0]
        fn = call.get('function') or {}
        if fn.get('name') == 'submit':
            kind = 'submit'
        else:
            try:
                cmd = json.loads(fn.get('arguments') or '{}').get('command', '')
            except (TypeError, ValueError):
                cmd = ''
            kind = 'edit' if _EXPLORE_TURN_EDIT_RE.search(cmd) else 'explore'
        turns.append((kind, pair))
        i += len(pair)

    explore_idx = [j for j, (kind, _) in enumerate(turns) if kind == 'explore']
    if len(explore_idx) <= k:
        return msgs                                # already at/under the cap: no-op

    rng = random.Random(_row_hash(msgs))
    rest = explore_idx[1:]
    keep_rest = set(rng.sample(rest, min(k - 1, len(rest)))) if k > 1 else set()
    keep = {explore_idx[0]} | keep_rest

    out = list(head)
    for j, (kind, pair) in enumerate(turns):
        if kind != 'explore' or j in keep:
            out.extend(pair)
    return out


def _convert(msgs, obs_max_chars=3000):
    """SWE-ZERO (blok bash) -> percakapan tool-calling. None kalau barisnya tak bisa dipetakan.

    Aturannya sengaja ketat: satu giliran asisten = tepat satu blok bash. 97% giliran sumber
    sudah begitu, dan giliran ber-blok-ganda tidak punya pemetaan yang jujur ke kontrak
    "tepat SATU tool call per giliran" — memaksakannya justru mengajarkan bentuk yang salah.
    """
    firsts = [m for m in msgs if m.get('role') == 'user']
    if not firsts:
        return None
    out = [{'role': 'system', 'content': _SYSTEM},
           {'role': 'user', 'content': _INSTANCE.format(task=_task_text(firsts[0].get('content')))}]

    seen_first_user = False
    pending_id = None
    idx = 0
    saw_exec = False
    saw_submit = False
    saw_edit = False
    for m in msgs:
        role, content = m.get('role'), m.get('content')
        if role == 'system':
            continue
        if role == 'user':
            if not seen_first_user:
                seen_first_user = True
                continue
            if pending_id is None:
                return None                      # observasi tanpa tool call sebelumnya
            obs = _OBS_HEAD.sub('', content or '', count=1).strip()
            # Potong dump `cat`/`grep` yang panjang. Trainer FRONT-truncate, jadi observasi
            # raksasa di tengah mendorong giliran submit — giliran yang justru harus dipelajari
            # — keluar dari jendela token. Scaffold mini-swe-agent sendiri memotong keluaran
            # tool, jadi memotong di sini justru MENDEKATKAN latih ke eval, bukan menjauhkan.
            if obs_max_chars and len(obs) > obs_max_chars:
                obs = obs[:obs_max_chars] + '\n... [output dipotong]'
            out.append({'role': 'tool', 'tool_call_id': pending_id, 'content': obs})
            pending_id = None
            continue
        if role != 'assistant':
            return None
        if saw_submit:
            return None                          # ada giliran setelah submit
        blocks = _blocks(content)
        if len(blocks) != 1:
            return None
        thought = _THOUGHT_HEAD.sub('', (content or '').split('```')[0]).strip()
        cmd = blocks[0]
        if _is_submit_cmd(cmd):
            call = _call(idx, 'submit', {})
            saw_submit = True
        else:
            call = _call(idx, 'execute_bash', {'command': cmd})
            saw_exec = True
            if _EDIT_RE.search(cmd):
                saw_edit = True
        out.append({'role': 'assistant', 'content': thought, 'tool_calls': [call]})
        pending_id = call['id']
        idx += 1

    if not (saw_exec and saw_submit and saw_edit):
        # saw_edit WAJIB: arc explore->submit tanpa sunting adalah kegagalan "no patch"
        # yang terukur pada base. Melatihnya berarti menyuling skor 0.
        return None
    if out[-1].get('role') != 'assistant':
        return None
    return _subsample_exploration(out)


def _row_chars(msgs):
    return sum(len(m.get('content') or '') +
               len(json.dumps(m.get('tool_calls') or '')) for m in msgs)


def _load_swe_zero():
    """Muat SWE-ZERO yang sudah diunduh validator lewat MINER_DATASETS.

    Pakai loader bersama (scripts/shared/miner_datasets.py) yang sama dipakai
    combine_datasets._read_miner_datasets. Layout SWE-ZERO sendiri (data/train.jsonl +
    sampling_audit.json) sudah termuat oleh loader lama; yang berubah di sini: satu jalur
    muat untuk semua dataset, dan kegagalan tidak lagi ditelan diam-diam (dulu `except:
    continue` tanpa log). Tiap kandidat dilog: not mounted / load failed: <err> / shape mismatch.
    """
    import os
    root = os.environ.get('MINER_DATASETS_DIR')
    names = [n.strip() for n in os.environ.get('MINER_DATASETS', '').split(',') if n.strip()]
    if not root or not names:
        return None
    from shared.miner_datasets import load_miner_dataset_dir
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            print(f'[swe] miner dataset {name}: not found (not mounted at {path})', flush=True)
            continue
        try:
            ds = load_miner_dataset_dir(path)
        except Exception as exc:
            print(f'[swe] miner dataset {name}: load failed: {type(exc).__name__}: {exc}', flush=True)
            continue
        cols = set(ds.column_names)
        if 'messages' in cols and ('instance_id' in cols or 'trajectory_format' in cols):
            print('[swe] using miner dataset ' + name + ': rows=' + str(len(ds)) +
                  ' cols=' + str(sorted(cols)), flush=True)
            return ds
        print('[swe] miner dataset ' + name + ' cols=' + str(sorted(cols)) +
              ' not SWE-ZERO shape; skip', flush=True)
    return None


def _filter(ds, args):
    """Konversi + saring. `Accepted` dipisah karena ia satu-satunya penanda mutu yang ada.

    Dataset TIDAK punya kolom resolved/reward/score, jadi 'Accepted' (10.000 baris,
    trajectory_format='mini-swe-agent-plus') adalah sinyal terbaik yang kita punya —
    dan 300/300 sampelnya benar-benar menyunting berkas, lawan 289/300 pada 'Submitted'.
    Versi lama berkas ini membuang SELURUH baris itu lewat cek format di satu baris.
    """
    accepted, submitted = [], []
    per_instance = {}
    scanned = dropped_fmt = dropped_conv = dropped_len = 0
    ceiling = args.target_samples * 3 if args.target_samples > 0 else None
    for ex in ds:
        scanned += 1
        fmt = ex.get('trajectory_format')
        if fmt is not None and fmt not in _ACCEPTED_FORMATS:
            dropped_fmt += 1
            continue
        raw = ex.get('messages')
        if not isinstance(raw, list) or len(raw) < args.min_messages:
            dropped_conv += 1
            continue
        msgs = _convert(raw)
        if msgs is None:
            dropped_conv += 1
            continue
        if _row_chars(msgs) > args.max_chars:
            dropped_len += 1
            continue
        iid = ex.get('instance_id') or ''
        if per_instance.get(iid, 0) >= args.per_instance:
            continue
        per_instance[iid] = per_instance.get(iid, 0) + 1
        is_acc = (ex.get('exit_status') or '').strip().lower() == 'accepted'
        (accepted if is_acc else submitted).append({'messages': msgs})
        if ceiling and (len(accepted) + len(submitted)) >= ceiling:
            break
    print('[swe] converted SWE-ZERO: scanned=' + str(scanned) +
          ' kept=' + str(len(accepted) + len(submitted)) +
          ' (accepted=' + str(len(accepted)) + ' submitted=' + str(len(submitted)) + ')' +
          ' dropped: format=' + str(dropped_fmt) + ' shape=' + str(dropped_conv) +
          ' too_long=' + str(dropped_len), flush=True)
    return accepted, submitted


def _fallback_or_die(reason, args):
    """Fallback self-gen melatih DISTRIBUSI YANG BERBEDA dari eval.

    swe_infinite menilai perbaikan bug di repo nyata sampai 25 giliran; fallback ini
    membungkus tugas bash sepele dalam pakaian MiniSWE. Ia ada sebagai lantai keandalan —
    task yang menghasilkan nol baris gagal total dan memberi 0 untuk SETIAP env di task itu,
    dan itu lebih buruk. Tapi ia tak boleh menyala DIAM-DIAM, jadi ia berteriak, dan
    SWE_ALLOW_SELFGEN=0 mengubahnya jadi kegagalan keras untuk run tergate.
    """
    import os
    allow = os.environ.get('SWE_ALLOW_SELFGEN', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    banner = '=' * 78
    for line in (
        banner,
        '!!! SWE FALLBACK AKTIF — DATA LATIH BUKAN DISTRIBUSI EVAL !!!',
        '!!! sebab : ' + reason,
        '!!! akibat: model dilatih pada tugas bash sepele, bukan perbaikan bug repo nyata',
        '!!! cek   : MINER_DATASETS memuat SWE-ZERO? filter (--max-chars/--per-instance) terlalu ketat?',
        banner,
    ):
        print(line, flush=True)
    if not allow:
        raise RuntimeError(
            'SWE fallback diblokir oleh SWE_ALLOW_SELFGEN=0 — ' + reason +
            '. Perbaiki sumber data alih-alih melatih distribusi yang salah.')
    return _selfgen_rows(args.target_samples if args.target_samples > 0 else 4000, args.seed)


def _selfgen_rows(n, seed):
    """Trajektori self-gen, kini dalam bentuk tool-calling yang sama."""
    if n <= 0:
        return []
    try:
        from envs.selfgen_bash import generate_tasks
    except ImportError:
        from selfgen_bash import generate_tasks
    rows = []
    # Index rather than destructure: this tree's generate_tasks yields (query, gold, obs) while
    # the tree this file came from yields a fourth near-miss element. Destructuring would raise
    # ValueError here -- and here is the reliability floor, the path taken when the corpus is
    # missing entirely, so it is the one place a crash cannot be tolerated.
    for task in generate_tasks(n, seed=seed):
        (query, gold, obs) = (task[0], task[1], task[2])
        exec_call = _call(0, 'execute_bash', {'command': gold})
        submit_call = _call(1, 'submit', {})
        rows.append({'messages': [
            {'role': 'system', 'content': _SYSTEM},
            {'role': 'user', 'content': _INSTANCE.format(task=query)},
            {'role': 'assistant', 'content': 'I will run the command that produces the required result.',
             'tool_calls': [exec_call]},
            {'role': 'tool', 'tool_call_id': exec_call['id'], 'content': obs},
            {'role': 'assistant', 'content': 'The output is correct, so I will submit now.',
             'tool_calls': [submit_call]},
        ]})
    return rows


def _build_rows(args):
    ds = _load_swe_zero()
    if ds is None:
        return _fallback_or_die('SWE-ZERO tidak ditemukan di MINER_DATASETS', args)

    accepted, submitted = _filter(ds, args)
    if not accepted and not submitted:
        return _fallback_or_die(
            'SWE-ZERO ADA tapi 0 baris lolos konversi — ini kasus paling berbahaya: '
            'bentuk dataset berubah atau konversi tool-calling kita meleset', args)

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    rng.shuffle(submitted)
    requested = args.target_samples if args.target_samples > 0 else (len(accepted) + len(submitted))

    # `Accepted` didahulukan dan digandakan selama masih muat: ia satu-satunya tingkat mutu
    # yang bisa kita bedakan, dan 100% sampelnya benar-benar menyunting berkas.
    acc_weight = 2 if len(accepted) * 2 <= requested else 1
    rows = list(accepted) * acc_weight
    rows += submitted[:max(0, requested - len(rows))]
    rng.shuffle(rows)
    n_acc = min(len(accepted) * acc_weight, len(rows))
    print('[swe] mix: ' + str(n_acc) + ' accepted (x' + str(acc_weight) + ') + ' +
          str(len(rows) - n_acc) + ' submitted', flush=True)
    out = rows[:requested] if requested > 0 else rows
    if requested > 0 and len(out) < int(requested * 0.9):
        print('[swe] WARN under-delivery: requested=' + str(requested) + ' delivered=' + str(len(out)) +
              ' (baris nyata < target; tak ada top-up self-gen di jalur ini). '
              'Set SFT_TARGET_OVERRIDE=' + str(len(out)) + ' agar epoch mengikuti volume nyata.',
              flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_path', required=True, help='DatasetDict save dir')
    p.add_argument('--target_samples', type=int, default=0, help='Cap on total rows (0 = use all kept).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-instance', type=int, default=4,
                   help='Max kept rows per instance_id (anti-overfit).')
    p.add_argument('--max-chars', type=int, default=150000,
                   help='Drop converted trajectories longer than this. SENGAJA LONGGAR. '
                        'Yang mengikat bukan cap ini melainkan jendela token trainer (4096 utk 3B), '
                        'dan trainer FRONT-truncate — menyimpan EKOR. Diukur 15 Agu (n=3.698): meski '
                        'hanya 7%% baris submitted / 0%% accepted muat utuh di 4096 token, giliran '
                        'MENYUNTING tetap berada di dalam ekor pada 98%% / 99%% baris, karena '
                        'trajektori nyata menyunting berulang menjelang submit. Cap ketat justru '
                        'membuang tingkat `Accepted` (1.799 dari 3.000 gugur di cap 40.000) tanpa '
                        'menyelamatkan pelajaran apa pun.')
    p.add_argument('--min-messages', type=int, default=3,
                   help='Drop degenerate trajectories with fewer messages.')
    p.add_argument('--num_games', type=int, default=0, help='Ignored (caller-compat).')
    p.add_argument('--time_limit_seconds', type=int, default=0, help='Ignored (caller-compat).')
    args = p.parse_args()

    rows = _build_rows(args)
    if not rows:
        raise RuntimeError('mastering_swe produced 0 rows — neither SWE-ZERO nor the self-gen fallback yielded data.')

    tools_json = json.dumps(TOOLS)
    for r in rows:
        r['tools_json'] = tools_json

    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_list(rows)
    n_val = max(1, int(len(dataset) * VALIDATION_RATIO)) if len(dataset) >= 2 else 0
    if n_val and len(dataset) - n_val >= 1:
        splits = dataset.train_test_split(test_size=n_val, seed=args.seed)
        dd = DatasetDict({'train': splits['train'], 'validation': splits['test']})
    else:
        dd = DatasetDict({'train': dataset, 'validation': dataset.select(range(min(1, len(dataset))))})
    dd.save_to_disk(args.output_path)
    print('[swe] saved ' + str(len(dd['train'])) + ' train / ' + str(len(dd['validation'])) +
          ' val tool-calling trajectories to ' + args.output_path, flush=True)


if __name__ == '__main__':
    main()
