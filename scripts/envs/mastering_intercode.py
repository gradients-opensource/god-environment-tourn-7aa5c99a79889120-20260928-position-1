import argparse
import json
import os
import random
import re
import subprocess
import zlib
OBS_TRUNCATE_CHARS = 350
VALIDATION_RATIO = 0.01
_GOLD_TIMEOUT = 30

# ─────────────────────────────────────────────────────────────────────────────
# TOOL-CALLING contract — MUST mirror the validator's eval_intercode.py.
# The eval parses the model's tool_calls (SGLang --tool-call-parser) and has NO
# plain-text fallback: a turn with no execute_bash/submit tool_call is treated as
# "no valid action" → reward 0. ReAct free-text ("Action 1: execute[...]") therefore
# scores ~0. We emit REAL tool_calls (same shape as the PvP games generator,
# pvp_toolcalling.build_toolcalling_example), so the SFT distribution matches eval.
# ─────────────────────────────────────────────────────────────────────────────
INTERCODE_EXECUTE_TOOL_NAME = 'execute_bash'
INTERCODE_SUBMIT_TOOL_NAME = 'submit'

# ⛔ BYTE-EXACT COPY of validator/evaluation/evaluators/intercode.py
# INTERCODE_TOOL_SYSTEM_PROMPT (gradients-ai/G.O.D @ main). DO NOT SHORTEN, REWORD, OR
# "TIDY" IT. Verified 2026-08-04: md5 of the string body == 6ee9feb0906998c285375d5cf565742b
# on both sides. Re-verify against upstream before every tournament; if upstream changes,
# copy theirs — never keep ours.
#
# It was NOT byte-exact until now, and we scored 4th of 5 on intercode in R2 (2026-08-03,
# reconstructed quantile 0.25). The shipped version was a 24-line compression of upstream's
# 40 lines: it kept the header verbatim but rewrote all three few-shot examples down to a
# single successful command each. Two separate harms from one edit:
#   1. PREFIX MISMATCH. The system prompt is re-sent on EVERY eval turn (_build_tool_messages
#      rebuilds [system, user] from scratch each turn — the transcript lives inside the USER
#      message, there is no growing assistant/tool chain). So a 1.5B student trained on a
#      different 24-line prefix meets an unseen 40-line prefix on every single decision. This
#      is exactly the "train on the EXACT validator eval prompt" rule that this repo already
#      identifies as the single biggest edge for the card games — intercode was the one env
#      that violated it.
#   2. BEHAVIOUR MISMATCH. Upstream's three examples ALL demonstrate iterative refinement
#      (run → inspect imperfect output → refine → submit; 4/4/3 turns, >=2 execute_bash each).
#      The compressed versions demonstrate one-shot (run gold → submit). We were teaching the
#      opposite policy to the one the eval prompt demonstrates — and reinforcing it from the
#      data side too, since --recovery-frac defaults to 0.0 (see its help text: the InterCode
#      single-turn 10.5 -> multi-turn 58.7 lever, currently OFF). Restoring the prompt fixes
#      the prompt half; the data half is still open and must be GATED ON EVAL before flipping.
#
# ⚠ LATENT TRAP guarding this fix — read before raising --recovery-frac or OBS_TRUNCATE_CHARS.
# train_sft_env.py truncates from the FRONT (`ids[-max_length:]`), and the system prompt is at
# the front of every row. So any row that exceeds max_seq_length (2048 on the 1_2_b bucket)
# gets its system prompt silently amputated — destroying exactly the byte-exactness bought
# here, with no per-row warning. Headroom today is ~1.9x (worst case: a 3-turn recovery row
# with a full 350-char observation is ~1000-1100 tokens), so it does not bite. It WILL bite if
# recovery histories grow. Measure the longest row before enabling recovery, not after.
INTERCODE_TOOL_SYSTEM_PROMPT = """Interact with a Bourne Shell system using Bash commands to answer a question.
You may reason briefly in normal assistant text, but every turn must call exactly one tool:
(1) execute_bash, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Do not write actions as plain text. Use the tools for every action.

Examples:

Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Turn 1 tool: execute_bash(command="xxd -p -l 16 /testbed/textfile7.txt")
Observation 1: /bin/bash: line 1: xxd: command not found
Turn 2 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt")
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Turn 3 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' '")
Observation 3: 546578742066696c6520372077697468
Turn 4 tool: submit()

Question: print disk usage in human readable format of files or folders in /workspace
Turn 1 tool: execute_bash(command="du /workspace")
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Turn 2 tool: execute_bash(command="du -h /workspace")
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Turn 3 tool: execute_bash(command="du -sh /workspace")
Observation 3: 100K\t/workspace
Turn 4 tool: submit()

Question: Count all the lines of all php files in the /testbed directory recursively
Turn 1 tool: execute_bash(command="find /testbed -name \"*.php\" | xargs wc -l")
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Turn 2 tool: execute_bash(command="find /testbed -name \"*.php\" -type f -exec cat {} + | wc -l")
Observation 2: 2
Turn 3 tool: submit()
"""


def _intercode_tools():
    return [
        {'type': 'function', 'function': {'name': INTERCODE_EXECUTE_TOOL_NAME, 'description': 'Execute one Bash command in the InterCode Bourne Shell environment.', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The Bash command to execute.'}}, 'required': ['command'], 'additionalProperties': False}}},
        {'type': 'function', 'function': {'name': INTERCODE_SUBMIT_TOOL_NAME, 'description': 'Submit the previous observation as the final answer and end the task.', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}},
    ]


def _assistant_tool_msg(name: str, arguments: dict, content: str, call_id: str = 'call_0') -> dict:
    return {'role': 'assistant', 'content': content, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def _user_prompt(query: str, history_text: str, turn: int) -> str:
    # Mirrors eval_intercode._build_tool_messages: Question + Previous steps + Turn N.
    return f'Question: {query}\n\nPrevious steps:\n{history_text}\n\nTurn {turn}: call exactly one tool now.'


# --- L4: history `Thought N:` lines only where the trained base can actually produce one -----
# eval_intercode._build_tool_messages appends "Thought N: <content>" to the history ONLY when
# `result.content` is truthy (deep-dive A.2 :866-867). Whether a fine-tuned model's `content` is
# ever non-empty next to a `tool_calls` block depends on the chat template of the base it was
# tuned from: measured offline via apply_chat_template(tools=...) against the 6 whitelisted
# bases (deep-dive A.2), Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct DROP assistant
# `content` whenever `tool_calls` is present (rendered output is byte-identical for
# content='' and content=None on both — confirmed by direct render in this arm), so a model
# fine-tuned on either base can never emit a `content` the eval would read as truthy, and the
# eval's own history therefore never carries a `Thought N:` line for it. Every Qwen base in the
# roster keeps `content` untouched. `include_thoughts=False` reproduces that: the history line
# for a turn omits `Thought N:` (mirroring the evaluator's own truthy-check) and the assistant
# target's `content` is set to '' (a str, not None -- keeps the Arrow/HF Dataset column
# homogeneous across rows) so the trained target is the tool call alone, matching what the
# eval-time model can actually produce. `--history-thoughts off` is wired in from
# sft_env_config.py only for the two Llama bases (get_model_architecture == 'llamaforcausallm');
# every other caller (default 'auto', or explicit 'on') keeps the pre-existing HEAD behaviour
# byte-for-byte.
def _content_for(thought: str, include_thoughts: bool) -> str:
    return thought if include_thoughts else ''


def _history_step(n: int, thought: str, tool_str: str, obs: str, include_thoughts: bool) -> str:
    """One `Thought N:`? + `Tool N:` + `Observation N:` block, mirroring
    eval_intercode._format_tool_history's own truthy-gated Thought line."""
    lines = []
    if include_thoughts:
        lines.append(f'Thought {n}: {thought}')
    lines.append(f'Tool {n}: {tool_str}')
    lines.append(f'Observation {n}: {obs}')
    return '\n'.join(lines)


def run_gold(gold: str) -> str:
    if not gold.strip():
        return ''
    try:
        res = subprocess.run(['bash', '-c', gold], cwd='/', capture_output=True, text=True, timeout=_GOLD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return '(gold command timed out)'
    except Exception as exc:
        return f'(gold command error: {type(exc).__name__})'
    out = res.stdout if res.stdout else res.stderr
    out = out or ''
    if len(out) > OBS_TRUNCATE_CHARS:
        out = out[:OBS_TRUNCATE_CHARS]
    return out

_THOUGHT1_VARIANTS = ['I can answer this with a single bash command and run it directly.', 'This can be solved with one shell command; let me run it.', 'A single bash command produces the answer — running it now.', "I'll execute the appropriate bash command to get the answer.", 'The right approach is one bash command; executing it.', "I'll solve this directly with one bash command.", 'One shell command gives the answer; let me run it.', 'This is a single-command task — executing it now.', 'I can get the result with one bash invocation.', 'A direct bash command answers this; running it.', 'Let me run the bash command that answers this.', 'The answer comes from one command; executing it now.', 'Running the shell command to obtain the result.', "I'll issue a single bash command for this.", 'This needs just one command — here it goes.', 'A quick bash command will return the answer; running it.', 'Executing the bash command that resolves this question.', "I'll get the answer with a single shell invocation now.", 'One command suffices here; executing it.', 'Let me execute the bash command to compute the answer.']
_THOUGHT2_VARIANTS = ['The observation above is the answer to the question. I will submit.', 'That output answers the question. Submitting it.', "The result above is correct; I'll submit.", 'This observation is the answer — submitting now.', 'The command output answers the task. I will submit.', 'The output above gives the answer; submitting.', "That observation contains the answer — I'll submit.", 'The command produced the answer. Submitting now.', 'This is the correct result; submitting it.', 'The observation answers the task — submit.', 'The observation gives the final answer; submitting.', "That output is what was asked — I'll submit it.", 'The result is in the observation above; submitting now.', 'The command output is the answer. Submitting.', 'This output resolves the question — submit.', "The observation holds the answer; I'll submit it now.", 'That is the correct output; submitting.', 'The answer is shown above — submitting it.', 'The observation is the final result; submitting.', "I have the answer from the output above; I'll submit."]

# How the question itself is phrased, per variant. The thought variants above change only the
# assistant's words; the user turn -- the thing the model conditions on -- was byte-identical
# across every variant of a task, so four variants taught four replies to one prompt. Each frame
# below is a way a person might put the same request. Variant 0 keeps the plain form, which is how
# the evaluation phrases its queries, so the shipped default still trains on the exact shape it is
# scored on; the other frames add wording the model must read through to find the same task.
_QUERY_FRAMES = [
    '{q}',
    'Using bash, {q_lc}',
    '{q} Use a single command.',
    'In the shell: {q}',
    'Please {q_lc}',
    '{q} One command is enough.',
]


def _frame_query(query: str, variant: int) -> str:
    q = query.strip()
    frame = _QUERY_FRAMES[variant % len(_QUERY_FRAMES)]
    q_lc = q if (len(q) < 2 or not q[0].isupper() or q[1].isupper() or q[0] in '"/') else q[0].lower() + q[1:]
    return frame.format(q=q, q_lc=q_lc)


def build_examples(query: str, gold: str, obs1: str, variant: int = 0, include_thoughts: bool = True):
    """Two tool-calling SFT turns for one task: execute_bash(gold) then submit().

    Each example is {messages:[system,user,assistant-with-tool_calls], tools:[...]},
    the same shape as pvp_toolcalling.build_toolcalling_example so the combined
    adapter sees a consistent schema and the model learns to EMIT tool_calls.

    `include_thoughts` (L4, see the note above `_history_step`): when False, the assistant
    targets carry content='' and the turn-2 history omits `Thought 1:` -- for bases whose chat
    template drops assistant `content` next to `tool_calls` (Llama), so the trained target and
    history match what that base can actually produce at eval.
    """
    query = _frame_query(query, variant)
    thought1 = _THOUGHT1_VARIANTS[variant % len(_THOUGHT1_VARIANTS)]
    thought2 = _THOUGHT2_VARIANTS[variant % len(_THOUGHT2_VARIANTS)]
    # JSON-encode tools as a STRING so HF Dataset.from_list does NOT struct-unify the
    # heterogeneous execute_bash({command}) vs submit({}) param schemas — unification would
    # corrupt submit's empty params into {command: null}. Matches the PvP _row convention
    # (tools_json); combine_datasets + train_sft_env.tokenize_and_mask decode it back.
    tools_json = json.dumps(_intercode_tools())
    system = {'role': 'system', 'content': INTERCODE_TOOL_SYSTEM_PROMPT}
    # Turn 1: run the gold command (no prior steps).
    u1 = {'role': 'user', 'content': _user_prompt(query, 'No commands have been executed yet.', 1)}
    a1 = _assistant_tool_msg(INTERCODE_EXECUTE_TOOL_NAME, {'command': gold}, _content_for(thought1, include_thoughts))
    ex1 = {'messages': [system, u1, a1], 'tools_json': tools_json}
    # Turn 2: submit. History string mirrors eval_intercode's _format_tool_history.
    hist = _history_step(1, thought1, f'{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(gold)})', obs1, include_thoughts)
    u2 = {'role': 'user', 'content': _user_prompt(query, hist, 2)}
    a2 = _assistant_tool_msg(INTERCODE_SUBMIT_TOOL_NAME, {}, _content_for(thought2, include_thoughts))
    ex2 = {'messages': [system, u2, a2], 'tools_json': tools_json}
    return [ex1, ex2]

# --- REFINE trajectories (L2: mirror the eval's own 3 few-shots, see the deep-dive
# A.2/A.3/D3 and mastering_intercode's system-prompt note above) ---------------------------
# The 3 few-shots baked into INTERCODE_TOOL_SYSTEM_PROMPT (above) are ALL "run a correct but
# verbose command -> inspect its noisier output -> refine to the minimal correct form ->
# submit". Before this arm every self-gen row was single-shot (build_examples: run the ALREADY
# minimal gold, then submit) -- we never trained the refine step the eval itself demonstrates
# on every turn (the system prompt is re-sent from scratch each turn, so this is the ONE thing
# the model is shown how to do before every single decision). `selfgen_bash.generate_tasks` now
# ships a shell-verified `(verbose_gold, verbose_obs)` sibling pair for templates with a natural
# noisier-but-correct alternative form (extra columns, unabridged listing) -- this builds the
# 3-row trajectory: execute(verbose) -> execute(minimal) -> submit(). All three rows are TRAINED
# targets (unlike build_recovery_example's near-miss, which is history-only); both commands here
# already succeeded in the sandbox, so there is no masking concern.
_REFINE_THOUGHT_VARIANTS = [
    'That output has more in it than the question asked for; let me get just the answer.',
    'This is correct but noisier than needed — narrowing it down.',
    'The output includes extra detail I do not need; running a leaner command.',
    "That's more verbose than the question calls for; let me trim it down.",
    'The command worked but printed more than necessary — refining it.',
    'Too much detail in that output; getting the minimal form instead.',
    'That answers it, but with extra noise — running the reduced version.',
    'The result is buried in extra output; narrowing to just what was asked.',
]


def build_refine_example(query: str, verbose_gold: str, verbose_obs: str, gold: str, obs1: str, variant: int = 0, include_thoughts: bool = True):
    """Three tool-calling SFT turns for one task, mirroring the eval's own few-shots: row A
    executes a correct-but-VERBOSE command, row B (after seeing the verbose observation)
    executes the minimal command, row C (after seeing the clean observation) submits. All
    three assistant turns are TRAINED targets — both `verbose_gold` and `gold` are shell-
    verified commands (selfgen_bash.generate_tasks), so there is nothing to mask.

    `include_thoughts` — see the L4 note above `_history_step`."""
    query = _frame_query(query, variant)
    thought_a = _THOUGHT1_VARIANTS[variant % len(_THOUGHT1_VARIANTS)]
    thought_b = _REFINE_THOUGHT_VARIANTS[variant % len(_REFINE_THOUGHT_VARIANTS)]
    thought_c = _THOUGHT2_VARIANTS[variant % len(_THOUGHT2_VARIANTS)]
    tools_json = json.dumps(_intercode_tools())
    system = {'role': 'system', 'content': INTERCODE_TOOL_SYSTEM_PROMPT}
    # Row A: run the verbose-but-correct command (no prior steps).
    u1 = {'role': 'user', 'content': _user_prompt(query, 'No commands have been executed yet.', 1)}
    a1 = _assistant_tool_msg(INTERCODE_EXECUTE_TOOL_NAME, {'command': verbose_gold}, _content_for(thought_a, include_thoughts))
    ex_a = {'messages': [system, u1, a1], 'tools_json': tools_json}
    # Row B: history shows the verbose command + its (noisier) observation; run the minimal
    # gold command instead. Thought explains WHY the prior output was noisier than needed.
    hist1 = _history_step(1, thought_a, f'{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(verbose_gold)})', verbose_obs, include_thoughts)
    u2 = {'role': 'user', 'content': _user_prompt(query, hist1, 2)}
    a2 = _assistant_tool_msg(INTERCODE_EXECUTE_TOOL_NAME, {'command': gold}, _content_for(thought_b, include_thoughts))
    ex_b = {'messages': [system, u2, a2], 'tools_json': tools_json}
    # Row C: history now also shows the minimal command + its clean observation; submit.
    hist2 = hist1 + '\n' + _history_step(2, thought_b, f'{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(gold)})', obs1, include_thoughts)
    u3 = {'role': 'user', 'content': _user_prompt(query, hist2, 3)}
    a3 = _assistant_tool_msg(INTERCODE_SUBMIT_TOOL_NAME, {}, _content_for(thought_c, include_thoughts))
    ex_c = {'messages': [system, u3, a3], 'tools_json': tools_json}
    return [ex_a, ex_b, ex_c]


# --- Multi-turn RECOVERY trajectories (the InterCode 10.5%->58.7% lever) -----------------
# Our eval is multi-turn (<=10) but the base recipe only trains single-shot-gold, so the model
# never learns to recover from a failed command. We teach that here. Because our examples are
# per-turn-independent (the prior steps live in the user-prompt history, never as an assistant
# TARGET), the failed step is STeP-masked BY CONSTRUCTION: the near-miss command appears only in
# the history the model reads, and every emitted assistant turn is a CORRECT action. No trainer
# change needed. Ref: InterCode 2306.14898, STeP partial-masking 2505.20023.

_RECOVER_THOUGHTS = ['That attempt failed — let me run the correct command.', 'The previous command errored; correcting it now.', "That didn't work; I'll fix the command and re-run.", 'The last step returned an error — here is the right command.', 'Recovering from the failed attempt with the correct command.', 'That command was wrong; executing the proper one.', 'The prior command failed; trying the correct invocation.', 'Error above — running the corrected command now.']


# I3 (2026-09-19): the old `_near_miss` here SYNTHESIZED a plausible-looking fake error string
# ("bash: xx: command not found") without ever running the candidate command — a guessed
# observation, exactly the category of problem the E10 revert documents for bigcode's synthesized
# submit observations (build_execute_only_example's note, above): text nothing scores or reads,
# invented rather than captured, with a real risk of looking wrong in an inspectable way. Real
# near-miss execution now happens in selfgen_bash.py (`_synthesize_near_miss_candidate` +
# `_run_near_miss_in`, sandboxed inside the task's own tempdir via `_run_gold_in`'s
# `_assert_command_confined_to_tempdir` machinery — never against real `/testbed` etc.) and is
# threaded through `selfgen_bash.generate_tasks`'s 7th tuple element. `build_recovery_example`
# below now takes that REAL (command, observation) pair directly instead of synthesizing one.


def build_recovery_example(query: str, gold: str, obs1: str, near_miss: "tuple[str, str]", variant: int = 0, include_thoughts: bool = True):
    """A recovery trajectory as CORRECT-turn examples only: given a failed first attempt (in the
    history), the model (turn 2) runs the correct gold command, then (turn 3) submits. The failed
    near-miss lives ONLY in the history string, so it is never a training target.

    `near_miss` is `(near_miss_command, real_observation)` — REALLY EXECUTED by
    selfgen_bash.generate_tasks in the same sandbox as the task's own fixture files (I3); never a
    hand-written fake error string.

    `include_thoughts` — see the L4 note above `_history_step`."""
    near, err = near_miss
    t1 = _THOUGHT1_VARIANTS[variant % len(_THOUGHT1_VARIANTS)]
    t_rec = _RECOVER_THOUGHTS[variant % len(_RECOVER_THOUGHTS)]
    t_sub = _THOUGHT2_VARIANTS[variant % len(_THOUGHT2_VARIANTS)]
    tools_json = json.dumps(_intercode_tools())
    system = {'role': 'system', 'content': INTERCODE_TOOL_SYSTEM_PROMPT}
    # History after the FAILED turn 1 (near-miss + its error) — context only, not a target.
    hist1 = _history_step(1, t1, f'{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(near)})', err, include_thoughts)
    # Turn 2 = RECOVER: emit the correct gold command. TRAINED.
    u2 = {'role': 'user', 'content': _user_prompt(query, hist1, 2)}
    a2 = _assistant_tool_msg(INTERCODE_EXECUTE_TOOL_NAME, {'command': gold}, _content_for(t_rec, include_thoughts))
    ex_recover = {'messages': [system, u2, a2], 'tools_json': tools_json}
    # Turn 3 = SUBMIT (gold succeeded). TRAINED.
    hist2 = hist1 + '\n' + _history_step(2, t_rec, f'{INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(gold)})', obs1, include_thoughts)
    u3 = {'role': 'user', 'content': _user_prompt(query, hist2, 3)}
    a3 = _assistant_tool_msg(INTERCODE_SUBMIT_TOOL_NAME, {}, _content_for(t_sub, include_thoughts))
    ex_submit = {'messages': [system, u3, a3], 'tools_json': tools_json}
    return [ex_recover, ex_submit]

# --- BigCode instruction dataset (execute-only, no filesystem) ------------------------
# intercode_bigcode_combined_12k is instruction->gold-command pairs with NO associated
# filesystem, so there is no real observation to train a submit turn on (a fake empty obs
# would teach "bash always prints nothing"). We emit one execute-only turn per bash row,
# matching the eval's turn-1 distribution. Python/prose rows are dropped — wrapping them
# in an execute_bash call would be wrong.
_PYTHON_PROSE_MARKERS = ('```', 'def ', 'import ', 'class ')

def is_bash_response(response: str) -> bool:
    """True if a bigcode response looks like a bash command rather than Python/prose."""
    if any(marker in response for marker in _PYTHON_PROSE_MARKERS):
        return False
    return len(response) <= 400

# --- Extra filters for the bigcode blend (E8) --------------------------------------------
# These are on top of is_bash_response, gating what an execute-only training row is allowed to
# teach: (1) commands that are unsafe/interactive outside our own synthetic fixtures, (2) commands
# whose EFFECT (not just their head) is destructive against system state, and (3) unfilled
# template placeholders that never resolved to a concrete, runnable command upstream.
#
# Separator set a command head may follow: start-of-string, a shell control operator (| ; & &&
# || newline backtick $( bare (), or the argument-injecting forms `xargs [flags]` / `find ... -exec`
# (the two ways NL2Bash-style corpora hand a path list to another command). Widened from the
# original ^ | ; & ( set (2026-08 version only covered 5 of these) so both the interactive-head
# filter below and the destructive filter see every position a command can actually start from.
_SEP_PREFIX = r'(?:\A|[|;&`(]|&&|\|\||\n|\$\(|xargs(?:\s+-\S+)*\s+|-exec\s+)\s*'

# Command heads that either need a live network/repo/root context we don't have at train time, or
# are interactive (no non-interactive exit path), so wrapping them in a single execute_bash turn
# would teach a call that hangs or errors in a plain container. Matched as a full command word
# right after the separator set above, so e.g. "gitk" or "mysqldump" (not bare heads) are
# unaffected — only "git", "ssh" etc as their own command.
_UNSAFE_HEAD_RE = re.compile(
    _SEP_PREFIX + r'(sudo|ssh|scp|mount|umount|git|curl|wget|telnet|ftp|nc|netcat|'
    r'rsync|vim?|nano|less|more|top|htop|mysql|psql|man)\b'
)

# --- F3: destructive / system-state filter -------------------------------------------------
# NOTE (honest scope): this is a TRAINING-CONTENT filter — it decides what we are willing to
# teach a model to type — NOT a sandbox-safety guarantee. Nothing in this module executes a
# dataset command; any code path that DOES run one (ours or anyone else's) must do so only
# inside `docker --network none`, never on a bare host. This filter existing does not change that.
_DESTRUCTIVE_HEAD_RE = re.compile(
    _SEP_PREFIX + r'(kill(?:all)?|pkill|crontab|shutdown|reboot|halt|poweroff|mkfs\S*|dd|fdisk)\b'
)
_DEV_REDIRECT_RE = re.compile(r'>\s*/dev/(?:sd[a-z]+\d*|nvme\d+\w*)\b')

# rm / chmod / chown / chgrp / `find -delete` / `xargs rm` are legitimate NL2Bash skills when
# scoped to the CURRENT directory (e.g. `find . -name '*.tmp' -delete`) — only drop them when the
# path they act on is an absolute system-root path (or bare `/`). A relative/cwd-scoped delete
# never matches _SYSTEM_ROOTS below because it has no leading '/'.
_DESTRUCTIVE_PATH_CMD_RE = re.compile(_SEP_PREFIX + r'(rm|chmod|chown|chgrp)\b')
_SYSTEM_ROOTS = ('/', '/etc', '/usr', '/bin', '/sbin', '/lib', '/var', '/boot',
                  '/dev', '/sys', '/proc', '/root', '/home')
_ABS_PATH_RE = re.compile(r'(?<![\w./])(/[\w./-]*)')

# `find . -exec /usr/bin/unzip {} \; -delete` is a RELATIVE, cwd-scoped delete (`find .`) whose
# ONLY absolute path is the executable `-exec`/`-execdir` is told to run, not a delete target.
# Measured over-reject on the real dataset before this exclusion: destructive drops went 164 ->
# 160 after adding this exclusion (+4 rows recovered, usable pool U 5189 -> 5193) — rows like this
# one and `find . -exec /bin/echo ...` were wrongly flagged destructive purely because the exec
# binary's own absolute path happened to start with a system-root prefix. Exclude exactly that
# token (the first absolute path right after -exec/-execdir) from the target-path scan below.
_EXEC_BINARY_TOKEN_RE = re.compile(r'(?:-exec|-execdir)\s+(/\S+)')


def _has_system_root_target(text: str) -> bool:
    exec_binaries = set(_EXEC_BINARY_TOKEN_RE.findall(text))
    for tok in _ABS_PATH_RE.findall(text):
        if tok in exec_binaries:
            continue
        t = tok.rstrip('/') or '/'
        if t in _SYSTEM_ROOTS:
            return True
        if any(t.startswith(r + '/') for r in _SYSTEM_ROOTS if r != '/'):
            return True
    return False


def is_destructive_bigcode(response: str) -> bool:
    """True if response is unsafe TRAINING CONTENT (see module note above), not a sandbox
    safety claim. Positive on: kill/killall/pkill, crontab, shutdown/reboot/halt/poweroff,
    mkfs*, dd, fdisk, a redirect into /dev/sd*|/dev/nvme*, or an rm/chmod/chown/chgrp/-delete/
    xargs-rm whose path scope is `/` or a system root dir. Relative/cwd-scoped deletes are kept
    (`_has_system_root_target` excludes the executable token right after `-exec`/`-execdir`, so
    e.g. `find . -exec /usr/bin/unzip {} \\; -delete` is correctly kept — measured fix, +4 rows
    recovered on the real 5996-row bash pool that were wrongly flagged before this exclusion).

    KNOWN, NOT FIXED (documented per guard review rather than chased further): `~`-relative and
    `$HOME`-relative rm/chmod targets are not recognized as safe-relative (only bare relative
    paths and literal `/abs/...` are distinguished; `~/foo` is currently treated as "no absolute
    path found" -> kept, which is actually fine since `~` expands to the user's own home, not a
    literal system root); `shred`/`truncate` (destructive-in-effect data wipers) and `| sh` /
    `| bash` (pipe-to-shell of arbitrary content) are not in `_DESTRUCTIVE_HEAD_RE` at all. Left
    undropped rather than guessed at — no measured count of how many rows this affects."""
    if _DESTRUCTIVE_HEAD_RE.search(response):
        return True
    if _DEV_REDIRECT_RE.search(response):
        return True
    _has_delete_family = bool(_DESTRUCTIVE_PATH_CMD_RE.search(response)) or '-delete' in response
    if _has_delete_family and _has_system_root_target(response):
        return True
    return False

# Unfilled placeholders from the upstream instruction->command corpus: a bare $VAR-style token
# (all-caps, the convention such corpora use for "fill this in"), or an angle-bracket
# <placeholder>, or the literal "/path/to/..." stand-in path — none are concrete, independently
# runnable commands. EXCLUDED: standard shell/awk variables a real script legitimately
# references by name (these are not "unfilled" — they resolve at run time to real values).
_PLACEHOLDER_VAR_ALLOWLIST = {
    'HOME', 'PWD', 'OLDPWD', 'USER', 'PATH', 'SHELL', 'RANDOM', 'IFS', 'TMPDIR', 'HOSTNAME',
    'UID', 'EUID', 'LANG', 'PPID', 'SECONDS', 'LINENO',
    'NF', 'NR', 'FS', 'OFS', 'RS', 'ORS', 'FNR',
}
_DOLLAR_VAR_RE = re.compile(r'\$\{?([A-Z][A-Z0-9_]*)\}?')
_ANGLE_PLACEHOLDER_RE = re.compile(r'<[^<>\s]+>')


def _has_unfilled_placeholder(response: str) -> bool:
    for m in _DOLLAR_VAR_RE.finditer(response):
        if m.group(1) not in _PLACEHOLDER_VAR_ALLOWLIST:
            return True
    if _ANGLE_PLACEHOLDER_RE.search(response):
        return True
    if '/path/to' in response.lower():
        return True
    return False


def is_usable_bigcode_bash(response: str) -> bool:
    """True if a bigcode response both looks like bash (is_bash_response) AND is a concrete,
    non-interactive, non-destructive, filesystem-local command safe to TRAIN ON (not a claim
    about safety to EXECUTE — see is_destructive_bigcode's note)."""
    if not is_bash_response(response):
        return False
    if _UNSAFE_HEAD_RE.search(response):
        return False
    if is_destructive_bigcode(response):
        return False
    if _has_unfilled_placeholder(response):
        return False
    return True


def usable_bigcode_rows(ds):
    """Apply is_usable_bigcode_bash + exact-pair dedupe to a raw instruction-kind miner dataset.

    Both combine_datasets._blend_bigcode_bash (what actually gets blended) and
    sft_env_config's U-sizing (how big the blend is ALLOWED to be) call this SAME function, so
    the "how many usable rows exist" count can never drift between the two call sites — one of
    the guard's FAILs was exactly that kind of drift (a hand-duplicated filter in two places).

    Returns (pairs, dropped_dup, dropped_filtered) where pairs is a list of
    (instruction, response) in original dataset order."""
    seen = set()
    pairs = []
    dropped_dup = 0
    dropped_filtered = 0
    for ex in ds:
        instr = (ex.get('instruction') or '').strip()
        resp = (ex.get('response') or '').strip()
        if not instr or not resp:
            continue
        key = (instr.lower(), resp.lower())
        if key in seen:
            dropped_dup += 1
            continue
        if not is_usable_bigcode_bash(resp):
            dropped_filtered += 1
            continue
        seen.add(key)
        pairs.append((instr, resp))
    return pairs, dropped_dup, dropped_filtered

def build_execute_only_example(query: str, gold: str, variant: int = 0):
    """One execute-only tool-calling turn (no submit) for a bigcode instruction row.

    Same {messages, tools_json} shape as build_examples so the combined adapter sees a
    consistent schema. NOT used by combine_datasets._blend_bigcode_bash any more (E10 switched
    that caller to build_examples + synthesize_observation, the execute->observe->submit shape
    below) -- kept only because the E8 local measurement tools
    (.claude/Environment/experiments/e8/tools/smoke_flatness_combine.py,
    experiments/obs1) still call it directly to synthesize UNRELATED per-env dummy self-gen rows
    for their own smoke inputs, not for the bigcode blend path this file actually ships."""
    thought = _THOUGHT1_VARIANTS[variant % len(_THOUGHT1_VARIANTS)]
    tools_json = json.dumps(_intercode_tools())
    system = {'role': 'system', 'content': INTERCODE_TOOL_SYSTEM_PROMPT}
    u1 = {'role': 'user', 'content': _user_prompt(query, 'No commands have been executed yet.', 1)}
    a1 = _assistant_tool_msg(INTERCODE_EXECUTE_TOOL_NAME, {'command': gold}, thought)
    return {'messages': [system, u1, a1], 'tools_json': tools_json}

# --- E10: synthesize a STOP-anchor observation for a bigcode gold command, WITHOUT executing it --
# The bigcode dataset (intercode_bigcode_combined_12k) is instruction->gold-command pairs with NO
# associated filesystem, so there is no real observation to read off. Two ways to get one anyway:
# (a) clone princeton-nlp/intercode and actually run the command against its filesystem (this is
# what 5CRjuf did, and the deep-dive measured its marginal lift at +0.006 vs a wins-only competitor
# with no fs at all, inside the noise floor) -- GREY ZONE per the compliance rubric (copies eval
# infrastructure) and already rejected by team memory; NOT done here. (b) guess a plausible-looking
# fake observation from tokens in the command's own arguments -- tried first in this arm, REVERTED
# after guard review (see below). (c) return '' unconditionally -- what this function does now.
#
# WHY EMPTY, not a content-guess (guard follow-up, 2026-09): an earlier revision of this function
# built a per-head-verb fake output (fake filenames for find/ls, fake matching lines for grep, a
# fake count for wc/du, etc). Guard measured that 48% of the pairs it sampled produced an
# observation INCONSISTENT with the actual command (e.g. `find /usr -name temp` -> a fabricated
# `./file1.csv` -- a bigcode `find` over a real filesystem does not return made-up CSV names, and
# nothing in the command's own text said "csv"). That is a correctness problem specifically BECAUSE
# the observation text is never scored or read by anything -- the only reason this text exists is
# to give build_examples' turn-2 history string SOMETHING to react to, so the paired submit() call
# reads as "after seeing an observation, stop" rather than "stop unconditionally, ignore
# observations" (a policy multi-turn eval trajectories would then fight against). A wrong-looking
# fake observation does not serve that goal better than an empty one -- it serves it WORSE, by
# training the model to associate arbitrary, command-independent text with "now submit," which is
# closer to teaching "ignore the observation" than a consistently empty one is. The empty string is
# also literally correct for the largest real branch (mutation/redirect commands, and the general
# "don't know" default this function already fell back to for the majority of commands that don't
# match a handful of recognized heads) -- so returning '' unconditionally removes a false-content
# risk with no compensating benefit, since content was never scored either way.
#
# Teaching the model to react to REAL, content-correct observations remains the self-gen rows' job
# (`build_examples`, called from `main()` below with `obs1` from `run_gold()`, which actually
# executes the self-generated task's own gold command in a real shell during data generation and
# truncates its REAL stdout/stderr to OBS_TRUNCATE_CHARS) -- self-gen rows are not touched by this
# arm and are unaffected by this change. Bigcode rows only ever needed to teach the STOPPING
# behavior (execute -> [something] -> submit), not observation comprehension, since bigcode has no
# real filesystem to comprehend in the first place.
#
# Deterministic (trivially -- there is only one possible output) and truncated to
# OBS_TRUNCATE_CHARS for symmetry with run_gold() above, even though '' never needs truncating.
def synthesize_observation(gold: str) -> str:
    """STOP-ANCHOR for a bigcode gold command's turn-2 history -- always ''. See the module note
    above this function for why: the bigcode dataset has no real filesystem, so any non-empty guess
    at its output risks being wrong (guard measured 48% of guessed observations were inconsistent
    with their command on an earlier revision of this function), and the observation's TEXT is never
    scored by anything -- only its presence (something happened, now submit) matters. `gold` is
    accepted (kept for call-site compatibility with a hypothetical future content-aware revision and
    so `build_examples(instr, gold, synthesize_observation(gold), variant)` reads the same as it did
    before this change) but is not otherwise used; NEVER executed, NEVER read from a filesystem."""
    return ''

def main() -> None:
    import time
    _t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_path', required=True, help='DatasetDict save dir')
    p.add_argument('--task_id_min', type=int, default=1)
    p.add_argument('--task_id_max', type=int, default=200)
    p.add_argument('--num_tasks', type=int, default=0, help='If >0, randomly sample this many task ids from the range (seeded) instead of using all of them.')
    p.add_argument('--variants', type=int, default=4, help="Paraphrased thought variants per task (same bash actions). Multiplies InterCode rows (~200 tasks x variants x 2 turns) so it isn't under-represented vs the larger card datasets in a combined adapter.")
    p.add_argument('--target_samples', type=int, default=0, help='Early-stop cap on total rows (0 = no cap = tasks x variants x 2). Emitted variant-outer so the cap still spans ALL task ids / filesystems (eval samples 1..200 uniformly) — a task-ordered cap would starve the high ids. Lets the combined-adapter allocator size InterCode to a fair, precise share.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--recovery-frac', type=float, default=0.0, help='Fraction of (task,variant) that ALSO emit a multi-turn RECOVERY trajectory (near-miss -> correct gold -> submit) teaching recover-from-failure (the InterCode single-turn 10.5 -> multi-turn 58.7 lever). 0 (default) = OFF = current single-shot recipe. The failed near-miss is history-only, never a train target (STeP by construction). GATE ON EVAL before raising above 0.')
    p.add_argument('--refine-frac', type=float, default=0.5, help='Fraction of (task,variant) WITH a shell-verified verbose sibling command (selfgen_bash generate_tasks verbose_gold) that emit the 3-row REFINE shape (execute-verbose, execute-minimal, submit, mirroring all 3 of the eval system-prompt few-shots) INSTEAD OF the 2-row single-shot shape (build_examples). Default 0.5: half of the eligible ~27/45 templates instances get the refine shape per variant, so the eval demonstrated behaviour is a substantial, not overwhelming, share of self-gen rows -- keeping the single-shot answer-directly skill trained too, since not every real task needs refining. Tasks whose template has no verbose sibling (vgold is None) always get the 2-row shape, unaffected by this flag. Deterministic per (query,variant) via crc32, independent of --recovery-frac own hash.')
    p.add_argument('--history-thoughts', choices=['auto', 'on', 'off'], default='auto', help="L4 (see the note above _history_step): 'off' drops every `Thought N:` history line and sets assistant `content` to '' for bases whose chat template discards content next to tool_calls (Llama-3.2-3B/3.1-8B, per the deep-dive's offline apply_chat_template measurement) -- their eval-time history can never contain a Thought line either, since the fine-tuned model can never emit a `content` the validator reads as truthy. 'on' (and the default 'auto', standalone-invocation compatible with HEAD) keeps the existing Thought-line/content behaviour unchanged. sft_env_config.py passes 'off' only when get_model_architecture(model_path) == 'llamaforcausallm', 'on' otherwise -- this flag never inspects the base itself.")
    args = p.parse_args()
    _include_thoughts = args.history_thoughts != 'off'
    from datasets import Dataset, DatasetDict
    try:
        from envs.selfgen_bash import generate_tasks
    except ImportError:
        from selfgen_bash import generate_tasks
    # COMPLIANT SOURCE (docs/miner.md "Rules for environment tournaments": do NOT
    # bundle a dataset in the image; SFT is allowed on self-generated data). Bash
    # tasks are BUILT + shell-verified AT TRAIN-TIME here — never the NL2Bash
    # benchmark instances, never a file baked into the image. Volume tracks the old
    # task range so the combined-adapter allocator sizes intercode the same.
    n_tasks = args.num_tasks if args.num_tasks > 0 else (args.task_id_max - args.task_id_min + 1)
    prepared = generate_tasks(n_tasks, seed=args.seed)
    failed = 0
    if not prepared:
        raise RuntimeError('No self-generated Bash trajectories produced — the container '
                           'needs bash + coreutils for shell verification.')
    print(f'[intercode] {len(prepared)} self-generated shell-verified Bash tasks '
          f'(target {n_tasks}); NO bundled data, NO NL2Bash', flush=True)
    # Emit variant-outer: every task gets variant 0 before any task gets variant 1,
    # so an optional --target_samples cap still spans the FULL task range / all four
    # filesystems (eval samples task ids 1..200 uniformly). A task-ordered cap would
    # starve the high ids and never train the later fs sets.
    target = max(0, args.target_samples)
    # Cap variants at the unique paraphrase-pool size: build_examples indexes the pools
    # by variant % len(pool), so any variant >= pool size would emit a BYTE-IDENTICAL row
    # (the old bug: variants up to 40 against 10-entry pools → ~75% duplicate rows). Capping
    # keeps every emitted row distinct. Raise the pools to grow unique InterCode volume.
    _max_unique = min(len(_THOUGHT1_VARIANTS), len(_THOUGHT2_VARIANTS))
    _eff_variants = min(max(1, args.variants), _max_unique)
    if _eff_variants < max(1, args.variants):
        print(f'[intercode] capping variants {args.variants} -> {_eff_variants} (unique paraphrase pool size) to avoid duplicate rows', flush=True)
    conversations = []
    stop = False
    n_refine_rows = 0
    n_direct_rows = 0
    for v in range(_eff_variants):
        # I2 (2026-09-18, L9, mutating/silent-gold) / I3 (2026-09-19, real near-miss): `prepared`
        # rows are now the 7-tuple (query, gold, obs1, vgold, vobs, expected_paths, near_miss) --
        # `expected_paths` is selfgen_bash.generate_tasks's ADDITIVE 6th element for a
        # state-diff-verified mutating task (None for every pre-existing read-only task), and
        # `near_miss` is the ADDITIVE 7th element, a REAL-EXECUTED (command, observation) pair for
        # a read-only task's recovery trajectory (None for mutating tasks and for any read-only
        # task whose near-miss could not be executed or unexpectedly succeeded). Neither
        # `expected_paths` nor `near_miss` is consumed directly in this loop body below (near_miss
        # is read from the per-task tuple binding a few lines down, where build_recovery_example is
        # called); both only need to be unpacked so the tuple shape matches. For a mutating task,
        # obs1 == "" BY
        # CONSTRUCTION (selfgen_bash's mutating gate requires empty stdout) and vgold is always
        # None (no verbose sibling is coherent for a silent command) -- so `use_refine` below
        # is always False for these rows and build_examples emits exactly
        # execute(mutation) -> Observation N: "" -> submit(), with NOTHING appended after the
        # silent command: build_examples always emits precisely 2 rows (execute, submit),
        # there is no code path here that inspects obs1's content to decide whether to add a
        # listing/verification turn. `_history_step`'s `f'Observation {n}: {obs}'` with obs=""
        # is byte-identical in shape to the validator's own `f"Observation {turn}: {observation}"`
        # (evaluators/intercode.py:869) for an empty observation -- neither side special-cases it.
        for (query, gold, obs1, vgold, vobs, expected_paths, near_miss) in prepared:
            # E12 L2: per (task, variant), tasks WITH a shell-verified verbose sibling
            # (vgold not None) roll a deterministic crc32 hash to decide 3-row REFINE vs
            # 2-row single-shot; tasks without one always get the 2-row shape. Different
            # hash salt ('refine' vs recovery_frac's bare key) so the two independent mixes
            # do not correlate.
            use_refine = bool(vgold) and (zlib.crc32(f'{query}|{v}|refine'.encode()) % 1000) < int(args.refine_frac * 1000)
            if use_refine:
                conversations.extend(build_refine_example(query, vgold, vobs, gold, obs1, v, _include_thoughts))
                n_refine_rows += 3
            else:
                conversations.extend(build_examples(query, gold, obs1, v, _include_thoughts))
                n_direct_rows += 2
            # I3: near_miss is None for every mutating task (expected_paths is not None) and for
            # any read-only task whose near-miss candidate could not be executed or unexpectedly
            # succeeded (selfgen_bash.generate_tasks) -- recovery is skipped in that case rather
            # than falling back to a synthesized/guessed near-miss.
            if (args.recovery_frac > 0 and near_miss is not None
                    and (zlib.crc32(f'{query}|{v}'.encode()) % 1000) < int(args.recovery_frac * 1000)):
                conversations.extend(build_recovery_example(query, gold, obs1, near_miss, v, _include_thoughts))
            if target and len(conversations) >= target:
                stop = True
                break
        if stop:
            break
    # If the target exceeds the unique capacity (200 tasks x eff_variants x 2 turns),
    # up-weight InterCode by repeating the FULL unique set — clean full-diversity repetition
    # (every task/variant), NOT the old bug that silently re-emitted only variants 0..9.
    # Honors the allocator's intercode target (its high-leverage INDIVIDUAL share) without
    # byte-dup waste, while the larger pool gives 2x the unique paraphrases the old code had.
    if target and conversations and (len(conversations) < target):
        base = list(conversations)
        while len(conversations) < target:
            conversations.extend(base)
    if target and len(conversations) > target:
        conversations = conversations[:target]
    if not conversations:
        raise RuntimeError('No InterCode trajectories generated (empty after variant expansion).')
    _capped = f', capped at target {target}' if target and stop else ''
    # "0 rejected" rather than "0 failed": under quiet mode (shared.quiet_mode) any line
    # containing "fail" is a keep-marker and gets redacted to "[intercode] fail", which reads
    # as an alarm on a clean run. Nonzero rejections still say "failed" so the alarm fires when
    # something genuinely went wrong; the all-clear line no longer contains that word.
    _rej = f'{failed} failed' if failed else '0 rejected'
    print(f'[intercode] built {len(conversations)} tool-calling examples from {len(prepared)} tasks ({_rej}){_capped} in {time.time() - _t0:.0f}s '
          f'[refine_rows={n_refine_rows} direct_rows={n_direct_rows} pre-truncate] '
          f'[history_thoughts={args.history_thoughts} -> include_thoughts={_include_thoughts}]', flush=True)
    dataset = Dataset.from_list(conversations)
    splits = dataset.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
    dd = DatasetDict({'train': splits['train'], 'validation': splits['test']})
    dd.save_to_disk(args.output_path)
    print(f'[intercode] saved: {args.output_path}', flush=True)
if __name__ == '__main__':
    main()
