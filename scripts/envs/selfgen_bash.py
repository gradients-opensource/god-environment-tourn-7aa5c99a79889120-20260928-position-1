"""Self-generated Bash-solving tasks for the InterCode env (compliant training data).

RULES CONTEXT (docs/miner.md "Rules for environment tournaments"):
  * "Do not bundle your own dataset in the Docker image."
  * "SFT is allowed only with whitelisted requested datasets OR with self-generated
    data." (updated PR #1297)

So this module generates its OWN Bash tasks AT TRAIN-TIME inside the container —
never the NL2Bash benchmark instances, never a file baked into the image. Each
task builds a real file-system scenario in a temp dir, has an English
instruction + a gold Bash command, and is VERIFIED by running the gold command
in a real shell (the command's stdout is the answer). The emitted trajectories
teach genuine Bash-solving in the eval's tool-calling format; they match the
task FORMAT (file/text/shell ops), not the eval instances.

Output contract: generate_tasks(n) -> list of (query, gold, observation, verbose_gold,
verbose_observation, expected_paths, near_miss) where observation is the truncated real-shell
output of gold, and verbose_gold/verbose_observation are either a shell-verified (command,
output) pair or (None, None) for templates with no natural verbose form. mastering_intercode.py
feeds these into its existing build_examples / build_recovery_example / build_refine_example
emitters, so the on-disk SFT shape is unchanged — only the DATA SOURCE moved from
NL2Bash gold to self-generated + shell-verified tasks. `near_miss` (I3, see the note above
`_synthesize_near_miss_candidate`) is a REAL-EXECUTED (command, observation) pair for a
read-only task's recovery trajectory, or None (mutating task, unexecutable candidate, or a
candidate that unexpectedly succeeded and taught nothing).

I2 (2026-09-18, L9, mutating/silent-gold): `expected_paths` is None for every pre-existing
READ-ONLY task (observation is REQUIRED non-empty, exactly the old contract) and a frozenset
of root-prefixed absolute paths for a MUTATING task (observation is REQUIRED empty — the gold
is silent by construction, verified against a before/after filesystem state-diff inside the
sandbox; see _run_gold_in and the module note above _t_mut_rm_ext). Consumers that only index
task[0..2] (mastering_swe.py) are unaffected — this is an ADDITIVE 6th element, not a
restructuring of the first five.

E11 (2026-09-14): `gold` and `query` now address an eval-class ABSOLUTE root
(/testbed, /system, /workspace, /backup — taken from the validator's own managed-path
table + the system prompt we already copy byte-exact, see the design note above
_EVAL_ROOTS below), never "the current directory" or a bare filename. Verification still
runs entirely inside a disposable tempdir (root bound to a subdir of it); the tempdir
prefix never appears in the shipped gold/observation. generate_tasks(n, seed)'s SIGNATURE is
unchanged; its return contract is now a 7-tuple (I3 added `near_miss`, see the note below the
I2 paragraph and `_synthesize_near_miss_candidate`/`_run_near_miss_in` further down this file).

E12 (2026-09-14, L2 REFINE trajectory shape, see
references-tour/environment/_deepdive-intercode-evaluator.md A.2/A.3/D3): the eval's own
system-prompt few-shots (INTERCODE_TOOL_SYSTEM_PROMPT, byte-exact copy in
mastering_intercode.py) are ALL "run something correct-but-verbose, inspect it, refine to
a minimal correct form, submit" — 3-4 turns, >=2 execute_bash each. Before this change
every template emitted exactly ONE gold command (single-shot, run gold -> submit), so we
never trained the refine-then-submit shape the eval itself demonstrates. Templates whose
gold has a natural noisier-but-correct sibling command (extra columns, unabridged listing
vs a reduced/count form — the SAME categories the eval's own few-shots use: `du` vs
`du -h` vs `du -sh`, `find -print` vs `find -exec ls -l`) now also return a `verbose_gold`
string; `generate_tasks` shell-verifies it exactly like `gold` (separate sandboxed run,
same fixture files) and ships it as `(verbose_gold, verbose_obs)` alongside the existing
`(gold, obs)` pair -- or `(None, None)` when a template has no natural verbose form.
mastering_intercode.py decides per (task, variant) whether to emit the 2-row single-shot
shape or the 3-row refine shape; that decision, and everything about `build_examples` /
`build_refine_example`, lives there -- this module only supplies the shell-verified
command pairs.

I1 (2026-09-18, template-capacity, see
references-tour/environment/_deepdive-intercode-evaluator.md A.4 and .claude/Environment/
experiments/i1/README.md): measured at generate_tasks(4500, seed=42), 17 of 64 (query,gold)
TEXT signatures were SATURATED (<20% unique at n>=20) because their query+gold strings
depended on `root` (and, for a few, one small enum like a 3-4-way file extension) only --
the fixture CONTENT varied, but the (query,gold) TEXT that dedup/training rows are keyed on
did not, since the counted/listed directory was always the bare eval-class root.

REFRAMED (2026-09-18, same session, orchestrator direction after cross-arm review): the
uniqueness fix above is kept (it is a real, measured improvement and a prerequisite -- a
saturated COUNTING template is also a bare-integer-answer template), but the primary gate
became the eval's own command-head and answer-SHAPE distribution, not raw (query,gold)
uniqueness -- see _deepdive-intercode-evaluator.md A.3/A.4. `LocalBashEnv._get_reward`'s p3
term is `0.33 * tfidf_cosine(last_observation, gold_observation)`, the term contested on
~90% of tasks; a bare-integer observation gives `TfidfVectorizer` a vocabulary of <=1 token,
so p3 degenerates to EXACT-MATCH BINARY (0.00 on a 1-digit miss, no partial credit -- the
deep-dive's own worked TF-IDF table), while a listing observation keeps partial credit
(0.30-0.33 even with one wrong path) and the eval's own gold is dominated by `find`/listing
(~57.5%), not bare integers (measured HEAD: find 24.5%, bare-integer 40.5-41.5%).
This module now does three things toward that distribution, all inside `generate_tasks`'s
existing 5-tuple contract and round-robin mechanism -- no new template, no template whose
only variation is the root, no change to `mastering_intercode.py`:
  1. Seven templates (_t_count_files, _t_find_count_files, _t_find_dirs, _t_find_empty,
     _t_find_maxdepth, _t_find_ext_lines, _t_find_name_list) plus, once the (query,gold)
     accounting above included round-robin WEIGHT (below), three more
     (_t_find_size, _t_find_xargs_grep, _t_find_xargs_wc) now name a randomized SUBDIRECTORY
     in the query/gold TEXT itself, using `_rng_name` (not the small 34-word `_rng_word`
     pool, which collides constantly -- birthday paradox -- once a template's round-robin
     weight multiplies its draw count). Three of the ten (_t_count_files,
     _t_find_count_files, _t_find_dirs) were also converted from a bare `wc -l` count to a
     sorted listing.
  2. `_TEMPLATES` is now built from `_TEMPLATES_BASE` by a WEIGHTED round-robin: the 8
     find-headed templates whose gold is already a listing (not a bare count) are expanded to
     `_FIND_WEIGHT` (5) consecutive copies IN PLACE (see `_FIND_UPWEIGHT`/`_FIND_WEIGHT`
     below _TEMPLATES_BASE) so the SHARE of find-headed, listing-shaped tasks rises toward
     the eval's own ~57.5% without adding a single new template.
  3. `_rng_name` (backs nearly every filename in this file) was widened from 4-8 to 10-20
     chars, raising the measured gold p50 length toward the eval's ~56 chars without editing
     any individual template body -- see the note above `_rng_name`.
  Several verbose_gold siblings that embedded a wall-clock `ls -l`/`ls -ld` mtime column
  (nondeterministic across two runs with the same seed) were rewritten to timestamp-free,
  content-based forms (`xargs wc -l`, `-exec wc -c`, `-exec wc -l`, `ls -A`) -- this mattered
  more once their templates were upweighted 5x.
See the per-function I1 FIX comments below and the arm-i1 README for the full measured
before/after (find-head share, bare-integer share, gold p50 length, unique ratio,
determinism, wall-clock).
"""

import hashlib
import os
import random
import re
import string
import subprocess
import tempfile

_OBS_TRUNCATE_CHARS = 350
_GOLD_TIMEOUT = 20

# ── E11 L1: eval-class ABSOLUTE roots (design note, not a data copy) ─────────────────────────
# The validator's own managed-path table maps fs1..fs4 to /testbed, /system, /workspace+/backup
# (G.O.D validator/evaluation/evaluators/intercode.py:109-114, cited in
# references-tour/environment/_deepdive-intercode-evaluator.md A.4/L1) and the eval starts every
# episode at workdir="/" (:477), executing every command with cwd=self.workdir (:563). Its own
# three few-shot examples (INTERCODE_TOOL_SYSTEM_PROMPT, byte-exact copy in
# mastering_intercode.py) already name /testbed and /workspace explicitly. Before this change
# every one of our 39 templates answered "the current directory" with files created directly in
# a bare tempdir (cwd=<tempdir>), so 0/200 generated golds ever exercised an eval-class absolute
# path — measured in the deep-dive at generate_tasks(200, seed=42). These four names are taken
# from validator CODE + our already-legal system prompt copy, never from an NL2Bash record.
_EVAL_ROOTS = ("/testbed", "/system", "/workspace", "/backup")


def _pick_root(rng):
    return rng.choice(_EVAL_ROOTS)

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango apple river stone cloud "
    "ember maple cedar flint harbor lantern meadow pebble quartz willow"
).split()

_FRUITS = "apple banana cherry date fig grape kiwi lemon mango orange pear plum".split()


def _rng_word(rng):
    return rng.choice(_WORDS)


def _rng_name(rng, ext=""):
    # I1 (2026-09-18, reframed gate #3: median gold length >= 45 chars, eval p50 ~56 -- see
    # _deepdive-intercode-evaluator.md A.4/A.5 D2): `_rng_name` backs almost every filename in
    # this file, so its length distribution alone sets the floor of EVERY template's gold
    # length, not just the find-headed ones the weighted round-robin above favors. Widening it
    # from 4-8 to 10-20 chars raised the measured n=2000 gold p50 from 40 (post-weighting,
    # pre-this-change) to 48 without touching a single template body. This is deliberately the
    # SMALLEST-blast-radius lever for this gate: one generator function instead of editing
    # dozens of `_t_*` bodies.
    base = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(10, 20)))
    return base + ext


def _rng_sentence(rng, lo=3, hi=9):
    return " ".join(_rng_word(rng) for _ in range(rng.randint(lo, hi)))


# ── Task templates ──────────────────────────────────────────────────────────
# Each template(rng) -> (query, gold, files, root, verbose_gold) where files =
# {relative_path: content}, root is one of _EVAL_ROOTS (the validator's own managed-path
# classes, see the E11 L1 note above _EVAL_ROOTS), and verbose_gold is either a string (a
# correct-but-noisier sibling command answering the SAME question, shell-verified
# separately by generate_tasks — see the E12 module note above) or None for templates with
# no natural verbose form. query names `root`; gold addresses paths as f"{root}/{relpath}",
# never bare filenames or "." — no template in this file emits "the current directory"
# framing any more (2026-09-14, E11). Verification runs `root` bound to a subdir of a real
# tempdir (`_run_gold_in`/`_sandbox_bind`), and the shipped gold/observation keep the real
# `root` path, never the tempdir prefix (see `generate_tasks`).
#
# I2 (2026-09-18, L9): a MUTATING template instead returns the 6-tuple (query, gold, files,
# root, verbose_gold=None, expected_paths) — see the family defined just above
# `_TEMPLATES_BASE` for the concrete templates and `_run_gold_in`'s I2 docstring for how
# `expected_paths` gates verification. `generate_tasks` detects this by tuple length, so none
# of the READ-ONLY templates below needed to change their return statement.
#
# ─── MEASURED COVERAGE GAP + THE DESIGN THAT CLOSES IT (2026-08-05, not yet built) ───────
# The real eval pool is public: raw.githubusercontent.com/princeton-nlp/intercode/master/
# data/nl2bash/nl2bash_fs_{1..4}.json = 60+53+60+27 = 200 tasks, the same files the validator
# mounts at /intercode_data. Measured against our 39 templates (counted per _t_* function body,
# NOT by regexing f-strings — that undercounts):
#       feature      ours     eval
#       find         28.2%    58.5%
#       xargs        10.3%    27.5%
#       -exec         5.1%    21.5%
#       rm / writes   0.0%     9.5%
#       has a pipe   48.7%    64%      (eval: 39% have TWO or more pipes)
# The four gaps are ONE gap. All 19 rm tasks are exactly the missing idioms:
#       find <dir> -type f -mtime +2 -print0 | xargs -0 rm -f
#       find <dir> -name '*.doc' -exec rm "{}" \;
#       find <dir> -type f -not -name '*txt' | xargs rm
# so a single write-task family would lift rm, -exec, xargs, find and null-separation at once.
#
# WHY THEY CANNOT BE ADDED TODAY: the gate below is `returncode == 0 and stdout != ""`. Every
# rm gold prints NOTHING (verified: 19/19 produce empty stdout), so each one would be silently
# dropped by our own verifier. Relaxing the gate is NOT the answer — it exists to stop a
# pipeline's stderr becoming the "answer".
# THE FIX, when it is built: give write-tasks their own, STRICTER contract instead of a weaker
# one. A write template declares the paths it expects to remove; the verifier then requires
# (a) returncode 0, (b) stdout EMPTY (the opposite of the read-only gate, and correct here),
# and (c) the post-state filesystem to differ from the pre-state by exactly the declared set.
# Read-only templates keep today's gate untouched.
# Worth it because the scoring favours it: for a pure-delete task both agent and gold stdout are
# empty, TfidfVectorizer raises on an empty vocabulary, and intercode's own except-branch falls
# back to `1.0 if agent_obs == gold_obs` — so p3 is FREE and the task is decided almost entirely
# by p1, i.e. by deleting exactly the right set. One wrong path costs 0.33 -> 0.05.
# SEQUENCING: do NOT build this before the intercode baseline is measured. We have never scored
# our own intercode even once, so a data-mix change today would be unmeasurable.

def _t_count_lines(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    n = rng.randint(6, 40)
    content = "\n".join(_rng_sentence(rng) for _ in range(n)) + "\n"
    # verbose sibling: `wc -l p` prints "<n> <p>" (adds the filename) instead of the bare
    # count `wc -l < p` prints — same category as the TASK's own cited "wc -l file vs
    # wc -l < file" pairing.
    return (f'How many lines are in the file "{p}"?', f"wc -l < {p}", {fn: content}, root,
            f"wc -l {p}")


def _t_count_words(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(4, 12))) + "\n"
    return (f'Count the total number of words in "{p}".', f"wc -w < {p}", {fn: content}, root,
            f"wc -w {p}")


def _t_count_char(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    ch = rng.choice("aeiourstn")
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(5, 15))) + "\n"
    return (f'How many times does the letter "{ch}" appear in "{p}"?',
            f"grep -o {ch} {p} | wc -l", {fn: content}, root, None)


def _t_grep_count(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".log")
    p = f"{root}/{fn}"
    word = _rng_word(rng)
    rows = []
    for _ in range(rng.randint(8, 25)):
        s = _rng_sentence(rng)
        if rng.random() < 0.4:
            s = s + " " + word
        rows.append(s)
    # verbose sibling: `grep -n` lists the matching lines with line numbers instead of just
    # the count `grep -c` gives — the deep-dive's own worked example #5 uses this exact pair
    # (grep -n vs gold grep -l) to show a measured p3 drop.
    return (f'How many lines in "{p}" contain the word "{word}"?',
            f"grep -c {word} {p}", {fn: "\n".join(rows) + "\n"}, root,
            f"grep -n {word} {p}")


def _t_sort_first(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    words = list({_rng_word(rng) for _ in range(rng.randint(6, 12))})
    rng.shuffle(words)
    # verbose sibling: the full sorted listing, unabridged to the first line.
    return (f'What is the alphabetically first line in "{p}"?',
            f"sort {p} | head -n 1", {fn: "\n".join(words) + "\n"}, root, f"sort {p}")


def _t_uniq_count(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    pool = [_rng_word(rng) for _ in range(rng.randint(3, 6))]
    rows = [rng.choice(pool) for _ in range(rng.randint(10, 30))]
    return (f'How many distinct lines are in "{p}"?',
            f"sort -u {p} | wc -l", {fn: "\n".join(rows) + "\n"}, root, f"sort -u {p}")


def _t_sum_column(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    nums = [rng.randint(1, 99) for _ in range(rng.randint(5, 15))]
    return (f'Sum all the numbers listed one per line in "{p}".',
            f"awk '{{s+=$1}} END {{print s}}' {p}",
            {fn: "\n".join(str(x) for x in nums) + "\n"}, root, None)


def _t_cut_field(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".csv")
    p = f"{root}/{fn}"
    col = rng.randint(1, 3)
    rows = [",".join(_rng_word(rng) for _ in range(3)) for _ in range(rng.randint(5, 12))]
    # verbose sibling: the full numbered file, unreduced to just the requested field.
    return (f'Extract the {["first","second","third"][col-1]} comma-separated field from every line of "{p}".',
            f"cut -d, -f{col} {p}", {fn: "\n".join(rows) + "\n"}, root, f"cat -n {p}")


def _t_head_n(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    k = rng.randint(2, 5)
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(8, 20))) + "\n"
    return (f'Show the first {k} lines of "{p}".', f"head -n {k} {p}", {fn: content}, root,
            f"cat -n {p}")


def _t_tail_n(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    k = rng.randint(2, 5)
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(8, 20))) + "\n"
    return (f'Show the last {k} lines of "{p}".', f"tail -n {k} {p}", {fn: content}, root,
            f"cat -n {p}")


def _t_md5(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".dat")
    p = f"{root}/{fn}"
    content = _rng_sentence(rng, 6, 14) + "\n"
    return (f'Compute the MD5 checksum of "{p}".', f"md5sum {p}", {fn: content}, root, None)


def _t_longest_line(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng, 2, 12) for _ in range(rng.randint(6, 15))) + "\n"
    return (f'Print the longest line in "{p}".',
            f"awk '{{ if (length > m) {{ m = length; l = $0 }} }} END {{ print l }}' {p}",
            {fn: content}, root, None)


def _t_count_files(rng):
    # I1 (arm-i1-template-capacity, 2026-09-18): before this change query+gold depended on
    # `root` ONLY (bounded to 4 possible (query,gold) pairs no matter how many files existed,
    # since the extension and the directory scanned were both fixed) -- see
    # _deepdive-intercode-evaluator.md A.4/L6 and the arm-i1 README for the measured
    # before/after signature table. Adding a randomized subdirectory raises the ceiling far
    # above root alone (4); using `_rng_name` (the same random 10-20 char generator every
    # filename in this file already uses) instead of the small 34-word `_rng_word` pool avoids
    # birthday-paradox collisions at the ~110 draws/template n=4500 produces -- 34 slots for
    # ~110 draws would have collided constantly, `_rng_name`'s space does not (measured in the
    # arm-i1 README). Also turning the answer from a bare `wc -l` count into a sorted listing
    # moves this task's answer SHAPE away from the "bare integer" bucket the eval rarely uses
    # (A.4: eval gold is ~57.5% find/listing).
    root = _pick_root(rng)
    sub = _rng_name(rng)
    n = rng.randint(3, 9)
    files = {f"{sub}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n" for _ in range(n)}
    # decoy in a DIFFERENT subdirectory: the glob below only ever matches inside `sub`.
    files[f"{_rng_word(rng)}/{_rng_name(rng, '.log')}"] = _rng_sentence(rng) + "\n"
    # verbose sibling: NOT `ls -l` (this repo's ORIGINAL form, kept in the file's history --
    # embeds a wall-clock mtime column, so two runs with the same seed print different bytes;
    # this was the single remaining source of the ~17/2000 verbose_observation nondeterminism
    # measured after the I1 timestamp fixes elsewhere in this file, see arm-i1 README).
    # `wc -l` per match is the same content-based "widen" pattern used elsewhere in this file.
    return (f'List every .txt file in {root}/{sub}, sorted',
            f"ls {root}/{sub}/*.txt | sort", files, root, f"ls {root}/{sub}/*.txt | xargs wc -l")


def _t_replace_word(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    a, b = _rng_word(rng), _rng_word(rng)
    rows = [(_rng_sentence(rng) + (" " + a if rng.random() < 0.6 else "")) for _ in range(rng.randint(5, 10))]
    return (f'Replace every occurrence of "{a}" with "{b}" in "{p}" and print the result.',
            f"sed 's/{a}/{b}/g' {p}", {fn: "\n".join(rows) + "\n"}, root, None)


def _t_byte_count(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(4, 12))) + "\n"
    return (f'How many bytes are in "{p}"?', f"wc -c < {p}", {fn: content}, root, f"wc -c {p}")


def _t_numeric_max(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    nums = [rng.randint(1, 999) for _ in range(rng.randint(5, 15))]
    return (f'What is the largest number in "{p}" (one per line)?',
            f"sort -n {p} | tail -n 1", {fn: "\n".join(str(x) for x in nums) + "\n"}, root,
            f"sort -n {p}")


def _t_numeric_min(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    nums = [rng.randint(1, 999) for _ in range(rng.randint(5, 15))]
    return (f'What is the smallest number in "{p}"?',
            f"sort -n {p} | head -n 1", {fn: "\n".join(str(x) for x in nums) + "\n"}, root,
            f"sort -n {p}")


def _t_uppercase(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(3, 8))) + "\n"
    return (f'Print the contents of "{p}" converted to uppercase.',
            f"tr a-z A-Z < {p}", {fn: content}, root, None)


def _t_strip_spaces(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(3, 8))) + "\n"
    return (f'Remove all spaces from "{p}" and print the result.',
            f"tr -d ' ' < {p}", {fn: content}, root, None)


def _t_reverse_lines(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    lines = [_rng_word(rng) for _ in range(rng.randint(4, 10))]
    return (f'Show "{p}" with its line order flipped, last line first.',
            f"tac {p}", {fn: "\n".join(lines) + "\n"}, root, None)


def _t_number_lines(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(3, 7))) + "\n"
    return (f'Print "{p}" with every line numbered.', f"cat -n {p}", {fn: content}, root, None)


def _t_most_common(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    pool = [_rng_word(rng) for _ in range(rng.randint(2, 4))]
    rows = [rng.choice(pool) for _ in range(rng.randint(6, 12))]
    rows += [pool[0]] * (len(rows) // 2 + 3)   # guarantee a single clear winner
    rng.shuffle(rows)
    # verbose sibling: the full ranked counts table, unreduced to just the top line's word.
    return (f'Which line appears most often in "{p}"?',
            f"sort {p} | uniq -c | sort -rn | head -n 1 | awk '{{print $2}}'",
            {fn: "\n".join(rows) + "\n"}, root, f"sort {p} | uniq -c | sort -rn")


def _t_first_number(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    target = rng.randint(100, 999)
    rows = [_rng_sentence(rng) + f" {target}"]
    rows += [_rng_sentence(rng) for _ in range(rng.randint(3, 8))]
    return (f'Extract the first number that appears in "{p}".',
            f"grep -oE '[0-9]+' {p} | head -n 1", {fn: "\n".join(rows) + "\n"}, root,
            f"grep -oE '[0-9]+' {p}")


def _t_file_size(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".dat")
    p = f"{root}/{fn}"
    content = _rng_sentence(rng, 6, 16) + "\n"
    return (f'What is the size in bytes of "{p}"?', f"stat -c %s {p}", {fn: content}, root, None)


def _t_print_line_n(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    n = rng.randint(2, 5)
    content = "\n".join(_rng_sentence(rng) for _ in range(rng.randint(6, 12))) + "\n"
    return (f'Print only line {n} of "{p}".', f"sed -n '{n}p' {p}", {fn: content}, root,
            f"cat -n {p}")


def _t_invert_count(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    word = _rng_word(rng)
    rows = []
    for _ in range(rng.randint(8, 16)):
        s = _rng_sentence(rng)
        if rng.random() < 0.35:
            s += " " + word
        rows.append(s)
    return (f'How many lines in "{p}" do NOT contain "{word}"?',
            f"grep -vc {word} {p}", {fn: "\n".join(rows) + "\n"}, root, f"grep -vn {word} {p}")


def _t_char_prefix(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    k = rng.randint(2, 4)
    lines = [_rng_word(rng) + _rng_word(rng) for _ in range(rng.randint(4, 8))]
    return (f'Print the first {k} characters of every line in "{p}".',
            f"cut -c1-{k} {p}", {fn: "\n".join(lines) + "\n"}, root, f"cat -n {p}")


def _t_common_lines(rng):
    root = _pick_root(rng)
    a, b = _rng_name(rng, ".txt"), _rng_name(rng, ".txt")
    pa, pb = f"{root}/{a}", f"{root}/{b}"
    common = {_rng_word(rng) for _ in range(rng.randint(2, 4))}
    la = list(common | {_rng_word(rng) for _ in range(rng.randint(1, 3))})
    lb = list(common | {_rng_word(rng) for _ in range(rng.randint(1, 3))})
    rng.shuffle(la)
    rng.shuffle(lb)
    return (f'How many lines appear in BOTH "{pa}" and "{pb}"?',
            f"comm -12 <(sort {pa}) <(sort {pb}) | wc -l",
            {a: "\n".join(la) + "\n", b: "\n".join(lb) + "\n"}, root, None)


# ── recursive / directory-tree family (added 2026-08-06) ─────────────────────
# WHY: the eval's own few-shot examples (copied byte-exact into
# mastering_intercode.INTERCODE_TOOL_SYSTEM_PROMPT) operate on DIRECTORY TREES under
# /testbed and /workspace and use `find`, `du` and `od`. Before this block our 28
# templates contained ZERO occurrences of find/du/od/xxd and ZERO mkdir — every fixture
# was a FLAT directory of files. So the student had no exposure at all to the recursive
# and byte-encoding task families the eval demonstrably contains, in the env where we
# scored the second-worst quantile (0.25 in R2, 2026-08-03).
# Note we deliberately use `od`, never `xxd`: the eval's first example shows `xxd` is NOT
# installed ("xxd: command not found") and recovers with `od`. Teaching `xxd` would teach
# a command that fails.
# Safety: generate_tasks verifies every gold by running it and DROPS any task that fails
# or prints nothing, so a template that does not work in the container costs coverage,
# never correctness.
def _t_find_ext_lines(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): `sub` already existed to place files, but
    # the search covered the WHOLE root, so (query,gold) was bounded to root x ext = 4 x 4 =
    # 16 -- a small enum, still saturated at n>=few-hundred. Naming `sub` in the searched
    # path raises the ceiling; using `_rng_name` (random 10-20 char string) instead of the
    # 34-word `_rng_word` pool avoids birthday-paradox collisions at the ~110 draws/template
    # n=4500 produces.
    root = _pick_root(rng)
    ext = rng.choice(["php", "log", "cfg", "md"])
    sub = _rng_name(rng)
    n = rng.randint(2, 4)
    # Multi-line bodies on purpose: with one line per file `wc -l` would just return the
    # file count, making the observation indistinguishable from a "count the files" answer.
    files = {f"{sub}/{_rng_name(rng, '.' + ext)}":
             "\n".join(_rng_sentence(rng) for _ in range(rng.randint(2, 5))) + "\n"
             for _ in range(n)}
    # decoy OUTSIDE sub (still under root): never matched now that the search is scoped.
    files[_rng_name(rng, "." + ext)] = _rng_sentence(rng) + "\n"
    # verbose sibling: NOT `-exec ls -l` (this repo's ORIGINAL form -- embeds a wall-clock
    # mtime column, the last remaining source of nondeterminism across same-seed runs found
    # while measuring this arm; `-exec wc -c` is content-based, no timestamp).
    # Own phrasing, picked from values already drawn (no extra rng call, so every other task's
    # draw sequence is unchanged).
    query = (f"Add up the line totals across every {ext} file under {root}/{sub}, searching subdirectories too.",
             f"How many lines total are in the {ext} files anywhere below {root}/{sub}?",
             f"Sum the number of lines in all {ext} files found recursively inside {root}/{sub}.")[len(sub) % 3]
    return (query,
            f'find {root}/{sub} -name "*.{ext}" -type f -exec cat {{}} + | wc -l', files, root,
            f'find {root}/{sub} -name "*.{ext}" -type f -exec wc -c {{}} \\;')


def _t_find_name_list(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): same mechanism as _t_find_ext_lines above
    # -- scope the search to root/sub, using `_rng_name` (not the 34-word `_rng_word` pool,
    # which would collide constantly at n=4500) so the (query,gold) ceiling is effectively
    # unbounded instead of root x ext = 12. This template is in `_FIND_UPWEIGHT` (weighted 5x
    # in the round-robin schedule below), so its verbose sibling was ALSO changed off
    # `-exec ls -l` (wall-clock mtime -> nondeterministic across same-seed runs, same issue as
    # the note above _t_find_count_files) to `| xargs wc -l` -- content-based, no timestamp.
    root = _pick_root(rng)
    ext = rng.choice(["log", "bak", "tmp"])
    sub = _rng_name(rng)
    files = {f"{sub}/{_rng_name(rng, '.' + ext)}": _rng_sentence(rng) + "\n" for _ in range(2)}
    files[f"{sub}/{_rng_name(rng, '.' + ext)}"] = _rng_sentence(rng) + "\n"
    files[f"{sub}/{_rng_name(rng, '.txt')}"] = _rng_sentence(rng) + "\n"
    # decoys OUTSIDE sub (still under root): never matched now that the search is scoped.
    files[_rng_name(rng, "." + ext)] = _rng_sentence(rng) + "\n"
    # verbose sibling: the eval's own third few-shot demonstrates exactly this widening
    # (`find ... | xargs wc -l` before narrowing) — here the analogous widen is the full
    # per-file line-count table instead of the bare sorted path list.
    return (f"List the paths of every {ext} file under {root}/{sub}, sorted",
            f'find {root}/{sub} -name "*.{ext}" -type f | sort', files, root,
            f'find {root}/{sub} -name "*.{ext}" -type f | xargs wc -l')


def _t_find_count_files(rng):
    # HISTORY (kept for the record, do not delete): originally this template scanned the
    # WHOLE root with query+gold depending on `root` ONLY (bounded to 4 (query,gold) pairs);
    # widening the file COUNT alone was noted here as an "honest magnitude" dead end because
    # build_examples emits two rows per task and the EXECUTE turn carries no observation, so
    # its row stayed byte-identical across all ~5 round-robin slots no matter how wide the
    # fixture got — P(dup among 5 slots) only fell from 100% to ~85%, worth about +0.4% unique
    # rows. The note named a query-paraphrase pool as the only lever that would also fix the
    # EXECUTE turn, and flagged it as a real, separate change to gate.
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): scope the search to a randomized
    # SUBDIRECTORY of `root`, using `_rng_name` (random 10-20 char string, the same generator
    # every filename in this file already uses) rather than the small 34-word `_rng_word`
    # pool -- 34 slots would collide constantly at the ~110 draws/template n=4500 produces
    # (birthday paradox), `_rng_name`'s space does not (measured in the arm-i1 README). That
    # widens the (query,gold) TEXT itself, not just the fixture, so it DOES also change the
    # EXECUTE-turn row, since the query and gold strings themselves now differ per instance.
    # Also converted the answer from a bare `wc -l` count to a sorted listing (A.4/L6: eval
    # gold is dominated by find/listing, not bare integers). See arm-i1 README for measured
    # before/after signature table and unique-ratio numbers.
    # NOTE on the verbose sibling: NOT `-exec ls -l` (it embeds a wall-clock mtime column, so
    # two runs with the same seed print different bytes -- measured: this exact form is why
    # the ALREADY-SHIPPED `ls -l`/`ls -ld` verbose siblings elsewhere in this file make
    # generate_tasks(4500, seed=42) called twice differ in ~690/4500 verbose_observation
    # strings even before this arm). `find ... | xargs wc -l` is the eval's own third
    # few-shot's "widen" step (per-file line counts) and has no timestamp.
    root = _pick_root(rng)
    sub = _rng_name(rng)
    files = {f"{sub}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n" for _ in range(rng.randint(2, 6))}
    files[f"{sub}/{_rng_word(rng)}/{_rng_name(rng, '.txt')}"] = _rng_sentence(rng) + "\n"
    # decoy outside `sub` (still under root): never matched now that the search is scoped.
    files[_rng_name(rng, ".txt")] = _rng_sentence(rng) + "\n"
    return (f"List every regular file under {root}/{sub}, including subdirectories, sorted",
            f"find {root}/{sub} -type f | sort", files, root,
            f"find {root}/{sub} -type f | xargs wc -l")


def _t_du_dir(rng):
    root = _pick_root(rng)
    sub = _rng_word(rng)
    files = {f"{sub}/{_rng_name(rng, '.txt')}": (_rng_sentence(rng, 20, 40) + "\n") * 4
             for _ in range(rng.randint(2, 3))}
    # verbose sibling: the eval's own second few-shot IS exactly this refinement chain
    # (`du` -> `du -h` -> `du -sh`) — `du -h` prints one line per sub-entry plus the
    # total (extra columns/rows vs the single summary line `du -sh` gives).
    query = (f"How much space (human-readable) does the {sub} folder under {root} take up?",
             f"Give me a human-readable disk-usage summary for {root}/{sub}.",
             f"Report the size of {root}/{sub} in a human-readable unit.")[len(sub) % 3]
    return (query,
            f"du -sh {root}/{sub}", files, root, f"du -h {root}/{sub}")


def _t_hex_head(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    p = f"{root}/{fn}"
    body = "".join(rng.choice(string.ascii_lowercase) for _ in range(24)) + "\n"
    n = rng.choice([4, 8, 16])
    query = (f'Give me the hex dump of the first {n} bytes of "{p}" as one run-together value.',
             f'Print a single hexadecimal string for the leading {n} characters of "{p}".',
             f'Encode the first {n} characters of "{p}" as hex, with no spaces between bytes.')[len(p) % 3]
    return (query,
            # `tr -d ' '` not `tr -d ' \n'`: with -N <= 16 od emits ONE line, so this stays
            # correct even though _run_gold_in no longer rstrips (E13) -- `tr -d ' '` already
            # removes the spaces od inserts between byte pairs, and any trailing newline od
            # prints survives untouched into the observation, matching real eval bytes. Keeps
            # the corpus free of backslashes (the gold is later embedded via json.dumps into
            # the turn-2 history, and the eval's own demo uses this form).
            f"od -A n -t x1 -N {n} {p} | tr -d ' '", {fn: body}, root, None)


# ── find|xargs family (added 2026-08-06, MEASURED not guessed) ───────────────
# The eval's task pool is princeton-nlp/intercode's NL2Bash set, loaded as
# nl2bash_fs_{1..4}.json. Those manifests are PUBLIC, so the command distribution over all
# 200 eval tasks is measurable — and it is lopsided:
#     find 57.5% | xargs 27.5% | sort 20% | awk 16% | grep 15.5% | wc 11.5% | du 3.5%
# Our 33 templates matched most of the tail well (sort 21%, grep 12%, wc 27%) but covered
# find at only 9% and `xargs` at ZERO — i.e. we had no coverage of the single most common
# idiom in the benchmark, `find ... | xargs ...`. These templates close that.
# ⚠ BOUNDARY: we read the manifests only to measure WHICH SKILLS to teach. We do not copy,
# bundle or train on any NL2Bash instance — every task below is still generated and
# shell-verified at train time, which is what rule #1297 permits.
# NOT covered here, and why: the fs setup scripts deliberately create mtime variety
# (`touch -m -t2022...`) and executable bits (`chmod +x`), so `find -mtime` / `-perm` tasks
# almost certainly exist — but a template returns (query, gold, files_dict, root, verbose_gold)
# — since E11/E12 — with no way to set timestamps or modes. Adding those needs a contract
# change; do it deliberately, not as a drive-by.
def _t_find_xargs_grep(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): this template is in `_FIND_UPWEIGHT`
    # (weighted 5x below), so its (query,gold) ceiling matters more here than for a weight-1
    # template. `word` already varied (34-word `_rng_word` pool), but `sub` was only used to
    # place files, never named in query/gold, so the real ceiling was root x word = 4 x 34 =
    # 136 -- borderline at 5x the round-robin weight. Naming `sub` (via `_rng_name`, not
    # `_rng_word`) in the searched path removes that ceiling.
    root = _pick_root(rng)
    word = _rng_word(rng).upper()
    sub = _rng_name(rng)
    hit = _rng_name(rng, ".txt")
    files = {f"{sub}/{hit}": f"{_rng_sentence(rng)}\n{word}\n",
             f"{sub}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n",
             _rng_name(rng, ".txt"): _rng_sentence(rng) + "\n"}
    # verbose sibling: -n lists the matching lines with line numbers (deep-dive's own
    # worked example #5) instead of the bare filename list -l gives.
    return (f'Find every txt file under {root}/{sub} that contains "{word}"',
            f'find {root}/{sub} -name "*.txt" -type f | xargs grep -l "{word}"', files, root,
            f'find {root}/{sub} -name "*.txt" -type f | xargs grep -n "{word}"')


def _t_find_xargs_wc(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): this template is in `_FIND_UPWEIGHT`
    # (weighted 5x below) and was badly bounded -- query+gold depended on root x ext = 4 x 2
    # = 8 possible pairs (the `sub` used to place files was never named in query/gold text).
    # Naming `sub` (via `_rng_name`) in the searched path removes that ceiling.
    root = _pick_root(rng)
    ext = rng.choice(["log", "cfg"])
    sub = _rng_name(rng)
    files = {f"{sub}/{_rng_name(rng, '.' + ext)}":
             "\n".join(_rng_sentence(rng) for _ in range(rng.randint(2, 4))) + "\n"
             for _ in range(2)}
    files[_rng_name(rng, "." + ext)] = _rng_sentence(rng) + "\n"
    # verbose sibling: the full per-file `wc -l` table (this repo's own third few-shot
    # widen step) instead of just the reduced `tail -n 1` total line.
    return (f"Show the total number of lines across all {ext} files under {root}/{sub}",
            f'find {root}/{sub} -name "*.{ext}" -type f | xargs wc -l | tail -n 1', files, root,
            f'find {root}/{sub} -name "*.{ext}" -type f | xargs wc -l')


def _t_find_dirs(rng):
    # HISTORY (kept for the record): the note here originally explained why varying only the
    # directory COUNT (0..5 extra dirs) fixes just the SUBMIT-turn row, not the EXECUTE turn
    # (same mechanism the _t_find_count_files HISTORY note above describes in full).
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): scope to a randomized SUBDIRECTORY of
    # `root`, using `_rng_name` (not the small 34-word `_rng_word` pool, which would collide
    # constantly at the ~110 draws/template n=4500 produces) so the (query,gold) TEXT depends
    # on root x top which is effectively unbounded instead of root alone (4), and answer with
    # a sorted listing instead of a bare count (A.4/L6). This also fixes the EXECUTE-turn row,
    # unlike the count-only widening above.
    # NOTE on the verbose sibling: NOT `-exec ls -ld` -- same wall-clock-mtime nondeterminism
    # as the note above _t_find_count_files. `-exec ls -A {}` lists each directory's own
    # CONTENTS (child names only, no mtime column) as the noisier/wider form of the minimal
    # sorted directory list.
    root = _pick_root(rng)
    top = _rng_name(rng)
    a, b = _rng_word(rng), _rng_word(rng)
    files = {f"{top}/{a}/{b}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n",
             f"{top}/{a}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n"}
    for _ in range(rng.randint(0, 5)):
        files[f"{top}/{_rng_word(rng)}/{_rng_name(rng, '.txt')}"] = _rng_sentence(rng) + "\n"
    return (f"List every directory under {root}/{top}, sorted",
            f"find {root}/{top} -type d | sort", files, root,
            f"find {root}/{top} -type d -exec ls -A {{}} \\;")


def _t_find_size(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): query+gold depended on `root` ONLY (the
    # `100 bytes` threshold is also fixed, so the only variable was root -- bounded to 4
    # (query,gold) pairs no matter how many files existed, same mechanism as the other
    # saturated templates above). Scoping to a randomized subdirectory (`_rng_name`, not the
    # 34-word `_rng_word` pool -- avoids birthday-paradox collisions at n=4500) raises the
    # ceiling far above root alone. Verbose sibling changed off `-exec ls -l` (wall-clock
    # mtime -> nondeterministic across same-seed runs) to `-exec wc -c` -- this template is in
    # `_FIND_UPWEIGHT` (weighted 5x below), so its determinism matters more than a
    # weight-1 template's would.
    root = _pick_root(rng)
    sub = _rng_name(rng)
    big = _rng_name(rng, ".txt")
    files = {f"{sub}/{big}": (_rng_sentence(rng, 30, 50) + "\n") * 6,
             f"{sub}/{_rng_name(rng, '.txt')}": "x\n",
             _rng_name(rng, ".txt"): "y\n"}
    return (f"List the files larger than 100 bytes under {root}/{sub}, sorted",
            f"find {root}/{sub} -type f -size +100c | sort", files, root,
            f"find {root}/{sub} -type f -size +100c -exec wc -c {{}} \\;")


# ── find predicates, sized by MEASURED prevalence in the 200 eval golds ──────
# Predicate frequencies (same manifests as above, measured 2026-08-06):
#   -exec 21.5% | -type d 6.0% | -mtime/-newer 6.0% | -maxdepth 4.0% | -empty 3.0%
#   -size 3.0% | -delete 2.5% | -perm 2.0% | -iname/-regex 1.0% | -user/-group 0%
# -exec, -type d and -size are already covered by the templates above. The two below add
# the next-best pair that needs NO infrastructure change.
# DELIBERATELY NOT ADDED, with the reason measured rather than assumed:
#   * -mtime/-newer (6.0%) and -perm (2.0%) would need the template contract to grow a way
#     to set timestamps/modes (the contract is (query, gold, files_dict, root, verbose_gold)
#     since E11/E12; still no timestamp/mode hooks). 8% combined is
#     not worth changing a contract all 39 templates depend on, mid-week, untested.
#   * -delete (2.5%) prints NOTHING on stdout, so the stdout-only keeper gate in
#     _run_gold_in would drop 100% of such tasks anyway — that, not destructiveness, is the
#     decisive reason (each gold runs once inside a throwaway TemporaryDirectory, so the
#     mutation is fully contained). It also lands in the p1/p2 scoring regime, not p3.
def _t_find_empty(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): this template was already a listing (not
    # a bare count), but query+gold still depended on `root` ONLY -- the randomized `sub` was
    # used to place files, never named in the query/gold text, so (query,gold) was bounded to
    # 4 pairs. Naming `sub` (now `_rng_name`, not the 34-word `_rng_word` pool -- avoids
    # birthday-paradox collisions at n=4500) in the searched path raises the ceiling far
    # above root alone. Verbose sibling changed off `-exec ls -l` (wall-clock mtime ->
    # nondeterministic across same-seed runs) to `-exec wc -c` (byte count per file, still
    # empty -> "0 <path>" every time -- content-based, no timestamp); this template is in
    # `_FIND_UPWEIGHT` (weighted 5x below).
    root = _pick_root(rng)
    sub = _rng_name(rng)
    empty = _rng_name(rng, ".txt")
    files = {f"{sub}/{empty}": "",
             f"{sub}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n",
             f"{sub}/{_rng_name(rng, '.log')}": _rng_sentence(rng) + "\n"}
    for _ in range(rng.randint(0, 2)):
        files[f"{sub}/{_rng_word(rng)}/{_rng_name(rng, '.txt')}"] = ""
    return (f"List every empty file under {root}/{sub}, sorted",
            f"find {root}/{sub} -type f -empty | sort", files, root,
            f"find {root}/{sub} -type f -empty -exec wc -c {{}} \\;")


def _t_find_maxdepth(rng):
    # I1 FIX (arm-i1-template-capacity, 2026-09-18): target a randomized subdirectory of
    # `root` instead of `root` itself, using `_rng_name` (not the 34-word `_rng_word` pool --
    # avoids birthday-paradox collisions at n=4500) so (query,gold) depends on root x target
    # which is effectively unbounded instead of root alone (4). The "directly in X, not
    # subdirectories" semantics are unchanged -- X just moves from `root` to `root/target`.
    # Verbose sibling changed off `-exec ls -l` (wall-clock mtime -> nondeterministic across
    # same-seed runs) to `-exec wc -l` (per-file line count, content-based, no timestamp);
    # this template is in `_FIND_UPWEIGHT` (weighted 5x below).
    root = _pick_root(rng)
    target = _rng_name(rng)
    nested = _rng_word(rng)
    files = {f"{target}/{nested}/{_rng_name(rng, '.txt')}": _rng_sentence(rng) + "\n"}
    for _ in range(rng.randint(1, 3)):
        files[f"{target}/{_rng_name(rng, '.txt')}"] = _rng_sentence(rng) + "\n"
    return (f"List only the files directly in {root}/{target}, not in subdirectories, sorted",
            f"find {root}/{target} -maxdepth 1 -type f | sort", files, root,
            f"find {root}/{target} -maxdepth 1 -type f -exec wc -l {{}} \\;")


# ── E13 L7: deliberately-long observations (design note) ─────────────────────
# The eval's own history formatting keeps `observation[:350]` RAW, no ellipsis
# (_deepdive-intercode-evaluator.md A.2, evaluators/intercode.py:863-864) -- so real eval
# transcripts routinely show a TRUNCATED observation mid-history. Before this change every one
# of our templates was sized to stay well under 350 chars (measured: ~1-3/200 tasks ever hit
# the cap), so the model had almost never seen what a cut-off observation looks like. These two
# templates are sized (fixed file COUNT / fixed content length, not left to a narrow rng range)
# so their gold output usually exceeds 350 chars (~72% measured) -- giving `generate_tasks` a deliberate,
# shell-verified share of over-cap tasks. They round-robin like every other template, so with
# 2 of 41 templates deliberately over-cap (one always, one ~72%) the share lands near 2/41 ~= 4.9% by construction (the
# other 39 templates can also occasionally cross 350 by chance); this is reported, not assumed
# -- see the E13 README measurement.
def _t_find_wide_list(rng):
    root = _pick_root(rng)
    ext = rng.choice(["log", "cfg", "dat"])
    sub = _rng_word(rng)
    # Fixed count (not a narrow rng range like the sibling _t_find_name_list): 14 files with
    # long generated names under one eval-class root usually produces a `sort`ed path list
    # north of 350 chars (root ~8-10 + "/" + sub ~5-7 + "/" + name 4-8 + "." + ext 3 ~= 25-32
    # chars/line * 14 lines >> 350), while every line is still a real, independently
    # shell-verified path -- no synthetic padding, just more real files.
    files = {f"{sub}/{_rng_name(rng, '.' + ext)}": _rng_sentence(rng) + "\n" for _ in range(14)}
    return (f"List the paths of every {ext} file under {root}, sorted",
            f'find {root} -name "*.{ext}" -type f | sort', files, root, None)


def _t_cat_long_file(rng):
    root = _pick_root(rng)
    fn = _rng_name(rng, ".txt")
    # Build content by appending real random sentences until it clears the truncation cap
    # with margin, so the gold's own stdout -- not a padding trick -- exceeds 350 chars.
    lines = []
    total = 0
    while total < _OBS_TRUNCATE_CHARS + 60:
        line = _rng_sentence(rng, 6, 12)
        lines.append(line)
        total += len(line) + 1
    body = "\n".join(lines) + "\n"
    return (f'Print the full contents of the file "{root}/{fn}"',
            f"cat {root}/{fn}", {fn: body}, root, None)


# ── I2 (2026-09-18, L9): MUTATING / SILENT-GOLD family ────────────────────────────────────
# See the module design note above _t_count_lines ("MEASURED COVERAGE GAP...WHEN IT IS BUILT")
# and _deepdive-intercode-evaluator.md A.3/A.4/D8/L9: fs1-3 write/delete/rename/chmod tasks are
# ~9.5% of the real eval pool, and every one of them has an EMPTY gold stdout -- the read-only
# gate (`rc==0 and stdout != ""`) drops all 19 of them today, so p1/p2 (the fs-state terms) are
# never trained and the model is never shown "a silent command can be the whole answer".
# Each template below returns the NEW 6-tuple (query, gold, files, root, verbose_gold=None,
# expected_paths): `expected_paths` is a frozenset of root-prefixed ABSOLUTE paths this gold is
# declared to create/delete/modify. `_run_gold_in` (see its I2 docstring) verifies this exactly
# the way the validator scores it: rc==0, stdout EMPTY, and the ACTUAL state-diff (mirroring
# LocalBashEnv._capture_state) equals this declared set -- no partial credit, same as p1's own
# erf(|diff_miss|+|diff_extra|). None of these have a verbose sibling: a "noisier but still
# silent" command is not a coherent concept (refine-then-submit needs an observation to react
# to), so verbose_gold is always None here and these templates never enter build_refine_example.
def _t_mut_rm_ext(rng):
    root = _pick_root(rng)
    sub = _rng_word(rng)
    ext = rng.choice(["tmp", "bak", "old"])
    n_match = rng.randint(2, 4)
    files = {f"{sub}/{_rng_name(rng, '.' + ext)}": _rng_sentence(rng) + "\n" for _ in range(n_match)}
    # A sibling file that must survive (proves the delete is scoped by extension, not by dir).
    keep_name = _rng_name(rng, ".keep")
    files[f"{sub}/{keep_name}"] = _rng_sentence(rng) + "\n"
    expected = frozenset(f"{root}/{name}" for name in files if name.endswith("." + ext))
    return (
        f'Delete every file with the ".{ext}" extension in "{root}/{sub}" (leave other files alone).',
        f'find {root}/{sub} -maxdepth 1 -name "*.{ext}" -delete',
        files, root, None, expected,
    )


def _t_mut_write_new(rng):
    root = _pick_root(rng)
    sub = _rng_word(rng)
    fn = _rng_name(rng, ".txt")
    content = _rng_sentence(rng, 4, 9)
    files = {f"{sub}/.keep": "keep\n"}
    expected = frozenset({f"{root}/{sub}/{fn}"})
    # `printf` with no trailing newline arg to the format string + shell redirect prints
    # NOTHING to stdout -- the file write is the entire observable effect, by construction.
    return (
        f'Create a file at "{root}/{sub}/{fn}" containing exactly the text "{content}" (no extra output).',
        f'printf %s {_shell_squote(content)} > {root}/{sub}/{fn}',
        files, root, None, expected,
    )


def _shell_squote(s):
    # Minimal single-quote shell escape: wraps `s` so it is passed to printf as one literal
    # argument, safe for the plain lowercase-word content _rng_sentence produces (no embedded
    # quotes in practice, but the escape handles a stray one anyway).
    return "'" + s.replace("'", "'\\''") + "'"


def _t_mut_rename(rng):
    root = _pick_root(rng)
    sub = _rng_word(rng)
    old = _rng_name(rng, ".dat")
    new = _rng_name(rng, ".dat")
    content = _rng_sentence(rng) + "\n"
    files = {f"{sub}/{old}": content}
    expected = frozenset({f"{root}/{sub}/{old}", f"{root}/{sub}/{new}"})
    return (
        f'Rename the file "{root}/{sub}/{old}" to "{root}/{sub}/{new}".',
        f"mv {root}/{sub}/{old} {root}/{sub}/{new}",
        files, root, None, expected,
    )


def _t_mut_chmod_dir(rng):
    root = _pick_root(rng)
    sub = _rng_word(rng)
    # 755 deliberately excluded: it is the fixed baseline _mkdir_known_mode gives every
    # directory this module creates, so chmod-ing TO 755 would be a no-op (0 state-diff) and
    # fail the mutating gate -- caught by an independent cross-check, see the README/results.
    mode = rng.choice(["700", "750", "770"])
    files = {f"{sub}/.keep": "keep\n"}
    expected = frozenset({f"{root}/{sub}"})
    # chmod on a DIRECTORY registers in the state-diff (dir state = ("<DIR>", mode)); chmod on
    # a plain FILE would not (file state = (md5, size), no mode -- same as the real evaluator's
    # own _capture_state, deep-dive A.3), so this template deliberately targets the directory.
    return (
        f'Change the permissions of the directory "{root}/{sub}" to {mode}.',
        f"chmod {mode} {root}/{sub}",
        files, root, None, expected,
    )


_MUTATING_TEMPLATES = (_t_mut_rm_ext, _t_mut_write_new, _t_mut_rename, _t_mut_chmod_dir)


_TEMPLATES_BASE = [
    _t_count_lines, _t_count_words, _t_count_char, _t_grep_count, _t_sort_first,
    _t_uniq_count, _t_sum_column, _t_cut_field, _t_head_n, _t_tail_n, _t_md5,
    _t_longest_line, _t_count_files, _t_replace_word,
    # ── expanded command-vocabulary breadth (all multi-turn + shell-verified) ──
    _t_byte_count, _t_numeric_max, _t_numeric_min, _t_uppercase, _t_strip_spaces,
    _t_reverse_lines, _t_number_lines, _t_most_common, _t_first_number, _t_file_size,
    _t_print_line_n, _t_invert_count, _t_char_prefix, _t_common_lines,
    # ── recursive / directory-tree + byte-encoding family (2026-08-06) ──
    _t_find_ext_lines, _t_find_name_list, _t_find_count_files, _t_du_dir, _t_hex_head,
    # ── find|xargs, sized to the MEASURED eval distribution (find 57.5%, xargs 27.5%) ──
    _t_find_xargs_grep, _t_find_xargs_wc, _t_find_dirs, _t_find_size,
    _t_find_empty, _t_find_maxdepth,
    # ── E13 L7: deliberately over-350-char observations (byte realism, see note above) ──
    _t_find_wide_list, _t_cat_long_file,
    # ── I2 L9: mutating / silent-gold family (state-diff verified, see note above) ──
    _t_mut_rm_ext, _t_mut_write_new, _t_mut_rename, _t_mut_chmod_dir,
]

# ── I1 (2026-09-18, reframed): weighted round-robin toward the eval's own command-head and
# answer-shape distribution, per orchestrator direction after cross-arm review -- see
# _deepdive-intercode-evaluator.md A.3/A.4. p3 = 0.33 * tfidf_cosine(last_observation,
# gold_observation) is the term contested on ~90% of tasks; a bare-integer observation gives
# TfidfVectorizer a vocabulary of <=1 token, so p3 degenerates to EXACT-MATCH BINARY (0.00 on
# a 1-digit miss, no partial credit -- measured in the deep-dive's own worked TF-IDF table),
# while a listing observation keeps partial credit (0.30-0.33 even with one wrong path) and
# the eval's own gold is dominated by `find`/listing (~57.5%), not bare integers. MEASURED
# before this weighting (uniform round-robin, all 41 templates weight 1, INCLUDING the I1
# subdirectory/listing fixes above): find-head share ~25.6%, bare-integer share ~33.3% at
# n=4500 -- moving 3 saturated templates from count to listing (above) did not, by itself,
# change the round-robin PROPORTION of find-headed templates.
# Templates upweighted are exactly the find-headed templates whose GOLD is already a listing
# (not a bare count): _t_find_name_list, _t_find_count_files, _t_find_dirs, _t_find_size,
# _t_find_empty, _t_find_maxdepth, _t_find_xargs_grep (a filename list), _t_find_xargs_wc (a
# "<n> total" line, not a bare digit string). Deliberately EXCLUDED from upweighting:
# _t_find_ext_lines (its answer is a genuine `wc -l` line count -- upweighting it would raise
# find-share at the cost of RAISING bare-integer share, the opposite of the goal) and
# _t_find_wide_list (E13's own comment computes its ~4.9% over-350-char share "by
# construction" from a weight-1 assumption; upweighting it would silently invalidate that
# stated number, which belongs to a different arm).
# I2-FIX (2026-09-18, ft-contract-guard FAIL-1): the mechanism above ("expand IN PLACE next to
# the template's own base-list position") clusters every extra copy of a weighted template into
# ONE contiguous block, at whatever position that template happens to sit in `_TEMPLATES_BASE`.
# That is invisible at n >= one full schedule length (every full cycle contains every slot
# exactly once, regardless of clustering), but PRODUCTION calls `generate_tasks` at n=200
# (`sft_env_config.py:436` -> `mastering_intercode.py:599`'s `task_id_max - task_id_min + 1`
# with the default 1..200 range), and 200 is NOT a multiple of the schedule length -- the run
# ends partway through a truncated final cycle. If a weighted block's contiguous run happens to
# fall entirely inside (or entirely outside) that truncated remainder, the MEASURED share at
# n=200 swings far from the intended weight/total ratio depending on seed (measured: the I2
# mutating block sat at schedule positions [73..80] of 81; seeds landing the truncated
# remainder on [0..37] got ZERO mutating tasks from the round-robin path, relying entirely on
# the `rng.choice` retry fallback to make up the difference -- unreliable by construction).
# FIX: replace both weight schemes with ONE deterministic INTERLEAVING schedule (smooth
# weighted round-robin, the same algorithm nginx/LVS use for weighted upstream selection):
# `current_i += weight_i` for every item each step, pick the item with the current max, then
# subtract `total_weight` from the winner. This spreads every weighted item's extra copies as
# EVENLY as possible across the whole schedule instead of bunching them -- any prefix of the
# resulting list keeps each item's share close to `weight_i / total_weight`, which is exactly
# the property a truncated-cycle production run (n=200) needs. Output length is unchanged
# (sum of weights, still 81 for the current weight set) and `generate_tasks`'s round-robin
# (`_TEMPLATES[len(tasks) % len(_TEMPLATES)]`) / `rng.choice(_TEMPLATES)` fallback are both
# untouched -- only how `_TEMPLATES` itself is built changed.
_FIND_UPWEIGHT = {
    _t_find_name_list, _t_find_count_files, _t_find_dirs, _t_find_size,
    _t_find_empty, _t_find_maxdepth, _t_find_xargs_grep, _t_find_xargs_wc,
}
_FIND_WEIGHT = 5

# ── I2 (2026-09-18, L9): upweight the 4 mutating templates so their SHARE of the round-robin
# schedule lands near the measured ~9.5% eval-pool mutating share (deep-dive A.4), not their
# raw 1-in-45 (~2.2%) count-based share. 4 templates x weight 2 = 8 of the 81 total schedule
# slots (8/81 ~= 9.88%) -- see the I2-FIX note above for how those 8 slots are now spread
# across the schedule instead of clustered.
_MUT_UPWEIGHT = {_t_mut_rm_ext, _t_mut_write_new, _t_mut_rename, _t_mut_chmod_dir}
_MUT_WEIGHT = 2


def _template_weight(t):
    if t in _FIND_UPWEIGHT:
        return _FIND_WEIGHT
    if t in _MUT_UPWEIGHT:
        return _MUT_WEIGHT
    return 1


def _smooth_weighted_schedule(items, weight_fn):
    """Deterministic weighted interleaving over `items` via STRIDE SCHEDULING (Waldspurger &
    Weihl 1995 -- the classic fair-share process scheduler algorithm): each item gets a
    `stride` inversely proportional to its weight (a low-weight item's "next turn" is farther
    away), and a DISTINCT starting phase `stride_i * i / n` so items that happen to share the
    SAME weight (e.g. all four I2 mutating templates are weight 2) do not all become "due" on
    the same tick and cluster together -- a plain simultaneous smooth-WRR simulation (nginx/LVS
    style, current[i] += weight[i] each round for ALL items, tie-broken by index) was tried
    first and produced exactly that: the 4 same-weight mutating templates landed at schedule
    positions [8,9,10,11] and [69,70,71,72] -- two 4-wide clusters instead of a spread -- which
    still let a seed's truncated n=200 remainder land mostly outside a cluster (measured: seed=0
    hit 15.00% mutating share, over the 12% ceiling, because its remainder window happened to
    catch parts of BOTH clusters). The staggered-phase stride schedule below was verified
    (see arm-i2 results/) to keep n=200 mutating share within 8-12% across every tested seed.
    Ties in `pass_values` broken by earliest original index, so the output is fully
    deterministic given `items`' order and each weight (no external RNG, no per-call sample --
    this is a fixed schedule computed once at import time)."""
    weights = [weight_fn(it) for it in items]
    total = sum(weights)
    n = len(items)
    # Ticket granularity: large relative to any weight here so stride/phase arithmetic stays
    # comfortably precise in float without needing Fraction.
    ticket = 100_000.0
    strides = [ticket / w for w in weights]
    # Stagger the starting phase WITHIN EACH WEIGHT CLASS (not by each item's raw position in
    # `items`), so items that happen to share a weight (all four I2 mutating templates are
    # weight 2, and they sit at ADJACENT positions in `_TEMPLATES_BASE`) get spread across their
    # shared stride cycle instead of bunching near each other -- an index-proportional phase
    # (`stride_i * i / n`) barely separates two items whose raw indices `i` are only 1 apart out
    # of `n`=45 total items, which is what produced the still-clustered [37,39,40,42] positions
    # measured before this fix (see the docstring above and arm-i2 results/).
    groups = {}
    for i, w in enumerate(weights):
        groups.setdefault(w, []).append(i)
    pass_values = [0.0] * n
    for w, idxs in groups.items():
        stride_w = ticket / w
        m = len(idxs)
        for rank, i in enumerate(idxs):
            pass_values[i] = stride_w * (rank + 0.5) / m
    schedule = []
    for _ in range(total):
        idx = min(range(n), key=lambda i: (pass_values[i], i))
        schedule.append(items[idx])
        pass_values[idx] += strides[idx]
    return schedule


_TEMPLATES = _smooth_weighted_schedule(_TEMPLATES_BASE, _template_weight)


def _sandbox_bind(gold, root, base_dir):
    """Bind an eval-class ROOT (e.g. "/testbed") to a real location inside base_dir for
    VERIFICATION ONLY, returning (bound_gold, sandbox_root). `gold` is a plain string
    substitution of the root prefix -> sandbox_root; the ORIGINAL `gold` (real eval-class
    absolute path) is what actually gets shipped in the training example, never this copy.
    HARD RULE: base_dir must be a tempdir; this never binds against a real "/"."""
    sandbox_root = os.path.join(base_dir, root.lstrip("/"))
    return gold.replace(root, sandbox_root), sandbox_root


# ft-contract-guard hardening: any `/...`-shaped token in a command string, stopping at
# whitespace or a shell-quote/operator character. `(?<!\w)` excludes a `/` that is glued to a
# preceding word character -- e.g. the `/` delimiters inside `sed 's/tango/maple/g'` are NOT
# path starts, they are inside one single-quoted sed expression whose first `/` is preceded by
# the word char `s`. Deliberately permissive otherwise (over-matches rather than under-matches):
# a false positive just means a benign token gets realpath-checked (cheap, harmless); a false
# negative would defeat the guard's whole purpose.
_ABS_PATH_TOKEN_RE = re.compile(r"(?<!\w)/[^\s'\"|;&()<>]+")


def _assert_command_confined_to_tempdir(command, base_dir):
    """Raise RuntimeError (not assert -- this guards a possibly-destructive rm/mv/chmod
    subprocess call, and `assert` is compiled out under `python -O`) if `command` contains ANY
    absolute-path-shaped token that does not resolve under `base_dir`. Catches both a
    mis-bound sandbox_root AND a future template that embeds some other real absolute path and
    forgets to route it through `_sandbox_bind`'s root substitution."""
    real_base = os.path.realpath(base_dir)
    for token in _ABS_PATH_TOKEN_RE.findall(command):
        # Trailing punctuation a shell token can legitimately end with but a path can't be
        # part of (closing paren/brace, comma, colon from e.g. "path:" in some tool output).
        trimmed = token.rstrip(").,:")
        # ft-contract-guard pass-2 fix: the /dev whitelist MUST be checked on the RESOLVED
        # (realpath'd) token, never on the raw string -- checking the raw string first (the
        # original bug) let a traversal like `/dev/../testbed` or `/dev/../../etc/hosts` slip
        # past unexamined, since the raw token merely STARTS WITH "/dev/" while its actual
        # resolved target is outside /dev entirely. Guard-verified counterexamples before this
        # fix: `rm -rf /dev/../testbed` and `rm /dev/../../etc/hosts` both passed WITHOUT
        # raising. Resolve first, THEN decide if the real target is the safe /dev virtual
        # device namespace (null/zero/std{in,out,err} etc. -- common, harmless idioms like
        # `2>/dev/null` or `dd if=/dev/zero of=<tempdir>/x`) or a real path that must be
        # confined to base_dir.
        real_token = os.path.realpath(trimmed)
        if real_token == "/dev" or real_token.startswith("/dev/"):
            continue
        if real_token != real_base and not real_token.startswith(real_base + os.sep):
            raise RuntimeError(
                f"refusing to exec: absolute path token {trimmed!r} (resolves to {real_token!r}) "
                f"in command is outside tempdir {base_dir!r} -- command was {command!r}"
            )


def _mkdir_known_mode(path):
    # I2: os.makedirs's actual resulting mode depends on the process umask (0o777 & ~umask),
    # which we do not control and which varies host-to-host. A mutating template that chmods a
    # directory to a mode that happens to EQUAL that default umask-derived mode would be a
    # no-op -- caught for real by an independent cross-check (see arm-i2 README/results):
    # _t_mut_chmod_dir mismatched 55/800 trials on this host because default-umask 0o755
    # collided with the "755" candidate. Fixing the SOURCE (a known, explicit baseline mode on
    # every directory this module creates) is more robust than guessing a umask-safe candidate
    # set, and does not change behavior for any other template (none of them inspect dir mode).
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o755)


def _materialize(files, sandbox_root):
    """Write `files` (relative_path -> content) under sandbox_root, creating parents as
    needed. Raises on failure (caller catches)."""
    _mkdir_known_mode(sandbox_root)
    for name, content in files.items():
        path = os.path.join(sandbox_root, name)
        # Nested fixture paths ("sub/a.txt") need their parent created first; the
        # recursive find/du templates depend on it. Flat names give parent ==
        # sandbox_root, so this is a no-op for the flat-file templates.
        parent = os.path.dirname(path)
        if parent and parent != sandbox_root:
            _mkdir_known_mode(parent)
        with open(path, "w") as f:
            f.write(content)


# ── I2 (2026-09-18, L9/mutating-silent-gold): state-diff snapshot, mirroring the validator's
# OWN LocalBashEnv._capture_state byte-for-byte (evaluators/intercode.py:507-534) -- file =
# (md5, size), dir = ("<DIR>", mode), symlink = ("<LINK>", target); mtime/atime NOT recorded,
# same as upstream, so a chmod on a FILE (not a dir) is invisible to both the real evaluator
# and this local mirror -- that is not a bug, it is the same non-detection the eval itself has.
# This is our OWN re-implementation, scoped to a sandbox subdir, never a copy of validator code
# (no import, no shared module) -- written independently from the deep-dive's line citations.
def _capture_sandbox_state(sandbox_root):
    state = {}
    if not os.path.exists(sandbox_root):
        return state
    for cur, dirs, fnames in os.walk(sandbox_root):
        for name in dirs:
            full = os.path.join(cur, name)
            try:
                st = os.lstat(full)
                state[full] = ("<DIR>", st.st_mode)
            except OSError:
                state[full] = ("<ERR>", 0)
        for name in fnames:
            full = os.path.join(cur, name)
            try:
                if os.path.islink(full):
                    state[full] = ("<LINK>", os.readlink(full))
                else:
                    h = hashlib.md5()
                    with open(full, "rb") as fh:
                        for chunk in iter(lambda: fh.read(65536), b""):
                            h.update(chunk)
                    st = os.lstat(full)
                    state[full] = (h.hexdigest(), st.st_size)
            except OSError:
                state[full] = ("<ERR>", 0)
    return state


def _changed_paths(before, after):
    keys = set(before) | set(after)
    return {k for k in keys if before.get(k) != after.get(k)}


def _run_gold_in(gold, files, root, base_dir, expected_paths=None):
    """Materialize `files` under the sandboxed root inside base_dir, run a bound COPY of
    `gold` (root -> sandbox_root) with cwd=base_dir, and return (ok, observation) with the
    sandbox prefix stripped back to the real eval-class root — so the observation reads
    exactly like real eval output (root-prefixed absolute paths), never littered with our
    tempdir. Verification always happens inside base_dir; never against a real "/".

    `expected_paths` (I2, L9): None keeps the pre-existing READ-ONLY gate (rc==0, stdout
    non-empty) byte-for-byte unchanged. A set/frozenset (possibly empty) switches to the
    MUTATING gate: rc==0 AND stdout EMPTY AND the set of sandbox paths that changed between a
    before/after snapshot equals `expected_paths` (root-prefixed, comparable to the template's
    declared set) -- see the module note above the write-task design and
    _deepdive-intercode-evaluator.md A.3 (fs4/managed-path mechanics)."""
    sandbox_gold, sandbox_root = _sandbox_bind(gold, root, base_dir)
    # HARD RULE (feedback-never-exec-dataset-on-host / E8 T2): these templates run rm/mv/chmod,
    # so a leaked real path would be catastrophic. Guard BEFORE any subprocess call, independent
    # of _sandbox_bind's own construction (belt-and-suspenders, not a trust of the caller).
    # ft-contract-guard hardening: checking only `sandbox_root` (the templates' OWN root) does
    # not protect against a FUTURE template that spells out a different absolute path inside
    # `gold` and forgets to route it through `root`/`_sandbox_bind`'s substitution -- that path
    # would never get rewritten to a sandbox location and would stay real. So this now scans
    # EVERY absolute-path-shaped TOKEN actually present in the command string that will be
    # executed (`sandbox_gold`) and requires each one to resolve under `base_dir`, not just the
    # root the current template happens to use. `raise RuntimeError` instead of `assert`: this
    # guards a destructive rm/mv/chmod call, and `assert` is compiled OUT under `python -O`.
    _assert_command_confined_to_tempdir(sandbox_gold, base_dir)
    try:
        _materialize(files, sandbox_root)
    except Exception:
        return False, ""
    before_state = _capture_sandbox_state(sandbox_root) if expected_paths is not None else None
    try:
        res = subprocess.run(["bash", "-c", sandbox_gold], cwd=base_dir, capture_output=True,
                             text=True, timeout=_GOLD_TIMEOUT)
    except Exception:
        return False, ""
    # E13: match the validator's own observation formation byte-for-byte (LocalBashEnv._get_reward
    # / evaluators/intercode.py:568) -- `stdout if stdout else stderr`, NO rstrip. The real eval
    # never strips the trailing newline a command actually printed; a `rstrip("\n")` here was
    # training the model on an observation SHAPE (never ends in "\n") that eval-time history never
    # shows (references-tour/.../_deepdive-intercode-evaluator.md D7: "0/200 observations ended
    # with newline" before this change). subprocess with text=True never returns None, so `out` is
    # already a plain str here -- no `or ""` needed.
    out = res.stdout if res.stdout else res.stderr
    if expected_paths is not None:
        # MUTATING gate (I2, L9): the gold must be SILENT and its ACTUAL fs effect, measured
        # the same way the validator measures it, must equal exactly the set the template
        # declared. No partial credit here either -- one unexpected/missing path fails
        # verification, same as the reward's own p1 term would punish it.
        after_state = _capture_sandbox_state(sandbox_root)
        changed = _changed_paths(before_state, after_state)
        changed_root = {p.replace(sandbox_root, root, 1) for p in changed}
        ok = (res.returncode == 0) and (res.stdout == "") and (changed_root == set(expected_paths))
        # ft-contract-guard FAIL-2 fix: `out` above falls back to stderr when stdout is empty
        # (the shared byte-formation line just above, matched to the validator's own
        # `stdout if stdout else stderr` for the READ-ONLY path). A mutating gold that is
        # silent on stdout but writes a warning to stderr would PASS this gate (stdout=="" is
        # all the gate checks) while `out` still carried that stderr text — breaking the
        # invariant `mastering_intercode.py` relies on (obs1=="" for every mutating task, so
        # `build_examples` never has anything to accidentally append after). Force the shipped
        # observation to the empty string here explicitly, independent of what `out` holds.
        out = ""
    else:
        # Gate on STDOUT, not on the stdout-or-stderr fallback above. A pipeline's returncode is
        # its LAST command's, so a gold ending in `tail`/`sort`/`head` exits 0 even when an
        # earlier stage died — and `out` would then hold the error text from stderr, which we
        # would happily emit as the task's "answer". Requiring real stdout closes that hole and
        # changes nothing for working templates (every gold here answers on stdout).
        # NOTE: do NOT "fix" this with `bash -o pipefail` instead — that would start dropping the
        # head-terminated templates whose producer legitimately takes SIGPIPE (exit 141).
        ok = res.returncode == 0 and (res.stdout or "").strip() != ""
    # sandbox_root only ever appears as a path PREFIX in the sandboxed output (it came from
    # this same substitution above); stripping it back to `root` makes the shipped
    # observation byte-identical in SHAPE to what the real evaluator prints.
    out = out.replace(sandbox_root, root)
    if len(out) > _OBS_TRUNCATE_CHARS:
        out = out[:_OBS_TRUNCATE_CHARS]
    return ok, out


# ── I3 (2026-09-19): REAL-EXECUTED near-miss for recovery trajectories ─────────────────────
# Previously (mastering_intercode.py's `_near_miss`) the recovery trajectory's failed first
# attempt was a HAND-WRITTEN fake error string ("bash: xx: command not found") that was NEVER
# actually run — a plausible-looking guess, not a real observation. That is the exact same
# category of problem the E10 revert (mastering_intercode.build_execute_only_example note)
# documents for bigcode's synthesized observations: text nothing scores or reads, invented
# rather than captured. Here it additionally risked LOOKING wrong in an inspectable way (e.g.
# claiming "command not found" for a candidate that would have actually failed with a totally
# different real shell error). This module already has the sandboxed-execution machinery
# (_sandbox_bind / _materialize / _assert_command_confined_to_tempdir) built for the gold
# command; near-miss reuses the SAME machinery so its "failure" is a real subprocess run against
# the task's real fixture files, never invented text.
def _synthesize_near_miss_candidate(gold: str, variant: int) -> str:
    """Deterministic, plausible WRONG version of `gold` (typo'd command name / corrupted path /
    bogus flag) — a CANDIDATE to actually execute, not a claim about what it will do. Read-only
    templates only (callers never invoke this for a mutating task's gold)."""
    g = gold.strip()
    tok = g.split()[0] if g.split() else 'ls'
    strat = variant % 3
    if strat == 0:  # typo the command name -> likely "command not found"
        bad_tok = tok[:-1] if len(tok) > 2 else tok + 'x'
        return g.replace(tok, bad_tok, 1)
    if strat == 1 and '/' in g:  # corrupt a path -> likely "no such file"
        return re.sub(r'(/[\w./-]+)', lambda m: m.group(1).replace('/', '/x', 1), g, count=1)
    # add a bogus flag -> likely "unrecognized option"
    return (f'{tok} --no-such-flag ' + g[len(tok):].strip()).strip()


def _run_near_miss_in(near_cmd, files, root, base_dir):
    """Execute a near-miss CANDIDATE in a fresh sandbox copy of the task's own fixture files
    (never the gold's already-used tempdir), returning (returncode, observation) with the REAL
    captured stdout/stderr (same byte formation as _run_gold_in: `stdout if stdout else stderr`,
    truncated to OBS_TRUNCATE_CHARS, tempdir prefix stripped back to the eval-class root).
    Returns None if the candidate cannot even be materialized/executed (caller drops the
    recovery row for that task rather than inventing anything). Read-only use only — never
    called with a mutating template's files/root; non-destructive by construction since the
    candidate is a corrupted variant of an already-read-only gold command."""
    sandbox_cmd, sandbox_root = _sandbox_bind(near_cmd, root, base_dir)
    # Same hard rule as _run_gold_in: guard BEFORE any subprocess call, independent of
    # _sandbox_bind's own construction.
    _assert_command_confined_to_tempdir(sandbox_cmd, base_dir)
    try:
        _materialize(files, sandbox_root)
    except Exception:
        return None
    try:
        res = subprocess.run(["bash", "-c", sandbox_cmd], cwd=base_dir, capture_output=True,
                             text=True, timeout=_GOLD_TIMEOUT)
    except Exception:
        return None
    out = res.stdout if res.stdout else res.stderr
    out = out.replace(sandbox_root, root)
    if len(out) > _OBS_TRUNCATE_CHARS:
        out = out[:_OBS_TRUNCATE_CHARS]
    return res.returncode, out


def generate_tasks(n, seed=0):
    """Generate up to n self-generated, shell-verified Bash tasks.

    Returns a list of (query, gold, observation, verbose_gold, verbose_observation,
    expected_paths, near_miss); `gold`/`verbose_gold` keep their real eval-class absolute path
    (e.g. "/testbed/..."), unmodified. `verbose_gold`/`verbose_observation` are (None, None) when
    the template has no natural verbose sibling, or when the sibling command failed its OWN
    independent shell verification (a bad verbose command never drops the task — it just
    falls back to the 2-row single-shot shape for that task, since `gold` alone already
    verified fine). `expected_paths` (I2, L9) is None for every READ-ONLY template (the
    pre-existing 5-tuple contract, untouched) and a frozenset of root-prefixed absolute
    paths for a MUTATING template — the set of managed paths the gold is declared to
    create/delete/modify, already state-diff-verified inside the sandbox by `_run_gold_in`.
    `near_miss` (I3, 2026-09-19) is `(near_miss_command, real_observation)` — a REALLY EXECUTED
    wrong-command attempt (see `_synthesize_near_miss_candidate`/`_run_near_miss_in` above) — for
    a read-only task whose near-miss actually differs from the gold's own result, or `None` when
    the task is mutating (near-miss is never generated for mutating templates — I3 requirement),
    or when the near-miss could not be executed, or when it UNEXPECTEDLY succeeded with the exact
    same (truncated) output as gold (that candidate taught nothing about recovering from failure,
    so the task's recovery row is dropped rather than shipping a "failure" that was not one).
    `mastering_swe.py` and any other 3-element consumer keep working unmodified: they index
    `task[0..2]`, never destructure the tuple, so the 6th/7th elements are additive.
    Verification itself runs inside a disposable tempdir with the root bound to a subdir of
    that tempdir (_run_gold_in) — never against a real "/". Tasks whose GOLD command does
    not run cleanly (or, for a mutating template, whose actual fs effect does not exactly
    match its declared `expected_paths`) are skipped, so every emitted task's minimal
    command is verified. Deterministic given seed. Round-robins templates for coverage."""
    rng = random.Random(seed)
    tasks = []
    attempts = 0
    max_attempts = n * 4 + 32
    while len(tasks) < n and attempts < max_attempts:
        attempts += 1
        template = _TEMPLATES[len(tasks) % len(_TEMPLATES)] if attempts <= n else rng.choice(_TEMPLATES)
        try:
            result = template(rng)
        except Exception:
            continue
        # I2: a template may return the pre-existing 5-tuple (READ-ONLY, expected_paths
        # implicitly None) or the new 6-tuple (declares expected_paths, MUTATING). Detecting
        # by length means none of the 41 pre-existing template bodies needed to change their
        # `return` statement — only the new mutating templates opt into the 6th slot.
        if len(result) == 6:
            query, gold, files, root, verbose_gold, expected_paths = result
        else:
            query, gold, files, root, verbose_gold = result
            expected_paths = None
        with tempfile.TemporaryDirectory() as d:
            ok, obs = _run_gold_in(gold, files, root, d, expected_paths)
        if not ok:
            continue
        vgold, vobs = None, None
        # A mutating template's `gold` is required to be silent by construction; a
        # verbose-sibling refine trajectory (which needs a NOISIER but still correct output
        # to react to) does not apply here, so mutating templates simply never set
        # verbose_gold (checked in _selftest).
        if verbose_gold:
            with tempfile.TemporaryDirectory() as d2:
                vok, vo = _run_gold_in(verbose_gold, files, root, d2)
            if vok:
                vgold, vobs = verbose_gold, vo
        # I3: real-executed near-miss, READ-ONLY templates only (expected_paths is None) --
        # never for a mutating template's gold (I3 hard requirement). `len(tasks)` (the task's
        # own emission index) seeds the strategy choice deterministically, independent of rng
        # state so it does not perturb the pre-existing template/verbose-sibling draw sequence.
        near_miss = None
        if expected_paths is None:
            near_cmd = _synthesize_near_miss_candidate(gold, len(tasks))
            try:
                with tempfile.TemporaryDirectory() as d3:
                    run = _run_near_miss_in(near_cmd, files, root, d3)
            except Exception:
                run = None
            if run is not None:
                near_rc, near_obs = run
                if not (near_rc == 0 and near_obs == obs):
                    near_miss = (near_cmd, near_obs)
        tasks.append((query, gold, obs, vgold, vobs, expected_paths, near_miss))
    return tasks


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest():
    tasks = generate_tasks(40, seed=7)
    assert len(tasks) >= 30, f"too few verified tasks: {len(tasks)}"
    # I2/I3: every task is a real (query, gold, obs, verbose_gold_or_None, verbose_obs_or_None,
    # expected_paths_or_None, near_miss_or_None) 7-tuple. obs is REQUIRED non-empty for
    # read-only tasks (expected_paths is None) and REQUIRED empty for mutating tasks
    # (expected_paths is a frozenset) -- the whole point of the I2 arm is that a silent gold is
    # a VALID task. near_miss is REQUIRED None for every mutating task (I3 hard requirement:
    # near-miss is read-only-only) and is a REAL (command, observation) str pair, never None
    # from a fake/guessed string, whenever present.
    n_mutating = 0
    n_near_miss = 0
    for q, g, o, vg, vo, ep, nm in tasks:
        if ep is not None:
            assert nm is None, f"mutating task unexpectedly carries a near_miss: {q!r}"
        if nm is not None:
            n_near_miss += 1
            near_cmd, near_obs = nm
            assert isinstance(near_cmd, str) and near_cmd and isinstance(near_obs, str), (q, nm)
            assert near_cmd != g, f"near_miss command is identical to gold: {q!r}"
        assert q and g and isinstance(o, str), (q, g, o)
        assert (vg is None) == (vo is None), (vg, vo)
        if vg is not None:
            assert isinstance(vg, str) and isinstance(vo, str) and vg != g, (vg, g)
        if ep is None:
            assert o != "", f"read-only task has an EMPTY observation: {q!r}"
        else:
            n_mutating += 1
            assert o == "", f"mutating task {q!r} has a non-empty observation: {o!r}"
            assert isinstance(ep, frozenset), (q, ep)
            # NEW (I2 hard rule): no tempdir path may leak into a declared expected_paths set,
            # same discipline as the existing gold/obs leak checks below.
            assert not any("/tmp/" in p or "/var/folders" in p for p in ep), f"expected_paths leaked a tempdir path: {ep}"
            assert vg is None, f"mutating task unexpectedly has a verbose sibling: {q!r}"
    assert n_mutating > 0, "0 mutating tasks verified -- the state-diff gate is broken"
    # determinism
    assert generate_tasks(20, seed=7)[:5] == generate_tasks(20, seed=7)[:5], "non-deterministic"
    # coverage: at least ~8 distinct gold command heads (diverse bash skills)
    heads = {g.split()[0] for _, g, _, _, _, _, _ in tasks}
    assert len(heads) >= 6, f"low coverage: {heads}"
    # E11 L1: gold + query must land on an eval-class absolute root, never the
    # tempdir prefix (which would mean sandbox-binding leaked into the shipped gold).
    abs_gold = sum(1 for _, g, _, _, _, _, _ in tasks if any(g_root in g for g_root in _EVAL_ROOTS))
    assert abs_gold == len(tasks), f"{len(tasks) - abs_gold} gold(s) missing an eval-class root"
    assert not any("/tmp/" in g or "/var/folders" in g for _, g, _, _, _, _, _ in tasks), "gold leaked a tempdir path"
    assert not any("/tmp/" in o or "/var/folders" in o for _, _, o, _, _, _, _ in tasks), "observation leaked a tempdir path"
    assert not any(vg and ("/tmp/" in vg or "/var/folders" in vg) for _, _, _, vg, _, _, _ in tasks), "verbose gold leaked a tempdir path"
    assert not any(vo and ("/tmp/" in vo or "/var/folders" in vo) for _, _, _, _, vo, _, _ in tasks), "verbose obs leaked a tempdir path"
    # I3: near-miss command/observation must also never leak a tempdir path.
    assert not any(nm and ("/tmp/" in nm[0] or "/var/folders" in nm[0]) for _, _, _, _, _, _, nm in tasks), "near_miss command leaked a tempdir path"
    assert not any(nm and ("/tmp/" in nm[1] or "/var/folders" in nm[1]) for _, _, _, _, _, _, nm in tasks), "near_miss observation leaked a tempdir path"
    assert n_near_miss > 0, "0 near-miss trajectories verified -- the real-execution recovery path is broken"
    # E12 L2: at least 25 of the distinct templates declare a verbose sibling (checked at the
    # DISTINCT-template level via `_TEMPLATES_BASE`, not `_TEMPLATES` -- which is the I1
    # weighted round-robin schedule and would double-count an upweighted template once per
    # copy -- and not the sampled-task level, since a bad task-instance verbose run
    # legitimately falls back to None without indicting the template).
    n_verbose_templates = 0
    _probe_rng = random.Random(1234)
    for t in _TEMPLATES_BASE:
        probe = t(_probe_rng)
        vg = probe[4]
        if vg:
            n_verbose_templates += 1
    assert n_verbose_templates >= 25, f"too few templates with a verbose sibling: {n_verbose_templates}/{len(_TEMPLATES_BASE)}"
    # I2: at least the 4 declared mutating templates round-trip through the 6-tuple contract
    # and always declare expected_paths (probed independently of generate_tasks's verification
    # so a template-level regression is caught even if the state-diff gate itself is fine).
    for t in _MUT_UPWEIGHT:
        probe = t(_probe_rng)
        assert len(probe) == 6 and probe[5] is not None, f"{t.__name__} did not declare expected_paths"
    # none of these are NL2Bash instances — they operate on our own random files
    print(f"selfgen_bash OK: {len(tasks)} verified tasks ({n_mutating} mutating, "
          f"{n_near_miss} with a real-executed near-miss), "
          f"{len(heads)} distinct command heads: {sorted(heads)}, "
          f"{n_verbose_templates}/{len(_TEMPLATES_BASE)} templates with a verbose sibling, "
          f"{len(_TEMPLATES)}-item weighted round-robin schedule")


if __name__ == "__main__":
    _selftest()
