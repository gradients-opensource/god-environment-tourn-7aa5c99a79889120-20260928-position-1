# Quiet mode first, before the heavy libraries load: several of its controls are environment
# defaults those libraries read once at import time. Our own tagged diagnostics and anything
# on stderr still get through -- see scripts/shared/quiet_mode.py.
import shared.quiet_mode  # noqa: F401
import argparse
import json
import os
import sys
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

def _normalize(ds):
    cols = ds.column_names

    def _to_uniform(ex):
        if 'tools_json' in cols:
            tj = ex.get('tools_json')
        elif 'tools' in cols:
            tj = json.dumps(ex['tools'])
        else:
            tj = None
        return {'messages_json': json.dumps(ex['messages']), 'tools_json': tj}
    return ds.map(_to_uniform, remove_columns=cols)

def _read_miner_datasets():
    """Read the whitelisted datasets the validator pre-downloaded for this task.

    Contract (validator docs/miners.md + trainer/image_manager.py):
      MINER_DATASETS_DIR = cache root, MINER_DATASETS = csv of dir names (org--name).
      Each dataset is a raw HF snapshot_download (load via load_dataset, NOT
      load_from_disk). Returns [(kind, name, Dataset)] for split 'train'; kind is
      detected by columns: 'pvp_sft' (messages+tools, Dataset A) / 'instruction'
      (instruction+response, Dataset B) / 'unknown'. Empty when none requested."""
    root = os.environ.get('MINER_DATASETS_DIR')
    names = [n.strip() for n in os.environ.get('MINER_DATASETS', '').split(',') if n.strip()]
    if not root or not names:
        return []
    from shared.miner_datasets import load_miner_dataset_dir
    out = []
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            print(f'[combine] miner dataset {name}: not found (not mounted at {path})', flush=True)
            continue
        try:
            ds = load_miner_dataset_dir(path)
        except Exception as exc:
            print(f'[combine] miner dataset {name}: load failed: {type(exc).__name__}: {exc} (skip)', flush=True)
            continue
        cols = set(ds.column_names)
        # SWE-ZERO (messages + instance_id/trajectory_format) shares the 'messages' column with
        # pvp-tool-calling-sft; without this guard it is mis-typed as pvp_sft and blended into the
        # PvP/intercode training mix (requested_datasets is GLOBAL → SWE-ZERO is mounted for every
        # env task, not just swe_infinite). It is consumed only by envs.mastering_swe — skip it
        # here. Discriminator mirrors mastering_swe._load_swe_zero.
        if 'messages' in cols and ('instance_id' in cols or 'trajectory_format' in cols):
            print(f'[combine] miner dataset {name}: SWE-ZERO shape — skipped (mastering_swe only)', flush=True)
            continue
        if 'messages' in cols:
            kind = 'pvp_sft'
        elif {'instruction', 'response'} <= cols:
            kind = 'instruction'
        else:
            kind = 'unknown'
        print(f'[combine] miner dataset {name}: kind={kind} rows={len(ds)} cols={sorted(cols)}', flush=True)
        out.append((kind, name, ds))
    return out

def _blend_pvp_sft(ds, keep_envs, cap_total, seed=42):
    """Dataset A (pvp-tool-calling-sft): already in messages+tools eval format from a
    strong teacher (claude). Filter to the round's envs (its `env` column matches our
    env names, incl goofspiel), cap to ~cap_total rows as a warm-start dose, then map
    to the uniform {messages_json, tools_json} schema. Returns None if nothing kept."""
    cols = ds.column_names
    if 'env' in cols and keep_envs:
        ds = ds.filter(lambda ex: ex.get('env') in keep_envs)
    if len(ds) == 0:
        print(f'[combine] pvp_sft: 0 rows after env filter (envs={sorted(keep_envs)}); nothing to add', flush=True)
        return None
    if cap_total and len(ds) > cap_total:
        ds = ds.shuffle(seed=seed).select(range(cap_total))

    def _to_uniform(ex):
        tj = ex.get('tools')
        return {'messages_json': json.dumps(ex['messages']), 'tools_json': json.dumps(tj) if tj is not None else None}
    out = ds.map(_to_uniform, remove_columns=ds.column_names)
    print(f'[combine] pvp_sft warm-start: kept {len(out)} rows (envs={sorted(keep_envs)})', flush=True)
    return out


def _blend_bigcode_bash(ds, cap, include_thoughts=True):
    """Dataset B (intercode_bigcode_combined_12k): instruction->gold-command pairs. Filter with
    envs.mastering_intercode.usable_bigcode_rows (bash-only, deduped, unsafe/destructive/
    placeholder rows dropped — see that function for the full filter list), then emit ONE
    execute-only turn per usable pair: envs.mastering_intercode.build_examples(instr, gold, '',
    variant)[0] — the turn-1 (execute) example only, taking the first element of the SAME list
    build_examples returns for self-gen so the system prompt / tool schema / row shape stay
    byte-identical to self-gen's own execute turn (the `obs1` argument only feeds the discarded
    turn-2/submit example, so passing '' here is inert — never executed, never read).

    I3 REVERT of E10 (measured 2026-09-19 on a local GPU replay, see
    .claude/Environment/experiments/i3/README.md): E10 paired every bigcode row with a
    synthesize_observation()-based submit row so bigcode would "teach the stop-anchor shape
    too." Measured effect was the opposite of intended — 1,391/1,399 of our submit training
    rows became a trivial "1 command -> EMPTY observation -> submit" pattern, our trained
    model used a 2nd command in 0/20 eval tasks (base model: 19/20), and it regressed two
    tasks the base model solved via retry-after-error. The bigcode dataset has NO real
    filesystem, so its "observation" was always a synthesized, never-executed placeholder —
    pairing it with a submit target taught premature, unconditional stopping at production
    volume (bigcode was ~70% of intercode's training rows on some cells). Reverting to
    execute-only removes that submit-shape entirely from the bigcode blend; multi-turn submit
    behaviour is now taught ONLY from self-gen's real, shell-executed observations
    (mastering_intercode.build_recovery_example / build_refine_example), where "stop" is
    conditioned on what the sandbox actually printed. `build_execute_only_example` (this
    module's own hand-written execute-only row) is NOT reused here on purpose — this function
    prefers `build_examples(...)[0]`, the literal first element of the self-gen execute+submit
    pair-builder, precisely so a future edit to `build_examples` (system prompt, tool schema,
    thought-variant pools, `--history-thoughts` handling) can never make the bigcode execute
    turn drift from the self-gen execute turn by forgetting to mirror it into a second
    function — see GATE-0's `build_examples(...)[0] == build_execute_only_example(...)` check.

    `cap` is a ROW count == PAIR count now (1 row per pair): --intercode-bigcode-rows (the
    sft_env_config B design) or a legacy frac-of-generated-rows count pre-multiplied by the
    caller. Up to `min(cap, usable pool size)` pairs are taken, each contributing exactly one
    row — no even-rounding needed any more (E10's odd-cap/even-cap accounting in main(), below,
    is gone with it: a pair is never split because there is no second row to split off).

    Off by default (cap<=0). When enabled it augments intercode bash-fluency with an auxiliary
    bash-command subset — gate on eval before trusting it lifts the intercode score.

    `include_thoughts` (E14 L4 follow-up): threaded straight into build_examples, same meaning as
    the generator's own flag (see mastering_intercode.py's _history_step/_content_for note) — on
    bases whose chat template drops assistant `content` next to `tool_calls` (Llama), the bigcode
    blend's execute-turn `content` is dropped too (there is no turn-2 history line any more,
    since there is no turn 2)."""
    if cap <= 0:
        return None
    from envs.mastering_intercode import build_examples, usable_bigcode_rows
    pairs, dropped_dup, dropped_filtered = usable_bigcode_rows(ds)
    usable_total = len(pairs)
    n_pairs_wanted = cap
    rows = []
    for i, (instr, resp) in enumerate(pairs[:n_pairs_wanted]):
        ex_execute = build_examples(instr, resp, '', variant=i, include_thoughts=include_thoughts)[0]
        rows.append({'messages_json': json.dumps(ex_execute['messages']), 'tools_json': ex_execute['tools_json']})
    assert len(rows) <= cap, f'bigcode blend produced {len(rows)} rows, over its own cap {cap}'
    if not rows:
        # Keep-marker ("no usable"): the pre-fix wording ("0 usable rows after filter") contained
        # NO quiet_mode keep-marker (verified against shared.quiet_mode._marker_for) and would be
        # silently dropped in the tournament's default quiet mode even though B>0 was planned and
        # nothing got blended — exactly the "misleading" case the guard flagged.
        print('[combine] bigcode bash-subset: no usable rows after filter; nothing to add', flush=True)
        return None
    n_pairs_used = len(rows)
    if n_pairs_used < n_pairs_wanted:
        # Keep-marker ("fallback"): logged with the pair/row split so a shortfall is legible even
        # after quiet-mode's own-tag redaction reduces the line to "[combine] fallback" — this is
        # the ONLY branch that logs the split (quiet-mode rules: a clean run stays quiet; the
        # sft_env_config loud "!!! blend: pool=.../blend=on/off" flag line is untouched by this
        # arm and stays flag-only, it never carries U/B/pair numbers — see its own comment).
        print(f'[combine] bigcode bash-subset: shortfall — requested cap {cap} rows ({n_pairs_wanted} '
              f'pairs wanted), only {usable_total} usable pairs survived filter, blended '
              f'{n_pairs_used} pairs / {len(rows)} rows, falling back to {len(rows)}', flush=True)
    print(f'[combine] bigcode bash-subset: +{len(rows)} rows ({n_pairs_used} execute-only '
          f'rows, cap {cap}, usable pool {usable_total} pairs, dropped {dropped_dup} dup / '
          f'{dropped_filtered} unsafe-or-destructive-or-placeholder)', flush=True)
    return Dataset.from_list(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', required=True, help='Task identifier')
    parser.add_argument('--envs', required=True, help='Comma-separated env names (e.g. gin_rummy,liars_dice)')
    parser.add_argument('--output', required=True, help='Combined dataset output path')
    parser.add_argument('--input-dir', default='/workspace/scripts/datasets', help='Base directory for per-env datasets')
    parser.add_argument('--include-miner-datasets', action='store_true', help='Blend validator-pre-downloaded whitelisted datasets (MINER_DATASETS) into the combined train set.')
    parser.add_argument('--miner-pvp-frac', type=float, default=0.1, help='Cap for the pvp-tool-calling-sft warm-start: fraction of generated train rows it may add (note: 0 is treated as no cap by _blend_pvp_sft, not as disable).')
    parser.add_argument('--intercode-bigcode-frac', type=float, default=0.0, help='Legacy cap for the bigcode (instruction-kind) bash-subset blend: fraction of generated train rows it may add. 0 (default) = SKIP unless --intercode-bigcode-rows is also given. Only usable bash rows kept (see envs.mastering_intercode.usable_bigcode_rows); execute-only turns. GATE ON EVAL before trusting it helps.')
    parser.add_argument('--intercode-bigcode-rows', type=int, default=0, help='ABSOLUTE row-count cap for the bigcode blend (the sft_env_config B design: B = min(usable pool, I_sg - 0.25*I_sg), sized once at config time). Takes priority over --intercode-bigcode-frac when > 0; pass --intercode-bigcode-frac 0 alongside it. 0 (default) = use --intercode-bigcode-frac instead (legacy/manual path).')
    parser.add_argument('--history-thoughts', choices=['auto', 'on', 'off'], default='auto', help="E14 L4 follow-up: same meaning as envs.mastering_intercode's own --history-thoughts, threaded into _blend_bigcode_bash's build_examples call. 'off' drops the bigcode blend's `Thought N:` history line and sets assistant content='' (for Llama-family bases, whose chat template discards content next to tool_calls anyway). 'on'/the default 'auto' (HEAD-identical) keep the pre-existing behaviour. sft_env_config.py passes 'off' only when get_model_architecture(model_path) == 'llamaforcausallm'.")
    args = parser.parse_args()
    _include_thoughts = args.history_thoughts != 'off'
    env_names = [e.strip() for e in args.envs.split(',') if e.strip()]
    if not env_names:
        sys.exit('--envs must list at least one environment_name')
    train_parts = []
    val_parts = []
    loaded = []  # (env, row_count) parallel to train_parts, for the per-env row-floor below
    for env in env_names:
        per_env_path = os.path.join(args.input_dir, f'sft_env_{args.task_id}_{env}')
        if not os.path.isdir(per_env_path):
            # Stays a WARNING on purpose. A sys.exit here was tried on 2026-08-06 and REVERTED
            # the same day, because it is strictly dominated — worth recording so nobody retries:
            #   * It does NOT prevent a bad score for the dead env. A missing per-env DATASET does
            #     not remove that env from the tournament; the submitted model still plays it and
            #     is still scored on it. Only the TRAINING is missing. (The reasoning that
            #     justified the exit — "an env with no score takes a 0.0 quantile" — confused a
            #     missing dataset with a missing score.)
            #   * The exit is invisible anyway: text_trainer.run_cmd_with_log captures the return
            #     code and only writes it to a log, never checks it. Combine writing nothing just
            #     makes train_sft_env hit a missing --dataset-path, which fails the task and ships
            #     the UN-NOISED BASE MODEL for EVERY env — so it converts "one env untrained" into
            #     "all envs untrained".
            #   * The tool-calling schema is SHARED across envs (pvp_toolcalling.build_toolcalling_
            #     example), so training on the surviving env still teaches the format that stops
            #     forfeits in the dead one. Partial training beats none for the dead env too.
            # The all-envs-dead case is already covered by `if not train_parts` below.
            print(f'[combine] ⚠️ WARNING: {per_env_path} not found — {env!r} produced NO dataset. '
                  f'Training on the surviving envs only; {env!r} will play UNTRAINED. '
                  f'Check that env\'s generator log before trusting this run.', flush=True)
            continue
        dd = load_from_disk(per_env_path)
        train_ds = dd['train'] if 'train' in dd else dd
        train_parts.append(_normalize(train_ds))
        loaded.append((env, len(train_ds)))
        print(f'[combine] loaded {env}: train={len(train_ds)}' + (f" validation={len(dd['validation'])}" if 'validation' in dd else ''), flush=True)
        if 'validation' in dd:
            val_parts.append(_normalize(dd['validation']))
    if not train_parts:
        sys.exit('No per-env datasets found — combine aborted')

    # --- F2 guard fix: compute the ACTUAL bigcode blend for intercode BEFORE the row-floor below.
    # The floor used to run first, oversample intercode's (post-B-carve-out, now smaller) self-gen
    # count back up toward the OTHER envs' floor, and then the bigcode blend got concatenated ON
    # TOP of that already-restored count — silently re-inflating intercode past what
    # sft_env_config's B design intended (measured: +14.5% total / higher intercode share on a
    # real smoke run). Doing the blend first and feeding its ACTUAL (post-filter) row count into
    # the floor's accounting fixes this: the floor only tops up what's STILL missing after the
    # blend, never on top of it.
    _miner_names = [n.strip() for n in os.environ.get('MINER_DATASETS', '').split(',') if n.strip()] \
        if args.include_miner_datasets else []
    loaded_miner = _read_miner_datasets() if args.include_miner_datasets else []
    _gen_train_size_pre_floor = sum(n for (_e, n) in loaded)
    _bigcode_extra = None
    _intercode_blended_rows = 0
    _saw_instruction_kind = False
    # Flag for the loud audit line at the end of this function — see there for why this is a
    # flag ('skip'/'fail'/'none'/'ok'), never the row count itself.
    _instr_flag = 'skip'
    # E10: the row count the shortfall check below compares delivered rows against. Set inside the
    # loop (even-rounded, see there) so it survives past the loop for the post-loop shortfall check
    # — starts at 0 so "never attempted" and "attempted, 0 usable" both correctly read as no cap.
    _bigcode_cap_even = 0
    for (kind, name, ds) in loaded_miner:
        if kind != 'instruction':
            continue
        _saw_instruction_kind = True
        # Dataset B (intercode_bigcode_combined_12k): ~50% Python/prose by is_bash_response
        # (measured on the real 12000-row dataset: 5996/12000 pass is_bash_response, i.e.
        # ~50% dropped as Python/prose, not the ~64% an earlier comment here claimed),
        # off-distribution for the bash (execute_bash) intercode env. Default OFF (skip) so
        # it never silently mis-wraps. Two ways to opt in, checked in this priority order:
        #   1. --intercode-bigcode-rows B > 0 (sft_env_config's per-task B design: an
        #      ABSOLUTE row count sized from the round's own intercode target + the usable
        #      pool size, see sft_env_config.get_training_json). This is what a live task uses.
        #   2. --intercode-bigcode-frac > 0 (legacy/manual path: fraction of PRE-FLOOR generated
        #      rows — note this is measured before the row-floor now, a small, documented behavior
        #      change from the pre-fix version, which measured it post-floor; the live B-design
        #      path above never uses this branch, it always passes an absolute --intercode-bigcode-rows).
        # Either way, only rows surviving envs.mastering_intercode.usable_bigcode_rows are
        # ever blended (bash-only + dedup + unsafe/destructive/placeholder-filtered).
        # Gate on eval.
        if args.intercode_bigcode_rows > 0:
            cap = args.intercode_bigcode_rows
        elif args.intercode_bigcode_frac > 0:
            cap = int(_gen_train_size_pre_floor * args.intercode_bigcode_frac)
        else:
            cap = 0
        _requested_cap = cap
        # I3: no even-rounding needed any more -- _blend_bigcode_bash emits exactly one row per
        # pair (E10's 2-row execute+submit shape was reverted, see that function's docstring), so
        # a pair can never be "split" and any cap>=1 is directly usable.
        if cap >= 1:
            _bigcode_cap_even = cap
            try:
                blended = _blend_bigcode_bash(ds, cap, _include_thoughts)
            except Exception as exc:
                # FAIL-2: an exception here must skip the blend, not fail the whole combine step
                # (which would fail the whole training task closed for every env in the round).
                print(f'[combine] bigcode bash-subset: fail ({type(exc).__name__}: {exc}) — '
                      f'skip, no bigcode rows added', flush=True)
                blended = None
                _instr_flag = 'fail'
            else:
                if blended is not None:
                    _bigcode_extra = blended
                    _intercode_blended_rows = len(blended)
                    # ok means rows were REALLY concatenated (len>0); 0 usable rows after filter
                    # is 'none' (attempted, nothing survived), not 'ok'.
                    _instr_flag = 'ok' if len(blended) > 0 else 'none'
                else:
                    _instr_flag = 'none'
        else:
            # I3: cap<=0 is the only way to land here now -- 1 row per pair means any cap>=1 is
            # directly usable (no more "cap rounds below one pair" dead zone from the old 2-row
            # shape; `_requested_cap` no longer diverges from `cap`, since neither is rounded).
            print(f'[combine] instruction-kind miner dataset {name}: SKIPPED '
                  f'(--intercode-bigcode-frac=0 and --intercode-bigcode-rows=0, default)', flush=True)
            _instr_flag = 'skip'
    if (args.intercode_bigcode_rows > 0 or args.intercode_bigcode_frac > 0) and not _saw_instruction_kind:
        # sft_env_config sized a non-zero B/frac because 'intercode' is in this round's envs,
        # but the validator did not mount an instruction-kind (bigcode) dataset under
        # MINER_DATASETS for this task (not requested at registration, revision swap, etc).
        # Behave exactly like the frac=0/rows=0 baseline: self-gen only, no crash. NOTE: intercode's
        # self-gen target was ALREADY cut by the planned B at config time (sft_env_config), so this
        # path is a NET ROW-COUNT LOSS vs the true no-blend baseline, not a neutral no-op — flagged
        # explicitly below, not just by this one line.
        print('[intercode] bigcode not requested — falling back to self-gen', flush=True)
    # Compare against the planned B from the ARGS directly (I3: no even-rounding any more, 1 row
    # per pair), not the cap discovered in the loop: when no instruction-kind dataset was mounted
    # the loop never set _bigcode_cap_even, and that not-mounted case is exactly the largest
    # possible shortfall (all of B missing).
    _planned_even = args.intercode_bigcode_rows
    if _planned_even >= 1 and _intercode_blended_rows < _planned_even:
        # Guard-flagged case: B>0 was planned at config time but combine got fewer (or 0) usable
        # rows. self-gen was already shrunk by the FULL planned B, so any shortfall here is a real
        # missing-rows loss, not merely "nothing extra to add" — say so explicitly, with a
        # keep-marker. Deliberately does NOT use the words "fallback"/"falling back" here: this
        # arm's generic no-dataset-mounted line above ALSO contains "falling back", and
        # shared.quiet_mode._marker_for returns the FIRST matching marker in _KEEP_MARKERS order
        # (not the first one occurring in the text) — 'fallback'/'falling back' sort earlier in
        # that tuple than 'missing', so a version of this line that also said "falling back" was
        # redacted to the SAME "[intercode] falling back" as the generic line, making the two
        # cases indistinguishable in the tournament's default quiet mode. Using only "missing"
        # (verified against _KEEP_MARKERS: no earlier marker in the tuple appears in this text)
        # makes this line redact to the distinct "[intercode] missing".
        #
        # I3: `_planned_even` is now literally `args.intercode_bigcode_rows` (no rounding — 1 row
        # per pair, so B in rows == B in pairs and nothing can be "split"). Name kept for diff
        # minimality against the E10-era variable it replaces; it is no longer ever rounded.
        print(f'[intercode] bigcode shortfall: planned B={args.intercode_bigcode_rows} rows, only '
              f'{_intercode_blended_rows} usable rows actually blended — self-gen target was '
              f'already reduced by the full planned B, so {_planned_even - _intercode_blended_rows} '
              f'rows are MISSING from the round (using only what survived the filter)', flush=True)

    # Per-env row-FLOOR (anti-starvation). Envs generate in parallel into ONE shared time-box and are
    # raw-concatenated — a slow/heavy env (othello 'expert', clobber solver) can under-fill its target
    # while a fast env dominates the mix, silently under-training the slow one. Oversample (repeat rows
    # of) any env below _FLOOR_FRAC of the largest env so every env gets a fair share of the trainer's
    # attention. Never drops data — only duplicates the deficient env's rows.
    #
    # F2 REGRESSION FIX (guard round-3): the floor THRESHOLD must be computed from each env's
    # EFFECTIVE count (self-gen + already-planned bigcode for intercode), not raw self-gen alone.
    # An earlier version of this comment claimed "unaffected by any bigcode blend, so unrelated
    # envs' floor behavior is unchanged" — that was FALSE and caused a real regression: intercode
    # never under-fills (no time-box for it, sft_env_config.py's generation time-box branch
    # excludes it — see _build_generate_cmd's intercode arm, which has no --time_limit_seconds;
    # mastering_intercode.py's generator repeats its unique task set and truncates to the exact
    # target), so intercode is reliably the LARGEST env and usually sets `max(_counts)`. Computing
    # `_floor` from intercode's RAW self-gen count (post-B-cut, roughly halved) silently HALVED
    # the anti-starvation floor for every OTHER (genuinely time-boxed, under-filling) round-mate.
    # Measured (guard's g3run/result.json, real combine_datasets.py subprocess runs): R1
    # intercode+clobber @1GPU, clobber under-filled to 1000 -- control floors it to 1837, the
    # buggy pre-fix treatment left it at 1000 (total 5511 -> 4693, clobber got ~46% of its floor).
    # Fixed by including intercode's OWN effective count (self-gen + already-planned bigcode) in
    # the threshold: clobber floors to 1851 (vs control's 1837, +0.8% — v5 number, at v5's larger
    # B=2784; the small difference is the B rows themselves counting toward `max()`, which is
    # correct: intercode's real combined size, self-gen + bigcode, IS what round-mates should be
    # floored against).
    _FLOOR_FRAC = 0.5
    if len(train_parts) > 1:
        _counts = [n + (_intercode_blended_rows if _e == 'intercode' else 0) for (_e, n) in loaded]
        _floor = int(_FLOOR_FRAC * max(_counts))
        for _i, (_env, _n) in enumerate(loaded):
            _already = _intercode_blended_rows if _env == 'intercode' else 0
            _eff = _n + _already
            if 0 < _n and _eff < _floor:
                _target = max(_n, _floor - _already)
                if _target > _n:
                    train_parts[_i] = train_parts[_i].select([j % _n for j in range(_target)])
                    _note = f' (net of {_already} bigcode rows already planned for this env)' if _already else ''
                    print(f'[combine] row-floor: {_env} {_n} -> {_target} rows '
                          f'(oversampled x{_target / _n:.2f}, floor={_floor}){_note} so it is not starved', flush=True)
    combined_train = concatenate_datasets(train_parts).shuffle(seed=42)
    if val_parts:
        combined_val = concatenate_datasets(val_parts).shuffle(seed=42)
    else:
        val_size = min(40, max(1, len(combined_train) // 100))
        combined_val = combined_train.select(range(val_size))
    if args.include_miner_datasets:
        gen_train_size = len(combined_train)
        extra = []
        _pvp_blended_rows = 0
        # Flag for the loud audit line below — 'skip' unless a pvp_sft-kind dataset is actually
        # seen in loaded_miner (mirrors _instr_flag's default above).
        _pvp_flag = 'skip'
        for (kind, name, ds) in loaded_miner:
            if kind == 'pvp_sft':
                cap = int(gen_train_size * args.miner_pvp_frac) if args.miner_pvp_frac > 0 else 0
                try:
                    blended = _blend_pvp_sft(ds, set(env_names), cap)
                except Exception as exc:
                    print(f'[combine] pvp_sft warm-start: fail ({type(exc).__name__}: {exc}) — '
                          f'skip, no rows added', flush=True)
                    blended = None
                    _pvp_flag = 'fail'
                else:
                    if blended is not None:
                        extra.append(blended)
                        # += (not =): only one pvp-shaped dataset is reachable today (a second
                        # 'pvp_sft'-kind entry in loaded_miner isn't a real registration case —
                        # MINER_DATASETS is capped at 2 slots and the other slot is always
                        # instruction/SWE-ZERO-shaped in practice) but accumulating rather than
                        # overwriting is the correct operation if that ever changes, and it costs
                        # nothing today.
                        _pvp_blended_rows += len(blended)
                        _pvp_flag = 'ok' if len(blended) > 0 else 'none'
                    else:
                        _pvp_flag = 'none'
            elif kind == 'instruction':
                pass  # already handled above, BEFORE the row-floor (see _bigcode_extra)
            else:
                print(f'[combine] miner dataset {name}: kind=unknown, skipped', flush=True)
        if _bigcode_extra is not None:
            extra.append(_bigcode_extra)
        if extra:
            combined_train = concatenate_datasets([combined_train] + extra).shuffle(seed=42)
            print(f'[combine] miner blend: train {gen_train_size} -> {len(combined_train)}', flush=True)
        # Loud (contains '!!!', shared.quiet_mode._LOUD) audit line so a tournament log — which
        # under quiet mode redacts every other [combine] line to its tag plus one trigger word —
        # still lets a human confirm the blend step ran, without leaking the recipe. FLAGS ONLY,
        # not row counts: ft-contract-guard flagged an earlier draft of this line (raw
        # blended_pvp/blended_instr counts) as invertible — blended_pvp = int(gen_train_size *
        # 0.1) recovers the round's total generated rows, and (in sft_env_config's twin line) B
        # combined with the published _SELF_GEN_FLOOR_FRAC=0.25 recovers I_sg exactly, and U
        # fingerprints the mounted dataset. mounted/loaded are plain counts (0/1/2, since
        # MINER_DATASETS is capped at 2 slots by requested_datasets registration) — they reveal
        # nothing beyond what the round's own (already-public) dataset registration makes known.
        # pvp/instr are ok/none/skip/fail: 'ok' means rows were REALLY concatenated (>0), which is
        # the one thing worth confirming from a log — none of the four values can be inverted back
        # into a row count, a target, or which dataset was mounted.
        print(f'[combine] !!! miner-ds audit: mounted={len(_miner_names)} loaded={len(loaded_miner)} '
              f'pvp={_pvp_flag} instr={_instr_flag}', flush=True)
    dd = DatasetDict({'train': combined_train, 'validation': combined_val})
    dd.save_to_disk(args.output)
    print(f'[combine] saved combined dataset: train={len(combined_train)} val={len(combined_val)} envs={env_names} → {args.output}', flush=True)
if __name__ == '__main__':
    main()
