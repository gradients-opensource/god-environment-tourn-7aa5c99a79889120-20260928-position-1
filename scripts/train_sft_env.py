# Quiet mode first, before the heavy libraries load: several of its controls are environment
# defaults those libraries read once at import time. Our own tagged diagnostics and anything
# on stderr still get through -- see scripts/shared/quiet_mode.py.
from shared import quiet_mode
import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = os.environ.get('MINER_DATASETS_DIR', '/cache/miner_datasets')
DEFAULT_SFT_OUTPUT = '/tmp/sft_adapter'
DEFAULT_DATASET_DIRS = ['gradients-io-tournaments--env_training_gradients', 'Mahesh111000--Hanabi_dataset']
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_TARGET = 'all-linear'
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-05
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_MAX_STEPS = -1

def load_from_disk_dataset(dataset_path: Path):
    from datasets import load_from_disk
    if not dataset_path.exists():
        logger.warning('Dataset path %s does not exist', dataset_path)
        return None
    logger.info('Loading DatasetDict from %s', dataset_path)
    dd = load_from_disk(str(dataset_path))
    train_ds = dd['train'] if 'train' in dd else dd
    logger.info('  loaded %d training samples', len(train_ds))
    return train_ds

def load_preprocessed_datasets(cache_dir: Path, dataset_dirs: list):
    from datasets import Dataset, concatenate_datasets, load_dataset
    parts = []
    for name in dataset_dirs:
        preprocessed_path = cache_dir / name / 'preprocessed.jsonl'
        if not preprocessed_path.is_file():
            logger.warning('Skipping %s: %s not found', name, preprocessed_path)
            continue
        logger.info('Loading %s', preprocessed_path)
        ds = load_dataset('json', data_files=str(preprocessed_path), split='train')
        logger.info('  loaded %d samples from %s', len(ds), name)
        parts.append(ds)
    if not parts:
        logger.warning('No preprocessed datasets found in %s', cache_dir)
        return None
    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    logger.info('Combined SFT dataset: %d total samples', len(combined))
    return combined

def load_model_and_tokenizer(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if os.path.isdir(model_name_or_path):
        _lr = os.environ.get('LOCAL_RANK')
        # /cache/models is mounted READ-ONLY in the trainer container, so the cross-rank
        # merge-done sentinel MUST live on a writable, node-shared path (/tmp) — writing it
        # next to the model (/cache/models/<id>.merge_done) crashed every multi-GPU run with
        # OSError(Errno 30, read-only fs). The write is also guarded so it can never be fatal.
        _sentinel = os.path.join('/tmp', os.path.basename(model_name_or_path.rstrip('/')) + '.merge_done')
        if _lr in (None, '0'):
            if _lr == '0':
                try:
                    if os.path.exists(_sentinel):
                        os.remove(_sentinel)
                except OSError:
                    pass
            try:
                from trainer_downloader import _detect_and_merge_lora
                _detect_and_merge_lora(model_name_or_path)
            except Exception as exc:
                logger.warning('adapter-merge guard skipped (%s: %s); loading path as-is', type(exc).__name__, exc)
            if _lr == '0':
                try:
                    open(_sentinel, 'w').close()
                except OSError as exc:
                    logger.warning('merge-done sentinel write skipped (%s); other ranks fall through after timeout', exc)
        else:
            import time as _time
            for _ in range(900):
                if os.path.exists(_sentinel):
                    break
                _time.sleep(1)
    logger.info('Loading tokenizer from %s', model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info('pad_token was None, set to eos_token')
    tokenizer.padding_side = 'right'
    logger.info('Loading full model from %s', model_name_or_path)
    _local_rank = os.environ.get('LOCAL_RANK')
    _world_size = int(os.environ.get('WORLD_SIZE', '1'))
    # Under multi-GPU DDP (torchrun), do NOT pin device_map={'':rank}: it makes the HF
    # Trainer treat the model as model-parallel and BYPASSES trl's distributed guard
    # (the guard only fires when trl builds the model from a path, not for a
    # pre-instantiated object). That is the Qwen3 multi-GPU crash that fell every
    # tournament submission back to the un-noised-base fallback. Load WITHOUT device_map and let
    # the Trainer place each rank's replica on its LOCAL_RANK GPU (the standard DDP pattern).
    if _world_size > 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        _device_map = {'': int(_local_rank)} if _local_rank is not None else 'auto'
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=_device_map, trust_remote_code=True)
    if hasattr(model, 'config'):
        model.config.use_cache = False  # required with gradient_checkpointing
    logger.info('Model loaded: %s (world_size=%d, device_map=%s)', type(model).__name__, _world_size, 'none/DDP' if _world_size > 1 else 'set')
    return (model, tokenizer)

def build_lora_config(args):
    from peft import LoraConfig
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM')
    logger.info('LoRA config: r=%d alpha=%d dropout=%g target=%s', args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
    return config

def assert_toolcall_renders_object(tokenizer):
    """FORFEIT GUARD (deterministic): render a game_action tool_call through the ACTUAL base
    chat template and fail LOUD if `arguments` comes out as a double-encoded string literal
    instead of a JSON object. A model trained on string-literal arguments forfeits every PvP
    turn ("committed no legal move"). This catches that class before we ever ship such a model,
    on any base — the machine-check the pipeline was missing (CLAUDE.md: gate deterministik)."""
    # content is NON-None on purpose: content=None makes apply_chat_template raise on the Qwen2-7B
    # and Hermes-3 templates (they concat content), which previously sent this guard into its except
    # branch and silently no-op'd on 2 of the 7 bases. A non-None content lets the guard actually
    # run its double-encode check on all 7.
    msgs = [{'role': 'user', 'content': 'x'},
            {'role': 'assistant', 'content': 'I will make my move.',
             'tool_calls': [{'id': 'call_0', 'type': 'function',
                             # 987654321 is a SENTINEL, not a plausible action id. The tools schema
                             # below declares the property NAME action_id but no VALUE, and every
                             # pooled template serialises that schema into the prompt — so probing
                             # for the name proves nothing. These digits can only reach `rendered`
                             # through the tool CALL, which is what we actually need to verify.
                             'function': {'name': 'game_action', 'arguments': json.dumps({'action_id': 987654321})}}]}]
    for _c in msgs[1]['tool_calls']:            # apply the same decode the training path applies
        _fn = _c['function']
        if isinstance(_fn.get('arguments'), str):
            _fn['arguments'] = json.loads(_fn['arguments'])
    tools = [{'type': 'function', 'function': {'name': 'game_action', 'description': 'commit move',
              'parameters': {'type': 'object', 'properties': {'action_id': {'type': 'integer'}}, 'required': ['action_id']}}}]
    try:
        rendered = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        raise RuntimeError(
            f'FORFEIT GUARD: apply_chat_template raised {type(exc).__name__}: {exc} — this base template '
            f'cannot render a game_action tool_call with the current message shape; a model trained on it '
            f'would forfeit. Fix before training.') from exc
    if '987654321' not in rendered:
        # BLIND SPOT CLOSED 2026-08-05. The double-encode check below can only fire when the tool
        # call is PRESENT in the rendered text. A template that DROPS tool_calls renders no call at
        # all, so both substrings below are absent and this function fell through to its "✓" —
        # passing the single worst outcome it exists to catch. A model trained on tool-call-free
        # assistant turns forfeits EVERY PvP turn (core/pvp/bot.py raises InvalidActionForfeitError
        # on empty tool_calls, with no text fallback; our harness does the same).
        # ⚠ The predicate is the SENTINEL VALUE, deliberately NOT the string 'action_id'. A first
        # draft probed for the name and was VACUOUS: `tools=` is passed to apply_chat_template and
        # all six pooled templates serialise the schema (Qwen `tool | tojson`, Llama
        # `t | tojson(indent=4)`), so the property NAME is in `rendered` whether or not the call
        # survived. Only the VALUE is call-only. Do not "simplify" this back to a name check.
        # Still correct under double-encoding: a stringified "{\"action_id\": 987654321}" contains
        # the digits, so this passes and control falls through to the double-encode check below —
        # which is the intended ordering (absent call and string-encoded call are different faults).
        # Cost if it ever fires spuriously: run_sft exits non-zero, no success.txt, and
        # text_trainer ships the untrained base. That is a lost env — bad, but strictly better than
        # shipping a model that forfeits every turn. NOTE the 0-rows safety net further down is NOT
        # reached on this path: it lives after tokenize_and_mask, and this raises inside it.
        raise RuntimeError(
            'FORFEIT GUARD TRIPPED: the rendered chat template does not contain the probe tool '
            'call at all — this base template DROPS tool_calls, so every training row would teach '
            'a turn with no tool call and the model would forfeit every PvP turn. DO NOT train.')
    if '\\"action_id\\"' in rendered or '"arguments": "{' in rendered:
        raise RuntimeError('FORFEIT GUARD TRIPPED: game_action `arguments` rendered as a STRING '
                           'literal, not a JSON object → this model would forfeit every PvP turn. '
                           'The tokenize_and_mask arguments-decode is missing/ineffective for this '
                           'base template. DO NOT train — fix first.')
    logger.info('forfeit-guard: game_action arguments render as a JSON object ✓ (%s)',
                getattr(tokenizer, 'name_or_path', '?'))

# Probe values for the string-argument guard below. Two sentinels, one on each side of a newline,
# because the failure this catches is a template that renders the head of a multi-line command and
# loses the tail. The command also carries a double quote and a backslash: those are exactly the
# characters JSON escaping gets wrong, and a mangled escape produces a command the harness runs
# verbatim and the shell then rejects.
_BASH_PROBE_HEAD = 'ZZHEAD987654321'
_BASH_PROBE_TAIL = 'ZZTAIL987654322'

# ORDER MATTERS: the head sentinel must sit on the FIRST line and the tail on a LATER one. With
# both on the second line a truncating template loses them both and trips the dropped-call check
# instead, which reports the wrong fault -- equally fatal, but it sends the reader after the wrong
# bug. Verified by a fake template that truncates at the newline.
_BASH_PROBE_CMD = (
    'grep -rn "needle \\" quote" ' + _BASH_PROBE_HEAD + '\n'
    'echo ' + _BASH_PROBE_TAIL
)
_SWE_TOOLS = [
    {'type': 'function', 'function': {'name': 'execute_bash', 'description': 'run one bash command',
     'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                    'required': ['command']}}},
    {'type': 'function', 'function': {'name': 'submit', 'description': 'finish the task',
     'parameters': {'type': 'object', 'properties': {}}}},
]

def assert_bash_toolcall_renders_object(tokenizer, fatal: bool):
    """FORFEIT GUARD for the swe_infinite shape: a tool call whose argument is a STRING.

    The guard above proves a game_action call renders — but its only argument is an INTEGER, and
    integers survive escaping trivially. swe_infinite trains a different shape: execute_bash takes
    a command STRING full of quotes, backslashes and newlines, and submit takes an EMPTY object.
    Neither is exercised by the integer probe, and the harness is unforgiving about both: it
    discards bash-fence parsing entirely and raises FormatError the moment `tool_calls` is empty,
    with no fallback, so a template that drops or mangles these calls scores zero on every task.

    `fatal` is the caller's judgement about what is at stake. A base whose template cannot render
    the SWE shape should not cost us the six PvP envs that render fine, so the failure is loud but
    non-fatal unless SWE rows are actually in the dataset being trained.
    """
    def _render(msgs):
        for _m in msgs:
            for _c in (_m.get('tool_calls') or []):
                _fn = _c['function']
                if isinstance(_fn.get('arguments'), str):
                    _fn['arguments'] = json.loads(_fn['arguments'])
        return tokenizer.apply_chat_template(msgs, tools=_SWE_TOOLS, tokenize=False,
                                             add_generation_prompt=False)

    def _fail(msg):
        if fatal:
            raise RuntimeError('FORFEIT GUARD TRIPPED (swe_infinite): ' + msg + ' DO NOT train.')
        logger.warning('forfeit-guard (swe_infinite, non-fatal — no SWE rows in this dataset): %s', msg)

    base = [{'role': 'user', 'content': 'Fix the bug.'}]
    call_msgs = base + [{'role': 'assistant', 'content': 'I will look for the failing call site.',
                         'tool_calls': [{'id': 'call_0', 'type': 'function',
                                         'function': {'name': 'execute_bash',
                                                      'arguments': json.dumps({'command': _BASH_PROBE_CMD})}}]}]
    try:
        rendered = _render([dict(m) for m in call_msgs])
    except Exception as exc:
        return _fail(f'apply_chat_template raised {type(exc).__name__}: {exc} — this base cannot '
                     f'render an execute_bash call, so every SWE turn would carry no tool call.')

    # Same ordering as the integer guard: a dropped call and a string-encoded call are different
    # faults, and the dropped-call check has to come first because it makes the other one vacuous.
    if _BASH_PROBE_HEAD not in rendered:
        return _fail('the rendered template does not contain the probe command at all — this base '
                     'DROPS tool_calls, so every SWE row would teach a turn with no tool call.')
    if _BASH_PROBE_TAIL not in rendered:
        return _fail('the probe command survived only up to its newline — this template truncates '
                     'multi-line commands, and real trajectories edit files with multi-line '
                     'heredocs, so the command the model learns would be cut short.')
    if '"arguments": "{' in rendered or '\\"command\\"' in rendered:
        return _fail('execute_bash `arguments` rendered as a STRING literal rather than a JSON '
                     'object — the harness cannot read `command` out of that and raises FormatError.')

    # submit() carries no argument value, so there is no sentinel to look for and the name is
    # already in the serialised tools schema. Render the same turn with and without the call and
    # require them to differ: identical output means the template dropped the call.
    submit_call = base + [{'role': 'assistant', 'content': 'The fix is in place.',
                           'tool_calls': [{'id': 'call_0', 'type': 'function',
                                           'function': {'name': 'submit', 'arguments': '{}'}}]}]
    submit_plain = base + [{'role': 'assistant', 'content': 'The fix is in place.'}]
    try:
        with_call = _render([dict(m) for m in submit_call])
        without_call = _render([dict(m) for m in submit_plain])
    except Exception as exc:
        return _fail(f'apply_chat_template raised {type(exc).__name__}: {exc} on an empty-argument '
                     f'submit call — the episode could never terminate.')
    if with_call == without_call:
        return _fail('an empty-argument submit() call renders identically to no call at all — this '
                     'template drops zero-argument tool calls, so the model would never learn to '
                     'terminate an episode and every task would run to its 900s timeout.')
    logger.info('forfeit-guard: execute_bash string args and submit() render as JSON objects ✓ (%s)',
                getattr(tokenizer, 'name_or_path', '?'))

def _dataset_has_swe_rows(dataset, sample: int = 64) -> bool:
    """Whether any row trains the swe_infinite tool shape. Sampled: this only decides how loud the
    guard is, and scanning a full multi-env dataset to answer it would cost more than it saves."""
    try:
        for i in range(min(sample, len(dataset))):
            raw = dataset[i].get('messages_json') or ''
            if 'execute_bash' in raw:
                return True
            for m in (dataset[i].get('messages') or []):
                for c in (m.get('tool_calls') or []):
                    if (c.get('function') or {}).get('name') == 'execute_bash':
                        return True
    except Exception as exc:
        logger.warning('swe-row probe failed (%s: %s) — treating the SWE guard as fatal, which is '
                       'the safe direction.', type(exc).__name__, exc)
        return True
    return False

def tokenize_and_mask(dataset, tokenizer, max_length: int=4096):
    assert_toolcall_renders_object(tokenizer)   # forfeit guard — fail loud before training a broken model
    assert_bash_toolcall_renders_object(tokenizer, fatal=_dataset_has_swe_rows(dataset))
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def _process(example):
        try:
            # deepcopy: the args-decode below mutates tool_call['arguments'] IN PLACE. Without a copy
            # that mutation lands on the shared 'messages' column object that datasets.map keeps (this
            # _process returns no 'messages'), turning arguments from Value(string) into a struct whose
            # shape varies per row (game_action {action_id} vs memory {slot,content}). Under num_proc>1
            # the per-shard Arrow schemas then can't align → uncaught ValueError kills training whenever
            # the sparse reflection/memory rows happen to land in only some shards (data-order dependent).
            msgs = copy.deepcopy(example.get('messages'))
            if msgs is None and example.get('messages_json'):
                msgs = json.loads(example['messages_json'])
            tools = example.get('tools')
            if tools is None and example.get('tools_json'):
                tools = json.loads(example['tools_json'])
            # FORFEIT FIX: generators store tool_call arguments as a json.dumps STRING. The
            # chat template's `| tojson` filter (Qwen2.5/Llama, unconditional) would then
            # DOUBLE-ENCODE it → the model learns to emit `"arguments": "{\"action_id\": N}"`
            # (a string literal) instead of the object `{"action_id": N}` → the eval parser
            # can't read action_id → forfeit every turn. Decode back to a dict here so the
            # template renders a real JSON object. (Qwen3-4B's template is string-safe, so it
            # was silently immune — which is why intercode runs on Qwen3 looked fine.)
            for _m in (msgs or []):
                for _c in (_m.get('tool_calls') or []):
                    _fn = _c.get('function') or {}
                    if isinstance(_fn.get('arguments'), str):
                        try:
                            _fn['arguments'] = json.loads(_fn['arguments'])
                        except (ValueError, TypeError):
                            pass
            ids = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False)
            mask = [0] * len(ids)
            for (i, msg) in enumerate(msgs):
                if msg['role'] != 'assistant':
                    continue
                p = len(tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=True, add_generation_prompt=True))
                r = len(tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=True, add_generation_prompt=False))
                p = max(0, min(p, len(ids)))
                r = max(0, min(r, len(ids)))
                for j in range(p, r):
                    mask[j] = 1
            if len(ids) > max_length:
                # FRONT-truncate (keep the TAIL). The assistant response (the training target)
                # sits at the END; back-truncation (ids[:max_length]) cut it off for long rows
                # — liars rows with a big accumulated LONG_TERM memory block in the system
                # prompt exceed 2048 → the masked assistant tokens fell past the cutoff →
                # sum(mask)==0 → the row was silently dropped (measured 31% of liars rows).
                # Keeping the tail preserves the response AND the user state (both near the end).
                # Verified on real posterior rows: long-row drop 31%→0%.
                (ids, mask) = (ids[-max_length:], mask[-max_length:])
            if sum(mask) == 0:
                return {'input_ids': [eos_id], 'assistant_masks': [0], '_skip': True}
            return {'input_ids': ids, 'assistant_masks': mask, '_skip': False}
        except Exception as exc:
            logger.warning('tokenize: skip example due to %s: %s', type(exc).__name__, exc)
            return {'input_ids': [eos_id], 'assistant_masks': [0], '_skip': True}
    mapped = dataset.map(_process, num_proc=4, desc='tokenize+mask')
    if '_skip' in mapped.column_names:
        before = len(mapped)
        mapped = mapped.filter(lambda x: not x['_skip'], num_proc=4)
        mapped = mapped.remove_columns(['_skip'])
        dropped = before - len(mapped)
        if dropped > 0:
            logger.warning('tokenize: dropped %d/%d (%.2f%%) malformed/empty examples', dropped, before, 100 * dropped / max(before, 1))
    return mapped

def run_sft(args, dataset, model, tokenizer, peft_config, train_request=None):
    from trl import SFTConfig, SFTTrainer
    dataset = tokenize_and_mask(dataset, tokenizer, args.max_seq_length)
    logger.info('Tokenized + assistant-masked dataset: %d samples', len(dataset))
    if len(dataset) == 0:
        # 0-rows safety net (reliability-first / T0). Every example can drop at tokenize when a
        # {gin_rummy, clobber} pair lands on a tool_calls-dropping base (Qwen2-7B / Hermes-3): those
        # envs' content=None game_action rows raise in apply_chat_template (verified) and are skipped,
        # leaving an EMPTY dataset. SFTTrainer on 0 rows can hard-crash -> FAILED task / -1. Ship the
        # untrained base instead so the run still yields a SERVABLE model (the base keeps its own
        # tool-calling; a degraded-but-alive submission beats a crash). Do NOT "fix" the drop by coercing
        # content='' — that trains the model to emit an EMPTY assistant turn on those bases (CPU-verified),
        # i.e. it poisons toward forfeit; leaving the rows dropped preserves base tool-call ability.
        logger.warning('tokenize: 0 rows survived (all examples dropped) — shipping untrained base (no SFT) to avoid an empty-dataset crash')
        try:
            # RANK-0 ONLY: this branch runs on every rank under torchrun (tokenize is pre-shard so all
            # ranks hit len==0 identically); concurrent multi-GB safetensors writes to the same
            # output_dir would corrupt it. Mirror the merge-sentinel idiom at line ~61. torchrun waits
            # for all ranks to exit, so rank-0's write completes before the outer copy — no barrier needed.
            if os.environ.get('LOCAL_RANK') in (None, '0'):
                if peft_config is not None:
                    from peft import get_peft_model
                    get_peft_model(model, peft_config).save_pretrained(args.output_dir)
                else:
                    model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        except Exception as exc:  # the safety net must never itself crash the run
            logger.warning('zero-rows base save failed: %s', exc)
        return {'zero_rows_skip': True, 'train_runtime': 0.0}
    _world_size_estimate = max(1, int(os.environ.get('WORLD_SIZE', '1')))
    _steps_per_epoch_est = max(1, len(dataset) // (args.batch_size * args.grad_accum_steps * _world_size_estimate))
    _total_steps_est = int(_steps_per_epoch_est * max(args.epochs, 1))
    warmup_dynamic = max(10, min(args.warmup_steps, _total_steps_est // 15))
    logger.info('Warmup: dynamic=%d (capped %d, estimate %d steps total)', warmup_dynamic, args.warmup_steps, _total_steps_est)
    sft_config = SFTConfig(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum_steps, learning_rate=args.learning_rate, weight_decay=0.0, warmup_steps=warmup_dynamic, lr_scheduler_type='cosine_with_min_lr', lr_scheduler_kwargs={'min_lr_rate': args.min_lr_rate}, bf16=True, tf32=True, logging_steps=10, save_strategy='no', save_only_model=True, save_total_limit=1, eval_strategy='no', report_to=args.report_to, max_length=args.max_seq_length, packing=False, gradient_checkpointing=args.gradient_checkpointing, max_steps=args.max_steps if args.max_steps > 0 else -1, optim='paged_adamw_8bit', use_liger_kernel=args.use_liger_kernel, ddp_find_unused_parameters=False)
    logger.info('SFTConfig: epochs=%d lr=%g batch=%d ga=%d max_seq=%d max_steps=%s', args.epochs, args.learning_rate, args.batch_size, args.grad_accum_steps, args.max_seq_length, args.max_steps if args.max_steps > 0 else 'unlimited')
    callbacks = []
    if train_request is not None:
        from finetuning.customized_trainer import CustomEvalSaveCallback, WhenToEvalHandler
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        total_steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum_steps * world_size))
        total_steps_all_epochs = int(total_steps_per_epoch * args.epochs)
        periodic_save_steps = train_request.get('periodic_save_steps', -1)
        max_steps_cfg = train_request.get('max_steps', -1)
        save_before_remaining_time = train_request.get('save_before_remaining_time', 5)
        checking_step = train_request.get('checking_step', 70)
        if checking_step >= total_steps_per_epoch:
            checking_step = max(total_steps_per_epoch - 2, 1)
        logger.info('Callback wired: end_time=%s submission_dir=%s periodic_save_steps=%s save_before_remaining_time=%s checking_step=%s steps_per_epoch=%d', train_request.get('end_time'), train_request.get('submission_dir'), periodic_save_steps, save_before_remaining_time, checking_step, total_steps_per_epoch)
        callbacks.append(CustomEvalSaveCallback(WhenToEvalHandler(train_request['end_time'], save_before_remaining_time, periodic_save_steps=periodic_save_steps, steps_per_epoch=total_steps_per_epoch, max_steps=max_steps_cfg), train_request['submission_dir'], args.output_dir, train_request['model_name'], max_steps_cfg, checking_step=checking_step, total_steps_all_epochs=total_steps_all_epochs, end_time=train_request['end_time'], checking_mode=train_request.get('checking_mode', 'none')))
    else:
        logger.info('No --request-path; running standalone (no save callback)')
    # Gradient-checkpointing correctness, made explicit instead of relying on trl
    # version behavior: trl 0.27 auto-enables input grads ONLY for the LoRA/peft path;
    # the --no-lora full-FT path (sub-2B models) needs this or backward throws
    # "element 0 of tensors does not require grad". use_reentrant=False is the torch>=2 safe mode.
    if args.gradient_checkpointing:
        sft_config.gradient_checkpointing_kwargs = {'use_reentrant': False}
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config, callbacks=callbacks or None)
    logger.info('Starting SFT training...')
    train_result = trainer.train()
    logger.info('SFT training done: %s', train_result.metrics)
    logger.info('Saving LoRA adapter to %s', args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _added_tokens = os.path.join(args.output_dir, 'added_tokens.json')
    if os.path.exists(_added_tokens):
        os.remove(_added_tokens)
        logger.info('Removed added_tokens.json (clean LoRA; vocab unchanged)')
    logger.info('SFT adapter + tokenizer saved')
    return train_result.metrics

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-name-or-path', required=True, help='Base model (HF repo or local path)')
    p.add_argument('--dataset-path', default=None, help='HF DatasetDict directory (self-gen trajectory mode). If set, overrides --cache-dir/--datasets.')
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    p.add_argument('--datasets', default=None, help='Comma-separated dataset dirs (org--name). Default: MINER_DATASETS env var, then built-in defaults.')
    p.add_argument('--output-dir', default=DEFAULT_SFT_OUTPUT, help=f'Where to save adapter or model (default: {DEFAULT_SFT_OUTPUT})')
    p.add_argument('--epochs', type=float, default=DEFAULT_EPOCHS)
    p.add_argument('--learning-rate', type=float, default=DEFAULT_LR)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum-steps', type=int, default=4)
    p.add_argument('--warmup-steps', type=int, default=20)
    p.add_argument('--min-lr-rate', type=float, default=0.25, help='cosine_with_min_lr floor as fraction of peak LR (0.25 = decay to 25%% of peak instead of 0)')
    p.add_argument('--max-seq-length', type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS, help=f'Max training steps (cap budget). -1 = unlimited. Default {DEFAULT_MAX_STEPS}.')
    p.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing (slower but lower mem)')
    p.add_argument('--use-liger-kernel', action='store_true', help='Enable Liger fused kernels incl. fused-linear cross-entropy: computes loss WITHOUT materializing the full (batch x seq x vocab) logits tensor → prevents the OOM that scales with vocab/batch.')
    p.add_argument('--no-lora', action='store_true', help='Disable LoRA, do Full FT (self-gen high-data mode)')
    p.add_argument('--lora-r', type=int, default=DEFAULT_LORA_R)
    p.add_argument('--lora-alpha', type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument('--lora-dropout', type=float, default=0.0)
    p.add_argument('--lora-target-modules', default=DEFAULT_LORA_TARGET)
    p.add_argument('--report-to', default='none', help='wandb / tensorboard / none')
    p.add_argument('--request-path', default=None, help='JSON path with train_request (end_time, submission_dir, periodic_save_steps, save_before_remaining_time, checking_step). When set, wires CustomEvalSaveCallback so latest checkpoint is copied to submission_dir on periodic saves and force-saved before end_time, preventing empty-submission on validator timeout. When None, callback is skipped (standalone mode).')
    return p.parse_args()

def main():
    # basicConfig installs a root handler AND resets the root level, which would undo the clamp
    # quiet mode applied at import and put this module's whole INFO trail -- model path, dataset
    # path, submission_dir, output_dir, metrics -- straight onto stderr. Handlers write through
    # stream.write(), so they bypass the print filter entirely and nothing downstream would catch
    # it. Pick the level from quiet mode instead of hardcoding INFO, which also leaves a developer
    # running with TRAIN_QUIET=0 the full trail they expect.
    # NEVER below WARNING. logging filters at Logger.isEnabledFor() BEFORE any handler runs, so an
    # ERROR floor drops every logger.warning() in this process silently -- on a channel the print
    # filter never sees, since log records do not go through builtins.print. The warnings in this
    # module are the forfeit class: dataset path missing, no preprocessed datasets, adapter-merge
    # guard skipped, zero rows survived and shipping an untrained base. Losing the last one means
    # shipping an untrained model with a log that says nothing.
    logging.basicConfig(level=logging.WARNING if quiet_mode.is_enabled() else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s', stream=sys.stderr)
    args = parse_args()
    if args.dataset_path:
        logger.info('Self-gen trajectory mode: dataset_path=%s', args.dataset_path)
        dataset = load_from_disk_dataset(Path(args.dataset_path))
    else:
        if args.datasets:
            dataset_dirs = [d.strip() for d in args.datasets.split(',') if d.strip()]
        else:
            env_val = os.getenv('MINER_DATASETS', '')
            if env_val:
                dataset_dirs = [d.strip() for d in env_val.split(',') if d.strip()]
                logger.info('Using MINER_DATASETS env var: %s', dataset_dirs)
            else:
                dataset_dirs = DEFAULT_DATASET_DIRS
                logger.info('No --datasets/MINER_DATASETS; defaulting to: %s', dataset_dirs)
        cache_dir = Path(args.cache_dir)
        dataset = load_preprocessed_datasets(cache_dir, dataset_dirs)
    if dataset is None or len(dataset) == 0:
        logger.warning('No SFT data found — exiting 0 (GRPO phase will start from base model). Verify trajectory generation or preprocessing ran successfully.')
        return 0
    train_request = None
    if args.request_path:
        request_path = Path(args.request_path)
        if request_path.is_file():
            try:
                with open(request_path, 'r') as f:
                    train_info = json.load(f)
                train_request = train_info.get('train_request') or train_info
                logger.info('Loaded train_request from %s', request_path)
            except Exception as exc:
                logger.warning('Failed to parse %s (%s); proceeding without callback', request_path, exc)
        else:
            logger.warning('--request-path %s not found; proceeding without callback', request_path)
    (model, tokenizer) = load_model_and_tokenizer(args.model_name_or_path)
    peft_config = None if args.no_lora else build_lora_config(args)
    if args.no_lora:
        logger.info('Full FT mode (--no-lora): all base model params trainable')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_sft(args, dataset, model, tokenizer, peft_config, train_request)
    logger.info('SFT phase complete. Adapter at: %s', args.output_dir)
    logger.info('Metrics: %s', metrics)
    success_marker = output_dir / 'success.txt'
    success_marker.write_text('ok')
    logger.info('Wrote success.txt to %s', success_marker)
    return 0
if __name__ == '__main__':
    sys.exit(main())
