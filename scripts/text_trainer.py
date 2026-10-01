# Quiet mode first, before the heavy libraries load: several of its controls are environment
# defaults those libraries read once at import time. Our own tagged diagnostics and anything
# on stderr still get through -- see scripts/shared/quiet_mode.py.
from shared import quiet_mode
import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from shared.state_manager import get_state, set_state
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
import pathlib
from finetuning import lr_utils
from shared import train_cst
from shared import training_paths as train_paths
from core.models.utility_models import TaskType
from core.config.sft_env_config import get_training_json as get_sft_env_training_json
from core.config.sft_env_config import supports_sft_trajectory
from pvp_infra.baseline_stats import load_baseline_stats, get_init_loss
from transformers import AutoConfig

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict=None):
    with open(log_file_path, 'w') as log_file:
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=process_env)
        for line in process.stdout:
            # One rule, both sinks. quiet_mode.relay decides what this line becomes: dropped,
            # passed through, or reduced to its tag and trigger word. The log file used to keep
            # verbatim what the filter had just removed from stdout, and the file outlives the run.
            relayed = quiet_mode.relay(line)
            if relayed is None:
                continue
            # _real_print, not print: print is already the quiet-mode shim, so re-entering it would
            # route this to stderr and quietly empty the stdout sink this line is meant for.
            quiet_mode._real_print(relayed, end='', file=sys.stdout, flush=True)
            log_file.write(relayed)
            log_file.flush()
        return_code = process.wait()
        log_file.write(f'\nProcess completed with return code: {return_code}\n')

def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    if cmd is None:
        return None
    for variant in {arg_name, arg_name.replace('_', '-')}:
        match = re.search(f'(?P<p>--{variant}(\\s+)([^\\s]+))(\\s+)', cmd)
        if match:
            left_index = match.start('p')
            right_index = match.end('p')
            return cmd[:left_index] + f' --{variant} {arg_value} ' + cmd[right_index:]
    return cmd

def extract_value_from_cmd(cmd: str, arg_name: str):
    if cmd is None:
        return None
    for variant in {arg_name, arg_name.replace('_', '-')}:
        match = re.search(f'(?P<p>--{variant}(\\s+)(?P<value>[^\\s]+))(\\s+)', cmd)
        if match:
            return match.group('value')
    return None

def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return 'Multiple architectures'
        return architectures[0].strip().lower()
    except Exception as e:
        if 'model type `gpt_oss`' in str(e):
            return 'GptOssForCausalLM'
        return 'Unknown'

def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == 'gptossforcausallm':
        return True
    return False
OOM_ERROR = 'torch.OutOfMemoryError: CUDA out of memory'
VLLM_OOM_ERROR = 'ValueError: No available memory for the cache blocks'

def get_error_type(log_path: str):
    with open(log_path, 'r') as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None

def extract_output_dir(train_cmd: str) -> str:
    match = re.search('--output_dir\\s+(.*?)\\s+', train_cmd)
    if match:
        return match.group(1)
    else:
        return None

# Environment tasks retry ONLY on an early CUDA OOM. An OOM surfaces in the first optimizer steps
# (model load + first micro-batch), so a failed attempt that ran longer than this was not an OOM
# we can fix by shrinking the micro-batch — and re-running the same command late in the window
# could overwrite a deadline-saved adapter with a fresh, worse one. 15 min covers model load,
# torchrun spin-up and the first steps on the slowest bucket.
ENV_OOM_RETRY_MAX_ATTEMPT_SECONDS = 900
# ...and only while enough of the absolute window is left for the retry to matter. end_time is fixed
# at task start (see the main block below) and shared by every attempt through the request json,
# so a retry never extends the window; but a retry that starts with only minutes left would hit
# end_time at once and publish a step-0-ish checkpoint over whatever the first attempt had saved.
ENV_OOM_RETRY_MIN_REMAINING_MINUTES = 25

def _env_remaining_minutes(train_cmd: str):
    """Minutes until the task's absolute end_time, read from the request json the command points
    at (--request-path). None when it cannot be determined; callers must treat None as 'do not
    retry' — an unreadable request is not a reason to gamble the window."""
    try:
        request_path = extract_value_from_cmd(train_cmd, 'request_path')
        with open(request_path, 'r') as f:
            request = json.load(f)
        # The env request json is the dict get_training_json returns: end_time sits under
        # 'train_request' (same lookup train_sft_env.py uses to read it).
        end_time = (request.get('train_request') or request)['end_time']
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        return (end_time - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception as exc:
        print(f'Environment task: cannot read end_time from the request json ({type(exc).__name__}: {exc})', flush=True)
        return None

def _shrink_env_train_cmd_after_oom(train_cmd: str):
    """Halve --batch-size and double --grad-accum-steps (same effective batch, so the bucket's
    LR pairing is preserved). These are the flag names _build_run_cmd (sft_env_config) actually
    emits; the generic branch below looks for per_device_train_batch_size, which the env command
    never carries, so before this the env OOM path could not read the batch back at all."""
    current = extract_value_from_cmd(train_cmd, 'batch_size')
    if current is None or int(current) <= 1:
        return None
    new_batch = int(current) // 2
    train_cmd = replace_args_in_cmd(train_cmd, 'batch_size', str(new_batch))
    accum = extract_value_from_cmd(train_cmd, 'grad_accum_steps')
    if accum is not None:
        train_cmd = replace_args_in_cmd(train_cmd, 'grad_accum_steps', str(int(accum) * 2))
    print(f'OOM: reducing --batch-size {current} -> {new_batch}, --grad-accum-steps {accum} -> {int(accum) * 2 if accum is not None else accum}', flush=True)
    return train_cmd

def run_training(train_cmd: str, log_path: str, task_id: str, retries: int, task_type: str, expected_repo_name: str, wandb_mode: str='offline', wandb_project: str=None, wandb_entity: str=None):
    last_attempt_seconds = 0.0
    for i in range(retries):
        print(f'************* Training attempt {i + 1}/{retries} for task {task_id}*************', flush=True)
        if i > 0 and task_type == TaskType.ENVIRONMENTTASK.value:
            error_type = get_error_type(log_path) if os.path.exists(log_path) else None
            if error_type != OOM_ERROR:
                print(f'Environment task: previous attempt did not fail with CUDA OOM ({error_type}); not retrying', flush=True)
                return False
            if last_attempt_seconds > ENV_OOM_RETRY_MAX_ATTEMPT_SECONDS:
                print(f'Environment task: OOM after {last_attempt_seconds:.0f}s (> {ENV_OOM_RETRY_MAX_ATTEMPT_SECONDS}s); too late to retry safely', flush=True)
                return False
            remaining = _env_remaining_minutes(train_cmd)
            if remaining is None or remaining < ENV_OOM_RETRY_MIN_REMAINING_MINUTES:
                print(f'Environment task: {remaining if remaining is None else round(remaining)} min left before end_time (< {ENV_OOM_RETRY_MIN_REMAINING_MINUTES}); not retrying', flush=True)
                return False
            shrunk = _shrink_env_train_cmd_after_oom(train_cmd)
            if shrunk is None:
                print('Environment task: --batch-size already 1 (or absent), cannot reduce further', flush=True)
                return False
            train_cmd = shrunk
        elif i > 0:
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(train_cmd, 'per_device_train_batch_size')
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(f'Reducing batch size from {current_batch_size} to {new_batch_size}', flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, 'per_device_train_batch_size', str(new_batch_size))
                    else:
                        print(f'batch size is 1, cannot reduce further', flush=True)
                        if task_type == TaskType.GRPOTASK.value:
                            train_cmd = replace_args_in_cmd(train_cmd, 'use_vllm', 'False')
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f'VLLM OOM error, disable VLLM', flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, 'use_vllm', 'False')
        if os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('STARTING TRAINING')
        # The trainer container is airgapped from huggingface.co (DNS fails) — the model is
        # already local in /cache/models, so force offline so from_pretrained loads it directly
        # instead of burning retries on a HEAD check it can never complete.
        training_env_vars = {'WANDB_MODE': wandb_mode, 'WANDB_RUN_ID': f'{task_id}_{expected_repo_name}', 'WANDB_NAME': f'{task_id}_{expected_repo_name}', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        wandb_token = os.environ.get('WANDB_API_KEY') or os.environ.get('WANDB_TOKEN')
        if wandb_token:
            training_env_vars['WANDB_API_KEY'] = wandb_token
        if wandb_mode == 'online':
            if wandb_project:
                training_env_vars['WANDB_PROJECT'] = wandb_project
            if wandb_entity:
                training_env_vars['WANDB_ENTITY'] = wandb_entity
        attempt_started = time.time()
        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        last_attempt_seconds = time.time() - attempt_started
        output_dir = extract_value_from_cmd(train_cmd, 'output_dir')
        if os.path.exists(os.path.join(output_dir, 'success.txt')):
            return True
        time.sleep(5)
    return False

def patch_wandb_symlinks(base_dir: str):
    for (root, _, files) in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                target_path = os.readlink(full_path)
                print(f'Symlink: {full_path} → {target_path}')
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f'Failed to unlink {full_path}: {e}')
                    continue
                if os.path.exists(target_path):
                    print('Copying real file')
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f'Failed to copy: {e}')
                else:
                    print('Target not found, creating dummy')
                    pathlib.Path(full_path).touch()

def delete_poor_checkpoints(train_runs: list[dict]):
    lowest_loss = min([run['current_loss'] for run in train_runs])
    for run in train_runs:
        if run['current_loss'] > lowest_loss:
            if os.path.exists(run['output_dir']):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run['output_dir'])

def get_log_scale(task_type: str):
    # EnvTask uses the SFT-family default (0.18); .get avoids a KeyError crash in the
    # continue-mode LR search when task_type isn't in the map.
    log_scale_map = {TaskType.INSTRUCTTEXTTASK.value: 0.18, TaskType.DPOTASK.value: 0.18, TaskType.GRPOTASK.value: 0.2, TaskType.CHATTASK.value: 0.18}
    return log_scale_map.get(task_type, 0.18)

def _dir_has_safetensors(dir_path: str) -> bool:
    """Mirror of the per-subdir test in the validator uploader.

    trainer/containers/uploader.py::detect_subfolder walks the DIRECT child dirs of the
    folder it is asked to upload and does, name-agnostically:

        has_checkpoint_files = False
        for file in os.listdir(item_path):
            if file.endswith('.safetensors'):
                has_checkpoint_files = True
                break
        if has_checkpoint_files:
            return item_path

    The FIRST subdir holding a *.safetensors REPLACES the root folder as the thing that
    gets uploaded. Because the test never looks at the name, checkpoint-<N>/ (HF periodic
    save), last-checkpoint/ (HF Trainer _push_from_checkpoint), best_model/ and
    global_step<N>/ (DeepSpeed) all hijack the upload identically.
    """
    try:
        for file in os.listdir(dir_path):
            if file.endswith('.safetensors'):
                return True
    except OSError:
        return False
    return False

def find_safetensors_subdirs(base_folder: str) -> list:
    """Every direct child dir of base_folder that uploader.detect_subfolder() would latch
    onto. detect_subfolder returns only the first one; we return all of them, because
    deleting the first would just promote the next."""
    if not os.path.isdir(base_folder):
        return []
    try:
        entries = sorted(os.listdir(base_folder))
    except OSError:
        return []
    found = []
    for item in entries:
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path) and _dir_has_safetensors(item_path):
            found.append(item_path)
    return found

def _sweep_stale_tmp_copies(directory: str) -> None:
    """Remove any leftover '<name>.tmp-<pid>' staging file/dir from a previous
    _atomic_copy_file attempt that crashed or was killed mid-copy. These must never
    survive into an upload - the uploader has no concept of them, but they are dead
    weight at best and, if a future rename step is ever added elsewhere, a source of
    exactly the corruption this fix is meant to prevent."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if '.tmp-' in name:
            path = os.path.join(directory, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f'[text_trainer] swept stale tmp copy artifact: {path}', flush=True)
            except OSError as sweep_exc:
                print(f'Warning: failed to sweep stale tmp artifact {path}: {type(sweep_exc).__name__}: {sweep_exc}', flush=True)

def _atomic_copy_file(src: str, dst: str) -> None:
    """Copy a single file into dst without ever leaving dst truncated/partial.

    Old behaviour was shutil.copy2(src, dst): it writes straight over an existing dst
    in place. A kill landing mid-copy (validator SIGKILL at hours*3600, ~3 minutes after
    end_time - see customized_trainer.py's end_time handling) leaves a half-written file;
    for a *.safetensors that means the eval load fails outright -> score None.

    New behaviour: stage the copy at dst + '.tmp-<pid>' in the SAME directory (same
    filesystem, so the swap below is a rename, not a copy), then os.replace(tmp, dst) -
    a single atomic rename on POSIX. dst is only ever touched by that one rename, so it
    always holds either the previous complete content or the new complete content, never
    a partial write. If the copy itself raises partway through, the tmp file is removed
    and dst is left exactly as it was.
    """
    # Staged NEXT TO submission_dir (its parent), not inside it: same filesystem so the rename
    # stays atomic, but a kill mid-copy leaves the partial file outside the folder the uploader
    # ships. Inside it, the leftover would be uploaded (the uploader's ignore_patterns do not
    # cover it) and the post-kill sweep never runs.
    parent, sub = os.path.split(os.path.dirname(os.path.abspath(dst)))
    tmp_dst = os.path.join(parent, f'.{sub}.{os.path.basename(dst)}.tmp-{os.getpid()}')
    try:
        shutil.copy2(src, tmp_dst)
    except Exception:
        if os.path.exists(tmp_dst):
            try:
                os.remove(tmp_dst)
            except OSError:
                pass
        raise
    os.replace(tmp_dst, dst)

def copy_run_output_to_submission(run_output_dir: str, submission_dir: str):
    """Copy the finished trainer output into submission_dir.

    Always overwrite with the FINAL trainer.save_model() output. The old skip-if-populated
    guard meant that whenever the periodic-save callback had already written a checkpoint,
    we shipped that intermediate checkpoint and threw the finished model away.
    dirs_exist_ok overwrites directory trees in place; top-level FILES go through
    _atomic_copy_file (stage-then-rename) instead of copy2-in-place, so a kill mid-copy of
    a top-level weight file can never leave a truncated file behind.

    The skip rule is NAME-AGNOSTIC and mirrors uploader.detect_subfolder(): skip any
    SUBDIR that directly contains a *.safetensors, whatever it is called. Such a dir would
    (a) be uploaded on top of the final weights, doubling the transfer against a few-minute
    upload reserve, and worse (b) be picked by detect_subfolder() INSTEAD of the root, so
    we would silently ship intermediate weights. Root-level files are always copied;
    subdirs WITHOUT safetensors (runs/ tensorboard logs, etc.) are copied normally.
    .cache is skipped as before.

    Returns (copied_count, skipped_names).
    """
    os.makedirs(submission_dir, exist_ok=True)
    _sweep_stale_tmp_copies(submission_dir)
    _parent, _sub = os.path.split(os.path.abspath(submission_dir))
    for _n in os.listdir(_parent):   # staging leftovers from _atomic_copy_file (see there)
        if _n.startswith(f'.{_sub}.') and '.tmp-' in _n and not os.path.isdir(os.path.join(_parent, _n)):
            try:
                os.remove(os.path.join(_parent, _n))
            except OSError:
                pass
    copied = 0
    skipped = []
    # adapter_config.json LAST: the evaluator decides "this is a LoRA" from that file alone
    # (validator model_checks.check_for_lora). Landing it before adapter_model.safetensors opens a
    # window where the repo reads as a LoRA with no adapter weights and fails to load; landing it
    # last means the repo is always either the previous complete state or the new complete one.
    for name in sorted(os.listdir(run_output_dir), key=lambda n: (n == 'adapter_config.json', n)):
        src = os.path.join(run_output_dir, name)
        if name == '.cache':
            skipped.append(name)
            continue
        if os.path.isdir(src) and (name.startswith('checkpoint-') or _dir_has_safetensors(src)):
            # Name-agnostic, mirroring the uploader's own subfolder detection, PLUS the legacy
            # name rule. A checkpoint-* directory without safetensors is still skipped: the
            # uploader's ignore_patterns are root-relative fnmatch, so 'optimizer.pt' never matches
            # 'checkpoint-50/optimizer.pt' and the optimizer state would be uploaded with it.
            skipped.append(name)
            continue
        dst = os.path.join(submission_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            _atomic_copy_file(src, dst)
        copied += 1
    return (copied, skipped)

def purge_hijacking_subdirs(submission_dir: str) -> list:
    """Second guard, run AFTER the copy: re-check submission_dir with the uploader's own
    rule and delete anything that would hijack the upload. This covers writers other than
    the copy above (e.g. a periodic-save callback pointed straight at submission_dir)."""
    removed = []
    if find_safetensors_subdirs(submission_dir) and (not _dir_has_safetensors(submission_dir)):
        # Root holds NO weights, so the safetensors subdir(s) are the only ones that exist and
        # detect_subfolder() will ship one of them. Deleting here would leave nothing to upload,
        # which is strictly worse than shipping an intermediate checkpoint. Never purge blind.
        print(f'Purge SKIPPED: {submission_dir} has no root *.safetensors - the safetensors subdir(s) are the ONLY weights present and detect_subfolder() will ship them; deleting them would leave nothing to upload.', flush=True)
        return removed
    for path in find_safetensors_subdirs(submission_dir):
        print(f'WARNING: {path} contains *.safetensors - uploader.detect_subfolder() would upload THAT subdir INSTEAD of {submission_dir}, shipping the WRONG weights. Removing it.', flush=True, file=sys.stderr)
        try:
            shutil.rmtree(path)
            removed.append(os.path.basename(path))
            print(f'WARNING: removed upload-hijacking subdir {path}', flush=True)
        except Exception as rm_exc:
            print(f'WARNING: FAILED to remove upload-hijacking subdir {path}: {type(rm_exc).__name__}: {rm_exc} - the upload may ship the WRONG weights', flush=True, file=sys.stderr)
    if removed:
        _removed_list = ', '.join(removed)
        print(f'Purged {len(removed)} upload-hijacking subdir(s) from {submission_dir}: {_removed_list}', flush=True)
    return removed

def main():
    print('---STARTING TEXT TRAINING SCRIPT---', flush=True)
    parser = argparse.ArgumentParser(description='Text Model Training Script')
    parser.add_argument('--task-id', required=True, help='Task ID')
    parser.add_argument('--model', required=True, help='Model name or path')
    parser.add_argument('--dataset', required=True, help='Dataset path or HF dataset name')
    parser.add_argument('--dataset-type', required=True, help='JSON string of dataset type config')
    parser.add_argument('--task-type', required=True, choices=['InstructTextTask', 'DpoTask', 'GrpoTask', 'ChatTask', 'EnvTask'], help='Type of task')
    parser.add_argument('--file-format', required=False, choices=['csv', 'json', 'hf', 's3'], help='File format', default='s3')
    parser.add_argument('--hours-to-complete', type=float, required=True, help='Number of hours to complete the task')
    parser.add_argument('--expected-repo-name', help='Expected repository name')
    parser.add_argument('--max-data-size', type=int, help='Max data size to use for training', default=-1)
    parser.add_argument('--max-steps', type=int, help='Max steps to use for training', default=-1)
    parser.add_argument('--retries', type=int, help='Number of retries', default=5)
    parser.add_argument('--min-steps', type=int, help='Min steps to use for training', default=100)
    parser.add_argument('--reg-ratio', type=float, help='Reg ratio to use for training', default=1.0)
    parser.add_argument('--wandb-mode', type=str, choices=['offline', 'online', 'disabled'], help='Wandb mode: offline (default), online, or disabled', default='offline')
    parser.add_argument('--wandb-project', type=str, help='Wandb project name (required for online mode)', default=None)
    parser.add_argument('--wandb-entity', type=str, help='Wandb entity/team name (optional for online mode)', default=None)
    args = parser.parse_args()
    if args.wandb_mode == 'online' and (not args.wandb_project):
        args.wandb_project = 'Gradients-On-Demand'
        print(f"Info: --wandb-mode is set to 'online' but --wandb-project not provided. Using default project: '{args.wandb_project}'", flush=True)
    original_model_name = args.model
    original_task_type = args.task_type
    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f'Error creating dataset type object: {e}')
    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    print(f'submission_dir: {submission_dir}', flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)
    output_dir = f'/workspace/scripts/soutputs/{args.task_id}'
    os.makedirs(output_dir, exist_ok=True)
    end_time = datetime.now(timezone.utc) + timedelta(hours=args.hours_to_complete - 3 / 60)
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    print('end_time: ', end_time, flush=True)
    ds_folder = 'datasets'
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f'training_request_{args.task_id}.json')
    model_path = str(train_paths.get_text_base_model_path(original_model_name))
    # B3 (audit 24 Agu) — ASURANSI KILL. Penulis PERTAMA ke submission_dir hanya
    # customized_trainer.on_save, dan save paling cepat menyala di periodic_save_steps=100
    # (train_sft_env memakai save_strategy='no', jadi HF Trainer tak pernah menyimpan sendiri).
    # Hard-kill kapan pun SEBELUM optimizer step ke-100 meninggalkan submission_dir KOSONG ->
    # uploader melempar ValueError -> runtime.py menyetel success=False -> NOL upload -> skor 0.0
    # (ingat: skor hilang = 0.0 mentah lalu di-rank, bukan 0.5, bukan diekslusi).
    # Ini bukan skenario teoretis: fase pra-training tidak seluruhnya berbatas waktu —
    # swe_infinite dan intercode tidak di-time-box.
    # Kita pra-publikasikan bobot base apa adanya sebagai jaring: worst case berubah dari
    # SKOR 0 menjadi SKOR BASE-MODEL.
    # copy_run_output_to_submission does NOT clear the directory first, it copies over the top.
    # So record exactly what was placed here and remove it before the training output is copied in.
    # Without that, full base weights can end up mixed with the LoRA adapter and confuse the
    # uploader's detection.
    _prepub_names = []
    try:
        if os.path.isdir(model_path) and (not os.listdir(submission_dir)):
            for _n in os.listdir(model_path):
                _s = os.path.join(model_path, _n)
                _d = os.path.join(submission_dir, _n)
                if os.path.isdir(_s):
                    shutil.copytree(_s, _d, dirs_exist_ok=True)
                else:
                    shutil.copy2(_s, _d)
                _prepub_names.append(_n)
            purge_hijacking_subdirs(submission_dir)
            print(f'[text_trainer] PRA-PUBLIKASI {len(_prepub_names)} entri base sebagai asuransi kill',
                  flush=True)
    except Exception as _pp:
        _prepub_names = []
        print(f'[text_trainer] pra-publikasi dilewati: {type(_pp).__name__}: {_pp}',
              flush=True, file=sys.stderr)
    is_openai = False
    if is_openai_model(model_path):
        print('Upgrading python packages for openai model', flush=True)
        run_cmd_with_log('pip uninstall -y transformers && pip install transformers==4.55.0', os.path.join(ds_folder, f'upgrade_transformers.log'))
        run_cmd_with_log('pip uninstall -y deepspeed && pip install deepspeed==0.17.4', os.path.join(ds_folder, f'upgrade_deepspeed.log'))
        run_cmd_with_log('pip install kernels==0.9.0', os.path.join(ds_folder, f'install_kernel.log'))
        is_openai = True
    baseline_stats = load_baseline_stats()
    if baseline_stats is not None:
        init_loss = get_init_loss(baseline_stats)
        print(f"[BASELINE_STATS] task_type={baseline_stats.get('task_type')} init_loss={init_loss}", flush=True)
    train_info = {'model_name': original_model_name, 'model_path': model_path, 'task_id': args.task_id, 'dataset': dataset_path, 'hours_to_complete': args.hours_to_complete, 'expected_repo_name': args.expected_repo_name, 'end_time': end_time, 'dataset_type': dataset_type_dict, 'submission_dir': submission_dir, 'output_dir': output_dir, 'adjust_batch_size': True, 'request_path': request_path, 'max_data_size': args.max_data_size, 'max_steps': args.max_steps, 'wandb_log_dir': train_cst.WANDB_LOGS_DIR, 'min_steps': args.min_steps, 'is_openai': is_openai, 'reg_ratio': args.reg_ratio, 'find_lk_lr': True, 'checking_mode': 'first_time'}
    env_unsupported_fallback = False
    if args.task_type != TaskType.ENVIRONMENTTASK.value:
        sys.exit(f'Environment-only miner: unsupported task type {args.task_type!r}')
    if args.task_type == TaskType.ENVIRONMENTTASK.value:
        env_names = dataset_type_dict.get('environment_names')
        if env_names and isinstance(env_names, list) and (len(env_names) > 0):
            candidate_envs = list(env_names)
        else:
            candidate_envs = [dataset_type_dict.get('environment_name', '')]
        supported_envs = [e for e in candidate_envs if supports_sft_trajectory(e)]
        dropped_envs = [e for e in candidate_envs if not supports_sft_trajectory(e)]
        if dropped_envs:
            print(f'[text_trainer] Dropping envs with no SFT generator: {dropped_envs}. Training on supported subset: {supported_envs}.', flush=True, file=sys.stderr)
        tokenize_cmd = ''
        train_cmd = ''
        if supported_envs:
            dataset_type_dict['environment_names'] = supported_envs
            train_info['dataset_type'] = dataset_type_dict
            try:
                train_info = get_sft_env_training_json(train_info)
                tokenize_cmd = train_info['generate_cmd']
                train_cmd = train_info['run_cmd']
            except Exception as exc:
                print(f'[text_trainer] env SFT plan build failed ({type(exc).__name__}: {exc}); un-noised base.', flush=True, file=sys.stderr)
                env_unsupported_fallback = True
        else:
            print(f'[text_trainer] No SFT generator for {candidate_envs}; un-noised base.', flush=True, file=sys.stderr)
            env_unsupported_fallback = True
    else:
        raise ValueError(f'Task type {args.task_type} not supported')
    with open(request_path, 'w') as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)
    if tokenize_cmd:
        run_cmd_with_log(tokenize_cmd, os.path.join(ds_folder, f'tokenize_{args.task_id}.log'))
    original_train_cmd = train_cmd
    train_success = False
    state = get_state()
    state = {}
    set_state(state)
    state['mode'] = 'initial'
    set_state(state)
    count = 0
    while True:
        if env_unsupported_fallback:
            print(f'Skipping training loop for task {args.task_id}; emitting un-noised fallback submission.', flush=True, file=sys.stderr)
            break
        state = get_state()
        train_cmd = original_train_cmd
        c_train_info = copy.deepcopy(train_info)
        final_output_dir = None
        if args.task_type == TaskType.GRPOTASK.value or args.task_type == TaskType.ENVIRONMENTTASK.value:
            state['mode'] = 'finish'
            c_train_info['train_request']['checking_mode'] = 'none'
        elif state['mode'] == 'initial':
            c_train_info['train_request']['checking_mode'] = 'first_time'
        elif state['mode'] == 'continue':
            c_train_info['train_request']['checking_mode'] = 'second_time'
            n_runs = state['next_runs']
            if 'lrs' not in state:
                current_lr = float(state['train']['lr'])
                state['lrs'] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
                assert len(state['lrs']) == n_runs, f"Number of learning rates {state['lrs']} should be equal to number of runs {n_runs}"
                state['runs'] = []
            set_state(state)
            state['runs'].append(state['train'].copy())
            delete_poor_checkpoints(state['runs'])
            if len(state['runs']) < n_runs:
                index = len(state['runs'])
                current_lr = state['lrs'][index]
                train_cmd = replace_args_in_cmd(train_cmd, 'learning_rate', str(state['lrs'][index]))
            else:
                c_train_info['train_request']['checking_mode'] = 'none'
                index = np.argmin([run['current_loss'] for run in state['runs']])
                print(f"BL;{index};{state['runs'][index]['current_loss']}; {state['lrs'][index]}", flush=True)
                train_cmd = state['runs'][index]['train_cmd']
                final_output_dir = state['runs'][index]['output_dir']
                state['mode'] = 'finish'
        else:
            assert state['mode'] == 'finish'
            break
        set_state(state)
        if train_cmd:
            run_output_dir = output_dir + f'_{count}' if not final_output_dir else final_output_dir
            train_cmd = replace_args_in_cmd(train_cmd, 'output_dir', run_output_dir)
            current_request_path = os.path.join(ds_folder, f'training_request_{args.task_id}_{count}.json')
            with open(current_request_path, 'w') as f:
                json.dump(c_train_info, f, indent=4, ensure_ascii=False)
            train_cmd = replace_args_in_cmd(train_cmd, 'request_path', current_request_path)
            state['train'] = {'train_cmd': train_cmd, 'log_path': os.path.join(ds_folder, f'train_{args.task_id}.log'), 'lr': extract_value_from_cmd(train_cmd, 'learning_rate'), 'output_dir': run_output_dir}
            state['train']['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            set_state(state)
            log_path = state['train']['log_path']
            # Environment tasks: exactly one extra attempt, and run_training only takes it after an
            # EARLY CUDA OOM (see ENV_OOM_RETRY_MAX_ATTEMPT_SECONDS) with a halved --batch-size.
            # Any other failure still ends the task after the first attempt, as before.
            effective_retries = 2 if args.task_type == TaskType.ENVIRONMENTTASK.value else args.retries
            success = run_training(train_cmd, log_path, args.task_id, effective_retries, args.task_type, args.expected_repo_name, args.wandb_mode, args.wandb_project, args.wandb_entity)
            time.sleep(5)
            if not success:
                print(f'Training failed for task {args.task_id} at count={count}', flush=True, file=sys.stderr)
                break
            else:
                try:
                    if os.path.isdir(run_output_dir):
                        # Copy the training output FIRST, then remove any pre-publication base
                        # entries that were NOT overwritten by that copy. Same-name entries are
                        # already atomically replaced inside copy_run_output_to_submission
                        # (_atomic_copy_file / copytree), so removing them separately here would
                        # be redundant; entries only the pre-publication step wrote (no
                        # counterpart in run_output_dir) are stale and safe to drop now that
                        # submission_dir has real training-output weights on disk. Doing the
                        # removal AFTER the copy - not before, as before this fix - means
                        # submission_dir is never weight-empty: it always holds either the
                        # pre-published base or the (being-)copied training output, never neither.
                        copied, skipped = copy_run_output_to_submission(run_output_dir, submission_dir)
                        print(f'Copied {copied} entries from {run_output_dir} → {submission_dir} (final model overwrites any periodic checkpoint)', flush=True)
                        if skipped:
                            _skipped_list = ', '.join(skipped)
                            print(f'Skipped {len(skipped)} entries that must not ship (safetensors-bearing subdirs would hijack uploader.detect_subfolder(); kept in {run_output_dir}): {_skipped_list}', flush=True)
                        if _prepub_names:
                            try:
                                run_output_names = set(os.listdir(run_output_dir))
                            except OSError:
                                run_output_names = set()
                            stale_prepub = [n for n in _prepub_names if n not in run_output_names]
                            for _n in stale_prepub:
                                _p = os.path.join(submission_dir, _n)
                                try:
                                    shutil.rmtree(_p) if os.path.isdir(_p) else os.remove(_p)
                                except FileNotFoundError:
                                    pass
                            if stale_prepub:
                                print(f'[text_trainer] pra-publikasi dibersihkan ({len(stale_prepub)} entri stale, tidak tertimpa salinan baru)',
                                      flush=True)
                            _prepub_names = []
                    else:
                        print(f'Warning: run_output_dir {run_output_dir} does not exist post-training; submission_dir will remain whatever the callback wrote.', flush=True)
                except Exception as copy_exc:
                    print(f'Warning: failed to copy {run_output_dir} → {submission_dir}: {type(copy_exc).__name__}: {copy_exc}', flush=True)
                try:
                    purge_hijacking_subdirs(submission_dir)
                except Exception as sweep_exc:
                    print(f'Warning: hijack sweep of {submission_dir} failed: {type(sweep_exc).__name__}: {sweep_exc}', flush=True)
                print(f'Training successfully done for task {args.task_id} at count={count}', flush=True)
                break
        count += 1
    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f'Training failed for task {args.task_id}', flush=True, file=sys.stderr)
    else:
        print(f'Training successfully done for task {args.task_id}', flush=True)
        train_success = True
    if not train_success:
        # Fallback when training produced no usable submission.
        # Do NOT add noise to the base model. The validator's env-trainer guidance is
        # explicit: noised-base submissions forfeit turn after turn, and for continuation
        # (R2+) the model starts from a good checkpoint so noise destroys real work. Submit
        # the UN-NOISED starting model instead — for a fresh task that is the base (plays at
        # base level instead of forfeiting); for a continuation it is the previous adapter
        # (preserves the prior round's training). Same submission shape the old noise path
        # produced, minus the destructive noise. Reason is written to a file so it survives
        # the log tail; the actual crash traceback remains in train_{task_id}.log.
        reason = ('env plan unsupported or build-failed' if env_unsupported_fallback
                  else 'training failed or produced no submission')
        fb_log = os.path.join(ds_folder, f'fallback_{args.task_id}.log')
        try:
            os.makedirs(submission_dir, exist_ok=True)
            if os.path.isdir(model_path):
                copied = 0
                for name in os.listdir(model_path):
                    src = os.path.join(model_path, name)
                    dst = os.path.join(submission_dir, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                    copied += 1
                purge_hijacking_subdirs(submission_dir)
                outcome = f'submitted UN-NOISED starting model: copied {copied} entries {model_path} -> {submission_dir}'
            else:
                outcome = f'model_path {model_path} is not a directory; cannot copy starting model, submission_dir left as-is'
        except Exception as fb_exc:
            outcome = f'fallback copy FAILED: {type(fb_exc).__name__}: {fb_exc}'
        try:
            with open(fb_log, 'w') as _fb:
                _fb.write(f'[text_trainer] FALLBACK task={args.task_id}\nreason={reason}\nmodel_path={model_path}\nsubmission_dir={submission_dir}\n{outcome}\n')
        except Exception:
            pass
        print(f'[text_trainer] FALLBACK task={args.task_id} ({reason}): {outcome}', flush=True, file=sys.stderr)
    # B1 (audit 24 Agu): dipanggil SETELAH bobot ada di submission_dir, dan main() tak punya
    # Top-level try/except. The os.walk/os.readlink/.touch() calls inside are NOT otherwise
    # guarded, and if one raises, the exit code is non-zero, runtime.py:1135 `if success:` is never
    # reached, the upload never runs, and the score is 0 even though training finished and the
    # weights exist. The validator is asymmetric here: a timeout forces success=True and still
    # uploads, a crash does not. This block is purely wandb log cosmetics and must never be the
    # reason a finished submission is lost.
    try:
        patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)
    except Exception as _wb:
        print(f'[text_trainer] wandb symlink patch dilewati: {type(_wb).__name__}: {_wb}',
              flush=True, file=sys.stderr)
if __name__ == '__main__':
    main()
