from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import TrainerCallback, TrainerState, TrainerControl
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from shared.state_manager import get_state, set_state
MAX_TRIES = 9
MIS_MATCH_VOCAB_SIZE_MODELS = ['NousResearch/Nous-Capybara-7B-V1', 'berkeley-nest/Starling-LM-7B-alpha', 'NousResearch/Hermes-2-Theta-Llama-3-8B', 'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4']
ERROR_GENERATION_CONFIG_MODELS = ['lmsys/vicuna-7b-v1.5', 'lmsys/vicuna-13b-v1.5', 'NousResearch/Nous-Hermes-llama-2-7b', 'defog/llama-3-sqlcoder-8b']
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
print(f'LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py', flush=True)

def _atomic_publish(src_dir: str, dst_dir: str) -> None:
    """Publish the contents of src_dir into dst_dir without ever leaving dst_dir
    absent or half-written.

    Old behaviour was rmtree(dst_dir) then copytree(src_dir, dst_dir): for the
    whole copy, dst_dir either doesn't exist or holds a partial model, and a
    kill (validator SIGKILL, OOM, etc.) landing in that window turns a
    would-be-scored submission into nothing.

    New behaviour: copy into a sibling temp dir first, then swap it onto
    dst_dir with os.replace (a single rename, atomic on POSIX for dirs on the
    same filesystem). dst_dir is only ever touched by that one rename, so it
    always contains either the previous complete publish or the new complete
    publish - never a partial one. If the copy itself raises partway through,
    the temp dir is cleaned up and dst_dir is left exactly as it was.
    """
    dst_dir = dst_dir.rstrip('/') or dst_dir
    parent = os.path.dirname(os.path.abspath(dst_dir)) or '.'
    base = os.path.basename(dst_dir)
    tmp_dir = os.path.join(parent, f'.{base}.tmp-{os.getpid()}')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    try:
        shutil.copytree(src_dir, tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    old_dir = None
    if os.path.exists(dst_dir):
        old_dir = os.path.join(parent, f'.{base}.old-{os.getpid()}')
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        os.replace(dst_dir, old_dir)
    os.replace(tmp_dir, dst_dir)
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)

def _sync_end_time_across_ranks(local_is_end_time: bool) -> bool:
    """Make the end_time decision agree across all ranks in a torchrun job.

    end_time is a per-rank wall-clock check (WhenToEvalHandler / this callback's
    caller), so two ranks can disagree by one step: rank A crosses the threshold
    at step N and exits the training loop, while rank B is still at step N and
    proceeds into the next forward/backward. Any collective inside that next
    step (gradient all_reduce, distributed save, etc.) then has one rank missing
    and hangs until the validator's wall-clock SIGKILL.

    Fix: every rank calls this every on_step_end with its own local boolean, we
    all_reduce(MAX) a 1/0 flag, and every rank uses the SAME synced result. If
    torch.distributed isn't available/initialized or world_size<=1 (no
    torchrun, or a single process), this is a no-op passthrough of the local
    value - identical to today's single-process behaviour.
    """
    dist = getattr(torch, 'distributed', None)
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return local_is_end_time
    if dist.get_world_size() <= 1:
        return local_is_end_time
    use_cuda = torch.cuda.is_available() and dist.get_backend() == 'nccl'
    device = torch.device(f'cuda:{LOCAL_RANK}') if use_cuda else torch.device('cpu')
    flag = torch.tensor([1 if local_is_end_time else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())

class CustomEvalSaveCallback(TrainerCallback):

    def __init__(self, function_when_to_evaluate: Callable, submission_dir: str, output_dir: str, original_model_name: str, max_steps: int=-1, checking_step: int=100, total_steps_all_epochs: int=-1, end_time: str='', checking_mode: str='none'):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time

    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get('eval_loss', None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.checking_step and self.checking_mode == 'first_time':
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state['train']['start_time'], '%Y-%m-%d %H:%M:%S')
            start_train_time_obj = datetime.datetime.strptime(my_state['train']['start_train_time'], '%Y-%m-%d %H:%M:%S')
            log_content = f'Checking the model at step: {state.global_step}'
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f'\nPreparation time: {preparation_time}'
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f'\nTime so far: {time_so_far}'
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f'\nTime for one step: {time_for_one_step}'
            log_content += f'\nTotal steps all epochs: {self.total_steps_all_epochs}'
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f'\nTotal remaining training time: {total_remaining_training_time}'
            end_time_obj = datetime.datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f'\nTotal remaining time: {total_remaining_time}'
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar)
            n = int(n)
            my_state['check_details'] = {'now': str(now.strftime('%Y-%m-%d %H:%M:%S')), 'start_time': str(start_time_obj.strftime('%Y-%m-%d %H:%M:%S')), 'start_train_time': str(start_train_time_obj.strftime('%Y-%m-%d %H:%M:%S')), 'checking_step': self.checking_step, 'checking_mode': self.checking_mode, 'estimation_of_steps': n, 'preparation_time': preparation_time, 'time_so_far': time_so_far, 'time_for_one_step': time_for_one_step, 'total_remaining_training_time': total_remaining_training_time, 'total_remaining_time': total_remaining_time, 'end_time': self.end_time}
            if n > 0:
                log_content += f'\nEstimated number of steps to complete the training: {n}'
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = 'no'
                last_log = state.log_history[-1]
                my_state['train']['current_loss'] = last_log['loss']
                my_state['mode'] = 'continue'
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f'\nFinal number: {n + 1}'
                my_state['next_runs'] = n + 1
            else:
                print(f'Time is not enough so we will finish the training', flush=True)
                my_state['mode'] = 'finish'
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)
            return control
        elif state.global_step == self.checking_step and self.checking_mode == 'second_time':
            log_content = f'Checking the model at step: {state.global_step} where check_mode=second_time'
            my_state = get_state()
            current_loss = state.log_history[-1]['loss']
            my_state['train']['current_loss'] = current_loss
            control.should_training_stop = True
            current_is_the_best = False
            current_min_loss = min([run['current_loss'] for run in my_state['runs']])
            if current_loss <= current_min_loss:
                if len(my_state['runs']) + 1 == my_state['next_runs']:
                    print(f"Current loss: {my_state['train']['current_loss']} is greater than: {current_min_loss}", flush=True)
                    current_is_the_best = True
            if current_is_the_best:
                control.should_training_stop = False
                my_state['mode'] = 'finish'
            else:
                control.should_save = False
                args.save_strategy = 'no'
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        # Collective sync MUST run every step, for every rank, regardless of what this
        # rank's own reason is - a collective call that only some ranks make is exactly
        # the desync this is meant to prevent.
        end_time_synced = _sync_end_time_across_ranks(when_to_eval['reason'] == 'end_time')
        if end_time_synced and when_to_eval['reason'] != 'end_time':
            print(f'***SYNC: another rank crossed end_time at step {state.global_step}; stopping in lockstep', flush=True)
            when_to_eval = {'eval': True, 'reason': 'end_time'}
        if when_to_eval['eval']:
            if args.do_eval:
                print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
                control.should_evaluate = True
                control.should_save = True
            else:
                control.should_evaluate = False
                control.should_save = True
                self.save_only = True
            if when_to_eval['reason'] == 'end_time':
                # Always stop on end_time, even if a periodic checkpoint already exists.
                # Previously should_training_stop was only set in the "no checkpoint yet"
                # branch below, so once a periodic save had happened training kept running
                # past end_time until the validator's wall-clock SIGKILL — trainer.save_model()
                # and the final clean copy in text_trainer.py never ran, so the published
                # submission was whatever mid-schedule checkpoint on_save last wrote.
                control.should_training_stop = True
                if not self.has_checkpoint:
                    print(f'No checkpoint found, just save the model at step: {state.global_step}', flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        self.save_only = False
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return
        print(f'GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}', flush=True)
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info['loss']:
            print(f'Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}', flush=True)
            self.best_checkpoint_info = {'loss': eval_loss, 'step': state.global_step}
            self.update_best_checkpoint = True
        elif self.best_checkpoint_info is not None:
            print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)

    # Publish is atomic (see _atomic_publish above): stage the full copy into a sibling
    # tmp dir, then a single os.replace swaps it onto submission_dir - never an rmtree of
    # the destination before the replacement exists. Extra disk used is ONE extra copy of
    # submission_dir for the duration of the copy (LoRA: well under 1GB; full-FT of a ~1.5B
    # model: ~3GB; task disk budget is 1TB), and the swap itself is O(1) (a rename, not a
    # copy). Real copy-time and peak-disk for the largest model in the pool are still
    # SMOKE-NEEDED on a GPU box.
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f'Stop training because of max steps: {self.max_steps}', flush=True)
            control.should_training_stop = True
        self.has_checkpoint = True
        if not is_main_process(LOCAL_RANK):
            # HF's Trainer._save_checkpoint runs (and, for distributed strategies, does its
            # own internal collective gathering of shards) on ALL ranks BEFORE on_save
            # fires, so checkpoint-{step} is already complete on disk by the time we get
            # here - no new barrier needed for rank-0 to safely read it.
            # Only rank-0 (is_main_process) publishes to submission_dir: on_save fires on
            # every rank, and letting every rank run _atomic_publish/write loss.txt against
            # the SAME submission_dir means multiple processes race the same
            # os.path.exists -> os.replace(dst, old-<pid>) sequence (TOCTOU), so a second
            # rank's os.replace can hit a dst that the first rank already moved -> crash.
            # Non-main ranks still reset the same instance flags rank-0 resets below, so
            # this callback's local state (self.save_only / self.update_best_checkpoint)
            # stays consistent with rank-0's for the next on_step_end/on_evaluate call.
            self.save_only = False
            self.update_best_checkpoint = False
            return
        if self.save_only:
            print(f'Only save the model at step: {state.global_step}, no evaluation', flush=True)
            current_step = state.global_step
            _atomic_publish(os.path.join(self.output_dir, f'checkpoint-{current_step}'), self.submission_dir)
            self.update_best_checkpoint = False
            with open(os.path.join(self.submission_dir, 'loss.txt'), 'w') as f:
                f.write(f'{current_step},no_eval')
            self.save_only = False
            return
        if self.update_best_checkpoint and is_main_process(LOCAL_RANK):
            print(f'Copy the best checkpoint to the submission directory at step: {state.global_step}', flush=True)
            best_eval_loss = self.best_checkpoint_info['loss']
            _atomic_publish(os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"), self.submission_dir)
            self.update_best_checkpoint = False
            with open(os.path.join(self.submission_dir, 'loss.txt'), 'w') as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss}")

def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool:
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    end_time = end_time.replace(tzinfo=timezone.utc)
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result = time_diff.total_seconds() < minutes * 60
    if result:
        print(f'*** current time: {now} end_time: {end_time} time_diff: {time_diff}', flush=True)
    return result

class WhenToEvalHandler:

    def __init__(self, end_time: str, save_before_remaining_time: int=3, periodic_save_steps: int=-1, steps_per_epoch: int=-1, max_steps: int=-1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and (global_step > 1):
            return {'eval': True, 'reason': 'epoch'}
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and (global_step > 1):
            return {'eval': True, 'reason': 'periodic'}
        if self.save_before_remaining_time > 0 and (not self.run_eval):
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f'***ALERT: The time is about to run out need to eval & save the model', flush=True)
                self.run_eval = True
                return {'eval': True, 'reason': 'end_time'}
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f'Stop training because of max steps: {self.max_steps}', flush=True)
            return {'eval': True, 'reason': 'max_step'}
        return {'eval': False, 'reason': 'none'}

def init_wandb(train_request: Dict):
    return True
    task_id = train_request['task_id']
    expected_repo_name = train_request['expected_repo_name']
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = train_request['wandb_log_dir']
    os.environ['WANDB_RUN_ID'] = f'{task_id}_{expected_repo_name}'
    os.environ['WANDB_NAME'] = f'{task_id}_{expected_repo_name}'
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request['wandb_log_dir'], exist_ok=True)
    return True
