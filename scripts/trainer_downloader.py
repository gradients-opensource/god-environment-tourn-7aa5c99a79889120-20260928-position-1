# Quiet mode first, before the heavy libraries load: several of its controls are environment
# defaults those libraries read once at import time. Our own tagged diagnostics and anything
# on stderr still get through -- see scripts/shared/quiet_mode.py.
import shared.quiet_mode  # noqa: F401
import argparse
import asyncio
import json
import os
import shutil
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import CLIPTokenizer, T5TokenizerFast
from core.models.utility_models import TaskType
from shared import train_cst as cst
LORA_ADAPTER_CONFIG = 'adapter_config.json'

CHAT_TEMPLATE_FILE = 'chat_template.jinja'


def _read_chat_template(source_dir: str) -> str | None:
    """Chat template from a merged/adapter dir, or None. A LoRA adapter frequently ships its
    chat template ONLY as a standalone chat_template.jinja (not inline in tokenizer_config.json),
    so a plain merge that rebuilds the tokenizer from the base silently loses it. Mirrors the
    validator's LoRA-merge chat-template carry-over (trainer/containers/downloader.py, #1294/#1295)."""
    jinja_path = os.path.join(source_dir, CHAT_TEMPLATE_FILE)
    if os.path.exists(jinja_path):
        with open(jinja_path) as f:
            template = f.read().strip()
        if template:
            return template
    config_path = os.path.join(source_dir, 'tokenizer_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            template = json.load(f).get('chat_template')
        if isinstance(template, str) and template.strip():
            return template
    return None


def _detect_and_merge_lora(model_dir: str) -> None:
    adapter_config_path = os.path.join(model_dir, LORA_ADAPTER_CONFIG)
    if not os.path.exists(adapter_config_path):
        return
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config.get('base_model_name_or_path')
    if not base_model_id:
        print(f'WARNING: {LORA_ADAPTER_CONFIG} missing base_model_name_or_path, skipping merge — adapter at {model_dir} cannot be used as base.', flush=True)
        return
    print(f'[downloader] LoRA adapter detected in {model_dir}', flush=True)
    print(f'[downloader] Base model: {base_model_id}', flush=True)
    base_dir = os.path.join(os.path.dirname(model_dir), f'_base_for_{os.path.basename(model_dir)}')
    if not os.path.exists(base_dir):
        snapshot_download(repo_id=base_model_id, repo_type='model', local_dir=base_dir, local_dir_use_symlinks=False)
    _detect_and_merge_lora(base_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[downloader] Merging LoRA on {device}', flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    lora_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if len(lora_tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(lora_tokenizer))
    merged = PeftModel.from_pretrained(base_model, model_dir)
    merged = merged.merge_and_unload(safe_merge=False)
    # Carry the adapter's chat template onto the merged tokenizer (validator parity, #1294/#1295):
    # an adapter often ships it ONLY as chat_template.jinja, which a base-rebuilt tokenizer drops —
    # losing it changes how every tool-calling prompt renders vs how the validator serves the lineage.
    chat_template = _read_chat_template(model_dir) or getattr(base_tokenizer, 'chat_template', None)
    if hasattr(merged, '_hf_peft_config_loaded'):
        merged._hf_peft_config_loaded = False
    merge_tmp = model_dir + '.merged_tmp'
    if os.path.exists(merge_tmp):
        shutil.rmtree(merge_tmp)
    os.makedirs(merge_tmp, exist_ok=True)
    merged.save_pretrained(merge_tmp, safe_serialization=True)
    target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= len(base_tokenizer) else base_tokenizer
    if chat_template and not getattr(target_tokenizer, 'chat_template', None):
        target_tokenizer.chat_template = chat_template
    target_tokenizer.save_pretrained(merge_tmp)
    if chat_template:
        with open(os.path.join(merge_tmp, CHAT_TEMPLATE_FILE), 'w') as f:
            f.write(chat_template)
    del base_model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(model_dir)
    os.rename(merge_tmp, model_dir)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    print(f'[downloader] LoRA merge complete → {model_dir}', flush=True)

async def download_axolotl_base_model(repo_id: str, save_dir: str) -> str:
    # EnvTask/continuation base download: writes to /cache/models/<repo--id> (EXACTLY the path
    # the trainer reads via training_paths.get_text_base_model_path) and, if the repo is a LoRA
    # adapter (R2+ continuation), merges it into its own base so the /cache-read-only, airgapped
    # trainer only ever loads a ready full model.
    model_dir = os.path.join(save_dir, repo_id.replace('/', '--'))
    if os.path.exists(model_dir):
        print(f'Model {repo_id} already exists at {model_dir}. Skipping download.', flush=True)
        _detect_and_merge_lora(model_dir)
        return model_dir
    snapshot_download(repo_id=repo_id, repo_type='model', local_dir=model_dir, local_dir_use_symlinks=False)
    _detect_and_merge_lora(model_dir)
    return model_dir

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task-type', required=True, choices=[TaskType.ENVIRONMENTTASK.value])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--file-format')
    parser.add_argument('--num-samples', type=int, default=-1)
    args = parser.parse_args()
    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f'Downloading datasets to: {dataset_dir}', flush=True)
    print(f'Downloading models to: {model_dir}', flush=True)
    if args.task_type == TaskType.ENVIRONMENTTASK.value:
        # Env-only miner: no external dataset to fetch (pyspiel/intercode self-generate
        # in-container); download the base model and merge any continuation adapter.
        model_path = await download_axolotl_base_model(args.model, model_dir)
    else:
        raise ValueError(f'Environment-only downloader: unsupported task type {args.task_type!r}')
    print(f'Model path: {model_path}', flush=True)
    print(f'Dataset path: {dataset_dir}', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
