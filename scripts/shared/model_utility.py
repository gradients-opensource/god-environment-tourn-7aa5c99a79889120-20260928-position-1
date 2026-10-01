import re
from huggingface_hub import HfApi
from transformers import AutoConfig
import glob
from safetensors.torch import load_file
from pathlib import Path
import torch
import os
import json
from transformers import AutoTokenizer
MODEL_CONFIG = {'facebook/opt-1.3b': {'model_size': 1300000000}, 'facebook/opt-3b': {'model_size': 3000000000}, 'facebook/opt-6.7b': {'model_size': 6700000000}, 'facebook/opt-13b': {'model_size': 13000000000}, 'EleutherAI/gpt-neo-1.3B': {'model_size': 1300000000}, 'EleutherAI/gpt-neo-125m': {'model_size': 125000000}, 'bigscience/bloom-560m': {'model_size': 560000000}, 'TinyLlama/TinyLlama_v1.1': {'model_size': 1100000000}}
hf_api = HfApi()

def get_model_architecture(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = config.architectures
        if len(architectures) > 1:
            return 'Multiple architectures'
        return architectures[0].strip().lower()
    except:
        return 'Unknown'

def get_use_liger(architecture: str) -> str:
    if architecture.lower() in ['qwen2forcausallm', 'llamaforcausallm', 'gemma2forcausallm', 'mixtralforcausallm', 'mistralforcausallm', 'qwen3forcausallm', 'phi3forcausallm', 'gemmaforcausallm']:
        return 'True'
    else:
        return 'False'

def count_params_from_safetensors(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, '*.safetensors'))
    if not shards:
        return None
    for shard_path in shards:
        print(f'Loading shard: {shard_path}')
        tensors = load_file(shard_path)
        total_params += sum((v.numel() for v in tensors.values()))
    return total_params

def count_params_from_bin(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, '*.bin'))
    if not shards:
        return None
    for shard_path in shards:
        print(f'Loading shard: {shard_path}')
        try:
            state_dict = torch.load(shard_path, map_location='cpu')
            total_params += sum((v.numel() for v in state_dict.values()))
        except Exception as e:
            print(f'cannot load {shard_path}: {e}')
            continue
    return total_params

def get_model_size_from_local_path(model_path: str) -> int:
    size = count_params_from_safetensors(model_path)
    if size is not None and size > 1000:
        print(f'Model size from safetensors: {size}')
        return size
    size = count_params_from_bin(model_path)
    if size is not None and size > 1000:
        print(f'Model size from bin: {size}')
        return size
    return None

def estimate_params_from_config(model_path: str):
    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception:
        return None
    h = getattr(config, 'hidden_size', None)
    layers = getattr(config, 'num_hidden_layers', None)
    vocab = getattr(config, 'vocab_size', None)
    if not (h and layers and vocab):
        return None
    inter = getattr(config, 'intermediate_size', None) or 4 * h
    per_layer = 4 * h * h + 3 * h * inter
    return int(vocab * h * 2 + layers * per_layer)

def get_gpu_count():
    return torch.cuda.device_count()

def get_model_num_params(model_id: str, model_path: str) -> int:
    if model_id in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]['model_size']
    try:
        size = get_model_size_from_local_path(model_path)
        if size is not None:
            return size
        raise Exception(f'Cannot get model size from {model_path}')
    except Exception as e:
        print(f'Error getting model size from safetensors: {e}')
    # config.json estimate BEFORE the name regex: config (hidden_size/layers/vocab)
    # is reliable and present even for anonymized per-task base models, whereas the
    # name regex mis-parses decimals ("1.5B"->5B) and fabricates from hex hashes
    # ("...-9b..."->9B). Name parse is now the last resort, decimal-aware + boundary-
    # bounded so it only matches real size tokens (e.g. "-7B-", "-1.5B-"), not hashes.
    est = estimate_params_from_config(model_path)
    if est:
        print(f'Model size estimated from config.json: {est}')
        return est
    model_size = re.search(r'(?:^|[-_/ ])(\d+(?:\.\d+)?)\s*[bB](?:[-_/.]|$)', model_id or '')
    if model_size:
        n = int(float(model_size.group(1)) * 1000000000)
        print(f'Model size from regex: {n}')
        return n
    print('Model size: could not detect — falling back to 7.6B (tournament base class)')
    return 7600000000

def disable_flash_attention(architecture: str, model: str) -> str:
    if model == 'microsoft/phi-2':
        return 'True'
    if 'falcon-rw' in model.lower():
        return 'True'
    if architecture.strip().lower() in ['gptneoforcausallm', 'bloomforcausallm', 'gptossforcausallm']:
        return True
    else:
        return False

def detect_param_count_bucket(model_path: str) -> str:
    size = get_model_size_from_local_path(model_path)
    if size is None:
        return 'unknown'
    params_b = size / 1000000000
    if params_b < 1:
        return '0_1_b'
    if params_b < 2:
        return '1_2_b'
    if params_b < 4:
        return '2_4_b'
    if params_b < 5:
        return '4_5_b'
    if params_b < 6:
        return '5_6_b'
    if params_b < 9:
        return '6_9_b'
    if params_b < 12:
        return '9_12_b'
    if params_b < 15:
        return '12_15_b'
    if params_b < 20:
        return '15_20_b'
    if params_b < 40:
        return '20_40_b'
    return '40_80_b'
