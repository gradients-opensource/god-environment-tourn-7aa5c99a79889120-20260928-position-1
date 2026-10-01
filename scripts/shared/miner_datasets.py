"""Shared loader for validator-mounted miner datasets (MINER_DATASETS_DIR / MINER_DATASETS).

The validator's downloader (G.O.D trainer/containers/downloader.py:download_miner_dataset)
snapshot_downloads the whole HF dataset repo as-is into
/cache/miner_datasets/<org>--<name>/. That directory therefore holds the actual data file(s)
PLUS repo metadata that rides along for free: README.md, .gitattributes, dataset_infos.json,
and (for some repos, e.g. intercode_bigcode_combined_12k) a `<name>.meta.json` sidecar.

A bare `load_dataset(<dir>, split='train')` needs a builder config to succeed and often
doesn't have one for a raw snapshot; the natural fallback is to glob for data files. But an
unfiltered `**/*.json` glob happily hands those metadata files to the json builder as if they
were rows too -> mismatched columns -> DatasetGenerationCastError -> the whole dataset silently
dropped. `find_data_files` below is the fix: pick the real data file(s) first (parquet over
jsonl over json, metadata/README excluded), and only fall back to directory auto-detect when
none are found.
"""
import glob
import os

# Sidecar/metadata files a raw HF snapshot may contain alongside real data files. None of
# these are miner-authored rows.
_METADATA_JSON_NAMES = {'dataset_info.json', 'dataset_infos.json'}


def _is_metadata_json(path):
    base = os.path.basename(path)
    return (
        base.lower().startswith('readme')
        or base.endswith('.meta.json')
        or base in _METADATA_JSON_NAMES
    )


def find_data_files(root):
    """Return (files, fmt) for the first data format present under root, else ([], None).

    Priority: parquet > jsonl > json (metadata/README-shaped .json excluded)."""
    parquet = sorted(glob.glob(os.path.join(root, '**', '*.parquet'), recursive=True))
    if parquet:
        return parquet, 'parquet'
    jsonl = sorted(glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True))
    if jsonl:
        return jsonl, 'json'
    json_files = sorted(
        f for f in glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
        if not _is_metadata_json(f)
    )
    if json_files:
        return json_files, 'json'
    return [], None


def load_miner_dataset_dir(path):
    """Load one validator-mounted miner dataset directory as its 'train' split.

    Tries the explicit data file(s) first (see find_data_files); falls back to bare-directory
    auto-detect only when no data files are found. Raises on failure -- callers decide how to
    log/skip, since a silently-swallowed load failure is exactly the bug this replaces."""
    from datasets import load_dataset
    files, fmt = find_data_files(path)
    if files:
        return load_dataset(fmt, data_files=files, split='train')
    return load_dataset(path, split='train')
