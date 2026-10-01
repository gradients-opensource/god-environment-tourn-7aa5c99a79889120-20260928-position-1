#!/usr/bin/env python3
"""Pre-submission artifact guard.

Read-only. Answers one question: *is what we are about to upload actually a
trained model?*  Two miners scored 0.0 in the Jul-27 tournament by uploading
weights byte-identical to the base model; this catches that class of failure
before the artifact leaves the box.

Checks (see check_* functions):
  (i)   full-weights: sha256 of every shard vs the base model's shards.
        All identical => the model was never trained => FAIL.
  (ii)  LoRA: adapter present, plausibly sized, not truncated, and NOT a
        mathematical no-op.  A freshly-initialised PEFT adapter has lora_B == 0,
        so dW = B @ A == 0 and the model behaves exactly like the base.
  (iii) required files for the detected path (config / tokenizer / chat template).
  (iv)  stray files that must not ship (optimizer state, checkpoint-* dirs, *.py).
  (v)   human-readable summary; exit code 1 if any FAIL.

Deliberately stdlib-only (hashlib/json/struct) so it can run in any container or
venv without installing anything.  The safetensors header is parsed directly
rather than via torch: an IEEE float of +0.0 is all-zero bits in fp32/fp16/bf16,
so "tensor is all zero" == "tensor's bytes are all zero", which needs no dtype
support and no GPU.

FAIL-CLOSED: when a definitive test cannot be performed (no --base-model, a
base dir with no shards, or a non-PEFT adapter naming) the verdict is FAIL, not
WARN.  A guard whose only job is to catch an untrained artifact must not return
exit 0 exactly in the case where it was unable to look.  Pass
--allow-unverified-base to downgrade those to WARN when you knowingly have no
base copy on the box.

Usage:
    python3 presubmit_check.py --submission-dir DIR --base-model DIR [--json]
    python3 presubmit_check.py --submission-dir DIR --allow-unverified-base
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import struct
import sys

# --- reporting primitives ---------------------------------------------------

FAIL, WARN, INFO, OK = "FAIL", "WARN", "INFO", "OK"

_ICON = {FAIL: "[FAIL]", WARN: "[WARN]", INFO: "[info]", OK: "[ ok ]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, check: str, msg: str) -> None:
        self.rows.append((level, check, msg))

    @property
    def failed(self) -> bool:
        return any(lvl == FAIL for lvl, _, _ in self.rows)

    @property
    def warned(self) -> bool:
        return any(lvl == WARN for lvl, _, _ in self.rows)


# --- helpers ----------------------------------------------------------------

_HASH_CHUNK = 8 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}PB"


def read_safetensors_header(path: str) -> tuple[dict, int, int]:
    """Return (header_dict, data_start_offset, file_size).

    safetensors layout: u64 LE header length | JSON header | raw tensor data.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError("file shorter than the 8-byte header length field")
        (hdr_len,) = struct.unpack("<Q", raw_len)
        if hdr_len <= 0 or hdr_len > size:
            raise ValueError(f"implausible header length {hdr_len} for a {size}B file")
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
    return header, 8 + hdr_len, size


def tensor_entries(header: dict) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict) and "data_offsets" in v]


def region_is_all_zero(fh, start: int, end: int) -> bool:
    """True iff every byte in [start, end) is 0x00. Streams; never loads it all."""
    fh.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = fh.read(min(_HASH_CHUNK, remaining))
        if not chunk:
            return True  # truncated; caller reports the truncation separately
        if any(chunk):
            return False
        remaining -= len(chunk)
    return True


def model_shards(directory: str) -> list[str]:
    """Top-level full-weight shards (adapters excluded)."""
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        if name.startswith("adapter_"):
            continue
        if os.path.isfile(os.path.join(directory, name)):
            out.append(name)
    return out


def resolve_upload_dir(local_folder: str) -> str:
    """Replicate the validator uploader's detect_subfolder().

    trainer/containers/uploader.py descends into the FIRST immediate subdirectory
    that directly contains a *.safetensors file and uploads that instead. A stray
    checkpoint-N/ therefore hijacks the upload, so the guard must inspect whatever
    the uploader would actually ship, not what we assume it ships.
    """
    if not os.path.isdir(local_folder):
        return local_folder
    for item in sorted(os.listdir(local_folder)):
        item_path = os.path.join(local_folder, item)
        if not os.path.isdir(item_path):
            continue
        try:
            entries = os.listdir(item_path)
        except OSError:
            continue
        if any(f.endswith(".safetensors") for f in entries):
            return item_path
    return local_folder


def has_chat_template(directory: str) -> tuple[bool, str]:
    if os.path.exists(os.path.join(directory, "chat_template.jinja")):
        return True, "chat_template.jinja"
    tc = os.path.join(directory, "tokenizer_config.json")
    if os.path.exists(tc):
        try:
            with open(tc, encoding="utf-8") as fh:
                if json.load(fh).get("chat_template"):
                    return True, "tokenizer_config.json:chat_template"
        except Exception:
            pass
    return False, ""


def detect_path(directory: str) -> str:
    """'lora' | 'full' | 'empty' | 'unknown'"""
    try:
        entries = os.listdir(directory)
    except OSError:
        return "empty"
    if not entries:
        return "empty"
    if "adapter_config.json" in entries or "adapter_model.safetensors" in entries:
        return "lora"
    if model_shards(directory):
        return "full"
    return "unknown"


# --- checks -----------------------------------------------------------------

# Minimum plausible LoRA adapter. Even r=8 on q/v of a 1B model clears this;
# anything smaller means a truncated or empty write.
MIN_ADAPTER_BYTES = 50 * 1024

REQUIRED_LORA = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json"]
REQUIRED_FULL = ["config.json", "tokenizer.json", "tokenizer_config.json"]

# Strays. Value = (is_fatal, note). Fatal ones either hijack or actually ship.
STRAY_GLOBS = [
    ("optimizer.pt", True, "optimizer state (uploader ignore_patterns would drop it, but it must never be written here)"),
    ("optimizer.bin", True, "optimizer state"),
    ("scheduler.pt", True, "LR scheduler state"),
    ("scaler.pt", True, "grad scaler state"),
    ("rng_state*.pth", True, "RNG state"),
    ("*.py", True, "source file — SHIPS to HF (not in uploader ignore_patterns)"),
    ("added_tokens.json", True, "added_tokens.json — vocab drift vs base; train_sft_env removes it deliberately"),
    ("*.bin", True, "legacy .bin weights — duplicate/ambiguous alongside safetensors"),
]
# Present-but-harmless: the uploader filters these, so they are noise, not risk.
BENIGN_GLOBS = ["trainer_state.json", "training_args.bin"]


def check_full_weights(directory: str, base_dir: str, rep: Report,
                       allow_unverified: bool = False) -> None:
    """(i) sha256 shards vs base — all identical means untrained.

    FAIL-CLOSED.  If the definitive comparison cannot be run (no base dir given,
    or the base dir contains no shards) the result is FAIL, not WARN: a WARN does
    not move the exit code, so an untrained artifact would pass with exit 0 —
    precisely the failure class this guard exists to stop.  ``allow_unverified``
    (CLI: --allow-unverified-base) downgrades it to WARN for the operator who
    deliberately accepts running without a base copy.
    """
    shards = model_shards(directory)
    if not shards:
        rep.add(FAIL, "full/shards", "no model *.safetensors shards found")
        return

    unverified = WARN if allow_unverified else FAIL
    base_shards = []
    can_compare = False

    if not base_dir or not os.path.isdir(base_dir):
        rep.add(unverified, "full/base-hash",
                f"base model dir unavailable ({base_dir!r}) — CANNOT prove the weights differ "
                f"from base, so an UNTRAINED artifact would go undetected. Pass --base-model "
                f"<dir>; use --allow-unverified-base only to accept that risk deliberately.")
    else:
        base_shards = model_shards(base_dir)
        if not base_shards:
            rep.add(unverified, "full/base-hash",
                    f"no shards in base dir {base_dir} — the untrained-artifact comparison "
                    f"CANNOT be performed. Point --base-model at a dir holding the base "
                    f"*.safetensors; use --allow-unverified-base to accept that risk deliberately.")
        else:
            can_compare = True

    if can_compare:
        base_hashes = {}
        for name in base_shards:
            base_hashes[sha256_file(os.path.join(base_dir, name))] = name

        identical, differing = [], []
        for name in shards:
            digest = sha256_file(os.path.join(directory, name))
            if digest in base_hashes:
                identical.append((name, base_hashes[digest], digest))
            else:
                differing.append((name, digest))

        total = len(shards)
        if len(identical) == total:
            rep.add(FAIL, "full/base-hash",
                    f"ALL {total} shard(s) are byte-identical to the base model — "
                    f"the model is UNTRAINED. This is the exact artifact that scores 0.0. "
                    + "; ".join(f"{a} == base/{b} ({d[:16]}…)" for a, b, d in identical))
        elif identical:
            rep.add(FAIL, "full/base-hash",
                    f"{len(identical)}/{total} shard(s) identical to base "
                    f"({', '.join(a for a, _, _ in identical)}) — partial/failed save, weights are inconsistent")
        else:
            rep.add(OK, "full/base-hash",
                    f"all {total} shard(s) differ from base "
                    + ", ".join(f"{n}={d[:12]}…" for n, d in differing))

    # Readability / truncation are independent of the base comparison and must run
    # even when it could not (D2: the old early `return`s skipped them entirely).
    # Every shard all-zero would also be a dead model.
    for name in shards:
        path = os.path.join(directory, name)
        try:
            header, data_start, size = read_safetensors_header(path)
        except Exception as exc:
            rep.add(FAIL, "full/readable", f"{name}: unreadable safetensors header: {exc}")
            continue
        entries = tensor_entries(header)
        end_max = max((v["data_offsets"][1] for _, v in entries), default=0)
        if data_start + end_max != size:
            rep.add(FAIL, "full/truncated",
                    f"{name}: header declares {human(data_start + end_max)} but file is {human(size)} — TRUNCATED")

    if os.path.exists(os.path.join(directory, "model.safetensors.index.json")) and len(shards) == 1:
        rep.add(WARN, "full/index", "index json present but only one shard — index may be stale")


def check_lora(directory: str, rep: Report, allow_unverified: bool = False) -> None:
    """(ii) adapter exists, sized sanely, not truncated, and not a no-op."""
    adapter = os.path.join(directory, "adapter_model.safetensors")
    legacy = os.path.join(directory, "adapter_model.bin")
    if not os.path.exists(adapter):
        if os.path.exists(legacy):
            rep.add(FAIL, "lora/present", "only legacy adapter_model.bin found; adapter_model.safetensors missing")
        else:
            rep.add(FAIL, "lora/present", "adapter_model.safetensors missing")
        return

    size = os.path.getsize(adapter)
    if size < MIN_ADAPTER_BYTES:
        rep.add(FAIL, "lora/size", f"adapter is {human(size)} — below the {human(MIN_ADAPTER_BYTES)} floor; empty or truncated write")
        return
    rep.add(OK, "lora/size", f"adapter_model.safetensors = {human(size)}")

    try:
        header, data_start, fsize = read_safetensors_header(adapter)
    except Exception as exc:
        rep.add(FAIL, "lora/readable", f"unreadable safetensors header: {exc}")
        return

    entries = tensor_entries(header)
    if not entries:
        rep.add(FAIL, "lora/readable", "adapter contains no tensors")
        return

    end_max = max(v["data_offsets"][1] for _, v in entries)
    if data_start + end_max != fsize:
        rep.add(FAIL, "lora/truncated",
                f"header declares {human(data_start + end_max)} of data but file is {human(fsize)} — TRUNCATED")
        return

    n_tensors = len(entries)
    zero_all, zero_b, total_b, zero_a, total_a = [], 0, 0, 0, 0
    with open(adapter, "rb") as fh:
        for name, meta in entries:
            start, end = meta["data_offsets"]
            is_zero = region_is_all_zero(fh, data_start + start, data_start + end)
            if is_zero:
                zero_all.append(name)
            if "lora_B" in name:
                total_b += 1
                zero_b += int(is_zero)
            elif "lora_A" in name:
                total_a += 1
                zero_a += int(is_zero)

    rep.add(INFO, "lora/tensors",
            f"{n_tensors} tensors (lora_A={total_a}, lora_B={total_b}), all-zero={len(zero_all)}")

    if len(zero_all) == n_tensors:
        rep.add(FAIL, "lora/nonzero", "EVERY adapter tensor is all-zero — the adapter is empty")
        return

    if total_b and zero_b == total_b:
        rep.add(FAIL, "lora/nonzero",
                f"all {total_b} lora_B tensors are all-zero — this is a freshly-initialised, "
                f"UNTRAINED adapter. dW = B@A = 0, so the merged model is bit-identical in behaviour "
                f"to the base model and scores 0.0. (Matches train_sft_env.py's zero-rows save path.)")
    elif total_b and zero_b:
        rep.add(FAIL, "lora/nonzero",
                f"{zero_b}/{total_b} lora_B tensors are all-zero — those layers learned nothing (partial save)")
    elif total_b:
        rep.add(OK, "lora/nonzero", f"all {total_b} lora_B tensors carry non-zero weights (adapter is trained)")
    else:
        # Non-PEFT naming: the definitive "is this adapter a mathematical no-op"
        # test cannot run. Same fail-closed rule as the full-weights hash check.
        rep.add(WARN if allow_unverified else FAIL, "lora/nonzero",
                f"no lora_B tensors by name; {len(zero_all)}/{n_tensors} tensors all-zero — "
                f"CANNOT verify the adapter is not a no-op (dW = B@A == 0). "
                f"Use --allow-unverified-base to accept that risk deliberately.")

    cfg_path = os.path.join(directory, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            rep.add(INFO, "lora/config",
                    f"r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
                    f"base={cfg.get('base_model_name_or_path')} targets={cfg.get('target_modules')}")
        except Exception as exc:
            rep.add(FAIL, "lora/config", f"adapter_config.json unparseable: {exc}")


def check_required(directory: str, kind: str, rep: Report) -> None:
    """(iii) required files for the detected path."""
    required = REQUIRED_LORA if kind == "lora" else REQUIRED_FULL
    missing = [f for f in required if not os.path.exists(os.path.join(directory, f))]

    if kind == "full" and not model_shards(directory):
        missing.append("model*.safetensors")

    if missing:
        rep.add(FAIL, "files/required", f"missing: {', '.join(missing)}")
    else:
        rep.add(OK, "files/required", f"all required {kind} files present")

    ok, where = has_chat_template(directory)
    if ok:
        rep.add(OK, "files/chat-template", f"found via {where}")
    else:
        rep.add(FAIL, "files/chat-template",
                "no chat template (neither chat_template.jinja nor tokenizer_config.chat_template) — "
                "the eval harness cannot render prompts; the model forfeits")

    # A degenerate tokenizer_config is a silent forfeit even when the file exists.
    tc = os.path.join(directory, "tokenizer_config.json")
    if os.path.exists(tc):
        tc_size = os.path.getsize(tc)
        if tc_size < 2048:
            rep.add(WARN, "files/tokenizer",
                    f"tokenizer_config.json is only {human(tc_size)} — suspiciously minimal "
                    f"(a full Llama/Qwen config is tens of KB); likely a degenerate re-save "
                    f"that dropped added_tokens_decoder / chat_template")

    if kind == "lora" and not os.path.exists(os.path.join(directory, "special_tokens_map.json")):
        rep.add(WARN, "files/tokenizer", "special_tokens_map.json missing")


def check_strays(directory: str, rep: Report, upload_dir: str) -> None:
    """(iv) files that must not be in a submission dir."""
    try:
        entries = sorted(os.listdir(directory))
    except OSError as exc:
        rep.add(FAIL, "files/stray", f"cannot list dir: {exc}")
        return

    fatal, benign = [], []
    for name in entries:
        full = os.path.join(directory, name)
        if os.path.isdir(full):
            if name.startswith("checkpoint-"):
                hijacked = os.path.abspath(upload_dir) == os.path.abspath(full)
                fatal.append(
                    f"{name}/ (intermediate checkpoint dir; the uploader's detect_subfolder() "
                    f"descends into the first subdir holding a .safetensors and uploads THAT"
                    + (" — IT IS WHAT WOULD SHIP HERE" if hijacked else "") + ")")
            elif name == ".cache":
                benign.append(f"{name}/ (HF download cache — copied in from the base model snapshot)")
            continue
        # Benign first: training_args.bin is a HF trainer artifact, not weights, and
        # would otherwise be caught by the *.bin rule below.
        if any(fnmatch.fnmatch(name, p) for p in BENIGN_GLOBS):
            benign.append(f"{name} (filtered by the uploader's ignore_patterns)")
            continue
        for pat, is_fatal, note in STRAY_GLOBS:
            if fnmatch.fnmatch(name, pat):
                (fatal if is_fatal else benign).append(f"{name} — {note}")
                break

    if fatal:
        rep.add(FAIL, "files/stray", "; ".join(fatal))
    else:
        rep.add(OK, "files/stray", "no shipping strays")
    for item in benign:
        rep.add(INFO, "files/stray", item)


def check_fallback_signature(directory: str, rep: Report) -> None:
    """Distinguish a trained artifact from text_trainer.py's base-model fallback copy.

    The fallback branch copies /cache/models/<base> verbatim into submission_dir.
    It leaves a recognisable fingerprint: the HF snapshot's .cache/huggingface/
    download metadata, and none of the markers a training run writes.
    """
    has = lambda n: os.path.exists(os.path.join(directory, n))
    training_markers = [n for n in ("success.txt", "loss.txt", "trainer_state.json",
                                    "training_args.bin", "README.md") if has(n)]
    snapshot_marker = os.path.isdir(os.path.join(directory, ".cache", "huggingface", "download"))

    if snapshot_marker and not training_markers:
        rep.add(FAIL, "provenance",
                "artifact carries a raw HF snapshot fingerprint (.cache/huggingface/download/) and "
                "NO training marker (no success.txt / loss.txt / trainer_state.json / README.md) — "
                "this is text_trainer.py's base-model FALLBACK copy, not a trained model")
    elif training_markers:
        rep.add(OK, "provenance", f"training markers present: {', '.join(training_markers)}")
    else:
        rep.add(WARN, "provenance",
                "no training markers (success.txt / loss.txt / trainer_state.json / README.md) — "
                "provenance unverifiable from files alone; rely on the hash check above")


# --- driver -----------------------------------------------------------------

def audit(submission_dir: str, base_model: str,
          allow_unverified: bool = False) -> tuple[Report, dict]:
    rep = Report()
    meta = {"submission_dir": submission_dir, "base_model": base_model}

    if not os.path.isdir(submission_dir):
        rep.add(FAIL, "dir", f"submission dir does not exist: {submission_dir}")
        meta["kind"] = "missing"
        return rep, meta

    upload_dir = resolve_upload_dir(submission_dir)
    meta["upload_dir"] = upload_dir
    if os.path.abspath(upload_dir) != os.path.abspath(submission_dir):
        rep.add(FAIL, "upload/target",
                f"the uploader would ship {upload_dir} instead of {submission_dir} "
                f"(detect_subfolder picked a nested dir) — checks below run against what SHIPS")

    kind = detect_path(upload_dir)
    meta["kind"] = kind

    if kind == "empty":
        rep.add(FAIL, "dir", "submission dir is EMPTY — nothing to upload; the validator "
                             "raises 'empty or only contains metadata files' and the task scores 0")
        return rep, meta
    if kind == "unknown":
        rep.add(FAIL, "dir", f"no adapter and no model shards found in {upload_dir} — "
                             f"contents: {sorted(os.listdir(upload_dir))[:12]}")
        return rep, meta

    rep.add(INFO, "path", f"detected {kind}-weights artifact")

    if kind == "lora":
        check_lora(upload_dir, rep, allow_unverified)
    else:
        check_full_weights(upload_dir, base_model, rep, allow_unverified)

    check_required(upload_dir, kind, rep)
    check_strays(upload_dir, rep, upload_dir)
    check_fallback_signature(upload_dir, rep)
    return rep, meta


def print_report(rep: Report, meta: dict, quiet_info: bool = False) -> None:
    print("=" * 78)
    print(f"submission : {meta['submission_dir']}")
    if meta.get("upload_dir") and meta["upload_dir"] != meta["submission_dir"]:
        print(f"WOULD SHIP : {meta['upload_dir']}")
    print(f"base model : {meta['base_model']}")
    print(f"artifact   : {meta.get('kind', '?')}")
    print("-" * 78)
    for level, check, msg in rep.rows:
        if quiet_info and level == INFO:
            continue
        print(f"  {_ICON[level]} {check:<22} {msg}")
    print("-" * 78)
    n_fail = sum(1 for l, _, _ in rep.rows if l == FAIL)
    n_warn = sum(1 for l, _, _ in rep.rows if l == WARN)
    verdict = "GAGAL / FAIL" if rep.failed else ("LULUS (with warnings)" if rep.warned else "LULUS / PASS")
    print(f"  VERDICT: {verdict}   ({n_fail} fail, {n_warn} warn)")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-submission artifact guard (read-only).")
    ap.add_argument("--submission-dir", required=True, nargs="+",
                    help="submission dir(s) to audit (the uploader's LOCAL_FOLDER)")
    ap.add_argument("--base-model", default="",
                    help="path to the base model dir, for the shard-hash comparison")
    ap.add_argument("--allow-unverified-base", action="store_true",
                    help="downgrade un-runnable definitive checks (missing/empty --base-model, "
                         "non-PEFT adapter naming) from FAIL to WARN. Off by default: the guard "
                         "is fail-closed.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON as well")
    ap.add_argument("--quiet-info", action="store_true", help="hide INFO rows")
    args = ap.parse_args()

    results, worst = [], 0
    for d in args.submission_dir:
        rep, meta = audit(d, args.base_model, args.allow_unverified_base)
        print_report(rep, meta, args.quiet_info)
        if rep.failed:
            worst = 1
        results.append({
            "submission_dir": d,
            "upload_dir": meta.get("upload_dir"),
            "kind": meta.get("kind"),
            "verdict": "FAIL" if rep.failed else "PASS",
            "rows": [{"level": l, "check": c, "message": m} for l, c, m in rep.rows],
        })

    if len(results) > 1:
        print("\nSUMMARY")
        print("-" * 78)
        for r in results:
            mark = "GAGAL" if r["verdict"] == "FAIL" else "LULUS"
            reasons = [x["message"].split(" — ")[0].split(" (")[0]
                       for x in r["rows"] if x["level"] == FAIL]
            head = f"  {mark:<6} {r['kind'] or '?':<7} {r['submission_dir']}"
            print(head + (f"\n           -> {reasons[0][:100]}" if reasons else ""))
        print("-" * 78)
        n_bad = sum(1 for r in results if r["verdict"] == "FAIL")
        print(f"  {len(results) - n_bad}/{len(results)} LULUS, {n_bad} GAGAL")

    if args.json:
        print("\n" + json.dumps(results, indent=2))
    return worst


if __name__ == "__main__":
    sys.exit(main())
