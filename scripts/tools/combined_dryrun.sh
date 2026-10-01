#!/usr/bin/env bash
# Combined multi-env DRY RUN — validate the full tournament path end-to-end:
#   generate per-env (tool-calling card + intercode) -> combine (mixed schema)
#   -> DDP train -> adapter. Confirms the integration + DDP launch + timing.
#
# SAFE: intercode data is self-generated at train-time (own bash tasks, shell-verified;
# no NL2Bash, nothing bundled). Small targets for speed. Re-runnable: skips
# generate+combine if the combined dataset already exists.
#   bash tools/combined_dryrun.sh
#   LIARS_TARGET=2000 bash tools/combined_dryrun.sh
set -euo pipefail

# H100 PCIe (no NVLink) hangs on the first DDP all-reduce with GPU P2P enabled — a
# well-documented failure mode on this hardware. Route NCCL through host memory instead; for LoRA
# the all-reduce is tiny (only r=128 adapter grads) so the speed hit is negligible.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

TASK="dryrun"
DSDIR="/workspace/dryrun_ds"
BASE="${BASE:-gradients-io-tournaments/swe-base-qwen3-8b-continuous}"  # PR #1358: base SWE baru (tetap overridable via env)
LIARS_TARGET="${LIARS_TARGET:-3000}"
mkdir -p "$DSDIR"

if [ -d "$DSDIR/sft_env_${TASK}" ]; then
    echo "=== [1-2/3] combined dataset already exists → skipping generate+combine ==="
else
    echo "=== [1/3] generate per-env: liars (tool-calling) + intercode (self-generated) ==="
    python -m envs.generate_pvp_toolcalling --game liars_dice --num_games 6000 \
        --wins-only --target_samples "$LIARS_TARGET" \
        --output_path "$DSDIR/sft_env_${TASK}_liars_dice"
    python -m envs.mastering_intercode \
        --output_path "$DSDIR/sft_env_${TASK}_intercode" --variants 4

    echo "=== [2/3] combine (cards tools_json + intercode messages → uniform schema) ==="
    python combine_datasets.py --task_id "$TASK" --envs liars_dice,intercode \
        --input-dir "$DSDIR" --output "$DSDIR/sft_env_${TASK}"
fi

echo "=== [3/3] DDP train on all visible GPUs, timed (NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}) ==="
NG="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')"
if [ "$NG" -gt 1 ]; then LAUNCH="torchrun --nproc_per_node=${NG}"; else LAUNCH="python"; fi
echo "GPUs=${NG}  launcher: ${LAUNCH}"
time ${LAUNCH} train_sft_env.py --model-name-or-path "$BASE" \
    --dataset-path "$DSDIR/sft_env_${TASK}" --output-dir /tmp/sft_dryrun \
    --epochs 1 --batch-size 6 --max-seq-length 4096 --learning-rate 4.302e-5 \
    --lora-r 128 --lora-alpha 512 --min-lr-rate 0.25 --gradient-checkpointing --report-to none

echo ""
echo "=== RESULT ==="
if [ -f /tmp/sft_dryrun/adapter_config.json ]; then
    echo "✅ Combined adapter trained end-to-end via DDP (generate→combine→train)."
    echo "   Adapter: /tmp/sft_dryrun/adapter_config.json"
else
    echo "⚠️ No adapter_config.json — inspect the train output above."
fi
