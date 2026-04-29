#!/usr/bin/env bash
# One-shot launcher for Stage 2.1 training (backbone fine-tune on one held-out
# task). Stage 2.1 uses the same DDP/NCCL/EMA plumbing as Stage 1, the only
# additions here over Stage 1's launcher are:
#   - a required TASK env var (the exact task string from meta/test_tasks.json)
#   - train.substep=2.1 + train.stage1_ckpt wiring
#
# Usage (from repo root):
#   DATASET=/scr2/zhaoyang/libero_wm \
#   STAGE1_CKPT=/scr2/zhaoyang/latest_stage1.pt \
#   TASK="KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \
#   bash scripts/train_stage2_1_launch.sh
#
# Override knobs:
#   NGPU=4 BATCH_SIZE=128 LR=1e-4 VAL_FRACTION=0.05 NUM_WORKERS=6
#   MASTER_PORT=29505 EPOCHS=300 CHECKPOINT_EVERY=10 SKIP_PREWARM=1

set -euo pipefail

: "${DATASET:?Set DATASET=/path/to/libero_wm}"
: "${STAGE1_CKPT:?Set STAGE1_CKPT=/path/to/stage1.pt}"
: "${TASK:?Set TASK=<exact task string from meta/test_tasks.json>}"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29505}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EPOCHS="${EPOCHS:-300}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
LR="${LR:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"

# 1) Shared run timestamp so all ranks log into the same hydra output dir.
export RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "RUN_TS=${RUN_TS}"
echo "TASK=${TASK}"

# 2) Prewarm OS page cache so the first epoch doesn't pay cold-cache tax.
if [[ "${SKIP_PREWARM:-0}" != "1" ]]; then
    bash "$(dirname "$0")/prewarm_cache.sh" "${DATASET}"
else
    echo "SKIP_PREWARM=1 — skipping page-cache prewarm."
fi

# 3) Launch torchrun. Note: data.selected_task must be a *literal* string that
#    matches meta/test_tasks.json — Hydra lets us pass it via double-quoted
#    override so spaces and colons survive.
echo
echo "Launching Stage 2.1 torchrun on ${NGPU} GPUs..."
exec torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    train_stage2.py \
    --config-name stage2_1_idm_config \
    train.stage1_ckpt="${STAGE1_CKPT}" \
    data.root="${DATASET}" \
    data.selected_task="${TASK}" \
    data.frame_gap=4 \
    data.stage2_val_fraction="${VAL_FRACTION}" \
    data.num_workers="${NUM_WORKERS}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}" \
    train.optimizer.lr="${LR}" \
    train.epochs="${EPOCHS}" \
    train.checkpoint_every="${CHECKPOINT_EVERY}" \
    "$@"
