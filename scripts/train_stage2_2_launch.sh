#!/usr/bin/env bash
# One-shot launcher for Stage 2.2 training (frozen Stage-2.1 backbone, train
# only the IDM action head on a single held-out task). Mirrors the Stage 2.1
# launcher but swaps in the Stage 2.2 config and the additional knobs the
# 2.2 trainer needs:
#   - STAGE2_1_CKPT (required) wires train.stage2_1_ckpt — the frozen backbone.
#   - Optional STAGE1_CACHE_DIR enables train.idm.use_stage1_cache so the
#     frozen AR decode is read from disk instead of recomputed each step.
#     Produce the cache with scripts/prebake_stage1_pred.py before enabling.
#
# Usage (from repo root):
#   DATASET=/scr2/zhaoyang/libero_wm \
#   STAGE2_1_CKPT=/scr2/zhaoyang/latest_stage2_1.pt \
#   TASK="KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \
#   bash scripts/train_stage2_2_launch.sh
#
# Override knobs:
#   NGPU=4 BATCH_SIZE=128 LR=1e-6 VAL_FRACTION=0.01 NUM_WORKERS=8
#   MASTER_PORT=29506 EPOCHS=100 CHECKPOINT_EVERY=10 SKIP_PREWARM=1
#   STAGE1_CACHE_DIR=/scr2/zhaoyang/stage1_pred_cache REQUIRE_CACHE_SHA1=1

set -euo pipefail

: "${DATASET:?Set DATASET=/path/to/libero_wm}"
: "${STAGE2_1_CKPT:?Set STAGE2_1_CKPT=/path/to/stage2_1.pt}"
: "${TASK:?Set TASK=<exact task string from meta/test_tasks.json>}"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29506}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
LR="${LR:-1e-6}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"

# Optional Stage-1 prediction cache (A1). Off by default — set
# STAGE1_CACHE_DIR to enable. REQUIRE_CACHE_SHA1=0 disables the backbone-hash
# guard (use only if you know the cache matches the backbone).
STAGE1_CACHE_DIR="${STAGE1_CACHE_DIR:-}"
REQUIRE_CACHE_SHA1="${REQUIRE_CACHE_SHA1:-1}"

# 1) Shared run timestamp so all ranks log into the same hydra output dir.
export RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "RUN_TS=${RUN_TS}"
echo "TASK=${TASK}"
echo "STAGE2_1_CKPT=${STAGE2_1_CKPT}"

# 2) Prewarm OS page cache so the first epoch doesn't pay cold-cache tax.
if [[ "${SKIP_PREWARM:-0}" != "1" ]]; then
    bash "$(dirname "$0")/prewarm_cache.sh" "${DATASET}"
else
    echo "SKIP_PREWARM=1 — skipping page-cache prewarm."
fi

# 3) Assemble Stage-1 cache overrides only when STAGE1_CACHE_DIR is set, so
#    the default invocation matches the config (use_stage1_cache=false).
CACHE_OVERRIDES=()
if [[ -n "${STAGE1_CACHE_DIR}" ]]; then
    if [[ "${REQUIRE_CACHE_SHA1}" == "1" ]]; then
        require_flag="true"
    else
        require_flag="false"
    fi
    CACHE_OVERRIDES+=(
        "train.idm.use_stage1_cache=true"
        "train.idm.stage1_cache_dir=${STAGE1_CACHE_DIR}"
        "train.idm.require_cache_sha1=${require_flag}"
    )
    echo "Stage-1 cache enabled: ${STAGE1_CACHE_DIR} (require_sha1=${require_flag})"
fi

# 4) Launch torchrun. data.selected_task must be a *literal* string that
#    matches meta/test_tasks.json — Hydra lets us pass it via double-quoted
#    override so spaces and colons survive.
echo
echo "Launching Stage 2.2 torchrun on ${NGPU} GPUs..."
exec torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    train_stage2.py \
    --config-name stage2_2_idm_config \
    train.stage2_1_ckpt="${STAGE2_1_CKPT}" \
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
    "${CACHE_OVERRIDES[@]}" \
    "$@"
