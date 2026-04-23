#!/usr/bin/env bash
# One-shot launcher for Stage 1 training on 6 GPUs.
#
# Handles three things the vanilla torchrun command doesn't:
#   1. Picks a single RUN_TS once — all ranks inherit it, avoiding the
#      "3 output folders from 3 different ${now:...} resolutions" problem.
#   2. Prewarms the OS page cache for all .pt files so the first epoch
#      doesn't suffer a cold-cache slowdown.
#   3. Runs torchrun with the documented good defaults.
#
# Usage (from repo root):
#   bash scripts/train_stage1_launch.sh
#
# Override anything via env vars:
#   DATASET=/path/to/libero_wm NGPU=4 bash scripts/train_stage1_launch.sh
#   SKIP_PREWARM=1 bash scripts/train_stage1_launch.sh   # skip cache prewarm

set -euo pipefail

DATASET="${DATASET:-/scr2/zhaoyang/libero_wm}"
NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29504}"
BATCH_SIZE="${BATCH_SIZE:-72}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"

# 1) Shared run timestamp — set once, inherited by all ranks forked by torchrun
export RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
echo "RUN_TS=${RUN_TS}"

# 2) Prewarm OS page cache (can be skipped if already warm from a previous run)
if [[ "${SKIP_PREWARM:-0}" != "1" ]]; then
    bash "$(dirname "$0")/prewarm_cache.sh" "${DATASET}"
else
    echo "SKIP_PREWARM=1 — skipping page-cache prewarm."
fi

# 3) Launch training
echo
echo "Launching torchrun on ${NGPU} GPUs..."
exec torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    train_stage1.py \
    --config-name stage1_wm_config \
    data.root="${DATASET}" \
    data.use_goal=true \
    data.frame_gap=4 \
    train.batch_size="${BATCH_SIZE}" \
    train.checkpoint_every="${CHECKPOINT_EVERY}" \
    data.num_workers="${NUM_WORKERS}" \
    "$@"
