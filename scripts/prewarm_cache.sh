#!/usr/bin/env bash
# Prewarm OS page cache for all prebaked .pt files before training starts.
#
# Why
# ---
# On the first training epoch, each random-access of a .pt file that isn't
# in the OS page cache hits physical storage (~10-30 ms on NVMe, much worse
# on NFS/ceph). That causes a visible "cache warmup" period where step_time
# is 2-5× slower than steady state.
#
# Reading every .pt through `cat > /dev/null` once (with parallel xargs)
# pulls every file's pages into the kernel page cache. Subsequent training
# reads are served from RAM.
#
# Requirements
# ------------
#   * Enough free RAM to hold the full dataset (~55 GB for LIBERO 6193 eps
#     at 128×128). If you have less RAM than dataset size, the kernel will
#     evict older pages during training anyway — this script then only helps
#     the first epoch partially, which is still worth it.
#
# Usage
# -----
#   bash scripts/prewarm_cache.sh /linting-fast-vol/libero_wm
#
# Typical wall time: 1-3 minutes on local NVMe.

set -euo pipefail

DATASET_ROOT="${1:-/linting-fast-vol/libero_wm}"
PARALLELISM="${PARALLELISM:-8}"

if [[ ! -d "${DATASET_ROOT}/data" ]]; then
    echo "error: ${DATASET_ROOT}/data does not exist" >&2
    exit 1
fi

echo "Prewarming page cache for .pt files under ${DATASET_ROOT}/data ..."
n=$(find "${DATASET_ROOT}/data" -name "*.pt" | wc -l)
echo "Found ${n} .pt files; streaming with ${PARALLELISM}-way parallelism"

t_start=$(date +%s)

find "${DATASET_ROOT}/data" -name "*.pt" -print0 \
    | xargs -0 -P "${PARALLELISM}" -n 50 cat > /dev/null

t_end=$(date +%s)
echo "done in $((t_end - t_start)) seconds."

# Report current page-cache usage so the user can verify the dataset fit.
if command -v free >/dev/null 2>&1; then
    echo
    free -h
fi
