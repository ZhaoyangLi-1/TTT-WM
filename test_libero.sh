#!/bin/bash
#SBATCH --job-name=libero-eval
#SBATCH --output=/scr2/zhaoyang/logs/%x-%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G

mkdir -p /scr2/zhaoyang/logs
echo "=== START ==="; date; hostname

source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh

# ============================================================================
# Config — edit these to eval a different checkpoint / task.
# ============================================================================
REPO=/scr2/zhaoyang/TTT-WM
HOST=127.0.0.1
PORT=7000

# Policy server (serve_policy.py) config.
CHECKPOINT="/scr2/zhaoyang/TTT-WM-outputs/stage2_idm/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate/stage2.2/2026-06-30/14-57-20_g12/checkpoints/keep-epoch=0500.ckpt"
MODEL_TYPE=auto                                   # auto | dp | stage1 | stage2 | pure_idm
# diffusion_policy source checkout (the dir that CONTAINS the `diffusion_policy` package).
DIFFUSION_POLICY_SRC=/scr2/zhaoyang/condBFNPol-me/src/diffusion-policy
NUM_INFERENCE_STEPS=10

# LIBERO rollout client (main.py) config.
TASK_SUITE=libero_90
TEST_TASKS_JSON="$REPO/examples/libero/test_tasks/kitchen_scene2.json"
NUM_TRIALS=50

# Save path
VIDEO_SAVE_PATH=/scr2/zhaoyang/TTT-WM-outputs/stage2_idm/videos
SUMMARY_OUT_PATH=/scr2/zhaoyang/TTT-WM-outputs/stage2_idm/results
RUN_NAME=idm_kitchen_scene2_img_50
MAX_STEPS=250
LIBERO_WM_ROOT=/scr2/zhaoyang/libero_wm
REPLAN_STEPS=6
# Route (b): start each rollout from a libero_wm demo's OWN init sim state (from
# the LIBERO-90 HDF5) and force that demo's terminal frame as goal -> (start,
# goal) is the exact within-trajectory pair the policy trained on. Root must
# contain <task>_demo.hdf5 (searched under libero_90/, libero_100/, or itself).
INIT_FROM_HDF5_ROOT=/scr2/zhaoyang/libero_90

# ============================================================================
# 1. Start the policy server (serve_policy.py) in the background (wm env).
# ============================================================================
(
  conda activate wm
  export DIFFUSION_POLICY_SRC
  export PYTHONPATH="$DIFFUSION_POLICY_SRC:$PYTHONPATH"
  cd "$REPO/examples"
  python serve_policy.py \
    --checkpoint "$CHECKPOINT" \
    --model-type "$MODEL_TYPE" \
    --host "$HOST" \
    --port "$PORT" \
    --diffusion-policy-src "$DIFFUSION_POLICY_SRC" \
    --use-ddim \
    --num-inference-steps "$NUM_INFERENCE_STEPS"
) > /scr2/zhaoyang/logs/server.log 2>&1 &
SERVER_PID=$!

# --- Make sure the server is cleaned up when the client exits ---
cleanup() {
  echo "cleaning up server $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# ============================================================================
# 2. Wait for the server to be ready (serve_policy exposes GET /healthz).
# ============================================================================
echo "waiting for server on port $PORT ..."
for i in $(seq 1 120); do
  if curl -s "http://$HOST:$PORT/healthz" > /dev/null 2>&1; then
    echo "server is up"
    break
  fi
  # If the server died early, stop waiting.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died before becoming ready, check server.log"
    exit 1
  fi
  sleep 2
done

# ============================================================================
# 3. Run the LIBERO rollout client (main.py) in the foreground (libero env).
#    main.py imports `from client import ...`, so run it from examples/libero.
# ============================================================================
(
  conda activate libero
  cd "$REPO/examples/libero"
  export PYTHONPATH="$PYTHONPATH:$REPO/third_party/libero"
  python main.py \
    --host "$HOST" \
    --port "$PORT" \
    --task-suite-name "$TASK_SUITE" \
    --test-tasks-json "$TEST_TASKS_JSON" \
    --num-trials-per-task "$NUM_TRIALS" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-path "$VIDEO_SAVE_PATH" \
    --summary-out-path "$SUMMARY_OUT_PATH" \
    --run-name "$RUN_NAME" \
    --max-steps "$MAX_STEPS" \
    --libero-wm-root "$LIBERO_WM_ROOT" \
    --init-from-hdf5-root "$INIT_FROM_HDF5_ROOT"
) > /scr2/zhaoyang/logs/client.log 2>&1

echo "=== DONE ==="