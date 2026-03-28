# TTT-WM

## 1. Setup

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Set environment variables for outputs and dataset paths:

```bash
# Directory for saving outputs (logs, checkpoints, etc.)
export TTT_WM_OUTPUTS_ROOT=/ariesdv0/zhaoyang/data

# Directory containing training and testing data
export TTT_WM_DATA_ROOT=/ariesdv0/zhaoyang/libero_combined
```

---

## 2. Training

If `TTT_WM_DATA_ROOT` is not set, you can specify the dataset path using `data.root`.

### Key Arguments

- `data.use_goal`: Whether to use the last frame as the goal (conditioning input)
- `data.frame_gap`: Temporal gap between sampled frames (also means how many action gaps)

### Run Training

```bash
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  train.py \
  --config-name config \
  data.root=/ariesdv0/zhaoyang/libero_combined \
  data.use_goal=false \
  data.frame_gap=8
```

---

## Notes

- Ensure the dataset path is correctly set via `TTT_WM_DATA_ROOT` or `data.root`.
- Adjust `--nproc_per_node` based on the number of available GPUs.
- Change `--master_port` if there are conflicts with other distributed jobs.
