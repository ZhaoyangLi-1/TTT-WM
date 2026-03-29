# TTT-WM

## 1. Setup

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

## 2. Dataset
Download via git-lfs

```bash
# install git-lfs if not installed
git lfs install

# clone dataset
git clone https://huggingface.co/datasets/JeffreyLii/libero_wm

# set data path
export TTT_WM_DATA_ROOT=your_folder_for_libero_wm
```

Set environment variables for outputs and logging:

```bash
# Directory for saving outputs (logs, checkpoints, etc.)
export TTT_WM_OUTPUTS_ROOT=your_folder_for_outputs

# Weights & Biases API key (for experiment tracking)
export WANDB_API_KEY=your_wandb_api_key
```


## 3. Training

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
  data.root=/ariesdv0/zhaoyang/libero_wm \
  data.use_goal=true \
  data.frame_gap=4
```

---

## Notes

- Ensure the dataset path is correctly set via `TTT_WM_DATA_ROOT` or `data.root`.
- Adjust `--nproc_per_node` based on the number of available GPUs.
- Change `--master_port` if there are conflicts with other distributed jobs.
