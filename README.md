# TTT-WM

## 1. Setup

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Install the external `diffusion_policy` package with the provided script:

```bash
DP_PATH=your_folder_for_diffusion_policy ./install_diffusion_policy.sh
```

This script clones `diffusion_policy` to `your_folder_for_diffusion_policy/diffusion-policy`, checks out commit `5ba07ac6661db573af695b419a7947ecb704690f`, ensures `diffusion_policy/__init__.py` exists, and installs it in editable mode.

If you want to use a different local checkout later, you can still override the import path:

```bash
export DIFFUSION_POLICY_SRC=/path/to/diffusion-policy
export PYTHONPATH="$DIFFUSION_POLICY_SRC:$PYTHONPATH"
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

# Directory for WANDB log
export WANDB_DIR=your_wandb_log_folder

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

### Run Diffusion Policy Training

`configs/dp_config.yaml` is the diffusion-policy training config. By default it assumes:

- parquet dataset root from `TTT_WM_DATA_ROOT`
- RGB observation column `image`
- action column `actions`
- single-camera `shape_meta` with action dim 7

Run training:

```bash
python train_dp.py \
  --config-name dp_config \
  dataset_root=/path/to/libero_wm
```

Multi-GPU training:

```bash
torchrun \
  --nproc_per_node=4 \
  --master_port=29511 \
  train_dp.py \
  --config-name dp_config \
  dataset_root=/path/to/libero_wm
```

Common overrides:

```bash
python train_dp.py \
  --config-name dp_config \
  dataset_root=/path/to/libero_wm \
  shape_meta.obs.image.shape=[3,128,128] \
  data.action_key=actions \
  data.test_task_count=3
```

### Run IDM Diffusion-Policy Training

`train.py` can also train the Stage 2 inverse-dynamics model with the
diffusion-policy action head. This is different from `train_dp.py`:

- `train_dp.py` trains a standalone diffusion policy from image observations
- `train.py` with `train.stage=2 train.idm_type=diffusion_policy` trains the
  Cosmos Stage 2 IDM by first predicting the next frame with Stage 1, then
  feeding the frame pair into the original diffusion-policy image architecture
  `[current frame, predicted next frame]`

In this Stage 2 setup, the Stage 1 model is used only to predict
`predicted_next_frame`. The diffusion action model then takes
`(s_t, predicted_next_frame)` as an extra two-image observation and predicts
the intermediate action sequence `a_{t:t+m-1}` with the original DP U-Net.

Requirements:

- run `bash /scr2/zhaoyang/TTT-WM/install_diffusion_policy.sh`
- provide a trained Stage 1 checkpoint via `train.stage1_ckpt`
- set `data.frame_gap` to the number of intermediate action steps predicted by IDM

Single-GPU example:

```bash
python train.py \
  --config-name config \
  train.stage=2 \
  train.idm_type=diffusion_policy \
  train.stage1_ckpt=/path/to/stage1_checkpoint.pt \
  data.root=/path/to/libero_wm \
  data.frame_gap=4
```

Multi-GPU example:

```bash
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  train.py \
  --config-name config \
  train.stage=2 \
  train.idm_type=diffusion_policy \
  train.stage1_ckpt=/path/to/stage1_checkpoint.pt \
  data.root=/path/to/libero_wm \
  data.frame_gap=4
```

Common overrides:

```bash
python train.py \
  --config-name config \
  train.stage=2 \
  train.idm_type=diffusion_policy \
  train.stage1_ckpt=/path/to/stage1_checkpoint.pt \
  data.root=/path/to/libero_wm \
  data.frame_gap=4 \
  train.freeze_backbone=true \
  train.idm_dp.num_train_timesteps=100 \
  train.idm_dp.down_dims=[256,512,1024]
```

Run offline checkpoint evaluation:

```bash
python test_dp.py \
  --checkpoint /path/to/run_or_ckpt \
  --config /scr2/zhaoyang/TTT-WM/configs/dp_config.yaml \
  --dataset-root /path/to/libero_wm \
  --split val \
  --output-json /tmp/dp_eval.json
```

---

## Notes

- Ensure the dataset path is correctly set via `TTT_WM_DATA_ROOT` or `data.root`.
- Adjust `--nproc_per_node` based on the number of available GPUs.
- Change `--master_port` if there are conflicts with other distributed jobs.
- Stage 2 IDM diffusion-policy training expects a valid Stage 1 checkpoint in `train.stage1_ckpt`.
- The default diffusion-policy install script places the editable checkout at `/opt/venv/src/diffusion-policy`.
- If you later switch to a different external `diffusion_policy` checkout, update `DIFFUSION_POLICY_SRC` or `runtime.diffusion_policy_src`; the TTT-WM diffusion-policy code does not hardcode the old path.
