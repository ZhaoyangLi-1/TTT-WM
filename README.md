# TTT-WM

## 1. Setup
```bash
uv pip install -r requirements.txt
# Replace TTT_WM_OUTPUTS_ROOT to the folder that stores outputs
export TTT_WM_OUTPUTS_ROOT=/ariesdv0/zhaoyang/data
# Replace TTT_WM_DATA_ROOT to the folder that stores training and testing data
export TTT_WM_DATA_ROOT=/ariesdv0/zhaoyang/libero_combined 
```

## 2. Training
Use data.root if you don't set TTT_WM_DATA_ROOT.
use_goal means whetehr we use last frame as input condition, and goal_tag

```bash
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  train.py \
  --config-name cosmos_config \
  data.root=/ariesdv0/zhaoyang/libero_combined \
  data.use_goal=false \
  data.goal_tag=no_goal \
  data.frame_gap=8
```