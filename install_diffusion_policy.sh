#!/bin/bash
set -e

DP_PATH="/opt/venv/src/diffusion-policy"
COMMIT="5ba07ac6661db573af695b419a7947ecb704690f"

echo ">>> Cloning diffusion_policy..."
git clone https://github.com/real-stanford/diffusion_policy.git "$DP_PATH"

echo ">>> Checking out commit $COMMIT..."
cd "$DP_PATH"
git checkout "$COMMIT"

# 3. 确保 __init__.py 存在
echo ">>> Ensuring __init__.py exists..."
touch "$DP_PATH/diffusion_policy/__init__.py"

echo ">>> Uninstalling old version..."
python -m pip uninstall -y diffusion_policy diffusion-policy || true

echo ">>> Installing in editable mode..."
python -m pip install -e "$DP_PATH"

echo ">>> Verifying installation..."
python -c "import diffusion_policy; print('diffusion_policy:', diffusion_policy.__file__)"

echo ">>> Done!"