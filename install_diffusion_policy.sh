#!/bin/bash
set -e

DP_PATH="${DP_PATH:-/opt/venv/src/diffusion-policy}"
COMMIT="5ba07ac6661db573af695b419a7947ecb704690f"
PKG_DIR="$DP_PATH/diffusion_policy"  

echo ">>> DP_PATH: $DP_PATH"

if [ -d "$PKG_DIR" ]; then
  echo ">>> $PKG_DIR 已存在,跳过安装。"
else
  echo ">>> Cloning diffusion_policy..."
  git clone https://github.com/real-stanford/diffusion_policy.git "$DP_PATH"

  echo ">>> Checking out commit $COMMIT..."
  cd "$DP_PATH"
  git checkout "$COMMIT"

  echo ">>> Ensuring __init__.py exists..."
  touch "$PKG_DIR/__init__.py"

  echo ">>> Uninstalling old version..."
  python -m pip uninstall -y diffusion_policy diffusion-policy || true

  echo ">>> Installing in editable mode..."
  python -m pip install -e "$DP_PATH"

  echo ">>> Verifying installation..."
  python -c "import diffusion_policy; print('diffusion_policy:', diffusion_policy.__file__)"
fi

export DIFFUSION_POLICY_SRC="$PKG_DIR"
export PYTHONPATH="$DIFFUSION_POLICY_SRC:$PYTHONPATH"
echo ">>> DIFFUSION_POLICY_SRC=$DIFFUSION_POLICY_SRC"

echo ">>> Done!"