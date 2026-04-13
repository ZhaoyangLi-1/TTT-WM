FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    DP_PATH=/opt/src/diffusion-policy \
    DIFFUSION_POLICY_SRC=/opt/src/diffusion-policy

WORKDIR /workspace/TTT-WM

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tmux \
    zip \
    rclone \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN nvcc -V

COPY requirements.txt install_diffusion_policy.sh ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt && \
    chmod +x install_diffusion_policy.sh && \
    DP_PATH="${DP_PATH}" bash ./install_diffusion_policy.sh

COPY . .

ENV PYTHONPATH=/workspace/TTT-WM:/opt/src/diffusion-policy:${PYTHONPATH}

CMD ["bash"]
