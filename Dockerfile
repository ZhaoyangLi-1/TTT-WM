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

WORKDIR /workspace
RUN git clone https://github.com/ZhaoyangLi-1/TTT-WM.git

WORKDIR /workspace/TTT-WM

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    wget \
    ffmpeg \
    git \
    htop \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    tmux \
    unzip \
    zip \
    rclone \
    rsync \
    vim \
    libglfw3 \
    libglew-dev \
    xvfb \
    libosmesa6-dev \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt && \
    python -m pip install --no-build-isolation flash-attn && \
    sed -i 's/\r//' install_diffusion_policy.sh && \
    chmod +x install_diffusion_policy.sh && \
    DP_PATH="${DP_PATH}" bash ./install_diffusion_policy.sh

ENV PYTHONPATH=/workspace/TTT-WM:/opt/src/diffusion-policy

CMD ["bash"]