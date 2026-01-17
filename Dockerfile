FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV XFORMERS_DISABLE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace

# --- System dependencies ---
RUN apt update && apt install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Python ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- PyTorch (CUDA 12.1, NO xformers) ---
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# --- Clone your repo ---
RUN git clone https://github.com/nina23-ctrl/vast-ai-generative-models.git
WORKDIR /workspace/vast-ai-generative-models

# --- Python deps ---
RUN pip install -r requirements.txt || true

RUN pip install \
    einops \
    fire \
    omegaconf \
    rembg \
    imageio \
    opencv-python \
    open_clip_torch \
    kornia \
    timm

# --- Default command ---
CMD ["/bin/bash"]
