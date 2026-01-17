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
    libglib2.0-0 \
    wget \
    ca-certificates \
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
RUN pip install --no-cache-dir onnxruntime-gpu
RUN pip install --no-cache-dir onnxruntime
RUN apt-get update && apt-get install -y nano



RUN pip install --upgrade pip && pip install --no-cache-dir \
    torch torchvision torchaudio \
    einops \
    fire \
    omegaconf \
    imageio imageio-ffmpeg \
    rembg \
    imageio \
    opencv-python \
    open_clip_torch \
    pytorch-lightning==2.1.3 \
    kornia \
    imwatermark \
    git+https://github.com/openai/CLIP.git \
    timm

# --- Default command ---
CMD ["/bin/bash"]
RUN ln -s /usr/bin/python3 /usr/bin/python

