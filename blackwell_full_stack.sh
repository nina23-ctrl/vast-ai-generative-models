#!/bin/bash
set -e

echo "==============================="
echo " RTX 5090 BLACKWELL STACK BUILD "
echo "==============================="

# -------------------
# ENV CONFIG
# -------------------
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="120"
export FORCE_CUDA=1
export CUDA_VISIBLE_DEVICES=0

# Memory / fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Attention backends
export TORCHINDUCTOR_DISABLE=1
export XFORMERS_FORCE_DISABLE_TRITON=1
export XFORMERS_MORE_DETAILS=1

echo "[ENV] Blackwell flags set"

# -------------------
# SYSTEM DEPS
# -------------------
apt-get update && apt-get install -y \
  build-essential \
  git \
  cmake \
  ninja-build \
  pkg-config \
  python3-dev

# -------------------
# PYTHON TOOLS
# -------------------
pip install --upgrade pip setuptools wheel
pip install ninja cmake pybind11

# -------------------
# TORCH STACK
# -------------------
echo "[TORCH] Installing cu128 stack"

pip uninstall -y torch torchvision torchaudio xformers || true

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# -------------------
# CORE DEPS
# -------------------
pip install \
einops \
omegaconf \
fire \
imageio \
imageio-ffmpeg \
opencv-python-headless \
onnxruntime-gpu \
pytorch-lightning==2.1.3 \
transformers \
safetensors \
huggingface_hub \
kornia \
open_clip_torch \
accelerate \
scipy \
tqdm \
ftfy \
regex \
matplotlib \
numpy \
sentencepiece \
protobuf \
av \
pillow \
ffmpeg-python \
rembg[gpu]

pip install git+https://github.com/openai/CLIP.git

# -------------------
# XFORMERS SOURCE BUILD
# -------------------
echo "[XFORMERS] Building from source for sm_120"

cd /workspace || exit 1

rm -rf xformers
git clone https://github.com/facebookresearch/xformers.git
cd xformers

pip install -v .

cd /workspace

# -------------------
# VERIFY STACK
# -------------------
echo "==============================="
echo " VERIFYING STACK "
echo "==============================="

python - << 'EOF'
import torch, xformers, open_clip, kornia

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("arch list:", torch.cuda.get_arch_list())
print("xformers:", xformers.__version__)
print("xformers CUDA kernels loaded:", hasattr(xformers, "ops"))
print("open_clip OK")
print("kornia OK")

if not hasattr(xformers, "ops"):
    raise RuntimeError("❌ xformers CUDA kernels NOT loaded")

print("✅ BLACKWELL STACK READY")
EOF

echo "==============================="
echo " SYSTEM READY FOR SVD / SV3D / SV4D "
echo "==============================="
