#!/bin/bash
# Setup script for RunPod GPU pods.
#
# Creates a venv, installs torch (matching the pod's CUDA toolkit),
# then installs the rest of the dependencies.
#
# Usage:
#   ssh runpod-<name> "cd /workspace/steering-plateaus && bash setup_runpod.sh"
#
# Prerequisites:
#   - RunPod pod with CUDA 12.4 image (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
#   - Code synced to /workspace/steering-plateaus/

set -e

VENV_DIR="/home/user/venv"
HF_CACHE="/workspace/hf_cache"

echo "=== Setting up RunPod environment ==="

# 1. Create venv on local disk (avoids network volume quota pressure)
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating venv at $VENV_DIR..."
    python -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 2. Install torch matching CUDA 12.4
echo "Installing torch (cu124)..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

# 3. Install all dependencies
echo "Installing project dependencies..."
pip install -r requirements_gpu.txt --quiet

# 5. Point HuggingFace cache to network volume (persists across pods)
mkdir -p "$HF_CACHE"
rm -rf /home/user/.cache/huggingface 2>/dev/null || true
ln -sfn "$HF_CACHE" /home/user/.cache/huggingface

# 6. HuggingFace login (for gated models like Gemma, Llama)
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "Logged in to HuggingFace"
else
    echo "WARNING: HF_TOKEN not set. Set it for gated models: export HF_TOKEN=hf_..."
fi

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch, transformers, sae_lens
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'sae_lens:       {sae_lens.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
t = torch.randn(2, device='cuda')
print(f'CUDA test:      OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
