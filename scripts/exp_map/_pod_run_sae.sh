#!/bin/bash
# Run on the SAE pod. Pipes everything to /workspace/sae_run.log.
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q "transformers<5" accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"
echo "[$(date)] building SAE direction cache"
python -u scripts/exp_map/sae_directions.py
echo "[$(date)] launching sweep_2d --directions_pkl"
python -u scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 \
  --directions_pkl results/exp_map/data/directions/dirs_gemma_L2_sae.pkl \
  --skip_existing
echo "[$(date)] done"
