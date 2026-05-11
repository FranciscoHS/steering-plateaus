#!/bin/bash
# Stage 1 of better-baselines (Plans/exp_map_better_baselines.md).
# Builds the three direction caches:
#   - dirs_gemma_L2_sae_eval.pkl    (top-33 SAE features over 30 anchors)
#   - dirs_gemma_L2_sae_fineweb.pkl (top-33 SAE features over 10k FineWeb)
#   - dirs_gemma_L2_pca_fineweb.pkl (top-33 PCs over the same 10k FineWeb)
# As a side-effect produces acts_gemma_L2_fineweb_10000.pkl on the
# shared /workspace volume, which sweep_2d pods later reuse.
#
# Wall: ~30 min (10k FineWeb activation extraction dominates).
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q "transformers<5" accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

echo "[$(date)] building SAE-eval direction cache"
python -u scripts/exp_map/sae_directions_top.py --rank_set anchors

echo "[$(date)] building SAE-FineWeb direction cache"
python -u scripts/exp_map/sae_directions_top.py --rank_set fineweb \
  --n_fineweb 10000

echo "[$(date)] building PCA-FineWeb direction cache"
python -u scripts/exp_map/pca_directions.py --target gemma --layer 2 \
  --n_fineweb 10000

echo "[$(date)] direction extraction done. Direction pkls:"
ls -la results/exp_map/data/directions/dirs_gemma_L2_sae_eval.pkl \
       results/exp_map/data/directions/dirs_gemma_L2_sae_fineweb.pkl \
       results/exp_map/data/directions/dirs_gemma_L2_pca_fineweb.pkl
echo "[$(date)] done"
