#!/bin/bash
# Stage 2 of better-baselines (Plans/exp_map_better_baselines.md).
# Runs sweep_2d on one family's direction pkl. Three independent
# instances of this script (one per pod) run the three baselines in
# parallel: sae_eval, sae_fineweb, pca_fineweb.
#
# Usage:
#   bash _pod_run_baselines_sweep.sh sae_eval
#   bash _pod_run_baselines_sweep.sh sae_fineweb
#   bash _pod_run_baselines_sweep.sh pca_fineweb
#
# Reads the direction pkl from the shared /workspace volume (produced
# by _pod_run_baselines_prep.sh on a separate pod). Wall: ~3.5 GPU-hr
# per pod on RTX 4090 (528 pairs × ~24 s/pair).
set -e
FAMILY="${1:-}"
case "$FAMILY" in
  sae_eval|sae_fineweb|pca_fineweb) ;;
  *) echo "ERROR: family must be one of sae_eval / sae_fineweb / pca_fineweb"
     exit 1;;
esac

export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus

DIRS_PKL="results/exp_map/data/directions/dirs_gemma_L2_${FAMILY}.pkl"
if [ ! -f "$DIRS_PKL" ]; then
  echo "ERROR: $DIRS_PKL not found. Did the prep pod finish?"
  exit 2
fi

echo "[$(date)] installing deps"
pip install -q "transformers<5" accelerate scipy
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

echo "[$(date)] launching sweep_2d for ${FAMILY}"
python -u scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 \
  --directions_pkl "$DIRS_PKL" \
  --skip_existing

echo "[$(date)] done"
