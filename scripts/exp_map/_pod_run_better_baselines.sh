#!/bin/bash
# Run on a single pod (sequential, ~11h on RTX 4090). Builds the three
# new direction caches (sae_eval, sae_fineweb, pca_fineweb) and runs the
# corresponding sweep_2d sets for the better-baselines beeswarm
# (Plans/exp_map_better_baselines.md).
#
# To shard across N parallel pods: pass `--shard i/N` to each sweep_2d
# invocation below (e.g. uncomment the loop). Direction extraction is
# cheap (<10 min total once activations are cached) so it's fine to run
# it on every pod — they all read the same canonical SAE / FineWeb
# inputs and produce identical pkls under matching paths.
#
# Logs to /workspace/better_baselines_run.log when launched via nohup
# (recommended for spot interruption resilience).
set -e
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus
echo "[$(date)] installing deps"
pip install -q "transformers<5" accelerate
echo "[$(date)] HF login"
python -c "from huggingface_hub import login; login(token='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')"

# 1) Build the three direction pkls. The first run extracts the 10k
# FineWeb activation cache; subsequent runs reuse it.
echo "[$(date)] building SAE-eval direction cache"
python -u scripts/exp_map/sae_directions_top.py --rank_set anchors

echo "[$(date)] building SAE-FineWeb direction cache"
python -u scripts/exp_map/sae_directions_top.py --rank_set fineweb \
  --n_fineweb 10000

echo "[$(date)] building PCA-FineWeb direction cache"
python -u scripts/exp_map/pca_directions.py --target gemma --layer 2 \
  --n_fineweb 10000

# 2) Sweeps. Each is C(33, 2) = 528 pairs × ~24 s/pair ≈ 3.5 GPU-hr.
# To shard across N pods, append --shard i/N (i in 0..N-1) to each.
for FAMILY in sae_eval sae_fineweb pca_fineweb; do
  echo "[$(date)] launching sweep_2d for ${FAMILY}"
  python -u scripts/exp_map/sweep_2d.py \
    --target gemma --layer 2 \
    --directions_pkl results/exp_map/data/directions/dirs_gemma_L2_${FAMILY}.pkl \
    --skip_existing
done

echo "[$(date)] done"
