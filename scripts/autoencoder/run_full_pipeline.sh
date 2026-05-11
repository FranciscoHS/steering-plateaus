#!/bin/bash
# Full pipeline: wait for 10M cache, create subsets, train 4 AEs sequentially, eval CE loss on each.
# Runs after the 10M caching job completes.

set -e
set -u

REPO=/workspace/steering-plateaus
CACHE_10M=$REPO/results/trained_autoencoder/l10_cache_10M/cached_activations.pt
CACHE_10M_META=$CACHE_10M.meta.json
SUBSET_ROOT=$REPO/results/trained_autoencoder
OUTPUT_DIR=$REPO/results/trained_autoencoder/l10_scaling

cd $REPO

echo "=== [$(date)] Pipeline starting ==="

# ── 1. Wait for 10M cache to finish ──
echo "=== [$(date)] Waiting for 10M cache ==="
while [ ! -f $CACHE_10M_META ]; do
  sleep 300
done
sleep 10  # allow any final fsync to settle
echo "=== [$(date)] 10M cache exists: $(ls -lh $CACHE_10M) ==="

# ── 2. Train 4 AEs sequentially using slices of the 10M cache ──
mkdir -p $OUTPUT_DIR

run_training () {
  local max_samples=$1
  local exp_name=$2
  echo "=== [$(date)] Training $exp_name (max_samples=$max_samples) ==="
  python -u scripts/autoencoder/hillclimb_autoencoder.py \
    --cache_path $CACHE_10M \
    --max_samples $max_samples \
    --output_dir $OUTPUT_DIR \
    --experiments $exp_name \
    2>&1 | tee $OUTPUT_DIR/${exp_name}_train.log
  echo "=== [$(date)] Finished training $exp_name ==="
}

run_training 4000000 l10_4M_drop
run_training 6000000 l10_6M_drop
run_training 8000000 l10_8M_drop
run_training 9664101 l10_10M_drop

# ── 4. Run CE eval on each ──
run_eval () {
  local exp_name=$1
  local model_path=$OUTPUT_DIR/${exp_name}_model.pt
  local out_path=$OUTPUT_DIR/${exp_name}_ce_eval.json
  echo "=== [$(date)] CE eval for $exp_name ==="
  python -u scripts/autoencoder/eval_ce_loss.py \
    --model Qwen/Qwen3-1.7B \
    --layer 10 \
    --model_path $model_path \
    --model_type autoencoder \
    --n_eval 1000 \
    --eval_start 10008000 \
    --output_path $out_path \
    2>&1 | tee $OUTPUT_DIR/${exp_name}_eval.log
  echo "=== [$(date)] Finished eval for $exp_name ==="
}

run_eval l10_4M_drop
run_eval l10_6M_drop
run_eval l10_8M_drop
run_eval l10_10M_drop

echo "=== [$(date)] Pipeline complete ==="
