#!/bin/bash
# Recovery pipeline: re-train 4M, 6M, 8M (which crashed on eval with old code)
# and run CE evals for all four sizes.

set -u

REPO=/workspace/steering-plateaus
CACHE_10M=$REPO/results/trained_autoencoder/l10_cache_10M/cached_activations.pt
OUTPUT_DIR=$REPO/results/trained_autoencoder/l10_scaling

cd $REPO

echo "=== [$(date)] Recovery pipeline starting ==="

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

run_eval () {
  local exp_name=$1
  local model_path=$OUTPUT_DIR/${exp_name}_model.pt
  local out_path=$OUTPUT_DIR/${exp_name}_ce_eval.json
  if [ ! -f $model_path ]; then
    echo "=== [$(date)] SKIP eval for $exp_name (no model at $model_path) ==="
    return
  fi
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

run_training 4000000 l10_4M_drop
run_training 6000000 l10_6M_drop
run_training 8000000 l10_8M_drop

run_eval l10_4M_drop
run_eval l10_6M_drop
run_eval l10_8M_drop
run_eval l10_10M_drop

echo "=== [$(date)] Recovery pipeline complete ==="
