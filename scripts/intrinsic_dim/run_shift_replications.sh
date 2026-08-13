#!/bin/bash
# Queue: separating-normal-shift replication (pipeline + 1D sweep) on
# qwen3-0.6b, llama-3.1-8b, gemma-2-9b. Run on the pod with nohup.
# Each stage checkpoints, so re-running the queue resumes where it stopped.
set -u
cd /workspace/steering-plateaus
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one () {
  local model=$1 dtype=$2 chunk=$3
  echo "=== $model pipeline ($(date)) ==="
  python -u scripts/intrinsic_dim/shift_replication_pipeline.py \
    --model "$model" --layer 2 --dtype "$dtype" --chunk "$chunk" \
    || { echo "PIPELINE FAILED: $model"; return 1; }
  echo "=== $model sweep ($(date)) ==="
  python -u scripts/superellipse/eval_real_vs_random_acts.py \
    --model "$model" --layer 2 --n_act 30 --seq_len 5 --max_angle 90 \
    --n_steps 46 --n_dirs 10 --random_tokens \
    --extra_anchors_pt "results/intrinsic_dim/${model}_shift/jac_optimized.pt" \
    --extra_anchors_key X_opt --extra_label jac-optimized \
    --extra_anchors_pt "results/intrinsic_dim/${model}_shift/shift_anchors.pt" \
    --extra_anchors_key X_shift --extra_label 'gauss + c*w' \
    --out "results/real_vs_random_acts/${model}_L2_jacopt_randtok_gaussshift" \
    || { echo "SWEEP FAILED: $model"; return 1; }
}

run_one qwen3-0.6b float32 25
run_one llama-3.1-8b bfloat16 8
run_one gemma-2-9b bfloat16 4
echo "=== queue done ($(date)) ==="
