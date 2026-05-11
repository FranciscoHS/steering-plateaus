#!/bin/bash
# Run CE loss eval for a trained model.
# Usage: run_final_eval.sh <model_type> <model_path> <output_name>
# Example: run_final_eval.sh sae results/final_comparison/sae_k32_50k/sae_weights.pt sae_k32

set -e
cd /workspace/steering-plateaus

MODEL_TYPE=$1
MODEL_PATH=$2
OUTPUT_NAME=$3

pip install -q transformers accelerate datasets

# HuggingFace login for gated models
python -c "from huggingface_hub import login; login(token='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')" 2>/dev/null || true

python -u scripts/autoencoder/eval_ce_loss.py \
    --model Qwen/Qwen3-1.7B \
    --layer 2 \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --n_eval 1000 \
    --eval_start 600000 \
    --seq_len 10 \
    --batch_size 32 \
    --output_path "results/final_comparison/eval_${OUTPUT_NAME}.json"
