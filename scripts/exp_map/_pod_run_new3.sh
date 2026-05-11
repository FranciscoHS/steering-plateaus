#!/bin/bash
# Run one config from Plans/exp_map_new3_directions.md on the current pod.
# Each invocation handles: extraction (if needed) + pair list + sweep_2d.
# All output goes to the shared network volume so PKLs collect across pods.
#
# Usage:
#   bash scripts/exp_map/_pod_run_new3.sh <CONFIG>
#
# CONFIG ∈ {gemma_L2_canonical, gemma_L2_M7, gemma_L2_M12,
#           gemma_L2_coskl, gemma_L2_additive,
#           gemma_L2_wiki_en, gemma_L2_wiki_zh, gemma_L2_code,
#           gemma_L5, gemma_L10, gemma_L20,
#           qwen_L2_canonical, llama_L2_canonical, mistral_L2_canonical,
#           aya_L2_canonical, yi_L2_canonical}
set -e
CONFIG="${1:?usage: $0 <CONFIG>}"
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
cd /workspace/steering-plateaus

echo "[$(date)] CONFIG=$CONFIG"
pip install -q transformers accelerate
python -c "from huggingface_hub import login; login(token='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')"

case "$CONFIG" in
  gemma_L2_canonical|gemma_L2_M7|gemma_L2_M12|gemma_L2_coskl|gemma_L2_additive|gemma_L2_wiki_en|gemma_L2_wiki_zh|gemma_L2_code)
    TARGET=gemma; LAYER=2;;
  gemma_L5)  TARGET=gemma; LAYER=5;;
  gemma_L10) TARGET=gemma; LAYER=10;;
  gemma_L20) TARGET=gemma; LAYER=20;;
  qwen_L2_canonical|llama_L2_canonical|mistral_L2_canonical|aya_L2_canonical|yi_L2_canonical)
    TARGET=${CONFIG%_L2_canonical}; LAYER=2;;
  *) echo "unknown CONFIG=$CONFIG"; exit 1;;
esac

# 1. ensure W/H/C are in the dirs cache for (TARGET, LAYER).
echo "[$(date)] ensuring directions Wealth/Health/Certainty in cache for $TARGET L=$LAYER"
python -u scripts/exp_map/_extract_new3.py --target "$TARGET" --layer "$LAYER"

# 2. emit the pair list. For the new non-gemma cross-model column we
#    apply the plot-time overlap filter (|<d_i, d_j>| < 0.10) so we
#    don't waste sweep time on pairs that would be dropped at plot
#    time. For gemma configs we keep the unfiltered 93-pair list to
#    stay consistent with the existing pkls.
SELECT_FLAGS=(--target "$TARGET" --layer "$LAYER")
case "$CONFIG" in
  qwen_L2_canonical|llama_L2_canonical|mistral_L2_canonical|aya_L2_canonical|yi_L2_canonical)
    SELECT_FLAGS+=(--overlap_max 0.10);;
esac
python -u scripts/exp_map/select_new3_pairs.py "${SELECT_FLAGS[@]}"
PAIRS_FILE="results/exp_map/data/pair_lists/new3_pairs_${TARGET}_L${LAYER}.csv"

# 3. dispatch sweep_2d with the right variant flags.
COMMON=(--target "$TARGET" --layer "$LAYER" --pairs_file "$PAIRS_FILE" --skip_existing)
case "$CONFIG" in
  gemma_L2_canonical) FLAGS=();;
  gemma_L2_M7)        FLAGS=(--measure_offset -7);;
  gemma_L2_M12)       FLAGS=(--measure_offset -12);;
  gemma_L2_coskl)     FLAGS=(--metrics "cos,kl,l2");;
  gemma_L2_additive)  FLAGS=(--mode additive);;
  gemma_L2_wiki_en)   FLAGS=(--anchor_source wiki_en);;
  gemma_L2_wiki_zh)   FLAGS=(--anchor_source wiki_zh);;
  gemma_L2_code)      FLAGS=(--anchor_source code);;
  gemma_L5|gemma_L10|gemma_L20) FLAGS=();;
  qwen_L2_canonical|llama_L2_canonical|mistral_L2_canonical|aya_L2_canonical|yi_L2_canonical)
    FLAGS=();;
esac

echo "[$(date)] sweep_2d ${COMMON[*]} ${FLAGS[*]}"
python -u scripts/exp_map/sweep_2d.py "${COMMON[@]}" "${FLAGS[@]}"
echo "[$(date)] done CONFIG=$CONFIG"
