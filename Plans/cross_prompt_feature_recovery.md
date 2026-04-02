# Plan: Cross-Prompt Feature Recovery & Alignment

## Goal
For two prompts with expected shared features ("He read the" / "I like to read"), run iterative MELBO at each, identify shared features via steering, and measure how aligned their direction vectors are. High alignment → LRH; low alignment → manifold picture.

## Step 1: Run Iterative MELBO at Each Prompt
Use `run_iterative_melbo_hf.py --custom_prompt "..."` for each prompt on isoplateau-rand pod.
- Model: qwen3-1.7b, inject_layer=7, measure_layer=-2
- No max_directions cap — let it run until reward drops below threshold
- Two runs:
  - `--custom_prompt "He read the"`
  - `--custom_prompt "I like to read"`

## Step 2: Cross-Prompt Comparison & Steering Script
Write `scripts/melbo/compare_cross_prompt.py` that:
1. Loads directions from two MELBO runs (two `results.pkl` files)
2. Computes all pairwise cosine similarities between direction sets → matrix
3. For each direction from both prompts: steers both prompts, generates completions (greedy, ~40 tokens)
4. Saves: cosine similarity matrix, all steered completions, raw data

Reuse from existing code:
- Steering hook from `steer_cross_prompt.py` (generalized to non-GPT2 via `_get_blocks()`)
- Cosine similarity matrix plotting from manifold scripts
- `src/model.py` for model loading

## Step 3: Analysis
- Inspect completions: identify which directions produce similar behavioral effects across prompts
- For matched features: report cosine similarity
- Cross-steering: does direction from prompt A produce same effect on prompt B?
  - Yes + high cosine → LRH
  - Yes + low cosine → manifold picture
  - No → not the same feature
