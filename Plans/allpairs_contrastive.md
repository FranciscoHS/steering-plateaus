# Plan: All-pairwise multi-contrastive feature estimation (10 languages)

## Context
We want to recover individual language feature directions using the all-pairwise estimator from Section 8 of multi_contrastive_projection.pdf. Instead of the star topology (everything contrasted against one target language), we use all C(10,2)=45 pairwise contrasts between 10 languages to recover each language's feature direction via least-squares:

**f̂_i = (1/k) Σ_{j≠i} v̄_{ij}**

where v̄_{ij} is the mean difference vector for pair (i,j) and k=10 is the number of languages. This gives O(εC) error rather than O(C/√k) from the star topology.

Languages (no English): zh, nl, es, fr, de, ja, it, ru, pt, ar

## Script
Single self-contained script: `scripts/directions/run_allpairs_contrastive.py`

## Flow

### 1. Setup
- Load Qwen3-0.6B at layer 14
- Load PROMPT_SETS and LANGUAGES from `data/multilingual_prompts.py`

### 2. Extract activations
- For each of 10 languages, extract final-token activations at layer 14 for all 36 prompts
- Use `get_acts()` pattern from `run_multi_contrastive.py` (simple loop: tokenize, forward, grab hidden_states[layer+1][:, -1, :])
- Store as `acts[lang]` → [36, d_model]

### 3. Compute all 45 pairwise mean difference vectors
- For each pair (i, j) where i < j: `v̄_{ij} = mean(acts[i] - acts[j])` over 36 prompts
- Note: v̄_{ji} = -v̄_{ij}, so only compute once per pair
- Store normalized versions for diagnostics

### 4. Apply Section 8 least-squares estimator
- For each language i: `f̂_i = (1/10) Σ_{j≠i} v̄_{ij}`
- Normalize to unit length: `f̂_i = f̂_i / ||f̂_i||`
- Note: the unnormalized f̂_i should sum to ~0 (translation ambiguity). Print this as sanity check.

### 5. Diagnostics
- Print norms of unnormalized f̂_i (should be roughly comparable per Assumption 2)
- Print 10x10 cosine similarity matrix between all f̂_i
- Check Σ f̂_i ≈ 0 before normalization
- Print cosine similarities between all 45 raw pairwise vectors and corresponding (f̂_i - f̂_j) to verify consistency

### 6. Perturbation sweeps
- Load 50 FineWeb sequences (seq_len=5, seed=42) as in previous experiments
- Extract contexts and activations at layer 14
- Compute reference logits and layer-26 hidden states (for L2)
- Sweep each of the 10 f̂_i directions (positive = toward that language) and their negatives (20 total)
- Sweep 10 random directions as baseline
- Use `forward_with_measure` pattern from `rerun_plateau_sweeps.py` for KL from logits + L2 at layer 26
- Angles: np.linspace(3, 180, 60), display clipped to 60°

### 7. Save results
- Save to `results/allpairs_contrastive/qwen3-0.6b/`
  - `directions.pkl`: all 10 f̂_i, raw pairwise vectors, cosine matrix, norms
  - `sweep_results.pkl`: KL and L2 arrays for all directions + random

### 8. Plot
- Two-panel plot (KL left, L2 right) with all 10 positive directions + random baseline
- Second two-panel plot with all 10 negative directions + random baseline
- Cosine similarity heatmap of the 10 recovered features

## Key files to reuse
- `src/model.py`: `load_model`, `_get_blocks`, `_get_final_norm`, `_get_lm_head`, `_is_rope_model`, `_get_position_embeddings`, `_run_block`
- `src/data.py`: `load_fineweb_fixed_length`
- `data/multilingual_prompts.py`: `PROMPT_SETS`, `LANGUAGES`
- Pattern from `scripts/directions/rerun_plateau_sweeps.py`: `forward_with_measure`, `build_perturbed_hidden`, `sweep_single_direction` (modified for measure_layer), `sweep_random_directions`, `extract_context_and_activation`

## Run
- Spin up a RunPod pod (RTX 4090)
- Sync script + src/ + data/ + scp results back
- Model is 0.6B so sweeps should be fast (~2-3 min for 20 directions + random)

## Verification
- f̂_i should sum to ~0 before normalization
- Cosine similarity matrix should show structure (e.g., romance languages clustering)
- KL curves for positive f̂_i (toward language) should differ from negative f̂_i (away from language)
- Compare with existing plateau-breaking directions where applicable
