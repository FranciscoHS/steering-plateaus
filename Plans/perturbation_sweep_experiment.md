# Plan: Perturbation Sweep Experiment

## Context
We want to directly measure the "activation plateau" phenomenon: perturb real vs random activations by varying magnitudes in random directions, measure the downstream effect on final-layer activations (L2 distance). This is the foundational experiment that validates the plateau hypothesis — previous work used Jacobian norms as a proxy.

We also want to support Qwen3-0.6B alongside GPT-2 Large to check if plateaus appear in a better small model.

## Files to modify/create
1. **`src/model.py`** — Add multi-model support (Qwen3-0.6B)
2. **`scripts/run_perturbation_sweep.py`** — New experiment script

## Changes to `src/model.py`

### Multi-model abstraction (minimal, functional)
Add helpers to dispatch on architecture:
```python
def _get_blocks(model): ...       # model.transformer.h or model.model.layers
def _get_final_norm(model): ...   # model.transformer.ln_f or model.model.norm
def _get_lm_head(model): ...      # model.lm_head (same for all)
def _is_rope_model(model): ...    # True for Qwen/Llama
```

### `load_model()` — Extend to support Qwen3
- Use `AutoModelForCausalLM` and `AutoTokenizer` instead of GPT-2-specific classes
- Map short names to HuggingFace names: `'qwen3-0.6b' → 'Qwen/Qwen3-0.6B'`
- Attach `model._plateau_info = {'model_type': 'gpt2'|'llama', 'd_model': int}` for dispatch
- Keep 3-tuple return `(model, tokenizer, device)` — no breaking change
- Set `tokenizer.pad_token = tokenizer.eos_token` if needed

### `forward_from_layer()` — Handle RoPE
For Llama/Qwen models, pass `position_ids = torch.arange(seq_len)` to each block. GPT-2 blocks don't need this (position embeddings baked in at layer 0).

### `forward_from_layer_to_final()` — Same treatment
This is the function we'll use for the perturbation sweep (and needed for future Jacobian computations on Qwen3).

### Leave unchanged
`get_layer0_output()`, `forward_from_layer1()` — GPT-2-specific legacy functions used by older scripts.

## Script: `scripts/run_perturbation_sweep.py`

### Metrics: L2 distance + KL divergence
Two complementary metrics per perturbation:
1. **L2 distance** on final-layer activations: `||f(a + eps*v) - f(a)||_2` where `f = forward_from_layer_to_final` (layer L → final block, no ln_f/lm_head). Measures internal representation change.
2. **KL divergence** on output logits: `KL(softmax(logits_perturbed) || softmax(logits_ref))`. Measures behavioral change.

Cheap to compute together — from `forward_from_layer_to_final` output, apply `ln_f + lm_head` to also get logits.

### Random activations: simple norm-matched
No Gaussian covariance fitting. Just `torch.randn(n, d_model)` rescaled so each vector has the same norm as a typical real activation. Simple and sufficient.

### Parameters
- `--model`: `gpt2-large` (default) or `qwen3-0.6b`
- `--layer`: 6 (default)
- `--seq_len`: 5
- `--n_act`: 100 (activations for sweep)
- `--n_ctx`: 100 (cross-contexts, separate FineWeb sequences)
- `--n_dirs`: 10 (random directions per activation per magnitude)
- `--n_mags`: 30 (magnitude points, log-spaced)
- `--mag_min`: 1e-3 (relative to mean activation norm)
- `--mag_max`: 10.0 (relative to mean activation norm)

### Three conditions
1. **real_same_ctx**: Real activation `a_i` with its own context `ctx_i`
2. **real_cross_ctx**: Real activation `a_i` with borrowed context `ctx_j` (j != i)
3. **random**: Norm-matched random activation `r_i` with borrowed context `ctx_j`

### Stages
**Stage 0: Extract activations**
- Load `n_act + n_ctx` FineWeb sequences (fixed length)
- Extract layer-L activations + prefix contexts for all
- Split: first `n_act` → real activations + same-contexts; last `n_ctx` → cross-contexts
- Compute `act_norm_mean` for magnitude normalization
- Sample norm-matched random activations: `randn` scaled to `act_norm_mean`
- Pre-sample random directions: `[n_act, n_dirs, d_model]` unit vectors (reused across conditions for paired comparison)

**Stage 1: Perturbation sweep — real, same-context**
For each activation `a_i` with context `same_contexts[i]`:
- Compute reference final-layer activation `f(a_i)` via `forward_from_layer_to_final`
- For each magnitude `eps`:
  - Batch perturbations: `a_i + eps * directions[i]` → `[n_dirs, 1, d_model]`
  - Batched `forward_from_layer_to_final` → `[n_dirs, d_model]`
  - Apply `ln_f + lm_head` to also get logits → `[n_dirs, vocab_size]`
  - L2 distances: `||f(a_i + eps*v_k) - f(a_i)||_2` for each direction k
  - KL divergences: `KL(softmax(logits_perturbed_k) || softmax(logits_ref))` for each direction k
- Store `l2[i, mag_idx, dir_idx]` and `kl[i, mag_idx, dir_idx]`
- Checkpoint every 10 activations

**Stage 2: Perturbation sweep — real, cross-context**
Same but pair `a_i` with `cross_contexts[i % n_ctx]`

**Stage 3: Perturbation sweep — random**
Same but use `random_activations[i]` with `cross_contexts[i % n_ctx]`

**Stage 4: Plot and save**
- Plot 1: median L2 distance (with IQR shading: shaded band between 25th and 75th percentile across activations) vs relative magnitude, three conditions, log-log scale
- Plot 2: same but for KL divergence
- Save results pickle with full L2 and KL arrays

### Efficiency
- 100 acts × 30 mags × 3 conditions = 9,000 batched forward passes (each batching 10 directions)
- Each is a partial forward through layers L+1 to end on a 5-token sequence
- Estimated: ~1-2 minutes per condition on A100

### Output
- `results/perturbation_sweep/{model_name}/results.pkl`
- `results/perturbation_sweep/{model_name}/perturbation_sweep.png`

## RunPod
- `rpc create --gpu_type "A100 SXM" --runtime 600` (10 hours)
- Run GPT-2 Large first, then Qwen3-0.6B on same pod

## Verification
1. Run full experiment on RunPod for both models
2. Check that plots show the expected pattern: flat L2/KL for real activations at small perturbations, immediate rise for random activations
