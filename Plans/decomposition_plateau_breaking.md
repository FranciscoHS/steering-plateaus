# Decomposition into plateau-breaking directions

## Idea

Greedy decomposition of an activation `a_target` into K unit feature
directions `{d_k}` and signed scalar coefficients `{α_k}` such that
`a_target ≈ Σ_k α_k d_k`. Each `d_k` is jointly required to be:

- **plateau-breaking** — interferes super-additively (low `R_mix / (R_f+R_c)`)
  with an averaged contrastive scaffold, as in
  `learn_feature_dir_vs_contrastive.py`,
- **aligned with the current residual** of `a_target` (extra overlap term),
- distinct from the scaffold and from previously found features.

Under the LRH the activation is a linear combination of features, so every
true feature in the decomposition should also have non-trivial projection
onto `a_target` — the overlap term picks the "plateau-breaking dimension
that actually shows up in this activation" rather than any random
plateau-breaking dimension in `R^d`.

## Inspiration

`scripts/delta_optimization/response_decomp.py` — same greedy structure
(α = d · r, residual update r ← r − α d, push-apart penalty against
previously found features). We swap their cosine-distance "response" term
for our superellipse proxy averaged over a contrastive scaffold.

## Loss per direction k

Minimise
```
L(d_k) = mean_c [R_mix / (R_f + R_c)]              # plateau-breaking proxy
       + λ_scaffold · mean_c (d_k · d_c)²           # stay away from scaffold
       + λ_prev     · mean_prev (d_k · d_prev)²     # stay away from prev features
       − λ_overlap  · (d_k · r̂_{k-1})               # align with current residual
```
where `r̂ = r / ||r||`, `R_f / R_c / R_mix` are the downstream L² responses
to a spherical perturbation at angle α along `d_k` / `d_c` / their tangent
bisector, computed on the opt-set activations and averaged over scaffold
directions.

## Greedy outer loop

```
r ← a_target;  features ← []
for k = 1, 2, ...:
    r̂ ← r / ||r||
    d_k ← optimize(L(d_k))                  # Adam + renormalise (same as before)
    α_k ← d_k · r                            # signed scalar projection
    features.append((d_k, α_k))
    r ← r − α_k · d_k
    if ||r|| / ||a_target|| < 0.05: break    # stop condition
    if k == K_max:                  break    # safety bound
```

## Settings (defaults)

- Model: `gemma-2-9b`, layer 15, measure_layer = 2nd-to-last (= 40).
- Opt-set activations: 10 FineWeb sequences (seq_len 10), `seed=42` — same
  as the previous runs. (For `R_f / R_c / R_mix` only.)
- Target activation: 1 separate FineWeb sequence at `target_seed = 999`,
  index 0 (held-out from the opt-set).
- Scaffold: same 10 names as before — `Chinese, Python, Gender, Age,
  Sentiment, Tense, Number, Formal, Refusal, Honesty` — extracted at L15
  via fsec-anonymous's cache (already built).
- Perturbation angle α = 30° (single, list-capable).
- λ_scaffold = 1.0, λ_prev = 1.0 (match prior runs).
- **λ_overlap = 0.1** — proxy term is O(0.05), overlap term is in [−1, 1];
  0.1 makes them comparable. CLI-sweepable.
- Stop condition: `||r|| / ||a_target|| < 0.05` (residual ≤ 5%).
- `K_max = 20` (safety cap; should not be hit at 5% threshold).
- Adam, lr = 3e-3, n_steps = 200 per direction.
- bfloat16, HF_HOME = /workspace/hf_cache on pod.

## New script

`scripts/superellipse/decompose_plateau_breaking.py`
Results → `results/decomp_plateau/<model>_L<layer>_a<angle>/`

Reuses verbatim from `learn_superellipse_dict.py`:
`extract_context_and_activation`, `pair_loss`, `pair_responses`.
Reuses verbatim from `learn_feature_dir_vs_contrastive.py`:
`_find_fsec_path`, `load_or_build_scaffold`.

### Flow

1. Load model + scaffold (build via fsec extract_all if missing — already
   built for our 10 names).
2. Load `n_act + 1` FineWeb sequences:
   - opt-set: first `n_act` from `--seed`,
   - target: index 0 from a separate `--target_seed` to keep it strictly
     disjoint from the opt-set across all seeds.
3. Extract layer-`L` activations for both sets; precompute `f_refs` for
   opt-set only.
4. Greedy loop: optimise each `d_k` with the loss above, update residual,
   stop on the 5% criterion (or `K_max`).
5. Save `results.pkl` + summary plot.

### Outputs (files, never stdout-only)

- `results.pkl`:
  - `features [K, d]` (unit vectors),
  - `alphas [K]` (signed scalar projections, in raw units),
  - `residual_norm_frac [K+1]` (`||r_k|| / ||a_target||`, starting at 1),
  - per-step loss histories for each k,
  - scaffold gram, feature-scaffold gram, feature-feature gram,
  - K_used (actual K reached when stop hit).
- `summary.png`:
  - residual-fraction trajectory `||r_k||/||a||` vs k (decreasing curve),
  - bar of `|α_k|` per feature,
  - loss curves (one panel per feature),
  - feature × scaffold overlap heatmap.

## What we'll learn

- Whether a small number of plateau-breaking + overlap-aligned features
  explain ≥ 95 % of `||a_target||`.
- How distinct the features end up being (feature-feature Gram).
- How they relate to the contrastive scaffold (feature-scaffold Gram).
