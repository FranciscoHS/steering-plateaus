# Plan: Direction Investigation Experiments

## Context
We have perturbation sweep results showing that PC1 and MELBO_1 break the KL plateau at ~60-80° while diff-of-means and random directions don't — despite cos(PC1, mean) = 0.98! This means a tiny orthogonal component distinguishes plateau-breaking from non-breaking directions. We want to understand this phenomenon better through four related investigations.

**Key cosine similarities (already computed):**
- cos(PC1, mean_dir) = 0.979
- cos(MELBO_1, mean_dir) = 0.897
- cos(MELBO_1, PC1) = 0.835
- MELBO_2 and MELBO_3 are orthogonal to both PC1 and mean (<0.04)

## What we'll build

### Script 1: `scripts/run_direction_investigations.py`
Handles Q1, Q2, Q3 in a single script (they share the same model/data setup).

**Phase A — Cosine similarities (Q3: MELBO vs PC1)**
- Load saved directions from `results/melbo_constrained/qwen3-0.6b/optimization_results.pkl`
- Print full pairwise cosine similarity matrix (mean, PC1, MELBO_1/2/3)
- Result: Already answered! cos(MELBO_1, PC1) = 0.84, so they're correlated but not the same. MELBO_1 is actually closer to mean (0.90) than to PC1 (0.84)

**Phase B — PC1 decomposition (Q2)**
- Decompose: `PC1 = α·mean + β·orth` where α = cos(PC1, mean) ≈ 0.979, β ≈ 0.204
- Compute `orth = normalize(PC1 - α·mean)` — the orthogonal component
- Create rotation interpolation: N directions rotating from mean toward PC1 in the {mean, orth} plane
  - `dir(φ) = cos(φ)·mean + sin(φ)·orth` (then normalize)
  - φ ranges from 0 (= mean_dir) through arccos(0.979) ≈ 11.7° (= PC1) to ~45° (well past PC1)
  - ~15-20 steps should be enough to see when plateau-breaking kicks in

**Phase C — Load model + activations**
- Load Qwen3-0.6B, layer 14
- Load FineWeb activations with **seed=123** (original used seed=42) for Q1 (generalization)
- Same n_act=50, seq_len=5
- Compute reference logits

**Phase D — Perturbation sweeps**
- Sweep directions: mean, PC1, MELBO_1, orth, random, plus all interpolated directions
- Reuse sweep infrastructure from `run_melbo_constrained.py` (`sweep_single_direction`, `sweep_random_directions`)
- Same angles: 60 points from 3° to 180°

**Phase E — Plots and save**
1. **Generalization plot (Q1):** mean, PC1, MELBO_1, random — on new activations. Does the same pattern hold?
2. **Orthogonal component plot (Q2a):** orth direction vs mean vs PC1 vs random. Does orth alone break the plateau?
3. **Rotation heatmap (Q2b):** KL vs perturbation angle, parametrized by rotation from mean→PC1. Where does plateau-breaking emerge?
4. Save all results to `results/direction_investigations/`

### Script 2: `scripts/run_melbo_unconstrained.py`
Handles Q4. Modified `run_melbo_constrained.py` where optimization is over full R^d_model, not span(V).

**Changes from constrained:**
- Replace `direction = V @ alpha_normed` with direct optimization of a d_model-dimensional vector
- No subspace projection — the optimizer finds any direction in activation space
- Keep everything else identical: same model, layer, activations, angle, n_steps, n_restarts

**After optimization:**
- Compare unconstrained direction with constrained MELBO_1: cosine similarity
- Run perturbation sweep with unconstrained direction
- Save results to `results/melbo_unconstrained/qwen3-0.6b/`

## Key files to reuse
- `src/model.py` — `load_model`, `forward_from_layer_allpos`, model utilities
- `src/data.py` — `load_fineweb_fixed_length`
- `scripts/run_melbo_constrained.py` — sweep functions (`sweep_single_direction`, `sweep_random_directions`, `build_perturbed_hidden`, `forward_to_hidden_and_logits`, `extract_context_and_activation`)

## Verification
1. Sync scripts to RunPod
2. Run Script 1: `python scripts/run_direction_investigations.py --model qwen3-0.6b --layer 14`
3. Run Script 2: `python scripts/run_melbo_unconstrained.py --model qwen3-0.6b --layer 14`
4. Sync results back, review plots
