# Plan: Superellipse p-fitting on J-Lens directions (Qwen3-8B, L20)

**Date:** 2026-07-15
**Goal:** Run the iso-plateau superellipse p-fitting machinery on pairs of J-Lens
token vectors, comparing English vs Chinese token directions.

## Background
- J-Lens (Anthropic global workspace paper; Neel Nanda's review + Qwen 3.6 27B
  replication) defines one vector per vocab token per layer: the token's
  unembedding direction pulled back through the fitted average Jacobian
  J_l = E[∂h_final/∂h_l].
- Pre-fitted lenses: HF `neuronpedia/jacobian-lens` (folder `qwen3-8b`);
  reference code: `github.com/anthropics/jacobian-lens` (`jlens` package).
- Neel's observations on Qwen 3.6 27B live at ~55–80% depth. Qwen3-8B has 36
  layers → that band is ~L20–29. We use **L20** (~56% depth), Francisco's call.

## Model / layer config
- Model: `qwen3-8b` (Qwen/Qwen3-8B, 36 layers, d_model 4096) — already in
  `src/model.py` registry.
- Perturbation layer: **L20**. Measure layer: n_layers − 2 = L34 (matching
  existing iso-plateau convention).

## Steps
1. **Download + inspect lens** (local, no model load): pull the `qwen3-8b`
   folder from `neuronpedia/jacobian-lens`, confirm format (per-layer J
   matrices), check d_model/layer count line up.
2. **Build direction sets** (pod): compute J-Lens vectors at L20 for selected
   tokens. Categories, ~10 pairs each:
   - EN×EN: concrete single-token English content words
   - ZH×ZH: multi-character single-token Chinese words (the "rich concept"
     tokens Neel highlights)
   Filter pairs to low cosine similarity so the two superellipse axes are
   distinct; record cosines. Save directions pkl in the format the sweep
   scripts expect.
4. **Iso-plateau sweeps** (pod): new script variant (do not overwrite gemma
   ones) adapted from `scripts/perturbation/sweep_isoplateau*.py`: 30 base
   activations from generic text prompts, 2D angle grid, L2 response at L34,
   checkpointed per pair. Then `fit_superellipse` per pair/threshold.
5. **Compare p across categories** — beeswarm/summary, same style as existing
   direction-family comparisons.

## Status (completed 2026-07-15)
- [x] Step 1: download/inspect lens (`results/jlens/qwen3-8b/lens_inspection.txt`)
- [x] Step 2: direction sets — 9 EN + 10 ZH pairs, |cos|≤0.2 (max matching;
      10 disjoint EN pairs under 0.2 don't exist — EN J-Lens vectors are more
      mutually correlated than ZH)
- [x] Step 3: sweeps + p fits — base acts: 30 FineWeb 5-token prompts
      (matching gemma baselines); thresholds recalibrated to qwen scale
      (noise floor ~10, axis max ~200 → L2 = 15/20/30/50/80/120)
- [x] Step 4: plots — beeswarm + per-pair shape panels in
      `results/jlens/qwen3-8b/`
