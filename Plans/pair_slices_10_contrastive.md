# 45 pair slices from 10 contrastive directions

## Goal
Generalize the pair iso-plateau result to 10 contrastive directions at once, visualized as 45 pair contours overlaid on one plot with $p=1/2/\infty$ references. This is the page-2 figure for the short Google Doc.

## Setup

- **Model:** qwen3-1.7b
- **Layer:** 14 (matches existing `allpairs_contrastive/qwen3-1.7b/`)
- **Measure layer:** penultimate ($n_{\text{layers}} - 2 = 26$)
- **Directions:** 10 language contrastive features from `results/allpairs_contrastive/qwen3-1.7b/directions.pkl` key `features` — keys zh, nl, es, fr, de, ja, it, ru, pt, ar. Pairwise |cos|: mean 0.24, max 0.69 (not tight but usable).
- **Prompts:** 30 FineWeb holdout activations (seed 42, seq_len 5, act_idx 0..29 — matching the isoplateau/MELBO convention).
- **On-sphere parametrization:** same as `sweep_isoplateau.py`. Per-activation Gram-Schmidt of the pair's two directions against $\hat a$ and against each other.

## Phase A: 10-direction sensitivity baseline (Page-1 Figure)

Single-direction sweeps along each of the 10 contrastive directions + ~10 random unit directions (Gram-Schmidt'd against $\hat a$), 30 activations, median response per metric.

Output: one panel showing median response vs. angle for contrastive (solid) and random (dashed) directions. Demonstrates contrastive directions are consistently more sensitive (smaller plateau-breaking angle) than random.

Cost: $(10 + 10) \times 30 \times 40$ angle steps $\approx$ 24k forward passes. Fast.

## Phase B: 45 pair slices (Page-2 Figure)

For each of $\binom{10}{2} = 45$ pairs $(i, j)$:
- For each of 30 activations, build a 2D sphere grid over $(\alpha_i, \alpha_j)$ with the existing `sweep_isoplateau.py` machinery: $p = R(\cos\alpha_{\text{tot}} \hat a + \sin\alpha_i \hat d_i^\perp + \sin\alpha_j \hat d_j^\perp)$.
- Measure cos and L2 at measure layer. (KL optional — omit for cost unless we want it.)
- Median across 30 activations → 2D response grid per metric.
- Extract the iso-plateau contour at calibrated threshold, normalize by axis intercepts, fit superellipse exponent $p$.
- Record $p_{\text{fit}}^{(ij)}$ plus normalized contour points.

**Grid resolution:** 25×25 on $[0, 25^\circ]$ per axis (isoplateau's default is 41×41 on $[0, 20^\circ]$; 25×25 is lighter and should be fine for fitting a superellipse that hits the contour in only a few quadrants anyway). Can bump up if fits look undersampled.

**Threshold calibration:** phase 0 — pick thresholds from a few example sweeps by inspection, or reuse the ratios `plot_isoplateau.py` uses on Gemma (`L2=50`, `cosine=0.01`) scaled appropriately. Fix one threshold each for cos and L2 and use them consistently across all 45 pairs.

**Cost:** 45 pairs × 25² × 30 activations ≈ 840k forward passes. qwen3-1.7b at ~1000 fwd/s on RTX 4090 ≈ ~15 min. Manageable.

## Phase C: overlay plot

Two panels (one per metric: cos, L2). Each panel:
- Reference shapes: $p=1$ diamond (dashed light), $p=2$ circle (dashed medium), $p=20$ square (dashed light) in the first quadrant.
- All 45 normalized pair contours overlaid as faint lines (one per pair).
- Median of fitted $p$'s with 5–95 percentile, annotated.
- Marker at $(1, 0)$ and $(0, 1)$ for individual-direction breakings.

Additionally: histogram/strip plot of the 45 fitted $p$'s, showing the distribution clusters near 2.

## Reuse

- `scripts/perturbation/sweep_isoplateau.py`: almost drop-in for the 2D pair sweep. Refactor to loop over 45 pairs and accept directions/activations as arguments rather than hard-coded.
- `scripts/perturbation/plot_isoplateau.py::fit_superellipse`: reuse for the per-pair $p$ fit.
- Reference-shape drawing code: reuse from `plot_isoplateau.py::superellipse_curve`.

New script locations:
- `scripts/perturbation/sweep_pair_slices_10contrastive.py` — runs Phase A + Phase B, saves pkl.
- `scripts/perturbation/plot_pair_slices_10contrastive.py` — makes the overlay plot + histogram.

## Outputs

- `results/pair_slices_10contrastive/qwen3-1.7b_L14/`
  - `phase_a_sensitivity.pkl` + `.png`
  - `pair_slices.pkl` — [45, 30, 25, 25] grids per metric, plus fitted $p$'s
  - `pair_slices_overlay.png` — the 45-contour overlay
  - `fitted_p_histogram.png` — distribution of $p$'s across pairs

## Decisions

1. Metrics: cos + L2 only, or also KL? (KL roughly doubles cost but gives a third independent check.)
2. Random baseline in Phase A: 10 random directions or just 2–3? (More is more convincing but cost is tiny either way.)
3. Threshold: calibrate automatically per-metric (50% of median max across contrastive sweeps), or hard-code values from existing Gemma isoplateau (`L2=50`, `cos=0.01`) and check they make sense?

## Open question on visualization

45 contours overlaid may look crowded. If so, alternatives:
- Median contour across pairs with a confidence band
- Heatmap of contour density
- Sample 10 random pairs rather than all 45
Let's start with the raw overlay and see.
