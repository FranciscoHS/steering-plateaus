# Plateau-breaking objectives tried (Qwen3-1.7B, L14→26)

Goal: build dictionary directions with high superellipse p (held-out eval-p).
All builds: random contrast partners, K, step-resampled; 1 fixed anchor/dir.

## 1. Ratio proxy (original) — `R_mix / (R_f + R_c)`, minimized
- Single input-angle surrogate. **Optimizes cleanly** (loss 0.5→~0.03, K=4 and
  K=10 identical), but only weakly tracks eval-p (Spearman ≈ −0.33).
- Eval-p of resulting dirs: **~2.2**.

## 2. soft_p — Gauss-Newton fit of p to the soft-intercept contour, maximized
- Eval-frame estimator (rays, soft intercept, geodesic norm, |x|^p+|y|^p=1 fit,
  cap at p=10). **Calibrates well as a measurement** (Spearman 0.86 vs eval-p).
- But **poor objective**: p̂ wanders/drifts down, doesn't climb. Likely the hard
  bracket selection + p-cap give a choppy/flat gradient.
- Eval-p of resulting dirs: **~2.2**.

## 3. diag_rho — maximize smooth normalized diagonal intercept ρ (monotone in p)
- Simplification: drop the fit + cap; smooth (all-radii) intercept; maximize
  ρ = normalized 45° intercept (p = 1/(½−log₂ρ)).
- **Climbs** (gradient problem fixed) but **noisy/non-monotonic** (e.g. dir 1
  peaks ~step 40 then falls) and **overshoots** ρ past √2 → degenerate corner
  (p_monitor saturates to ~1000, meaningless).
- Held-out eval-p: **~2.2** vs random partners (pooled 2.22) AND vs each other
  (pooled 2.02, n=6). High train-ρ does **not** survive eval.

## Standing puzzle / open question
Eval-p is **pinned at ~2.0–2.2 across all three objectives**, regardless of how
hard the training signal is pushed. Train/eval gap is not contrast-overfitting
(contrast is resampled each step). Candidate mechanisms (untested):
(a) overfitting to the single fixed training anchor (eval uses 20 held-out);
(b) train monitor ≠ eval estimator (1 ray / smooth intercept / coarse radii vs
full contour, box-masked, 20 anchors).

## Files
- calibration (soft_p): `results/dictionary/calib_softp_qwen_L14/`
- trajectories: `diag_softp_hist_K4/`, `diag_ratio_hist_K4_nc{4,10}/`,
  `diag_diagrho_hist_K4/` (each has `*trajectory*.png` + `histories.pkl`)
- eval-p of diag_rho dirs: `diag_diagrho_hist_K4/eval_vs_random/`, `eval_vs_self/`
