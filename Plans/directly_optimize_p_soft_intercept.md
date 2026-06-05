# Plan: Directly optimize p via a soft-intercept differentiable proxy (Qwen3-1.7B, L14)

## Motivation

The current dictionary-build objective minimizes `R_mix / (R_f + R_c)` — a
single-input-angle surrogate for the superellipse exponent p. We measured that
it only weakly tracks the eval-time p (Spearman ≈ −0.33 between per-atom proxy
loss and per-atom p), and that the resulting p sits at a constant ~2.3 across
thousands of directions, flat across build order (bands [0,440)…[3960,4400)
all ~2.1–2.4). Meanwhile p ≫ 2 directions demonstrably exist (per-pair fits up
to ~5–6), so high-p directions are reachable but not being targeted.

The loose coupling is structural, not noise: the current proxy lives in the
"fixed input angle → output response" frame, whereas p is defined in the dual
"fixed output level → input-angle iso-contour" frame. The two are related only
through the radial response profiles, which the single-angle proxy discards.

**Goal:** build a tighter, still-differentiable estimator of p that lives in
the same frame as the eval fit (references an output level, measures
input-angle intercepts), optimize it directly, and answer the real question:
**does pushing held-out p up drag feature-ness (SAE alignment, sparsity, 1D
responsiveness) up with it?** If we can raise p to e.g. 3.5 and SAE alignment
does not move, that is the cleanest refutation that p tracks features in LLMs.
If it does move, p is validated as a feature signal here.

## The estimator (per pair (d, c))

In the 2D (θ_d, θ_c) input-angle plane, sample the output L2 response along
`n_phi` rays spanning φ ∈ [0°, 90°] (φ=0 is the d-axis, φ=90 the c-axis,
intermediate rays mix d and c), each at `n_r` geodesic radii in [0, max_angle].

1. **Threshold T (mirror eval exactly).** Eval uses `thresh = 0.5 * base`,
   `base = min(axis-max responses)`. Replicate: T = 0.5 · min(max R on d-axis,
   max R on c-axis) over the sampled radii.
2. **Soft intercept.** Along each ray, find the radius r_φ where R(r_φ) = T by
   linear interpolation between the two bracketing samples — differentiable in
   the sampled R's (which are differentiable in d). No hard contour extraction,
   no root solver.
3. **Normalize like eval.** Apply the eval's exact-geodesic normalization
   (`sin(r)/sin(t_axis)` factor + per-axis scaling by the two axis intercepts)
   so train-p and eval-p are on the same footing.
4. **Recover p differentiably.** Fit `|x|^p + |y|^p = 1` to the normalized
   contour points by a few Newton steps on p (minimizing
   Σ_i (|x_i|^p + |y_i|^p − 1)²), all differentiable. Minimal fallback if the
   fit is unstable: 3-ray closed form using only the 45° point,
   p = 1 / (1/2 − log₂ ρ) with ρ the geodesic-normalized diagonal intercept.
5. **Per-atom objective.** Average p over the K=4 (per-step-resampled) contrast
   partners and over anchors. Maximize it.

   ```
   L(d_k) = − mean_{c, anchor} p̂(d_k, c)        # maximize p
          + lam_prev · mean_prev (d_k · d_prev)²  # diversity (unchanged)
   ```

Consider a bounded transform of p (or a cap) for optimization stability, since
p → ∞ as the diagonal intercept approaches the corner.

## Settings (this run)

- Model **Qwen3-1.7B**, layer **14**, measure_layer **26** — same as all prior
  comparisons. (L7 noted as a fallback if L14 is unconvincing.)
- **K (contrast partners) = 4**, per-step resampled (per-step resampling shown
  to generalize and avoid overfitting; K=4 not 1 per Francisco).
- **n_r = 6** radii per ray, **n_phi = 5** rays (0°, 22.5°, 45°, 67.5°, 90°),
  max_angle = 60°.
- Greedy build, lam_prev = 1.0, n_steps = 150, lr = 0.003 (lr/steps may need
  retuning — the landscape changes; check the first few atoms).
- bf16, one FineWeb activation per atom (seq_len 10, seed 42), as before.

## Compute

- **Points/step** (d-axis = φ=0 ray shared across contrasts):
  `n_r + K·(n_phi−1)·n_r = 6 + 4·4·6 = 102` (vs 30 now).
- **Memory:** ~40 MB/point marginal → ~+3 GB over the current 4.7 GB; ceiling
  ~400–450 points in 19.5 GB free. Not binding.
- **Time:** current regime is latency-bound (n1=3pts ≈ n10=30pts, both
  ~2.4 s/atom), so 30→~100 points is likely near-free. **First action: measure
  where per-step time stops being flat** — time a step at 30 / 100 / 300 / 600
  points. Everything below saturation is free radii.

## Plan of work

**Phase 0 — estimator + calibration (no build).** Implement `p̂(d, c)` as a
standalone differentiable function. On a sample of existing dict pairs, compare
`p̂` (train estimator) against the eval `fit_pairs` p on the *same* pairs.
Tune n_r / n_phi / T / normalization until `p̂` tracks eval-p well
(target Spearman ≫ the current −0.33; aim ≳ 0.7). This de-risks everything
before any optimization — if the surrogate isn't tight here, fix it before
spending compute. Also measure the per-step time curve (saturation point).

**Phase 1 — small build.** ~300–500 atoms, K=4, new proxy. Confirm time/memory,
watch p climb during optimization, sanity-check a few atoms' eval-p.

**Phase 2 — evaluate.** On the new dict:
- held-out p (fresh partners + fresh anchors) — Goodhart guard,
- pairwise p distribution vs the R_mix-proxy build at matched atom count,
- **feature tests: SAE alignment (best-match cosine vs random null), projection
  sparsity, 1D responsiveness** — the payoff.
- Plot p vs SAE-alignment per atom: does higher p ⇒ higher alignment?

**Decision.** If higher p ⇒ higher feature-ness, p is validated and we scale.
If p rises but feature tests stay flat, that's strong evidence p does not track
features in this LLM — report and reconsider the whole framing.

## Files

- **New** `scripts/superellipse/soft_p_proxy.py`: the differentiable `p̂(d, c)`
  estimator (sampling, soft intercept, geodesic normalization, p fit). Reuse
  `forward_from_layer_to_layer` and mirror the eval normalization in
  `canonical/superellipse.py` / `plot_boundary_pair.boundary_for_set`.
- **Modify** `scripts/superellipse/build_dictionary.py`: add
  `--objective {ratio, soft_p}` (default `ratio`); when `soft_p`, swap the
  proxy term in `optimize_one_atom` for `−mean p̂` and add `--n_radii`,
  `--n_rays`. Keep K via existing `--n_contrast` (=4) and `--contrast_resample
  step`. Output dir suffix `_softp`.
- **Reuse** `fit_pairs_dictionary.py`, `dict_sae_alignment.py`,
  `dict_projection_dist.py`, `eval_plateaus_1d_dict.py` for Phase 2 (no change).
- **New** `scripts/superellipse/calibrate_soft_p.py`: Phase-0 train-p vs eval-p
  correlation + per-step timing sweep.

## Risks

- **Train/eval mismatch.** If the proxy's threshold/normalization don't match
  the eval fit, `p̂` won't be the eval p. Mitigated by Phase 0 (calibrate before
  building).
- **Goodhart.** Optimizing ≈ the eval metric can yield fit artifacts (cf. the
  random-contrast overfitting). Mitigated by held-out p (fresh partners +
  anchors) and the independent feature tests — those, not the train p, are the
  verdict.
- **Soft-intercept failure modes.** A ray may never reach T (saturates below)
  or T may be set by a degenerate axis. Need clamps / a fallback T (e.g. from
  max response) and to skip/limit bad rays.
- **p-maximization instability.** Unbounded p; may need a cap or bounded
  transform, and to watch for degenerate maximizers.
- **It may still not tighten.** A few-radius soft intercept is still coarse; if
  Phase 0 can't get `p̂` to track eval-p, escalate to implicit-diff on the full
  grid (more faithful, ~5–10× cost) before concluding.

## What we'd learn

The decisive result is Phase 2's p-vs-feature-ness relationship. Everything
else (tighter proxy, higher p) is instrumental. If p and feature-ness move
together when we deliberately drive p up, the plateau/p program has real
purchase in LLMs; if they decouple, we've cleanly shown p is not the right
target here and should pivot (e.g. to the native single-direction 1D
responsiveness measure).
