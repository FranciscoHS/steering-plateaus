# Plan: p-fit sanity check on the denoising toy model — feature vs random

## Motivation

In the transformer setting (Qwen3-1.7B L14 and Gemma-2-9b L15) the canonical
exact-geodesic superellipse fit gives p ≈ 2 in every pair configuration we've
tried:

- dict vs dict ≈ 2.10
- dict vs contrastive ≈ 2.43
- contrastive vs contrastive ≈ 2.03 (Qwen) / TBD (Gemma)
- contrastive vs random ≈ 1.96
- dict-first-50 vs dict-first-50 ≈ 2.04

This could mean either (a) the p metric doesn't actually distinguish
"feature-like" pairs from arbitrary pairs in these models, or (b) what we're
recovering aren't features in the relevant sense. Hard to tell without a
ground-truth-features setting.

**Sanity check:** measure p in the denoising toy model — where the features
are exactly known (canonical basis of the input) and the model has been
trained to recover them. If p (feature × random) ≈ 2 there too, we know the
metric isn't picking up the structure we hoped, and we should reconsider the
proxy / fit pipeline before reading too much into the transformer results.

## Toy model

Use the existing trained tanh³ checkpoint at
`~/Research/denoising/runs/phase1_tanh3/model.pt`:

- `TwoLayerNet(input=1000, hidden=256, activation='tanh3')`
- d=1000, sparsity=2, noise_variance=0.03
- Trained with Adam, 10 epochs, λ_on=10
- Encoder-decoder cosine alignment cos_mean ≈ 0.91 (model cleanly recovered
  the canonical features)
- Features = canonical basis vectors e_0, …, e_999 of input space

(Alternative larger checkpoint: `runs/phase1_tanh3_10x_p4` — d=2560, H=256,
sparsity=5, 30 epochs, cos_mean ≈ 0.98. Better-trained but a different
regime. Default to `phase1_tanh3` for the closest comparison to the original
plan.)

## Experiment

For N pairs (default N=100):

1. Pick a random feature index `f ∈ {0, …, 999}`.
2. Sample a clean two-hot signal `x_clean` with active set S of size 2, with
   `f ∉ S` (the perturbation direction is then off-feature, matching the
   probe.py convention).
3. Sample a random unit direction `v ∈ R^1000` (Gaussian, normalized).
4. **2D L2 grid** on (α, β) ∈ [0, 60°-equivalent]² (Euclidean parametrization,
   not geodesic):
   - At each grid point: `Δ = model(x_clean + α·e_f + β·v) − model(x_clean)`,
     record `L2 = ||Δ||`.
   - 31×31 grid by default.
5. Fit p via the existing canonical machinery from steering-plateaus:
   `scripts.superellipse.canonical.superellipse.robust_p_fit` on the 2D L2
   grid, exact-geodesic=False (toy model uses Euclidean perturbations, no
   spherical embedding).
6. Record `p_fit`, `p_lo`, `p_hi`, and per-pair normalized boundary points
   for pooled bootstrap.

Aggregate: per-pair p distribution + pooled p with bootstrap CI across pairs.

## (α, β) parametrization

Euclidean — pick the same α range we use in the transformer pair fits (0 to
~`||x_clean||/2` or whatever lets the iso-contour close at a sensible
threshold). Easiest is to scan, find the α at which the response saturates
at some chosen threshold along the axes, and grid up to ~2× that. Save the
final grid extents to the results pkl.

We are *not* using geodesic exp-map here — toy model perturbations are in
flat input space, no anchor-norm sphere.

## What's reusable

From `~/Research/denoising`:
- `mft_denoising.nn.TwoLayerNet` — the model class.
- `mft_denoising.probe.make_clean` — builds the two-hot anchor `x_clean`.
- `mft_denoising.probe.sweep_2d` — close template; we'll replace it with a
  version that takes arbitrary direction vectors and records L2 of the full
  output diff (rather than two specific output coords).

From `~/Research/steering-plateaus/scripts/superellipse/canonical`:
- `superellipse.robust_p_fit` — fits p with bootstrap. Plug in directly.
- `superellipse.extract_contour`, `axis_intercept`, `fit_superellipse` —
  needed by the pooled-points CI step (mirrors
  `pool_p_contrastive_mlp.normalized_points`).

From `~/Research/steering-plateaus/scripts/superellipse`:
- `fit_pairs_dictionary.py` — closest existing driver structure to copy:
  per-pair p fit + pair-level summary + pooled CI + plots.

## New code

One new script:
`scripts/superellipse/fit_pairs_toymodel.py` (lives in steering-plateaus
since the fit machinery is here).

Contents:
1. Load the toy denoiser checkpoint (`mft_denoising.nn.TwoLayerNet`).
2. `sweep_2d_l2(model, x_clean, d1, d2, alpha_max, n_grid)` →
   `(alphas[n_grid], l2_grid[n_grid, n_grid])`.
3. Driver loop:
   - For each of N pairs:
     - Sample `x_clean` and `f` so `f ∉ S(x_clean)`.
     - Sample random unit `v`.
     - Run the 2D sweep along `(e_f, v)`.
     - Call `robust_p_fit(angles_deg=alphas, l2_grid_per_anchor=l2[None],
       exact_geodesic=False)`.
     - Record per-pair p + boundary points.
4. Aggregate + plot in the same style as `fit_pairs_dictionary.py`
   (histogram + p-vs-cos scatter + summary text).

Output: `results/dictionary/toymodel_phase2_jumprelu/fit_pairs_feature_vs_random/`
with `fit_pairs.pkl` and `p_distribution.png`.

## Settings (defaults)

- `--n_pairs 100`
- `--n_anchors 1` (one x_clean per pair — pairs vary the random direction
  and feature index, not the activation; we average over pairs not anchors)
- `--n_grid 31`
- `--alpha_max` — auto-tuned via a coarse 1D sweep along e_f to find the
  saturation scale; allow `--alpha_max_override` CLI flag.
- `--n_bootstrap 100` per pair
- `--n_pool_bootstrap 2000`
- `--seed 42`

## Compute

Trivial: forward passes on a 256-wide JumpReLU net are microseconds.
Probably <1 s per pair on CPU; total <2 min for 100 pairs. **Run locally**
(this is one of the exceptions: it's the denoising toy model, but it's a
tiny inference-only run, not training. Francisco's "don't run denoising
locally" note refers to the training scripts. Confirm with him if unclear,
otherwise just run it locally and report.)

(Edit: per repo memory `feedback_no_local_denoising_runs.md`, denoising
runs should go on a pod. To be safe, run on whatever idle pod is available;
the dict-eval / dict-plateau pods all have Python + torch and can hit
`~/Research/denoising` over the network volume.)

## What we'd learn

- **If feature-vs-random p > 2 cleanly**: the metric works in a setting with
  known features. Our transformer p≈2 results are then real signal (the
  directions we're recovering, including the contrastive scaffold itself,
  are not super-additively privileged pairwise in those transformers).
- **If feature-vs-random p ≈ 2**: the metric is broken (or never gives p>2
  for this kind of geometry). We need to revisit the fit pipeline before
  reading much into the transformer results.

## Risks / caveats

- **Choice of `alpha_max`**: too small and the iso-contour doesn't close;
  too large and we sweep way past saturation, polluting the contour with
  asymptotic behavior. Auto-tune via the 1D scan along e_f.
- **tanh³ smoothness**: smooth everywhere; no kink issue. Iso-contours should
  be well-behaved.
- **One feature in `S` near `f`**: low probability with d=1000 and S=2, but
  enforce `f ∉ S` strictly in the sampler.
