# Plan: single-30° ratio + activation resampling — scale to 8000 directions

## Objective (the recipe that works)
Build dictionary directions by minimizing the **single-angle (30°) ratio**
`R_mix / (R_f + R_c)` — i.e. `build_dictionary.py --objective ratio --angles 30`
— with:
- **per-step activation resampling**: `--resample_act step --n_act_per_step 5`
  (fresh batch of 5 activations from the pool every gradient step; this is the
  ingredient that moved held-out p up, not multi-magnitude),
- **per-step contrast resampling**: `--contrast_source random --n_contrast 4
  --contrast_resample step` (4 fresh random unit contrasts each step),
- `--lr 0.003 --n_steps 150`, Qwen3-1.7B, layer 14, measure_layer 26,
  `--lam_prev 1.0` diversity, bf16.

### Why this objective (evidence)
- Single-30° ratio calibrates to eval-p at **Spearman −0.84** (n=60 random
  pairs) — better than the 6-angle multi-magnitude (−0.76). High angles (50/60°)
  barely track and dilute the average.
- 4-direction build + eval (held-out anchors, seed 2024): **p ≈ 3.06 vs random
  partners** [CI 2.92, 3.25], **p ≈ 3.9 vs each other** — vs the old fixed-anchor
  ratio's ~2.2. The activation resampling is what raised p.
- single-30° is ~half the cost of multi-magnitude and avoids the vs-self
  over-suppression collapse (multi-mag fell to p≈1 vs self).
- Caveat: all eval so far is 4 dirs / 6–8 pairs — loose magnitudes.

## Step 0 — speed optimization (do first)
Implement **R_f sharing** in `batched_ratio_multiangle` / the ratio path: the
"d-alone" response (R_f) does not depend on the contrast, so compute it once per
(activation, angle) instead of once per contrast. Cuts points from
`n_act·n_ang·n_contrast·3` to `n_act·n_ang·(1 + 2·n_contrast)` (~25% fewer at
n_contrast=4). Verify it reproduces the same loss trajectory before building.

## Step 1 — build 100, check timing
Build 100 directions with the recipe above. Confirm:
- per-direction time (single-30° M=5 measured 9.8 s/atom before R_f sharing;
  expect lower after). Extrapolate to 8000.
- p still high vs random on a quick `fit_pairs_dictionary` eval (leg2=random)
  + vs self; trajectories sane.

Decision gate: if timing extrapolates to a tolerable 8000 run (a few–several
hours) and p holds, proceed.

## Step 2 — build 8000
Same recipe, `--K_dict 8000`, checkpoint every 100, resumable. Output dir
`results/dictionary/qwen3-1.7b_L14_K8000_ratio30_actresample`.

## Step 3 — evaluate (on the 8000)
- **Pairwise eval-p** at scale: `fit_pairs_dictionary.py` vs random partners and
  vs dict (tight statistics, build-order bands).
- **Reconstruction / OMP** (`eval_dictionary_omp.py`) — needs the full ~thousands.
- **Contrastive + SAE-latent alignment** at scale (`dict_vs_contrastive_align.py`,
  `dict_sae_alignment.py`) — self-consistency only (both are weak feature proxies).
- **1D single-direction + pairwise sweeps** (`eval_plateaus_1d_dict.py --pairwise`)
  on a sample — are they high-gain saturating directions, do mixes suppress.
- **Corpus-activation alignment** (`dict_max_activating.py`), **token / logit-lens
  overlap** (`logit_lens_decomp.py`, `topk_lasttoken_overlap.py`).
- (LRH synthesis test — build an activation from ~30 dirs, check for plateaus —
  optional; a negative is overdetermined, so low priority.)

## Open question to keep in mind
Good calibration ≠ good optimization outcome (cf. soft-p: calibrated 0.86,
optimized poorly). The build+eval at each scale is the real verdict, not the
calibration number.

## Knobs if needed
- `--n_act_per_step`: M=5 best convergence; M=2 ≈2.5× faster (quality tradeoff).
- Could revisit angle placement (30° was best single; a tight cluster near 30°
  is the fallback if single-angle is too noisy at scale).
