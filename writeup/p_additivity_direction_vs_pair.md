# Is superellipse p a property of the direction or of the pair?

Date: 2026-08-02. Model: qwen3-1.7b, perturb L14, measure L26 (SAE rows: L10).

## Question

Pairwise p fits give one p per pair (d_i, d_j). Do directions "carry" their own
p (each pair's p set by its two members), or is p pair-specific? Motivated by
the k-way superellipse discussion: the joint single-p model predicts all pairs
in a tuple share one p, so we first ask how much of p is per-direction at all.

## Method

`scripts/superellipse/p_additivity_analysis.py` on all-pairs fit caches
(20 directions -> 190 pairs, each direction in 19 pairs; fit protocol =
`fit_pairs_dictionary.py`, 20 held-out anchors, seed 2024, 31-step grid to 60°).

- Additive decomposition: log p_ij = mu + a_i + a_j, least squares.
- R^2 = share of pair-to-pair variance in log p carried by per-direction terms.
- Permutation null (2000 shuffles of p across pairs, same design): with ~n_dirs
  free params on ~n_dirs^2/2 points, chance R^2 is ~0.10 — the **excess over
  null** is the signal.
- Also: correlation of p between pairs sharing one direction (dyads), same null.

## Results (excess = additive R^2 minus null mean)

Optimized-direction populations (all ratio objective unless noted):
- **snap20k** (random partners, per-step act resample — current best recipe):
  p median 4.40, R^2 0.24, excess **0.14**, perm p 0.0005, shared-dir r 0.06
- **scaffold_lam100_20k** (contrastive scaffold partners): p median 5.16,
  R^2 0.16, excess **0.06**, perm p 0.031, shared-dir r 0.03
- **scaffold_grow** (partners = scaffold + all found dirs): p median 4.14,
  R^2 0.37, excess **0.27**, perm p <0.0005, shared-dir r 0.13
- **ratio K20 slice** (older randcontrast build, fixed anchor): p median 2.21,
  R^2 0.31, excess **0.21**, perm p <0.0005
- **soft-p K20**: p median 2.20, R^2 0.46, excess **0.36**, perm p <0.0005
- **scaffold K10** (45 pairs only, underpowered): excess -0.03, perm p 0.58

Reference populations:
- **contrastive 17** (136 pairs): p median 2.03, R^2 0.31, excess **0.19**,
  perm p <0.0005
- **SAE baseline L10** (80 latents x 8 latent partners, eval-frame estimator):
  shared-dir r **0.153**, perm p <0.0005
- **SAE p-opt lam10 L10**: shared-dir r **0.158**, perm p <0.0005
  (additive R^2 saturates at 1.0 for the SAE design — one param per unique
  partner; only the dyad correlation is valid there. r values are not directly
  comparable across designs.)

Takeaway shape: everywhere tested (except the underpowered K10 set), p has a
significant per-direction component, but the majority of the variance in
log p is pair-specific — "a bit of both", with pair-level dominating.

## Data locations

- `results/dictionary/p_additivity/` — first four populations (summary.txt/pkl, plot)
- `results/dictionary/p_additivity_new3/` — snap20k / scaffold_lam100 / scaffold_grow
- `results/dictionary/p_additivity_sae/` — SAE rows
- New all-pairs caches: `results/dictionary/fit_pairs_{snap20k,scaffold_lam100,scaffoldgrow}_first20_allpairs/`
- Extra caches pulled from the network volume (random-pair designs, for
  completeness): `fit_pairs_band_*`, `fit_pairs_snap4400`, `fit_pairs_combo2`
  under `qwen3-1.7b_L14_K8000_randcontrast_stepresample/`, and
  `fit_pairs_1100_vs_{dict,random}` under `..._ratio30_actresample/`.

## Side findings from the same session

- Pairing-variation check (the "pathological cancellation" hypothesis): the
  main builds DO vary partners during optimization — snap20k used 4 fresh
  random partners per gradient step; scaffold builds resample 4 of 17 scaffold
  dirs per step; only the old joint-pair scripts
  (`learn_superellipse_dict*.py`) use fixed pairings. Single-accomplice
  cancellation is excluded for the main dictionaries.
- scaffold_grow build confirmed: ratio objective, per-step act resampling,
  lam_prev=100, 7700/40000 directions done.

## Possible next steps (not launched)

- Second disjoint 20-direction slices (dirs 20-40) per build: doubles pairs and
  tests generality beyond first atoms.
- SAE latents refit in the L14 `fit_pairs_dictionary` protocol for an
  apples-to-apples row.
- k-way diagonal test (3- and 4-way equal mixes vs the
  t_pred = sqrt(k) * (sum_i r_i^-p)^(-1/p) prediction) — the original goal this
  analysis was scoping.
