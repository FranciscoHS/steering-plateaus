# Finding a feature direction by optimizing against contrastive directions

## Idea

Same structure as `Plans/feature_dir_vs_random.md` / `learn_feature_dir_vs_random.py`,
but the fixed scaffold of auxiliary directions is **10 contrastive directions
of diverse types** instead of random unit vectors.

- Fix `K = 10` unit contrastive directions `{d_c}`, held fixed.
- Optimize one direction `d_f` on the constant-norm sphere to minimize
  ```
  L(d_f) = mean_c mean_a mean_alpha   r_mix / (r_c + r_f)
         + contrast_orth_lam * mean_c (d_f . d_c)^2
  ```
  where, per activation `a`, contrastive direction `d_c`, and angle `alpha`:
  - `r_f`   = downstream L2 response to a spherical perturbation along `d_f`
  - `r_c`   = response along the contrastive direction `d_c`
  - `r_mix` = response along the equal-mix bisector of `d_f` and `d_c`.
- `L < 0.5` is the superellipse / p>2 (super-additive) regime; the hope is that
  pushing `d_f` to interfere maximally against a *contrastive* basis pulls it
  toward a model feature distinct from the scaffold.

## Configuration (decided)

- Model: `gemma-2-9B`.
- Perturbation layer: **15** (so both `--layer` and `--measure_layer`'s default
  policy still apply — measure_layer defaults to the 2nd-to-last layer).
- Contrastive directions: **extracted at L15** (same layer they're applied at).
- `n_feat = 1` (single direction, no gathering).
- `contrast_orth_lam = 1.0` (penalize overlap with the contrastive scaffold).
- `feat_orth_lam` argument is kept for symmetry but is a no-op when `n_feat = 1`.

## The 10 contrastive directions (one per family/axis)

Sourced from `fsec-anonymous/scripts/lib/registry.py`. Sign rule is `b - a`.

1. **Chinese** — language (en → zh), from `data.multilingual_prompts`.
2. **Python** — code (en → python), from `data.programming_prompts`.
3. **Gender** — male → female (`data.gender_prompts`).
4. **Age** — young → old (`data.age_prompts`).
5. **Sentiment** — positive → negative (`data.sentiment_prompts`).
6. **Tense** — past → future (`data.tense_prompts`).
7. **Number** — singular → plural (`data.number_prompts`).
8. **Formal** — casual → formal (built from `data.register_prompts`).
9. **Refusal** — comply → refuse (`data.refusal_prompts`).
10. **Honesty** — honest → dishonest (`data.honesty_prompts`).

## Extracting the scaffold at L15

`fsec-anonymous/scripts/lib/directions.py::extract_all` already does
diff-of-means → unit vector, with disk-caching at
`results/directions/dirs_gemma_L<layer>.pkl`. For gemma the cache currently has
`L2`, `L10`, `L20` (each contains all 10 wanted names) — **no `L15` yet**.

Plan: append `~/Research/fsec-anonymous` to `sys.path` and call
`scripts.lib.directions.extract_all(model, tokenizer, device, target='gemma',
layer=15, names=['Chinese','Python','Gender','Age','Sentiment','Tense','Number',
'Formal','Refusal','Honesty'])`. The function writes
`fsec-anonymous/results/directions/dirs_gemma_L15.pkl` and returns the dict;
restrict to the 10 chosen names and stack as `[10, 3584]`. If the pkl is
already present, the function will reuse it (and extend with missing names).

To avoid hard cross-repo coupling, the new script will accept a
`--scaffold_pkl` argument (defaults to the fsec-anonymous cache path) and a
`--scaffold_names` list (defaults to the 10 above).

## New script

`scripts/superellipse/learn_feature_dir_vs_contrastive.py`
Results → `results/feature_dir_vs_contrastive/<model>_L<layer>_a<angle>/`

### Flow
1. Load model; load `n_act` FineWeb sequences; extract layer-15 acts + ctxs.
2. Precompute `f_refs` (unperturbed downstream responses at `measure_layer`).
3. Load scaffold from `--scaffold_pkl` (build it via fsec-anonymous's
   `extract_all` if missing), restrict to `--scaffold_names`, normalize → `[K, d]`.
4. Random unit init for `d_f`; Adam + renormalize-each-step.
5. Loss = `mean_c pair_loss(d_f, d_c) + contrast_orth_lam * mean_c (d_f . d_c)^2`.
6. Log per step: L, ratio, mean overlap with scaffold, mean `r_f / r_c / r_mix`.
7. Diagnostics + plots + save.

### Outputs (files, never stdout-only)
- `results.pkl`: `d_f [d]`, `scaffold [K, d]`, `scaffold_names`, loss curve,
  per-step mean `r_f/r_c/r_mix`, final mean `r_f/r_c/r_mix` per scaffold dir,
  Gram of `d_f` vs scaffold.
- `summary.png`:
  - loss curve (total & ratio components),
  - per-scaffold-direction mean `r_f / r_c / r_mix` bars,
  - bar of `|d_f . d_c|` overlaps with each scaffold direction.

## Defaults
- `n_act = 10`, `alpha = [30]`, `n_steps = 200`, `lr = 3e-3`.
- `seed = 42` (data + init).
- `contrast_orth_lam = 1.0`, `feat_orth_lam = 1.0` (unused when `n_feat = 1`).
