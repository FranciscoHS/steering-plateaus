# Finding feature directions by optimizing against random directions

## Idea

Instead of jointly learning a pair (or growing a dictionary against previously
learned directions, as in `learn_superellipse_dict.py`), we use a *fixed*
scaffold of random unit directions as auxiliary objects, and optimize a single
candidate **feature direction** `d_f` against them.

- Draw `R = 10` random unit directions `{d_r}`, **held fixed** for the whole
  optimization (auxiliary scaffold, not optimized).
- Optimize one direction `d_f` on the constant-norm sphere to minimize
  ```
  L(d_f) = mean_r mean_a mean_α   r_mix / (r_r + r_f)
  ```
  where, per activation `a` and angle `α`:
  - `r_f`   = downstream L2 response to a spherical perturbation along `d_f`
  - `r_r`   = response along the random direction `d_r`
  - `r_mix` = response along the equal-mix bisector of `d_f` and `d_r`
- Average the ratio over all 10 random directions, all activations, all angles.

`L < 0.5` is the superellipse / p>2 (super-additive interference) regime we
want; `L = 0.5` is no-interference; `L > 0.5` is sub-additive. By optimizing
`d_f` to be maximally "interfering" against a *random* basis, the hope is that
`d_f` is pulled toward genuine feature directions of the model.

## Relationship to existing code (reuse, don't rebuild)

This is the **"grow" branch** of `optimize()` in `learn_superellipse_dict.py`
with `fixed_dirs = {random directions}` and `n_new = 1`. Reusable verbatim:
- `extract_context_and_activation` — final-token acts + prefix contexts.
- `tangent_dirs(a, d1, d2)` — orthogonalize both against `a`, return
  `[d1p, d2p, mix]`. Call as `tangent_dirs(a, d_f, d_r)` → `r_f, r_r, r_mix`.
- `pair_responses`, `pair_loss` — give exactly `R_mix/(R1+R2)`.
- `f_refs` precompute (reference response, no grad).

So the new script is mostly a thin driver around these, differing in:
1. the fixed directions are random unit vectors (seeded), and
2. we optimize one `d_f` at a time and **gather several** of them.

## "Gathering" multiple feature directions

`--n_feat F`: run the single-direction optimization `F` times, each an
independent run with a different init seed, against the **same** fixed set of
10 random directions. Collect the `F` resulting `d_f`'s.

To control whether gathered directions are distinct:
- `--feat_orth_lam` (default **1.0**): soft penalty `λ·(d_f · d_prev)^2`
  against **every** already-found feature direction, pushing each new one away
  from all the previous ones. On by default so gathered directions are
  distinct rather than collapsing onto the same minimum.
- `--rand_orth_lam` (default **0.0**): soft penalty against the random
  scaffold itself (random dirs are near-orthogonal in high-d anyway, so keep
  this at 0 unless we see `d_f` collapsing onto a random direction).

(Both penalties reuse the same mechanism as `lam` in the existing script.)

## New script

`scripts/superellipse/learn_feature_dir_vs_random.py`
Results → `results/feature_dir_vs_random/<model>_L<layer>_a<angle>/`

### Flow
1. Load model; load `n_act` FineWeb sequences; extract layer-`L` acts + ctxs.
2. Precompute `f_refs` (unperturbed downstream responses).
3. Draw `R=10` random unit directions in `R^d` (seeded, fixed).
4. For each feature run `j = 1..F`:
   - random unit init for `d_f` (seed = base_seed + j),
   - Adam + renormalize-each-step (Riemannian SGD on the sphere),
   - loss = `mean_r pair_loss(d_f, d_r)` + `feat_orth_lam·overlap(prev feats)`
     `+ rand_orth_lam·overlap(random dirs)`,
   - log per step: L, mean r_f / r_r / r_mix, mean |d_f·d_r|, mean |d_f·prev|,
   - checkpoint directions each run.
5. Diagnostics + plots + save.

### Generalization check (extra, cheap, worth it)
Hold out a **second** fixed set of `R_test=10` random directions never used in
training. After each `d_f` is found, evaluate `L(d_f)` against this held-out
set. If `d_f` is a real feature direction, its low loss should transfer to
fresh random directions, not just the training scaffold.

### Outputs (files, never stdout-only)
- `results.pkl`: feature dirs `[F, d]`, random scaffold `[R, d]`, held-out
  random `[R_test, d]`, per-run loss curves, per-run train & held-out final L,
  per-run mean r_f/r_r/r_mix, Gram among feature dirs, Gram feature×random.
- `summary.png`:
  - loss curves (all F runs),
  - bar/scatter of per-run final train L vs held-out L (vs the 0.5 line),
  - Gram heatmap among the F feature directions (do they cluster?),
  - histogram of mean r_f (single-direction responsiveness) per run.

## Defaults
- model `qwen3-1.7b`, layer 2, measure_layer = 2nd-to-last, FineWeb, seq_len 10.
- `n_act = 10`, `R = 10` random dirs, `R_test = 10` held-out, `α = [30]`.
- `n_feat = 5`, `n_steps = 200`, `lr = 3e-3`, `feat_orth_lam = 1.0`, `rand_orth_lam = 0`.
- All CLI-overridable; same conventions as `learn_superellipse_dict.py`.

## Open decisions to confirm before coding
1. **Gathering:** is `--n_feat` independent runs (init-seed only) the intended
   meaning of "gathering directions", or did you want a single `d_f` per run?
2. **Distinctness:** keep `feat_orth_lam = 0` (observe natural clustering) vs
   add a default push-apart penalty?
3. **Random scaffold reuse:** same 10 random dirs for every feature run
   (current plan) vs a fresh random set per run?
