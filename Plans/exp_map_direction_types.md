# exp_map robustness — direction-types column

This is the consolidated plan for the **direction type** column of the
fig4-style robustness beeswarm (`scripts/exp_map/plotting/plot_robustness_beeswarm.py`).
Self-contained so the next session can clear context.

## Where we are

- All-pairs E2 (`results/exp_map/data/sweeps_2d/`) is complete for the
  6-model default config (528 pairs/model, geodesic, L=2 perturb,
  penult measure, L^2 metric).
- Robustness sweeps for the *non-direction-type* columns are running
  on four new pods (`runpod-expmap-resume-{m7,additive,metric}` plus
  `runpod-expmap-retrieve` doing perturb L=5):
    - measure_offset = -7 (penult-5): ~431/528 done at session start,
      resuming the missing ~97
    - mode = additive: ~362/528, resuming ~166
    - metrics = l2,cos,kl: ~201/528, resuming ~327
    - perturb layer = 5: 422/528, resuming ~106 (~37 min)
  All use `--skip_existing`. Expected to finish within ~3 h wall.

- Other perturb-layer sweeps (L=10, L=20) are complete (528/528 each).
- Measure offset = -12 (penult-10) sweep is complete (528/528).
- A retrieval pod (`runpod-expmap-retrieve`) is up holding the
  network volume; an scp of all sweep pkls is in flight to local
  (background bash task `bdhe0886x`).
- Plotting script ready, currently shows whatever fits files exist.

## Direction-type column — 4 sub-categories (locked)

Comparison group sizes are matched: each family contributes ~33
directions → C(33, 2) = 528 SAE-SAE pairs (or DoM-DoM, MELBO-MELBO,
random-random) so the column has the same sample size as the existing
contrastive column.

1. **DoM (contrastive)** — already have. 33 directions, 528 pairs,
   gemma L=2 default. Fits already exist.
2. **SAE random latent** — sample 33 random latent indices from the
   default-width Gemma Scope L=2 SAE; decode each via `W_dec[k]`,
   L2-normalize. Pair them up (528 pairs). Tests "feature-basis
   aligned but not necessarily active at FineWeb anchors".
3. **MELBO** — for each of 30 FineWeb anchors run MELBO once at L=2
   (degree 10, maximize L^2 response at penult), yielding one optimized
   direction. Then take 33 such directions (run on 33 anchors? or 30 +
   3 reseeds — see decision below) and pair them up.
4. **Random isotropic** — sample 33 unit vectors from N(0, I/D),
   pair them up (528 pairs).

`p ≈ 2` should hold across all four if the iso-plateau is a structural
property of the residual stream. Any divergence is interesting.

## Implementation — three new scripts

All under `scripts/exp_map/`. They produce the same kind of
`sweep2d_<target>_L<layer>_<d1>__<d2>_fineweb_60deg<suffix>.pkl` files
that `fit_pairs.py` already consumes, so the plotting pipeline is
unchanged.

### 1. `scripts/exp_map/sae_directions.py`

Extracts and caches 33 SAE-decoder directions at L=2.

- Load Gemma Scope SAE: `google/gemma-scope-9b-pt-res` at the L=2
  release (use `sae_lens` per `CLAUDE.md`'s dependency note).
- Sample 33 latent indices uniformly from `[0, n_latents)` with a
  fixed seed (default 42). No reference to FineWeb or contrastive
  directions — fully blind.
- For each k, take `W_dec[k]` (shape D), L2-normalize. Save to
  `results/exp_map/data/directions/dirs_gemma_L2_sae.pkl`:
  ```
  {
    'directions': {f"sae_{k:06d}": tensor(D)},
    'family': 'sae_random',
    'sae_release': '...',
    'sae_id': '...',
    'latent_indices': [k_0, k_1, ..., k_32],
    'seed': 42,
    'layer': 2, 'model': 'gemma',
  }
  ```
- This script does *not* run any 2D sweep — it just builds the
  direction cache. The sweep step reuses `sweep_2d.py` with a flag
  pointing it at this cache (see "sweep_2d extension" below).

### 2. `scripts/exp_map/melbo_directions.py`

Runs MELBO at L=2 to extract 33 unsupervised steering directions.

- Use the standard MELBO setup (decision: cite Andrew Mack's setup).
  At L=2 with `degree=10`, optimize *maximize* L^2 of the activation
  at the penultimate layer (matching the metric used everywhere
  else). Keep optimizer hyperparams default until shown otherwise.
- Pick 33 distinct seed activations: 33 FineWeb anchors (sampled
  with seed=42 from the same FineWeb cache used for the default
  sweep). For each, run one MELBO optimization and take the
  resulting unit direction.
- Save to `results/exp_map/data/directions/dirs_gemma_L2_melbo.pkl`
  with the same schema as above (`family='melbo'`).
- Anchor-source decision (open): since each MELBO direction is tied
  to a single seed anchor, pairing them sweeps over directions that
  came from *different* anchors. The 2D sweep itself uses a fresh
  draw of 30 FineWeb anchors (same as the default sweep), unrelated
  to the seed activations. Acceptable, but document.
- Compute: MELBO is iterative, ~1-3 min per direction on a 4090
  depending on hyperparams. ~90 min for 33 directions.
- Implementation: thin wrapper over the existing repo MELBO
  utilities (search `scripts/melbo/` and the `melbo_pca_sweep.py` /
  `melbo_followup.sh` references on the pods for the reference
  setup).

### 3. `scripts/exp_map/random_directions.py`

Generates 33 isotropic random unit vectors at the gemma-9b D-dim.

- `torch.randn(33, D, generator=Generator().manual_seed(SEED))`
  followed by L2 normalization.
- D = gemma-2-9b residual dim (3584). Save to
  `results/exp_map/data/directions/dirs_gemma_L2_random.pkl` with
  same schema (`family='random'`).
- Trivial; ~5 s of compute.

## sweep_2d extension

`scripts/exp_map/sweep_2d.py` currently builds DoM directions via
`directions.extract_all`. We need a flag to load directions from a
pre-built cache instead.

- New CLI flag: `--directions_pkl PATH`. If supplied, the script:
    - Loads `pkl['directions']` (a dict {name: tensor})
    - Loads `pkl['family']` for the output suffix
    - Generates the pair list as `all_pairs_over(names)`
      (C(N, 2) over the direction names in the cache)
    - Otherwise behaves identically: same FineWeb anchors, same
      grid, same fit pipeline downstream.
- New filename suffix: `_dir{family}` so files don't clash with
  the canonical DoM run. e.g.
  `sweep2d_gemma_L2_sae_001234__sae_005678_fineweb_60deg_dirsae_random.pkl`.
- `fit_pairs.py` already accepts `--variant_suffix`, so the
  downstream is `fit_pairs.py --target gemma --variant_suffix _dirsae_random`.

The 33 directions × C(33, 2) = 528 pairs each gives ~3.5 h on a 4090
per family (matches the canonical run). Total ~10.5 h for SAE +
MELBO + Random, parallelisable across 3 pods.

## Plotting (no changes needed beyond config)

`plot_robustness_beeswarm.py` already has a `direction-type` column
sketch. Wire it up:

```python
def col_direction_type(target='gemma', layer=2):
    out = {}
    default = load_default_fits(target, layer)
    if default:
        out['DoM'] = [v for v in default.values() if isfinite(v)]
    for fam in ['sae_random', 'melbo', 'random']:
        fp = f'results/exp_map/data/fits/fits_{target}_L{layer}_dir{fam}.pkl'
        if exists(fp):
            full = load(fp)
            ps = [median_p_for_pair(v) for v in full.values()]
            out[fam] = [p for p in ps if isfinite(p)]
    return out
```

Add `direction-type` to the `--columns` allowlist.

## Order of operations (next session)

The resume pods (m7 / additive / metric / L=5) are independent of the
direction-type work. **Do not wait for them.** Start the
direction-type sweeps immediately on idle pods. As each resume pod's
sweep completes, run `fit_pairs.py` on it (CPU-only, fast) and the
beeswarm columns just light up.

Direction-type pipeline:

1. Write `sae_directions.py`, `melbo_directions.py`,
   `random_directions.py`.
2. Extend `sweep_2d.py` with `--directions_pkl`.
3. Cache the 3 new direction sets locally on the shared volume.
4. Launch 3 sweep jobs in parallel (one pod per family) with
   `--directions_pkl` pointing at each cache. Compute on each pod
   ~3.5 h.
5. After they finish: run `fit_pairs.py --variant_suffix _dir<family>`
   and re-render the beeswarm.

Once *all* sweeps finish (resume + direction-type): one final
beeswarm render covers every column.
