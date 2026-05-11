# Plan: better baselines for the direction-family beeswarm

## Motivation
The current direction-family panel compares Contrastive vs SAE-random vs
MELBO vs Random unit vectors. SAE-random is a weak baseline — it samples
33 SAE-decoder rows uniformly, with no reference to whether those
features ever activate. Stronger baselines:

1. **SAE-eval (top-33 by activation on the 30 eval anchors)** — the SAE
   features that actually fire on our evaluation set. Tests whether
   "real, evaluation-relevant SAE features" behave like Contrastive
   directions.
2. **SAE-FineWeb (top-33 by activation on ≥10k FineWeb prompts)** —
   the SAE features that fire most across a broad corpus. Tests whether
   broadly-active SAE features behave like Contrastive directions
   (without leaking the eval set).
3. **PCA-FineWeb (top-33 PCs of activation variance over the same
   FineWeb sample)** — directions of maximum activation variance.
   Doesn't depend on the SAE basis at all; the SAE's "feature axis"
   may itself be wrong, in which case PCA gives the right reference.

## Decisions already made
- **PCA on the eval set is degenerate** (30 samples, top-33 trivially
  span the whole sample subspace) — only the FineWeb-broad PCA variant
  is run.
- **33 directions per variant**, matching Contrastive's count. C(33, 2)
  = 528 pairs per variant, just like the existing SAE/MELBO/Random
  baselines.

## Final direction-family panel
Six columns, in order:

1. Contrastive (existing)
2. MELBO (existing)
3. SAE-eval (new)
4. SAE-FineWeb (new)
5. PCA-FineWeb (new)
6. Random (existing)

Optionally drop the existing SAE-random column since SAE-eval and
SAE-FineWeb subsume it. Decide after seeing the new beeswarm.

## Implementation

### New direction caches
Each saves to `results/exp_map/data/directions/dirs_gemma_L2_<tag>.pkl`,
matching the existing schema (33 unit vectors keyed by direction name).

| tag             | source                                                                             |
|-----------------|------------------------------------------------------------------------------------|
| `sae_eval`      | top-33 SAE features by mean last-token activation over the 30 anchor prompts       |
| `sae_fineweb`   | top-33 SAE features by mean last-token activation over 10k FineWeb prompts         |
| `pca_fineweb`   | top-33 right singular vectors of (centred) last-token activations from same 10k    |

#### `sae_directions_top.py` (new helper)
Modify the existing SAE loader to accept a `--rank_set` flag:

- `--rank_set anchors`: rank features by mean `|activation|` across the
  30 evaluation anchor prompts. Save as `dirs_gemma_L2_sae_eval.pkl`.
- `--rank_set fineweb`: rank features by mean `|activation|` across
  the cached 10k-prompt FineWeb sample (cf. `actlib.fineweb_acts` —
  needs an extension to return more than 30, or a fresh helper). Save
  as `dirs_gemma_L2_sae_fineweb.pkl`.

The activation pass is a single forward through the SAE encoder per
prompt; cheap once activations are cached.

#### `pca_directions.py` (new helper)
- Pull last-token activations for the 10k FineWeb prompts at L=2 (cache
  them in `results/exp_map/data/activations/acts_gemma_L2_fineweb_10k.pkl`).
- Centre, run `np.linalg.svd` on the (10k, d_model) matrix, take
  top-33 right singular vectors as the directions.
- Save as `dirs_gemma_L2_pca_fineweb.pkl`.

### Sweeps
Three new sweep sets, each over C(33, 2) = 528 pairs. Use the existing
`sweep_2d.py --directions_pkl <…>` pathway:

```
python scripts/exp_map/sweep_2d.py \
  --target gemma --layer 2 \
  --directions_pkl results/exp_map/data/directions/dirs_gemma_L2_sae_eval.pkl \
  --skip_existing
```

The `--directions_pkl` mode appends `_dir<family>` to the sweep filename
(e.g. `_dirsae_eval`), reads the family from the pkl's `family` field
(set during direction extraction). Filenames stay distinct from existing
`_dirsae_random`.

Per variant: 528 pairs × ~24 s/pair ≈ 3.5 GPU-hr.

### Fits
After sweeps land, refit thrfixed per variant. Add three rows to
`scripts/exp_map/refit_thrfixed_all.py`:

```
("gemma", 2, "_dirsae_eval",     "fineweb", "l2"),
("gemma", 2, "_dirsae_fineweb",  "fineweb", "l2"),
("gemma", 2, "_dirpca_fineweb",  "fineweb", "l2"),
```

Each row computes `T = 0.40 × median(min(axis_max))` over the variant's
own sweeps and runs `fit_pairs.py` with `--thresh_l2 T` and
`--variant_suffix _dir<family>`.

### Plot update
`plot_beeswarm_direction_types.py`: extend `FAMILIES` to include the new
suffixes, and **enable family-internal overlap filtering for every
non-Contrastive family** (currently only Contrastive is overlap-filtered;
the user wants every baseline filtered at |⟨d_i, d_j⟩| < 0.10 within
its own direction set):

```python
FAMILIES = [
    ("Contrastive",   "",                   True),
    ("MELBO",         "_dirmelbo",          True),
    ("SAE-eval",      "_dirsae_eval",       True),
    ("SAE-FineWeb",   "_dirsae_fineweb",    True),
    ("PCA-FineWeb",   "_dirpca_fineweb",    True),
    ("Random",        "_dirrandom",         True),
]
```

Filename gets a `_morebaselines` tag to avoid clobbering the existing
4-column figure.

#### Required code change to enable filtering for non-Contrastive families
The current `_load_dom_overlaps(target, layer)` only loads
`dirs_<target>_L<layer>.pkl` (the Contrastive cache). For per-family
filtering we need an equivalent that loads from
`dirs_<target>_L<layer>_<family_suffix>.pkl`. Add a helper:

```python
def _load_overlaps_from_pkl(path: str) -> dict[frozenset, float]:
    """|<d_i,d_j>| over the directions in a single pkl. Cached per
    path so repeated calls (per-column filter) don't re-compute."""
```

…and route each non-Contrastive column through it using the matching
direction pkl path. SAE-eval features may have high pairwise overlap
since they fire on similar contexts — the filter will drop those
correctly rather than carrying Gram-Schmidt artifacts forward.

Expected effect:
- SAE-eval / SAE-FineWeb: noticeable drop in n_kept (top-activating
  features are highly correlated). Could be 30–60% retention.
- PCA-FineWeb: ~100% retention (PCs are orthogonal by construction;
  pairwise overlap is exactly 0 up to numerical noise).
- MELBO: probably high retention (MELBO directions are designed to be
  diverse), but not guaranteed.
- Random: ~100% retention (high-dim random vectors are nearly
  orthogonal w.h.p.).

## Pod plan

GPU work is bounded by the three sweep sets (3 × ~3.5 GPU-hr).
Extraction (SAE encoder pass + PCA SVD) is cheap (<10 min total once
activations cached). The 10k FineWeb activation cache itself takes
~20 min on Gemma at L=2.

| pods | wall  |
|------|-------|
| 3    | ~4 h  | (one variant per pod)
| 1    | ~11 h | (sequential)

Recommended: 3 pods, each running one of `{sae_eval, sae_fineweb,
pca_fineweb}` — but they all need the 10k FineWeb activation cache
first, which should be built on a single pod and synced to the network
volume before launching the parallel sweeps. Total wall ≈ 4.5 h.

Sequence per pod:
1. (Once, on one pod) Extract & cache 10k FineWeb activations at gemma
   L=2 → shared volume.
2. (Once, on one pod) Build the three direction pkls. Each is fast
   (forward + SVD); doing all three on the same pod takes ~10 min.
3. (Three pods in parallel) Run sweep_2d.py on each direction pkl.
4. (Local) Refit thrfixed for each variant; render the new beeswarm.

## Sanity checks before kickoff
- Confirm the SAE used is at the same layer (L=2) as our perturbation
  layer; otherwise the "top SAE feature on activation X" isn't tied to
  the same activation distribution we're perturbing.
- Confirm the 10k FineWeb sample is large enough for stable PCA — top
  33 components should explain a sensible fraction of variance (will
  print). If too few samples, scale up.
- Sanity-check overlaps: SAE-eval top features may have very high
  pairwise overlap (they all activate on similar prompts), which would
  show up as a Gram-Schmidt artifact in the sweeps. We'll see this
  immediately in the per-pair overlap distribution and can decide
  whether to filter.

## Out of scope
- Ablations across SAE width / canonical L0.
- PCA on prompt-domain bilingual / code activations (different anchor
  source columns).
- Cross-model PCA / SAE baselines (gemma only — same scope as the
  existing direction-family figure).
