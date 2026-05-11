# exp_map robustness figure — plan

## Goal

Reproduce the figure 4-style beeswarm of fitted superellipse exponent
`p` across many axes of variation, but in the **exp-map chart** and
on the **exp_map workspace** data (locked at L=2 perturb, penult
measure, L^2 metric, geodesic perturbation, FineWeb anchors as the
default).

Each dot in each column = one fit on one *pair* (528 pairs from the
all-pairs Cartesian over the 33 directions in
`scripts/exp_map/lib/registry.py`).

Default ("baseline" config column) = **Gemma**, L=2 perturb, measure =
n_layers - 2, L^2 metric, geodesic exp-map perturbation, threshold =
50% of the (per-pair) `min(max(L2[:, 0]), max(L2[0, :]))`.

## Columns (one axis of variation per column)

For each column we vary exactly one axis from the default. Within a
column, the (sub-) groups share the same x-position (single beeswarm
cluster per column) and are distinguished by colour, with a small
legend per column listing the level names. *No horizontal separation
within a column* — that would bias visual reading toward one sub-group.
Different columns can have different N.

1. **Model** — gemma, qwen, llama, mistral, aya, yi.
   *(All 6 in flight; gemma + qwen done as of this writing.)*
2. **Perturbation layer** — gemma at L ∈ {2, 5, 10, 20} (4 sub-cols).
3. **Measurement layer** — gemma at L=2 perturb, measure layer
   ∈ {penult, penult-5, penult-10} (3 sub-cols).
4. **Metric** — gemma default, but with the contour fit done on
   {L^2, 1-cos sim, KL divergence}. Threshold per metric chosen
   independently using the same 50% rule.
5. **Threshold** — gemma default, threshold ∈ {25%, 50%, 75%}.
   *(Post-hoc on existing pkls — no new compute.)*
6. **Direction type** — defer until we discuss the direction-family
   sweep (SAE, MELBO, random vs DoM). Probably a follow-up.
7. **Perturbation method** — gemma default, additive
   (`a_new = a + α R d_perp`) instead of geodesic (`a_new = R(cos α â
   + sin α d_perp)`).

Reference dashed line at p = 2.

## What we already have

- Default config for **all 6 models** at the all-pairs Cartesian: in
  flight, qwen done. Goes into columns 1.
- All-pairs sweeps store `l2` only — **threshold sweep** can be run
  post-hoc on these without new compute (column 5).

## What needs new compute (Gemma only)

| Column                | sweeps                   | est. wall |
|-----------------------|--------------------------|-----------|
| 2. Perturb layer      | 3 × 528 pairs at L=5,10,20 | ~10 h sequential / ~3.5 h on 3 pods |
| 3. Measure layer      | 2 × 528 pairs (penult-5, penult-10) | ~7 h sequential / ~3.5 h on 2 pods |
| 4. Metric (cos, KL)   | 1 × 528 pairs (rerun L=2 default but record cos + KL) | ~4 h |
| 7. Perturb method     | 1 × 528 pairs (additive) | ~3.5 h |

Total: ~25 pod-hours of new compute, parallelisable across 4–7 pods.

## Code changes needed

### `scripts/exp_map/lib/sweep_core.py`
- Optional `measure_layer` arg already plumbed through `run_2d_per_anchors`.
- Add: `record_cosine` and `record_kl` flags. If true, also compute
  `1 - cos(ref_act, perturbed_act)` and KL between final-layer logits
  (analogous to `fig4_robustness_v2.sweep_2d_spherical(..., want_kl=True)`).
- Add: `mode='additive'` option. If additive, build perturbations as
  `a_new = a + α (R/normalisation) d_perp` instead of the exp-map step.
  Use the same orthonormalised `d1_perp, d2_perp` so the chart axes
  remain interpretable. Cap `α` at the same `max_angle` (in the
  additive case the angle units become "scaled-displacement units").

### `scripts/exp_map/sweep_2d.py`
- New flags: `--measure_layer N`, `--metrics l2,cos,kl`,
  `--mode {geodesic,additive}`.
- Output filename suffix encodes the variation: `sweep2d_gemma_L2_<d1>__<d2>_fineweb_60deg_M{measure}_{mode}_{metrics}.pkl`.

### `scripts/exp_map/fit_pairs.py`
- Accept `--metric {l2,cos,kl}` and pick the right grid in the pkl.
- Accept `--threshold_frac F` (default 0.5).
- Output suffix in fits pkl matches.

### `scripts/exp_map/plotting/plot_robustness_beeswarm.py`
- New script. Reads fits pkls + raw sweep pkls (for post-hoc threshold
  refits), produces the beeswarm.

## Order of operations

1. Wait for current 6-model all-pairs E2 to finish (column 1 + raw
   data for column 5).
2. Add cos+KL recording and additive mode to sweep_core / sweep_2d.
3. Launch the four Gemma-only sweeps in parallel on 4 pods (perturb
   layer × 3, measure layer × 2, metric, additive). One target per
   pod where possible.
4. Run `fit_pairs.py` on each output set.
5. Run plotting script.

## Direction-type column (deferred)

The fig4_v3 plot had {DoM, SAE, MELBO, Random×Random}. For exp_map we
already include {semantic-DoM, language-DoM, code-DoM} via
`registry.all_direction_names()`. Adding SAE / MELBO / random
families will require:

- SAE: load Gemma Scope at L=2, pick top-N latents, decode to D-dim
  steering vectors, register as a new family.
- MELBO: requires running the MELBO optimizer at L=2 to get a set of
  unsupervised steering vectors.
- Random: `n` unit vectors sampled from N(0, I/D).

Discuss before launching.
