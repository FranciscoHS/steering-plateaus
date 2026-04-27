# Exp-map workspace plan (v2 after critique)

## Why
Existing iso-plateau / pair-sweep code uses the per-axis chart `(sin α₁, sin α₂)`, which (a) caps perturbations at sin²α₁+sin²α₂≤1 (≤45° per axis simultaneously, ≤30° in practice) and (b) makes the fitted superellipse exponent `p` chart-dependent: the same physical L² function gives different `p` in different charts. We want a chart in which (i) chart distance from anchor = great-circle angle to the perturbed point, (ii) chart direction = sphere direction, (iii) no per-axis angle ceiling, (iv) the activation magnitude is preserved exactly. The exponential-map / tangent chart at â is the unique chart with all four properties.

Going forward, all new perturbation experiments use the exp-map chart. We re-run every experiment whose conclusions depend on superellipse-`p` or on the geometry of perturbation composition. Existing A-chart data is kept (read-only, in `results/fig4_robustness/`); nothing in the new workspace depends on it.

## Parametrization spec (canonical, locked)

For an anchor activation `a ∈ ℝᴰ` at perturbation layer `L`, with `R = ‖a‖` and `â = a/R`, and two contrastive directions `d₁, d₂ ∈ ℝᴰ`:

1. Project to the tangent plane at â and orthonormalize:
   ```
   d₁_⊥ = d₁ − (d₁·â)·â
   assert ‖d₁_⊥‖ > eps · ‖d₁‖              # eps = 1e-6
   d₁_⊥ /= ‖d₁_⊥‖

   d₂_⊥ = d₂ − (d₂·â)·â − (d₂·d₁_⊥)·d₁_⊥
   assert ‖d₂_⊥‖ > eps · ‖d₂‖
   d₂_⊥ /= ‖d₂_⊥‖
   ```
2. Parametrize a perturbation by tangent coordinates `(α₁, α₂) ∈ ℝ²` (radians):
   ```
   v = α₁·d₁_⊥ + α₂·d₂_⊥
   r = ‖v‖
   if r < eps_r:                           # eps_r = 1e-12
       new_a = R · â
   else:
       new_a = R · (cos r · â + sin r · v / r)
   ```
3. Optional safety clamp: `r ≤ π/2 − ε` to stay on the upper hemisphere (no equator-crossing).

Properties guaranteed by construction (in the `r > 0` branch):
- `‖new_a‖ = R` exactly.
- `arccos(â·new_a/R) = r` exactly (chart distance = great-circle angle).
- Direction in chart `(α₁, α₂)/r` = tangent direction `v/r` on the sphere. *Holds because d₁_⊥, d₂_⊥ are orthonormal by construction.*
- 1D rotation toward d₁ alone: set α₂ = 0 → `R·(cos α₁·â + sin α₁·d₁_⊥)`.
- 1D rotation toward an arbitrary unit direction `d̂` (e.g. composite): treat `d̂` as d₁ (use single-direction primitive).

Sign convention:
- DoM directions have no canonical sign. We fix the sign at extraction time: for an axis labeled `(side_a → side_b)` (e.g. `male → female`), the direction points `b − a` (i.e. `+α` rotates toward `b`). Document the sign rule in every PKL header. The chart is invariant under simultaneous `α_i ↦ −α_i` and `d_i ↦ −d_i`, so `p`-fits are sign-agnostic.

Composite direction for compositional steering:
- Use the *orthogonalized* sum: `d̂_comp = (d₁_⊥ + d₂_⊥) / ‖d₁_⊥ + d₂_⊥‖`. (Decision D6 below.)
- Record both raw-sum and orthogonalized-sum vectors in the PKL header so we can revisit if needed.

## Directory layout

```
results/exp_map/
├── README.md                       # spec + plan link
├── data/
│   ├── directions/                 # cached DoM directions per (model, layer, dir-name)
│   ├── activations/                # cached anchor activations per (model, layer, source-name)
│   ├── sweeps_1d/                  # 1D rotations along a single direction
│   └── sweeps_2d/                  # 2D iso-plateau pair sweeps in (α₁, α₂)
└── figures/

scripts/exp_map/
├── lib/
│   ├── parametrize.py              # exp-map perturbation primitives
│   ├── directions.py               # extract / cache DoM directions
│   ├── activations.py              # extract / cache anchor activations
│   └── superellipse.py             # iso-L² contour extraction + |x|^p+|y|^p=c fit
├── sweep_1d.py                     # CLI
├── sweep_2d.py                     # CLI
├── tests_parametrize.py            # unit tests (run before any sweeps)
├── validation.py                   # E4 validation suite
└── plotting/
    ├── plot_1d.py
    └── plot_2d_boundary.py
```

`fit_pairs.py`, `plot_p_summary.py`, `steer_generate.py`, `fits/` dir, `steering_gen/` dir: deferred until E1 and E2 land. Build the minimum first.

## File-naming convention

```
sweeps_1d/   sweep1d_<model>_L<layer>_<direction>_<source>_<max_angle>deg.pkl
sweeps_2d/   sweep2d_<model>_L<layer>_<d1>__<d2>_<source>_<max_angle>deg.pkl
```

`<source>` examples: `fineweb`, `male_holdout`, `positive_holdout`, `english_holdout`.

PKL header (every output file) includes:
- `chart`: `'exp_map_v1'`
- `parametrize_git_sha`: SHA of `lib/parametrize.py` at run time
- `prompt_set_hashes`: SHA of every prompt source file used
- `seed_anchors`, `seed_random_dirs`: integers
- `direction_signs`: `{name: 'side_a→side_b'}` mapping
- `eps_thresholds`: dict with eps, eps_r values used

## Decisions (locked, with rationale)

- **D1. L = 2 for all models.** No relitigation.
- **D2. 30 FineWeb / holdout anchors per pair/direction.** Don't change two things at once.
- **D3. 31×31 grid for 2D sweeps** (`α_i ∈ {0, 2°, 4°, ..., 60°}`). 41×41 is 1.7× compute for marginally smoother contours we then smooth in the fit. If `p` fits look noisy at 31, then bump.
- **D4. FineWeb-only anchors for 2D sweeps in this round.** Matched-source 2D as a follow-up only if FineWeb 2D shows a surprising asymmetry between d₁ and d₂.
- **D5. Re-critique the 12 holdout sets before launching E1.** Cost is small, risk of skipping is large (poisoned matched-source 1D plots).
- **D6. Composite direction = orthogonalized sum** `(d₁_⊥ + d₂_⊥)/‖·‖`. Record raw sum too.

## Robustness protocol for the `p ≈ 2` claim (pre-registered)

For every 2D pair sweep we report `p` as a *range*, not a single number, computed under all combinations of:
- **Threshold**: 50% and 75% of `min(max(L²_grid[:,0]), max(L²_grid[0,:]))`.
- **Angle range for fit**: `α_i ≤ 30°`, `α_i ≤ 45°`, `α_i ≤ 60°` (three cuts; exclude grid-boundary clipping).
- **Anchor bootstrap**: 200 bootstrap resamples of the 30 anchors; report median + [5th, 95th] CI of `p` for each (threshold, range) cell.

Headline statement target: "p ∈ [low, high] across threshold × range × bootstrap, for ≥X% of pairs across 6 models."

## Experiments

### E1. 1D single-direction sweeps (matched-source)
- Models: 6 (gemma, qwen, llama, mistral, aya, yi); layer L=2.
- Directions: 14 (Gender, Sentiment, Age, Era, Tense, TensePresent, Number, Person, Status, Honesty, HonestyShort, Refusal, Formal, Literary).
- Source per direction: matched holdout (see Plan v1 list, with `Formal ← CASUAL_PROMPTS_HOLDOUT`, `Literary ← SLANG_PROMPTS_HOLDOUT`).
- Sweep: α ∈ [0°, 75°], 51 steps. + 10 random unit vectors swept from the same source as baseline.
- Per (model, direction): ~13s. Total: 6 × 14 × 13s ≈ 18 min compute, ~3 min wall on 6 pods.

### E2. 2D iso-plateau pair sweeps (FineWeb anchors)
- Models: 6.
- Pairs: 20 (Tier 1, see Plan v1).
- Sweep: (α₁, α₂) ∈ [0°, 60°]², 31×31 grid, 30 FineWeb anchors per pair.
- Per pair: 30 anchors × 961 grid × bf16 forward through ~26-46 layers ≈ ~3 min on 4090.
- Total: 6 × 20 × 3 min = 6 hours sequential, ~1 hour wall on 6 pods.

### E3. Compositional steering generations (deferred)
Defer until E1 and E2 land. Plan structure preserved for later.

### E4. Validation (strengthened — runs before any production sweep)

Run `validation.py` after `tests_parametrize.py` passes:
1. **Identity test**: pick 1 model, 1 pair (gemma, Gender×Tense). Run 2D sweep at the slice α₂=0; run standalone 1D sweep along Gender at the same anchors and angles. Assert `max |L²_2d[:,0] − L²_1d| < 1e-4`. Catches grid/index/aggregation bugs.
2. **Permutation test**: swap (d₁, d₂) → (d₂, d₁) in `sweep_2d.py` for the same pair. Assert `L²_grid_swapped == L²_grid.T` to within fp noise. Catches axis-mixup bugs.
3. **Round-trip test**: for each grid point, recompute `arccos(â·new_a/R)` (great-circle distance) and project `new_a − cos r·R·â` back into (d₁_⊥, d₂_⊥) coords. Assert recovered `(α₁, α₂)` matches input to <1e-5 rad. End-to-end check of property 3.
4. **Zero test**: at α₁=α₂=0, the perturbed activation must equal the original anchor exactly. Free regression against any future bug.

## `tests_parametrize.py` (unit tests, runs before everything)

- Norm-matched: 100 random `(â, d₁, d₂, α₁, α₂)`, assert `‖new_a‖ == R` to fp tolerance.
- Chart-distance: assert `arccos(â·new_a/R) == ‖(α₁,α₂)‖` to fp tolerance.
- Single-direction reduction: with α₂=0, assert `new_a == R·(cos α₁·â + sin α₁·d₁_⊥)`.
- Sign symmetry: simultaneous flip `α_i → −α_i, d_i → −d_i` produces the same `new_a`.
- Degeneracy: with `d₁ ∝ â`, assertion fires; with `d₂ ∝ d₁`, assertion fires.
- Numerical edge: `α₁=α₂=1e-15` should fall into the `r < eps_r` branch and return `R·â` exactly.

## Order of operations

1. Implement `lib/parametrize.py`. Run `tests_parametrize.py`. Block on green.
2. Implement `lib/superellipse.py` (port from old code; re-run on a known A-chart contour to confirm fit code is right).
3. Implement `sweep_1d.py` and `sweep_2d.py` (thin wrappers).
4. Run `validation.py` (E4) on one (model, pair). Block on green.
5. Run E1 end-to-end on **one** (model, direction) — gemma, Gender. Eyeball plot. Block on sanity.
6. Launch E1 on all 6 pods in parallel.
7. Pull, plot E1.
8. Launch E2 on all 6 pods in parallel.
9. Pull, run robustness protocol on E2; report `p` distribution.
10. Decide on E3 based on E1/E2 outcomes.

Wall time end-to-end (excluding holdout re-critique and unit-test debug): ~2-3 hours.

## Disk / cost estimate

- 1D sweeps: 6 models × 14 directions × (51 angles × 30 anchors) float32 + (10 rand × 51 × 30) float32 ≈ 6 × 14 × (1.5K + 15K) × 4 bytes ≈ 6 MB total. Trivial.
- 2D sweeps: 6 models × 20 pairs × (961 × 30 × 4 bytes) ≈ 14 MB total. Trivial.
- Pod-hours: ~1 hour wall × 6 pods = 6 pod-hours. At spot-RTX-4090 rates this is ~$2-3.

## Failure modes (decided in advance)

- **If `p` shifts dramatically away from 2 in the new chart**: report it. The chart-shift is a known-direction effect (the old chart artificially compresses the diagonal at large angles, biasing `p` toward 2 at large radii and away from 2 at small radii). If `p_new` is consistently below or above 2 with tight CIs, that's a real result, not a failure — report it as "the previously reported p≈2 in (sin α₁, sin α₂) coords corresponds to p ≈ X in geodesic coords; the underlying claim ['Euclidean composition'] should be restated as [...]".
- **If new code disagrees with old code on overlapping physical points**: stop, debug. Both should give identical L² values at identical perturbed activations.
- **If a holdout source produces a 1D L²-vs-angle curve that looks pathological** (e.g. flat, or saturating at 1°): flag the holdout, swap in FineWeb for that direction, document.

## What we don't do

- Do not reuse / convert / depend on `results/fig4_robustness/` PKLs. Frozen historical record.
- Do not rebuild the deferred pieces (`fit_pairs`, `plot_p_summary`, `steer_generate`) until E1 + E2 land.
- Do not re-run the holdout-fix loop unless D5 turns up an issue. The sets are reasonable as-is.
