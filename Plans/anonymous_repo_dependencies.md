# Anonymous repo: figure → file dependency map

Audit of the live repo against `paper.tex` (ICML 2026 main text +
appendix). Output of step 1 of `Plans/anonymous_repo_plan.md`.

## Figures actually included by paper.tex

`grep includegraphics` returns exactly these 7 paths (no
`fig1_feature_directions.pdf`, no `fig1_intro_boundary.pdf`, no
`fig_single_direction_sensitivity.pdf` — those are stale leftovers in
the figures dir and can be ignored):

| line | path | role | producing script |
| ---- | ---- | ---- | ---------------- |
| 215 | `figures/combo_sweep_gemma_L2_Gender_Refusal_to25deg.pdf` | Fig 1a | `scripts/exp_map/plotting/plot_combo_sweep.py` |
| 224 | `figures/fig3_boundary_Gender_Refusal_gemma_L2.pdf` | Fig 1b | `scripts/exp_map/plotting/plot_fig3_boundary.py` |
| 245 | `figures/composition_table_gender_tense.pdf` | Fig 2 | `scripts/exp_map/plotting/plot_composition_table.py` |
| 272 | `figures/fig_beeswarm_directions` | Fig 3 (direction families) | `scripts/exp_map/plotting/plot_beeswarm_direction_types.py` (manual rename of `robustness_beeswarm_<...>_dirfamilies_morebaselines.pdf`) |
| 305 | `figures/fig4_n_robust_beeswarm.pdf` | Fig 4 (robustness) | `scripts/exp_map/plotting/plot_robustness_beeswarm.py` (manual rename of `robustness_beeswarm_<...>_<columns>.pdf`) |
| 393 | `figures/appendix_directions_overlap.pdf` | Appx (overlap heatmap) | `scripts/exp_map/plotting/build_directions_appendix.py` |
| 414 | `figures/appendix_residuals_beeswarm.pdf` | Appx (fit residuals) | `scripts/exp_map/plotting/plot_residuals_beeswarm.py` |

File-size cross-check (live `results/exp_map/figures/` vs paper
`figures/`) confirms the rename mapping:
- `fig4_n_robust_beeswarm.pdf` (166 050 B) ↔ `robustness_beeswarm_gemma_L2_ov0p1_excl-Formal-HonestyShort-TensePresent_nomwe_thrfixed_exact_model+perturb+measure+metric+threshold+method+anchor_source.pdf`
- `fig_beeswarm_directions.pdf` (66 246 B) ↔ `robustness_beeswarm_gemma_L2_ov0p1_nomwe_thrfixed_exact_dirfamilies_morebaselines.pdf`
- `appendix_residuals_beeswarm.pdf` (207 005 B) ↔ `residuals_beeswarm_gemma_L2_ov0p1_nomwe_thrfixed_exact.pdf`

## Per-script input audit

### Fig 1a — `plot_combo_sweep.py`
Reads exactly one pkl: `results/exp_map/data/sweeps_2d/sweep2d_gemma_L2_Gender__Refusal_fineweb_60deg.pkl` (116 KB).

### Fig 1b — `plot_fig3_boundary.py`
Same single pkl as Fig 1a.

### Fig 2 — `plot_composition_table.py`
Static figure: hard-codes generation samples (PROMPT + ROWS). No PKL input.

### Fig 3 — `plot_beeswarm_direction_types.py`
Reads (with canonical args `--use_thrfixed --exact --max_overlap 0.10 --exclude_dirs HonestyShort,TensePresent,Formal --exclude_mwe`):
- fits: `fits_gemma_L2_thrfixed_exact.pkl`,
        `fits_gemma_L2_dirmelbo_thrfixed_exact.pkl`,
        `fits_gemma_L2_dirsae_fineweb_thrfixed_exact.pkl`,
        `fits_gemma_L2_dirpca_fineweb_thrfixed_exact.pkl`,
        `fits_gemma_L2_dirrandom_thrfixed_exact.pkl`
- direction caches (for the per-family overlap filter):
        `dirs_gemma_L2.pkl`,
        `dirs_gemma_L2_melbo.pkl`,
        `dirs_gemma_L2_sae_fineweb.pkl`,
        `dirs_gemma_L2_pca_fineweb.pkl`,
        `dirs_gemma_L2_random.pkl`

### Fig 4 — `plot_robustness_beeswarm.py`
With canonical args (`--use_thrfixed --exact --max_overlap 0.10
--exclude_dirs Formal,HonestyShort,TensePresent --exclude_mwe`,
columns including `model,perturb,measure,metric,threshold,method,anchor_source`):
- model column: `fits_{gemma,qwen,llama,mistral,aya,yi}_L2_thrfixed_exact.pkl`
- perturb-layer column: `fits_gemma_L{2,5,10,20}_thrfixed_exact.pkl`
- measure-layer column: `fits_gemma_L2_M-7_thrfixed_exact.pkl`, `fits_gemma_L2_M-12_thrfixed_exact.pkl`
- metric column: `fits_gemma_L2_cos+kl+l2_metric_cos_thrfixed_exact.pkl`, `fits_gemma_L2_cos+kl+l2_metric_kl_thrfixed_exact.pkl`
- threshold column (in thrfixed mode): re-parses the same `fits_gemma_L2_thrfixed_exact.pkl` — no extra file
- method column: `fits_gemma_L2_additive_thrfixed_exact.pkl`
- anchor-source column: `fits_gemma_L2_srcwiki_en_thrfixed_exact.pkl`, `_srcwiki_zh`, `_srccode`
- direction caches for overlap filtering: `dirs_{gemma,qwen,llama,mistral,aya,yi}_L2.pkl`, `dirs_gemma_L{5,10,20}.pkl`

### Fig 6 (residuals) — `plot_residuals_beeswarm.py`
Reuses every column from Figs 4 and 5 — same fits + direction caches.
No additional inputs.

### Appendix overlap — `build_directions_appendix.py`
Single input: `dirs_gemma_L2.pkl`. Also writes `appendix_directions_table.tex` (in addition to the PDF).

## Allowlist

### Source code

```
src/__init__.py
src/model.py                  # load_model, forward_from_layer*, hooks
src/data.py                   # load_fineweb_fixed_length only — strip
                              #  chess/random/codeparrot helpers
```

```
scripts/exp_map/__init__.py
scripts/exp_map/lib/__init__.py
scripts/exp_map/lib/registry.py
scripts/exp_map/lib/directions.py
scripts/exp_map/lib/activations.py
scripts/exp_map/lib/parametrize.py
scripts/exp_map/lib/sweep_core.py
scripts/exp_map/lib/superellipse.py
```

```
scripts/exp_map/sweep_2d.py
scripts/exp_map/sweep_1d.py                   # only kept if Fig 4 holdout
                                              # is restored; see note below
scripts/exp_map/fit_pairs.py
scripts/exp_map/refit_thrfixed_all.py
scripts/exp_map/recommend_fixed_threshold.py
scripts/exp_map/melbo_directions.py
scripts/exp_map/pca_directions.py
scripts/exp_map/sae_directions_top.py
scripts/exp_map/random_directions.py
```

```
scripts/exp_map/plotting/__init__.py
scripts/exp_map/plotting/plot_combo_sweep.py
scripts/exp_map/plotting/plot_fig3_boundary.py
scripts/exp_map/plotting/plot_composition_table.py
scripts/exp_map/plotting/plot_beeswarm_direction_types.py
scripts/exp_map/plotting/plot_robustness_beeswarm.py
scripts/exp_map/plotting/plot_residuals_beeswarm.py
scripts/exp_map/plotting/build_directions_appendix.py
```

Note: `plot_holdout_sweeps.py`, `sweep_1d.py`, and the `sweeps_1d/`
cache are *not* shipped — Fig 4 in the paper is the robustness
beeswarm, not the holdout sweep. Decision confirmed by Francisco.

### Prompt files (`data/`)

Required by `lib/registry.py` for the 33 directions:
```
data/__init__.py
data/gender_prompts.py            data/male_prompts_holdout.py
data/sentiment_prompts.py         data/positive_prompts_holdout.py
data/age_prompts.py               data/young_prompts_holdout.py
data/era_prompts.py               data/medieval_prompts_holdout.py
data/tense_prompts.py             data/past_prompts_holdout.py
data/tense_present_prompts.py     data/present_prompts_holdout.py
data/number_prompts.py            data/singular_prompts_holdout.py
data/person_prompts.py            data/first_prompts_holdout.py
data/status_prompts.py            data/royal_prompts_holdout.py
data/honesty_prompts.py           data/honest_prompts_holdout.py
data/honesty_short_prompts.py     data/honest_short_prompts_holdout.py
data/refusal_prompts.py           data/comply_prompts_holdout.py
data/register_prompts.py          data/casual_prompts_holdout.py
                                  data/slang_prompts_holdout.py
data/wealth_prompts.py            data/rich_prompts_holdout.py
data/health_prompts.py            data/healthy_prompts_holdout.py
data/certainty_prompts.py         data/certain_prompts_holdout.py
data/multilingual_prompts.py
data/programming_prompts.py
```

(`Formal` and `Literary` semantic directions both use
`register_prompts.py` for the contrastive sides; their matched
holdouts are `casual_prompts_holdout.py` / `slang_prompts_holdout.py`
respectively.)

### Drop list (prompt files)
Drop the MWE files: `mwe_prompts.py`, the entire `data/mwe/`
directory, plus every other prompt file *not* in the list above
(animal_prompts, alpaca_data.json, bilingual_*, coding_prompts,
genre_prompts, harmful_behaviors.csv, othello_sae, python_prompts,
spanish_prompts, etc.).

### Result PKLs to ship

Fast path (reviewers-without-GPU). Lives under `results/`:

`results/directions/`
```
dirs_gemma_L2.pkl
dirs_qwen_L2.pkl, dirs_llama_L2.pkl, dirs_mistral_L2.pkl,
  dirs_aya_L2.pkl, dirs_yi_L2.pkl
dirs_gemma_L5.pkl, dirs_gemma_L10.pkl, dirs_gemma_L20.pkl
dirs_gemma_L2_melbo.pkl
dirs_gemma_L2_pca_fineweb.pkl
dirs_gemma_L2_sae_fineweb.pkl
dirs_gemma_L2_random.pkl
```
(11 MB total in the live repo.)

`results/fits/`
```
# Model column
fits_{gemma,qwen,llama,mistral,aya,yi}_L2_thrfixed_exact.pkl   (×6)
# Perturb-layer column
fits_gemma_L{5,10,20}_thrfixed_exact.pkl                        (×3)
# Measure-layer column
fits_gemma_L2_M-7_thrfixed_exact.pkl                            (×1)
fits_gemma_L2_M-12_thrfixed_exact.pkl                           (×1)
# Metric column
fits_gemma_L2_cos+kl+l2_metric_cos_thrfixed_exact.pkl           (×1)
fits_gemma_L2_cos+kl+l2_metric_kl_thrfixed_exact.pkl            (×1)
# Method column
fits_gemma_L2_additive_thrfixed_exact.pkl                       (×1)
# Anchor-source column
fits_gemma_L2_srcwiki_en_thrfixed_exact.pkl                     (×1)
fits_gemma_L2_srcwiki_zh_thrfixed_exact.pkl                     (×1)
fits_gemma_L2_srccode_thrfixed_exact.pkl                        (×1)
# Direction-family column
fits_gemma_L2_dirmelbo_thrfixed_exact.pkl                       (×1)
fits_gemma_L2_dirpca_fineweb_thrfixed_exact.pkl                 (×1)
fits_gemma_L2_dirsae_fineweb_thrfixed_exact.pkl                 (×1)
fits_gemma_L2_dirrandom_thrfixed_exact.pkl                      (×1)
```
(20 fits files; ~14 MB out of the 48 MB total fits cache.)

`results/sweeps_2d/`
- All non-MWE sweep2d pkls (13 058 files, ~1.6 GB) — ship the lot so
  the refit step (`fit_pairs.py` / `refit_thrfixed_all.py`) is
  reproducible end-to-end without a GPU.
- Drop only the MWE-prefixed sweeps (any pair where one or both
  directions match `aair_h_*`, `aair_l_*`, `persona_*`, `syc_*`, `caa_*`):
  ~1 033 files, since the registry-anonymised scripts won't reference
  them.

Fast-path payload total: ~1.7 GB (mostly sweep2d). Probably exceeds
4open.science's per-file/per-repo cap — plan B (sweeps on anonymous
Zenodo deposit) is the likely landing spot. Confirm cap before final
upload.

### Slow-path notes

Two slow paths are supported:

1. **Refit only** (CPU, no GPU): rerun `refit_thrfixed_all.py --exact`
   over the bundled sweep_2d pkls; this regenerates every fits file
   the figure scripts read. ~minutes.
2. **Re-sweep from model weights** (GPU): rerun `sweep_2d.py` on the
   six target models, then refit. README will spell out wallclock:
   - `sweep_2d.py` over the 528-pair canonical run: ~6h on a single
     80 GB H100 (≈40 s/pair).
   - All 6 cross-model runs: ~36 h on the same.

## Files to drop entirely

- `_pod_run_*.sh` shell scripts (all of `scripts/exp_map/`).
- `CLAUDE.md`, `Plans/`, `writeup/`, `notebooks/`, `results/exp_map/figures/` (paper outputs go in the user's own paper-build dir, not the anonymous code repo).
- All `scripts/perturbation/`, `scripts/melbo/`, `scripts/directions/`, `scripts/plotting/`, `scripts/analysis/` — these are predecessor / parallel-line code paths not used by the paper figures (verified: none of the seven figure-producing scripts imports from them).
- `scripts/exp_map/_extract_new3.py`, `anchor_floor_pilot.py`,
  `anchor_subsample_stability.py`, `coherence_boundary.py`, `comp_steer.py`,
  `debug_floor.py`, `download_mwe.py`, `dump_coherence_outliers.py`,
  `overlap_vs_n.py`, `per_anchor_decomposition.py`,
  `select_mwe_pairs.py`, `select_new3_pairs.py`,
  `sweep_1d_at_anchors.py`, `sweep_1d_mwe.py`, `tests_parametrize.py`,
  `validation.py`, `verify_fp64.py` — exploratory / MWE-only / legacy.
- `scripts/exp_map/sae_directions.py` — superseded by `sae_directions_top.py` (the latter is what produced `dirs_gemma_L2_sae_fineweb.pkl`); confirm before dropping.
- All `results/exp_map/data/sweeps_2d/*.pkl` *except* the single Fig 1 pair.
- All `results/exp_map/data/sweeps_1d/`.
- `results/exp_map/data/{activations,coherence,comp_steering,pair_lists}/`.
- `results/exp_map/data/directions/dirs_*sae*.pkl` other than `dirs_gemma_L2_sae_fineweb.pkl` (we keep the one used by the paper's SAE column; eval-prompt and random-row variants are baselines we no longer use).

## Decisions (Francisco, 2026-05-04)

- Drop `plot_holdout_sweeps.py`, `sweep_1d.py`, and the `sweeps_1d/`
  cache — not in the paper.
- Ship all non-MWE sweep_2d pkls (~13 058 files, ~1.6 GB). The
  reviewer-friendly path stays "refit from sweeps", not "rerun
  sweeps from model weights". Likely needs anonymous Zenodo as the
  final host (4open.science size cap to be confirmed).
- Adopt option (b) for Fig 4 / Fig 3 names: ship a `render_figures.sh`
  that runs each plotting script and copies/renames its output to the
  paper-side filename (`fig4_n_robust_beeswarm.pdf`,
  `fig_beeswarm_directions.pdf`). Keeps the plotting scripts
  unchanged.

## Open items

- **Composition-table source generations:** `plot_composition_table.py` is purely typographic — the actual generations were produced elsewhere and pasted in. If reviewers are expected to regenerate them, we need to ship a `comp_steer.py`-style runner; otherwise the static script is enough.
