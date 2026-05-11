# Plan: Extract Vanilla AE + Intrinsic Dim Estimation to New Repo

## Goal

Move the vanilla-autoencoder training/analysis code and the intrinsic-dimensionality estimation code out of `steering-plateaus` into a new repo. Keep all SAE / dictionary-learning / pareto-plateau / probe / steering work in `steering-plateaus`.

Working name for the new repo: **`activation-manifold-tools`** (pick whatever).

## What moves vs. what stays

### Moves to new repo

**Vanilla AE training & analysis (~12 files):**
- `scripts/jacobian/train_autoencoder.py`
- `scripts/jacobian/hillclimb_autoencoder.py`
- `scripts/jacobian/sweep_autoencoder_bottleneck.py`
- `scripts/jacobian/quick_overfit_check.py` *(but it also touches TopKSAE — see "Bridge files" below)*
- `scripts/jacobian/create_subsets.py`
- `scripts/jacobian/plot_ae_vs_sae_scaling.py` *(compares to SAE — see "Bridge files")*
- `scripts/jacobian/plot_bottleneck_combined.py`
- `scripts/jacobian/plot_training_curves.py`
- `scripts/jacobian/plot_final_comparison.py`
- `scripts/jacobian/run_full_pipeline.sh`, `run_recovery_pipeline.sh`, `run_final_eval.sh`

**Intrinsic dim (~17 files):**
- `scripts/jacobian/intrinsic_dim_twonn.py`
- `scripts/jacobian/intrinsic_dim_layer_sweep.py`
- `scripts/jacobian/intrinsic_dim_gaussian_baseline.py`
- `scripts/jacobian/intrinsic_dim_robustness.py`
- `scripts/jacobian/intrinsic_dim_data_sources.py`  ← **cross-cut, see below**
- `scripts/jacobian/twonn_append.py`, `twonn_append_100k.py`, `twonn_vs_n.py`, `twonn_vs_n_traincache.py`
- `scripts/jacobian/cluster_intrinsic_dim.py`, `cluster_intrinsic_dim_sweep.py`, `cluster_id_n2000.py`
- `scripts/jacobian/plot_id_vs_cluster_size_l10.py`
- `scripts/jacobian/collect_layer10_activations.py`

**Jac-matched intrinsic dim (~3 files, gray zone — moving):**
- `scripts/jacobian/intrinsic_dim_jac_filtered.py`
- `scripts/jacobian/intrinsic_dim_jac_matched.py`
- `scripts/jacobian/intrinsic_dim_jac_optimized.py`
- These use `forward_from_layer_to_final` + `_get_blocks` etc. from `src.model` and consume jacobian pickle outputs. Move them, vendor the needed `src.model` helpers into the new repo, and pass jacobian pickle paths via CLI args.

**Results dirs:**
- `results/trained_autoencoder/` — 7.9 GB. Don't commit. Move on disk only (rsync/mv).
- `results/intrinsic_dim/` — 334 MB. Same.

### Stays in `steering-plateaus`

- All SAE work: `train_sae.py`, `perturbation/sweep_sae_recon.py`, `benchmark_dictionary.py`, `eval_ce_loss.py`, pareto plateau scripts, `sweep_directions_comparison.py`, `perturbation_sweep_dict_comparison.py`
- All dictionary learning: `prune_dictionary{,_fp}.py`, `feature_decomp.py`, `batch_decomp.py`, `case_study_*.py`, `dictionary_connected_components.py`, `run_full_decomp_pipeline.py`
- Probes: `run_probe_sae_vs_ae.py`, `run_probe_language.py`, `eval_recon_ood.py`, `gen_recon_examples.py` (they consume AE weights but *the question they answer is plateau-research*)
- Latent steering: `on_manifold_steering.py`, `latent_steering.py`, `latent_interpolation.py`, `eval_ae_ce.py`
- Everything jacobian / delta optimization / perturbation / anisotropy / boundary / melbo / refusal / autointerp / etc.

## Pre-split cleanup in `steering-plateaus` (do these first)

These are improvements worth doing regardless of whether the split happens, but they unblock a clean split.

### Step 1 — Factor `FlexAutoencoder` and `TopKSAE` into `src/autoencoders.py`

Currently `FlexAutoencoder` is defined inline in 5+ scripts (`probes/run_probe_sae_vs_ae.py`, `probes/run_probe_language.py`, `steering/latent_interpolation.py`) and similar `Autoencoder` classes in `steering/{latent_steering,on_manifold_steering,eval_ae_ce}.py`. `TopKSAE` is in `train_sae.py` and copy-pasted elsewhere.

Action: create `src/autoencoders.py` containing:
- `FlexAutoencoder` (canonical version from `hillclimb_autoencoder.py`)
- `TopKSAE` (from `train_sae.py`)
- `load_ae(path, device)` and `load_sae(path, device)` helpers (lift from `run_probe_sae_vs_ae.py`)

Then replace inline definitions with `from src.autoencoders import FlexAutoencoder, TopKSAE, load_ae, load_sae`. This is the loading-side API the plateau repo will use after the split.

**Verify:** re-run `probes/run_probe_sae_vs_ae.py` end-to-end on existing weights to confirm nothing broke.

### Step 2 — Factor `intrinsic_dim_data_sources.py` into a shared spot

It's imported by `probes/eval_recon_ood.py` and `probes/gen_recon_examples.py` via `from scripts.jacobian.intrinsic_dim_data_sources import SOURCE_LOADERS`. Once it moves to the new repo, those imports break.

Action: move `SOURCE_LOADERS` and the loader functions into `src/data_sources.py` in the plateau repo. The new repo will vendor its own copy.

Update the two probe scripts to use `from src.data_sources import SOURCE_LOADERS`.

### Step 3 — Cross-script imports inside the moving set

`quick_overfit_check.py` imports `FlexAutoencoder` from `hillclimb_autoencoder.py` and `TopKSAE` from `train_sae.py`. After Step 1 above, it should import from `src.autoencoders` instead — making it portable.

`sweep_autoencoder_bottleneck.py` imports `DeepAutoencoder, cache_activations` from `train_autoencoder.py`. That's fine — both move together; just preserve the same relative import.

## New repo layout

```
activation-manifold-tools/
├── src/
│   ├── __init__.py
│   ├── autoencoders.py        # FlexAutoencoder, DeepAutoencoder, load_ae
│   ├── model_utils.py         # vendored: load_model, _get_blocks,
│   │                          # forward_from_layer_to_final, _is_rope_model,
│   │                          # _get_position_embeddings, _run_block
│   ├── data.py                # vendored: load_fineweb_fixed_length
│   └── data_sources.py        # vendored copy of SOURCE_LOADERS
├── scripts/
│   ├── autoencoder/
│   │   ├── train.py           # renamed from train_autoencoder.py
│   │   ├── hillclimb.py
│   │   ├── sweep_bottleneck.py
│   │   ├── overfit_check.py
│   │   ├── create_subsets.py
│   │   ├── plot_*.py
│   │   └── pipelines/         # the .sh files
│   └── intrinsic_dim/
│       ├── twonn.py
│       ├── layer_sweep.py
│       ├── gaussian_baseline.py
│       ├── robustness.py
│       ├── twonn_*.py
│       ├── cluster_*.py
│       ├── collect_activations.py
│       └── jac_filtered.py    # the 3 jac-matched scripts
├── results/                   # not committed; AE checkpoints + dim CSVs land here
├── requirements.txt
├── README.md
└── CLAUDE.md                  # carry over relevant bits from plateau repo
```

## Bridge files — decide explicitly

A few scripts mix AE + SAE. Decide one of the three options per file:

- `quick_overfit_check.py` — moves. Drop the SAE comparison or vendor `TopKSAE` into the new repo's `src/autoencoders.py` too.
- `plot_ae_vs_sae_scaling.py` — moves (it's primarily AE; consume SAE eval results via CSV/pickle path on disk).
- `eval_ce_loss.py` — **stays** in plateau repo (it's a CE-loss benchmark across AE/SAE/dict; the plateau repo has more of the comparison set). New repo can have a simpler AE-only CE eval if needed.

## Migration order

1. **Pre-split cleanup (Steps 1–3 above) in plateau repo.** Commit. Sanity-run one AE-bridge script and one dim script. *Don't touch the new repo yet.*
2. **Create new repo locally.** Initial commit with skeleton layout (empty `src/`, empty `scripts/`).
3. **Vendor utilities into new repo's `src/`.** Copy the needed functions from `src/model.py`, `src/data.py`, `src/data_sources.py`. Keep just what's actually imported by the moving scripts — don't drag the full files.
4. **Copy scripts** from plateau repo into the new layout. Update import paths (`from src.model import load_model` → `from src.model_utils import load_model`).
5. **Smoke test in new repo:** train a tiny AE end-to-end (e.g., k=10, max_samples=100k) on a small model. Run `twonn.py` on a small sample. Confirm both work without reaching back to plateau repo.
6. **Move results dirs (NOT commit).** `mv ~/Research/steering-plateaus/results/trained_autoencoder ~/Research/activation-manifold-tools/results/`. Same for `results/intrinsic_dim`. Update any hardcoded paths in moved scripts (most use CLI args, but spot-check).
7. **Delete moved scripts from plateau repo.** Update `scripts/INDEX.md` to remove the deleted entries. Update `CLAUDE.md` if any references die.
8. **Verify plateau repo still works:** run one of each bridge script (`probes/run_probe_sae_vs_ae.py`, `steering/on_manifold_steering.py`, `probes/eval_recon_ood.py`) pointing `--ae_path` / `--sae_path` to the new locations. These take checkpoint paths as CLI args, so the bridge is just "pass `--ae_weights /path/to/new/repo/results/trained_autoencoder/...`".
9. **Commit + push both repos.**

## Cross-repo coupling after the split

After the move, the plateau repo still loads vanilla-AE checkpoints (via `probes/run_probe_sae_vs_ae.py` etc.). The coupling is:

- **On-disk weights format** (state_dict + config dict in `.pt`) — stable, decoupled.
- **`FlexAutoencoder` class definition** — needs to match between repos. Lives in `src/autoencoders.py` in both. Diverges over time unless you maintain it.

If `FlexAutoencoder` architecture changes in the new repo and old checkpoints get retrained, the plateau-repo `FlexAutoencoder` needs the matching update. Mitigation: keep architecture frozen unless there's a specific reason to change it; if you change it, bump a `version` field in the saved config and have both repos handle both versions.

## Risks / things to watch

- **Lost git history for moved files.** Mitigation: if you care, use `git filter-repo` to extract history into the new repo. Cheap if you do it before deletion.
- **7.9 GB of AE results.** Don't commit. Move on disk separately. Note location in the new repo's README so future-you can find it.
- **`intrinsic_dim_jac_*` need jacobian pickles.** Those live in `results/` in the plateau repo. Either copy the specific pickles to the new repo's `results/`, or pass paths via CLI.
- **`run_full_pipeline.sh` has hardcoded `/workspace/steering-plateaus` paths.** Update to `/workspace/activation-manifold-tools` (or whatever) after move.
- **CLAUDE.md in the new repo** should carry over: RunPod usage, scp not rsync, HF token, the older CUDA image guidance, etc. — basically all the "remote experiments" + "RunPod CLI" + "what to avoid" sections.
- **`scripts/INDEX.md` in plateau repo** doesn't currently list most of the moving scripts (the AE/dim work was added without index entries), so the plateau-repo INDEX won't lose much. The new repo should start with its own INDEX.md.

## Reversibility

This is reversible-but-painful. If after a few weeks you decide the split was a mistake, you can move everything back, but you'd lose any divergent edits in the new repo. So: don't make architectural changes in the new repo for the first few days post-split — let it settle.

## Decision points before starting

- Repo name?
- Do you want git history preserved (`git filter-repo`), or is a fresh history fine?
- Where do you want the AE results dir to physically live (same drive, same `~/Research/` tree, or somewhere else)?
- Do you want to also extract the **dictionary-learning** work (`prune_dictionary*`, decomp scripts) into the new repo, or is that staying with the plateau story? Current plan keeps it in plateau repo — it's coupled to the pareto-plateau-vs-recon narrative.
