# Scripts Index

Quick reference: "I want to do X — which script do I run?"

## Perturbation sweeps & plateau characterization
- **Run a perturbation sweep:** `perturbation/run_perturbation_sweep_spherical.py` (spherical, go-to version) or `perturbation/run_perturbation_sweep_linear.py` (linear)
- **Analyze plateau structure:** `perturbation/run_plateau_analysis.py`
- **Run Gemma sweeps:** `perturbation/run_gemma_sweeps.sh`

## Jacobian analysis
- **Compute Jacobian eigenspectrum:** `jacobian/run_jacobian_v2.py`
- **Optimize delta directions:** `jacobian/run_delta_optimization_v2.py`
- **Analyze optimized deltas:** `jacobian/analyze_delta.py`
- **Neuron gate switch analysis:** `jacobian/run_gate_switches.py`

## Anisotropy measurements
- **2D KL heatmap (single pair):** `anisotropy/run_anisotropy_heatmap.py` (KL) or `anisotropy/run_anisotropy_heatmap_l2.py` (L2)
- **Anisotropy across prompt pairs:** `anisotropy/run_anisotropy_across_pairs.py` (KL) or `anisotropy/run_anisotropy_across_pairs_l2.py` (L2)
- **Pairs × layers sweep:** `anisotropy/run_anisotropy_pairs_x_layers.py` (KL) or `anisotropy/run_anisotropy_pairs_x_layers_l2.py` (L2)
- **Per-layer anisotropy:** `anisotropy/run_anisotropy_per_layer.py`

## Boundary point analysis
- **Find boundary via interpolation:** `boundary/run_boundary_interpolation.py`
- **Probe boundary orthogonally:** `boundary/run_boundary_orthogonal.py`
- **Random unrestricted boundary search:** `boundary/run_boundary_random_unrestricted.py`
- **Boundary via MELBO:** `boundary/run_boundary_melbo.py`
- **Boundary via SVD (Mandarin):** `boundary/run_boundary_svd_mandarin.py`
- **Fixed-reference boundary:** `boundary/run_boundary_fixedref.py`
- **Steer along boundary directions:** `boundary/steer_boundary_dirs.py`

## MELBO direction discovery
- **Constrained MELBO (within subspace):** `melbo/run_melbo_constrained.py`
- **Unconstrained MELBO (full space):** `melbo/run_melbo_unconstrained.py`
- **Diverse directions:** `melbo/run_melbo_diverse.py`
- **Diverse directions (capped objective):** `melbo/run_melbo_diverse_capped.py`
- **Animal prompts:** `melbo/run_melbo_animal.py`
- **Dog breed prompts:** `melbo/run_melbo_dogs.py`
- **Dog breed prompts (capped objective):** `melbo/run_melbo_dogs_capped.py`
- **K=10 span experiment:** `melbo/run_melbo_k10_span.py`
- **Rotation analysis:** `melbo/run_melbo_rotation.py`
- **Diagnose MELBO animal discrepancy:** `melbo/diagnose_melbo_animal.py`
- **Interpret MELBO dogs results:** `melbo/interpret_melbo_dogs.py`
- **Print MELBO results:** `melbo/print_results.py`

## Steering experiments
- **Language steering (multi-contrastive):** `steering/run_steering_comparison.py`
- **Multi-contrastive steering:** `steering/run_steering_multi_contrastive.py`
- **Mandarin steering:** `steering/run_steering_mandarin_fraction.py`, `steering/run_steering_mandarin_extended.py`
- **Russian steering:** `steering/run_steering_russian.py`, `steering/run_steering_russian_extra.py`
- **Spanish/Arabic steering:** `steering/run_steering_spanish.py`, `steering/run_steering_arabic.py`
- **PC1 orthogonal steering:** `steering/run_steering_pc1_orth.py`
- **Steering vs plateau:** `steering/run_steering_vs_plateau.py`
- **Steer with optimized delta:** `steering/steer_with_delta.py`
- **Basic steering experiment:** `steering/steering_experiment.py`
- **Generate steered text:** `steering/steering_generate.py` (linear) or `steering/spherical_steering_generate.py` (spherical)
- **Overshoot analysis:** `steering/steering_overshoot.py`
- **Steer with diverse MELBO dirs:** `steering/run_steer_melbo_diverse.py`
- **Steer with diverse capped MELBO dirs:** `steering/run_steer_melbo_diverse_capped.py`
- **Steer with capped MELBO dog dirs:** `steering/steer_melbo_dogs_capped.py`
- **Fine-grained steering sweep (capped dirs):** `steering/run_steer_fine_sweep.py`
- **Topical steering:** `steering/gender_steering.py`, `steering/sentiment_steering.py`, `steering/refusal_steering.py`

## Direction decomposition & SVD
- **Direction decomposition:** `directions/run_direction_decomposition_v2.py`
- **Direction investigations:** `directions/run_direction_investigations.py`
- **SVD steering:** `directions/run_svd_steering.py` or `directions/run_svd_steering_bootstrap.py` (bootstrap)
- **SVD outlier analysis:** `directions/run_svd_outlier_analysis.py`
- **Multi-contrastive directions:** `directions/run_multi_contrastive.py`
- **Scaling tests:** `directions/run_scaling_test.py`, `directions/run_scaling_validation.py`
- **Orthogonal comparison:** `directions/run_orth_comparison.py`

## Path optimization & interpolation
- **Path optimization:** `paths/path_optimization_experiment.py`
- **Interpolation experiment:** `paths/interpolation_experiment.py`

## Coherence & generation analysis
- **Coherence analysis:** `coherence/run_coherence_analysis.py`
- **Quick steering tests:** `coherence/quick_orth_steer.py`, `coherence/quick_unconstrained_steer.py`
- **MELBO steering test:** `coherence/test_melbo_steering.py`
- **Small magnitude test:** `coherence/test_small_mag_steering.py`

## Refusal experiments
- **Refusal plateau analysis:** `refusal/refusal_plateau_analysis_v2.py`

## Auto-interpretability
- **Compute projections (GPU):** `autointerp/01_compute_projections.py` — extract 50k activations, project onto 100 MELBO directions
- **Explain + simulate (API):** `autointerp/02_explain_and_simulate.py` — Claude describes features and predicts activations
- **Score results (local):** `autointerp/03_score.py` — Spearman correlation between predicted and actual
- **Compute projections, long-context (GPU):** `autointerp/01_compute_projections_longctx.py` — all-position activations for 128-token passages
- **Explain + simulate, long-context (API):** `autointerp/02_explain_and_simulate_longctx.py` — per-token predictions with highlighted passages
- **Score results, long-context (local):** `autointerp/03_score_longctx.py` — per-token Spearman, comparison with short-context

## Plotting
- Coherence plots: `plotting/plot_arabic_coherence.py`, `plotting/plot_russian_coherence.py`, `plotting/plot_spanish_coherence.py`
- KL vs CJK: `plotting/plot_kl_vs_cjk.py`, `plotting/plot_kl_vs_cjk_v2.py`
- MELBO rotation: `plotting/plot_melbo_rotation.py`, `plotting/plot_rotation_improved.py`
- Multi-contrastive: `plotting/plot_multi_contrastive.py`
- Sweep overlay: `plotting/plot_gen_sweep_overlay.py`
- Path relative distance: `plotting/plot_path_relative_distance.py`
- Scaling test: `plotting/plot_scaling_test.py`

## Archive (superseded / one-off)
- `archive/run_experiment.py` — early example script
- `archive/run_delta_optimization.py` — superseded by v2
- `archive/run_direction_decomposition.py` — superseded by v2
- `archive/refusal_plateau_analysis.py` — superseded by v2
- `archive/patch_melbo_rerun.py` — one-off patch
- `archive/patch_plateau_sweeps.py` — one-off patch
- `archive/check_animal_tokens.py` — one-off sanity check
- `archive/check_dutch.py` — one-off sanity check
- `archive/sentiment_sanity_check.py` — one-off sanity check
