# Scripts Index

Quick reference: "I want to do X — which script do I run?"

## Perturbation sweeps & plateau characterization
- **Run a perturbation sweep:** `perturbation/run_perturbation_sweep_spherical.py` (spherical, go-to version) or `perturbation/run_perturbation_sweep_linear.py` (linear)
- **Analyze plateau structure:** `perturbation/run_plateau_analysis.py`
- **Real vs random points sweep (any model):** `perturbation/sweep_random_points.py`
- **Real vs random points sweep (chess only):** `perturbation/sweep_random_points_chess.py`
- **Run Gemma sweeps:** `perturbation/run_gemma_sweeps.sh`

## J-Lens superellipse experiment (qwen3-8b)
Superellipse p-fitting on J-Lens (Jacobian Lens) token-direction pairs,
EN vs ZH. See `Plans/jlens_superellipse_qwen3_8b.md`.
- **Download/inspect pre-fitted lens:** `jlens/download_inspect_lens.py`
- **Build direction pairs (pod):** `jlens/build_jlens_directions.py`
- **Iso-plateau sweeps + p fits (pod):** `jlens/sweep_isoplateau_jlens.py`
- **Beeswarm + shape panels:** `jlens/plot_jlens_p.py`
- **Single-pair shape plot (original style):** `jlens/plot_jlens_shape_single.py`
- **Sweep L2-range diagnostic:** `jlens/inspect_sweep_range.py`

## Jacobian analysis
- **Compute Jacobian eigenspectrum:** `jacobian/run_jacobian_v2.py`
- **Analyze optimized deltas:** `jacobian/analyze_delta.py`
- **Neuron gate switch analysis:** `jacobian/run_gate_switches.py`
- **Subspace overlap analysis:** `jacobian/subspace_overlap_analysis.py`
- **Compare spectra:** `jacobian/compare_spectra.py`
- **Compute activation PCs:** `jacobian/compute_activation_pcs.py`
- **KL-based direction selection:** `jacobian/kl_selection.py`
- **Min-Jacobian reconstruction:** `jacobian/reconstruct_min_jacobian.py`
- **Reproducibility (SVD):** `jacobian/reproducibility_svd.py`, `reproducibility_svd_kl.py`, `reproducibility_svd_recon.py`

## Delta optimization
Scripts that optimize delta directions in activation space (jacobian-flavored,
but spun out into their own dir because there are many versions).
- **Latest delta opt:** `delta_optimization/run_delta_optimization_v9.py` (v2..v9 available)
- **Per-activation optimization:** `delta_optimization/per_activation_opt_v9.py`
- **Joint vs per-activation sweep:** `delta_optimization/sweep_per_act_vs_joint.py`
- **Response analysis:** `delta_optimization/response_decomp.py`, `delta_optimization/response_sweep.py`
- **Delta vs random perturbation:** `delta_optimization/perturbation_response_delta_vs_random.py`
- **Entropy / shifted-activation analysis:** `delta_optimization/entropy_histogram_shifted.py`, `delta_optimization/generate_from_shifted.py`
- **Steer with optimized delta:** `delta_optimization/steer_delta_spherical.py`, `delta_optimization/steer_with_opt_delta.py`, `delta_optimization/steer_v{7,9}_direction.py`
- **Steer with response directions:** `delta_optimization/steer_response_dirs.py`, `delta_optimization/steer_shared_direction.py`
- **Direction sweeps:** `delta_optimization/sweep_v{6,7}_direction.py`
- **Token-boost v9:** `delta_optimization/token_boost_v9.py`

## Autoencoder training & analysis
Vanilla (non-sparse) autoencoder training and reconstruction-quality analysis.
- **Train AE:** `autoencoder/train_autoencoder.py` (older DeepAutoencoder) or `autoencoder/hillclimb_autoencoder.py` (FlexAutoencoder hillclimb sweep)
- **Cache subsets:** `autoencoder/create_subsets.py`
- **Bottleneck sweep:** `autoencoder/sweep_autoencoder_bottleneck.py`
- **CE-loss eval:** `autoencoder/eval_ce_loss.py`
- **Quick overfit check:** `autoencoder/quick_overfit_check.py`
- **Pipelines:** `autoencoder/run_full_pipeline.sh`, `run_recovery_pipeline.sh`, `run_final_eval.sh`
- **Plots:** `autoencoder/plot_training_curves.py`, `plot_final_comparison.py`, `plot_bottleneck_combined.py`, `plot_ae_vs_sae_scaling.py`

## Dictionary learning (SAE + pruned dicts)
SAE training and pruned-dictionary work (MP/FP).
- **Train TopK SAE:** `dictionary_learning/train_sae.py`
- **Prune dictionary (MP):** `dictionary_learning/prune_dictionary.py`
- **Prune dictionary (FP):** `dictionary_learning/prune_dictionary_fp.py`
- **Benchmark dictionary:** `dictionary_learning/benchmark_dictionary.py`
- **Connected components:** `dictionary_learning/dictionary_connected_components.py`
- **Feature decomposition:** `dictionary_learning/feature_decomp.py`, `batch_decomp.py`, `case_study_decomp.py`, `case_study_extra.py`, `run_full_decomp_pipeline.py`
- **Compare real vs GLP:** `dictionary_learning/compare_real_vs_glp.py`
- **Pareto recon vs plateau:** `dictionary_learning/pareto_plateau_vs_recon.py` (KL) or `pareto_plateau_vs_recon_l2.py` (L2)
- **Perturbation sweeps:** `dictionary_learning/perturbation_sweep_dict_comparison.py`, `perturbation_sweep_reconstructions.py`
- **Sweep directions comparison:** `dictionary_learning/sweep_directions_comparison.py`
- **Plots:** `dictionary_learning/plot_fp_vs_mp_benchmark.py`, `plot_residual_curve.py`, `plot_sweep_cosine.py`

## Intrinsic dimensionality estimation
TwoNN-based intrinsic dim, clustered ID, and jacobian-matched ID.
- **TwoNN baseline:** `intrinsic_dim/intrinsic_dim_twonn.py`
- **Layer sweep:** `intrinsic_dim/intrinsic_dim_layer_sweep.py`
- **Gaussian baseline:** `intrinsic_dim/intrinsic_dim_gaussian_baseline.py`
- **Robustness:** `intrinsic_dim/intrinsic_dim_robustness.py`
- **Across data sources:** `intrinsic_dim/intrinsic_dim_data_sources.py`
- **Jacobian-matched ID:** `intrinsic_dim/intrinsic_dim_jac_filtered.py`, `intrinsic_dim_jac_matched.py`, `intrinsic_dim_jac_optimized.py`
- **TwoNN vs n:** `intrinsic_dim/twonn_vs_n.py`, `twonn_vs_n_traincache.py`, `twonn_append.py`, `twonn_append_100k.py`
- **Cluster ID:** `intrinsic_dim/cluster_intrinsic_dim.py`, `cluster_intrinsic_dim_sweep.py`, `cluster_id_n2000.py`
- **Collect layer activations:** `intrinsic_dim/collect_layer10_activations.py`
- **Plot ID vs cluster size:** `intrinsic_dim/plot_id_vs_cluster_size_l10.py`

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
