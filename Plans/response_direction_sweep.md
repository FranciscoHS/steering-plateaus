# Response-Direction Perturbation Sweep

## Question
Standard perturbation experiments in this repo perturb along a direction at layer L and
measure the **magnitude** of the downstream effect (L2 distance at a later layer). Here we
instead ask about the **direction** of the downstream response: as we increase the
perturbation magnitude, does the direction of the response stay fixed, or does it rotate?

Define the downstream response vector at a capture layer:
    r(eps) = resid_capture(act + eps·d) - resid_capture(act)
Baseline = response direction at the smallest perturbation (linear regime):
    b = r(eps_min) / ||r(eps_min)||
Track:
    cos(eps) = <r(eps), b> / ||r(eps)||      (direction stability)
    ||r(eps)||                                 (magnitude, recorded for free)

cos≈1 → response direction is stable as we push harder; a drop → the response rotates
(nonlinearity / saturation onset).

## Conditions
- **Contrastive**: 10 language directions from
  `results/allpairs_contrastive/qwen3-1.7b/directions.pkl` (layer 14, d=2048, unit-norm),
  applied to every activation.
- **Random**: n_dirs isotropic unit vectors sampled per activation.

## Method (reuses run_perturbation_sweep.py machinery)
- Model: qwen3-1.7b, perturb at layer 14.
- Activations: FineWeb fixed-length (seq_len=5), last-token resid at layer 14.
- Capture layer: final block by default (forward_from_layer_to_final); configurable
  (forward_from_layer_to_layer) via --capture_layer.
- Magnitudes: log-spaced relative to mean act norm, mag_min=1e-3, mag_max=8, n_mags=30.
  Baseline at mag_min.
- For each (activation, direction): compute r(eps) at all magnitudes, then cos(eps) vs
  baseline and ||r(eps)||.

## Aggregation / plots
- Per (activation × direction) cos(eps) curve; median + IQR over the pool, one line per
  condition. Optional per-language breakdown for contrastive.
- Companion panel: ||r(eps)|| vs eps (median/IQR).
- Save results pkl + png to results/response_direction_sweep/qwen3-1.7b/.

## Defaults
- capture_layer = final, mag_min=1e-3, mag_max=8, n_mags=30, seq_len=5, n_act=50,
  n_dirs=10 (random), seed=42.
- Runs on RunPod GPU pod (real Qwen3 inference).
