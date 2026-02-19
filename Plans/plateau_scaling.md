# Plan: Plateau scaling with composite feature directions

## Context
We have 10 recovered language feature directions from the all-pairwise contrastive experiment (`results/allpairs_contrastive/qwen3-0.6b/directions.pkl`). The error correction hypothesis predicts that the model can correct for perturbations along feature directions. When perturbing along the normalized sum of n features, each individual feature only receives ~θ/√n of perturbation (where θ is the total perturbation angle). The model can independently correct each feature's small perturbation, so the plateau extends further:

- Perturb along 1 feature: each feature gets full θ perturbation → plateau up to θ₁
- Perturb along sum of n features (normalized): each feature gets ~θ/√n → plateau up to √n × θ₁
- **Prediction: plateau size grows as √n**

Verifying this scaling would be strong evidence for an error-correcting structure in the residual stream.

## Measurement: plateau size
**Threshold-crossing:** plateau_size(direction) = angle (degrees) at which mean KL divergence first exceeds 0.5.

From existing data at selected angles (positive single features):
- 12°: KL ≈ 0.03–0.09 (random ≈ 0.023)
- 21°: KL ≈ 0.12–1.14 (random ≈ 0.076)
- 30°: KL ≈ 0.34–3.0 (random ≈ 0.182)

So KL > 0.5 plateau sizes for single features are roughly 17–30°.

## Script
`scripts/directions/run_plateau_scaling.py`

## Flow

### 1. Load existing data
- Load 10 normalized feature directions from `results/allpairs_contrastive/qwen3-0.6b/directions.pkl`
- Load model (Qwen3-0.6B), FineWeb activations (50 seqs, seq_len=5, seed=42)
- Compute reference logits and layer-26 hidden states

### 2. Generate composite directions
For each n from 1 to 10:
- Generate combinations of n features from the 10 available
- n=1: all 10; n=2: all 45; n=8: all 45; n=9: all 10; n=10: 1
- n=3..7: sample 45 random combinations each (seed=0 for reproducibility)
- For each combination: d = Σ f̂_i / ||Σ f̂_i|| (sum and normalize)
- Total: ~336 composite directions

### 3. Perturbation sweeps
- Sweep each composite direction using existing `sweep_single_direction` infrastructure
- Also reuse random baseline from `results/allpairs_contrastive/qwen3-0.6b/sweep_results.pkl`
- Angles: np.linspace(3, 180, 60), as before
- Estimated runtime: ~336 × 15s ≈ 85 minutes on RTX 4090

### 4. Extract plateau sizes
For each composite direction:
- Compute mean KL over 50 activations at each angle
- Interpolate to find the angle where mean KL first crosses 0.5
- If threshold never crossed within 60°, record 60° (right-censored)

### 5. Aggregate and fit
For each n:
- Collect plateau sizes across all combinations of that n
- Compute mean ± std
- Fit plateau_size(n) = a * n^α to estimate α (prediction: α ≈ 0.5)

### 6. Save
Save to `results/allpairs_contrastive/qwen3-0.6b/`:
- `plateau_scaling.pkl`: all sweep results, plateau sizes, fits
- Checkpoint after each n value (in case of interruption)

### 7. Plot
**Plot 1: KL curves by n** — For each n, show the mean KL curve (averaged over all combinations of that n), colored by n (1–10 colorscale). Plus random baseline. Display clipped to 60°.

**Plot 2: Plateau size vs n** — Main result:
- Points: individual combination plateau sizes (small dots, jittered)
- Thick line + error bars: mean ± std per n
- Dashed line: best-fit a * n^α
- Annotate fitted α
- Also show on log-log axes (slope should be +0.5 if prediction holds)

## Key files to reuse
- `results/allpairs_contrastive/qwen3-0.6b/directions.pkl`: 10 feature directions
- `results/allpairs_contrastive/qwen3-0.6b/sweep_results.pkl`: random baseline (reuse, don't re-sweep)
- Helper functions copied from `scripts/directions/run_allpairs_contrastive.py`: `forward_with_measure`, `build_perturbed_hidden`, `sweep_single_direction`, `extract_context_and_activation`
- `src/model.py`, `src/data.py`

## Run
- Use existing pod (l5w6vv9u1p44qw, ~5.5h remaining)
- Sync script, run, pull results back
- ~85 min runtime fits comfortably

## Verification
- n=1 results should match single-feature sweeps from previous experiment
- Plateau sizes should increase with n
- Log-log slope should be near +0.5 if √n prediction holds
