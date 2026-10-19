# Do Jacobian-minimized random activations sit on plateaus?

## Motivation

We study **activation plateaus**. Real (in-distribution) activations are resistant to
perturbation: if you move a real residual-stream activation a little, the model's later
activations and outputs barely change, until the perturbation is large enough to cross
some boundary. Randomly sampled activations don't behave this way. This raises the
hypothesis that plateaus are **necessary and sufficient** for an activation being "real".

We have also observed that real activations generally have a lower Jacobian norm
||J||_F (of the rest of the network with respect to the activation) than randomly
sampled ones. So we asked: **if we take randomly sampled activations and minimize their
Jacobian norm, do we get plateaus?**

## Result

Apparently yes. In perturbation sweeps, the Jacobian-minimized points respond like real
activations, not like random ones. This calls into question the "sufficient" part of the
hypothesis. Another explanation remains possible: the optimization may have just pushed
the points into plateaus that genuinely exist.

Median L2 change at the measurement layer (Qwen3-1.7B, perturbing at layer 2, measuring
at layer 26; 30 anchors x 10 random directions):

- 10°: real 90.6, random 802.7, jac-opt 47.7
- 30°: real 298.2, random 1254.0, jac-opt 161.8
- 90°: real 1920.4, random 1442.5, jac-opt 1849.6

Plot: `results/real_vs_random_acts/qwen3-1.7b_L2_jacopt/sweep.png`

### Sanity checks

These check that the optimization didn't do something trivial (200 points per set):

- **No blown-up coordinates.** This is measured by the participation ratio over standard-basis coordinates: 1 means one-hot, 2048 means perfectly uniform. The medians are real 37.2, cov-matched Gaussian 50.7 and jac-opt 156.8, so the optimized points are *more* spread out, not concentrated on a few coordinates.
  Plot: `results/intrinsic_dim/participation_ratio/pr_hist.png`
- **No collapse onto a single point.** These are pairwise L2 distances within each set. The medians are real 34.9, Gaussian 35.6 and jac-opt 70.2. Note that the jac-opt points have about 1.9x the mean norm of real activations (54.7 vs 29.3). After rescaling them to the real mean norm, the jac-opt median is 37.6 (min 27.7), comparable to the other sets.
  Plots: `results/intrinsic_dim/pairwise_dist/` and `results/intrinsic_dim/pairwise_dist_rescaled/`

## Setup

Qwen3-1.7B. The activations are the last-token residual stream after layer 2. ||J||_F is
the Frobenius norm of the Jacobian from that activation to the final block's last-token
output, estimated with Hutchinson's estimator.

The optimization starts from Gaussian samples with the same mean and covariance as real
activations. It runs Adam for 200 steps (lr 0.1) on ||J||_F + λ||x − x_start||², with
λ = 0.001. Each point is patched into the last position of a real 10-token FineWeb-Edu
context.

## Reproducing

```bash
pip install -r requirements.txt
# run everything from the repo root; steps 1-3 need a GPU
python scripts/1_sample_real_and_gaussian.py      # real acts + cov-matched Gaussian pool
python scripts/2_minimize_jacobian.py             # minimize ||J||_F on 200 Gaussian points
python scripts/3_perturbation_sweep.py --model qwen3-1.7b --layer 2 --n_act 30 --seq_len 5 \
    --max_angle 90 --n_steps 46 --n_dirs 10 \
    --extra_anchors_pt results/intrinsic_dim/jac_optimized_data.pt \
    --extra_anchors_key X_opt --extra_label jac-optimized \
    --out results/real_vs_random_acts/qwen3-1.7b_L2_jacopt
python scripts/4_check_participation_ratio.py     # sanity: blown-up coordinates
python scripts/5_check_pairwise_distances.py      # sanity: collapse
python scripts/5_check_pairwise_distances.py --rescale
```

What step 3 does: each anchor a is rotated on the sphere of radius ||a|| towards a random
direction by 0–90°. The script records the L2 distance between the perturbed and
unperturbed last-token residual at the measurement layer. The random baseline is
isotropic, scaled to each real anchor's norm (`--rand_mode norm`, the default). All three
conditions share the same contexts and directions.

## Caveat

The contexts used during optimization are not saved. In step 3, optimized point i is
evaluated in FineWeb context i of that run, which is generally not the context it was
optimized in.

## Data

- `results/intrinsic_dim/jac_optimized_data.pt` (step 2 output) contains:
  - `X_opt` [200, 2048]: the optimized points.
  - `X_gauss_subset` [200, 2048]: their starting points.
  - `history`: the loss curves.
  - `id_before` / `id_after`: TWO-NN intrinsic dimension of the points before and after optimization.
- `results/real_vs_random_acts/qwen3-1.7b_L2_jacopt/results.pkl` (step 3 output) contains:
  - `real_l2`, `rand_l2`, `extra_l2` (jac-opt): arrays of shape [30 anchors, 10 directions, 46 angles].
  - `angles_deg`: the perturbation angles.
  - `args`: the arguments of the run.
- `results/intrinsic_dim/{participation_ratio,pairwise_dist,pairwise_dist_rescaled}/`: sanity-check outputs (`results.pkl` plus a plot each).
- Not included: `results/intrinsic_dim/jac_filtered_data.pt` (step 1 output, 176 MB, over
  GitHub's file limit). It holds the 5000 real activations, the 20000-point Gaussian
  pool and their Jacobian norms. Steps 2, 4 and 5 read it. Step 1 regenerates it,
  or ask us for a copy.

## Code

- `scripts/`: the pipeline above, numbered in run order.
- `src/model.py`: model loading and partial forward passes (run from layer L onward).
- `src/data.py`: FineWeb loading and covariance-matched random sampling.
