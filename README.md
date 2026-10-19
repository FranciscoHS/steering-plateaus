# Jacobian-minimized activations: generation and perturbation sweeps

This is a small extract of a larger research repo. We sample random points in a model's
residual-stream activation space and run gradient descent on them to minimize the
Jacobian norm of the rest of the network. We then run perturbation sweeps that compare
how the network responds near these points with how it responds near real and random
activations.

Default setting: Qwen3-1.7B, points at layer 2 (last token), Jacobian taken from layer 2
to the final block.

## Pipeline

Run everything from the repo root.

1. `python scripts/1_sample_real_and_gaussian.py`
   - Collects 5000 real last-token activations at layer 2 (FineWeb-Edu, 10 tokens).
   - Samples 20000 Gaussian points with the same mean and covariance.
   - Estimates ||J||_F for both sets (Hutchinson estimator, finite differences).
   - Writes `results/intrinsic_dim/jac_filtered_data.pt`.
   - The script also runs a TWO-NN intrinsic-dimension comparison, which was its original purpose. It isn't needed for the rest of the pipeline.
2. `python scripts/2_minimize_jacobian.py`
   - Takes the first 200 Gaussian points, each patched into a real FineWeb context.
   - Runs gradient descent (Adam, 200 steps, lr 0.1) to minimize ||J||_F + λ||x − x_start||² with λ = 0.001.
   - Writes `results/intrinsic_dim/jac_optimized_data.pt`. Key `X_opt` holds the [200, 2048] optimized points; `X_gauss_subset` holds their starting points.
3. `python scripts/3_perturbation_sweep.py --model qwen3-1.7b --layer 2 --n_act 30 --seq_len 5 --max_angle 90 --n_steps 46 --n_dirs 10 --extra_anchors_pt results/intrinsic_dim/jac_optimized_data.pt --extra_anchors_key X_opt --extra_label jac-optimized --out results/real_vs_random_acts/qwen3-1.7b_L2_jacopt`
   - Rotates each anchor along random directions on its norm sphere and records the L2 change of the residual stream at a later layer, as a function of angle.
   - Conditions: real activations, random activations (`--rand_mode norm` or `cov`), and the Jacobian-optimized points.
   - All conditions share the same contexts and the same random directions.

## Results included

- `results/intrinsic_dim/jac_optimized_data.pt`: output of step 2. It holds the 200 optimized points (`X_opt`), their starting points (`X_gauss_subset`), the loss history, and TWO-NN ID before and after optimization.
- `results/real_vs_random_acts/qwen3-1.7b_L2_jacopt/`: output of step 3 (`results.pkl`, `sweep.png`), produced by the exact command above. It compares 30 anchors x 10 random directions, rotated 0-90 degrees, with L2 measured at layer 26. The random baseline is isotropic and norm-matched per anchor (`--rand_mode norm`, the default).
- Not included: `jac_filtered_data.pt` (step 1 output, 176 MB). Step 1 regenerates it, or ask us for a copy.

## Caveat

The contexts used during optimization (step 2) are not saved. In step 3, optimized point
i is evaluated in FineWeb context i of that run, which is generally not the context it was
optimized in.

## Code

- `src/model.py`: model loading and partial forward passes (run from layer L onward).
- `src/data.py`: FineWeb loading and covariance-matched random sampling.
