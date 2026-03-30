# Plan: Random Direction Baseline for Jacobian Reconstruction

## Goal
Run the same Jacobian-minimizing reconstruction but with random orthonormal directions instead of MELBO directions. This is a control experiment to test whether the MELBO directions are special for reconstruction.

## What I'll do

1. **Modify `scripts/jacobian/reconstruct_min_jacobian.py`** to add a `--random_directions` flag
   - When set: instead of loading directions from the pkl, generate `n_dirs` random orthonormal directions (via QR decomposition of a random matrix)
   - `n_dirs` taken from the pkl (31 in this case) so the comparison is fair
   - Everything else identical: same layer, same context, same activation for norm target, same hyperparameters

2. **Run on the pod** with:
   ```
   python scripts/jacobian/reconstruct_min_jacobian.py \
     --direction_pkl results/iterative_melbo_hf/qwen3-1.7b_L2_meas-2_angle0.1_act0/results.pkl \
     --layer 2 --seq_len 1 --random_directions \
     --results_dir results/reconstruct_min_jacobian/qwen3-1.7b_L2_31dirs_random
   ```

3. **Compare** Jacobian norms: MELBO reconstruction (60.7) vs random reconstruction vs real (72.7) vs random point (51.6)

## Files to modify
- `scripts/jacobian/reconstruct_min_jacobian.py`: add `--random_directions` flag, generate random orthonormal directions when set

## Expected output
- Same format as before: Jacobian norms, entropy, top tokens, coefficients, plot
- Key comparison: can random directions achieve the same Jacobian norm as MELBO directions?
