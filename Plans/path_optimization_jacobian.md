# Path Optimization to Minimize Jacobian Norm

## Context

We want to find paths in activation space (layer 2 of Qwen3-0.6B) that stay within low-Jacobian-norm regions — i.e., paths that avoid boundary regions between plateaus. The idea: instead of naively slerping between two points (which cuts through boundaries), optimize a flexible path (parameterized by knots) that minimizes the Jacobian norm along the way while keeping the path short.

## Step 1: Verify plateau structure at layer 2

**Goal:** Confirm that slerp interpolation between "The house was big" and "The house was in" shows clear plateau->boundary->plateau structure when interpolating at layer 2 (the existing run used layer 1, which showed abnormal behavior).

**Implementation:** Re-run the existing `scripts/interpolation_experiment.py` with `--layer 2`. Just layer 2, nothing else.

```bash
python scripts/interpolation_experiment.py --model qwen3-0.6b --layer 2
```

**Show results to Francisco before proceeding to Step 2.**

## Step 2: Path optimization experiment

### 2a. New utility: `forward_from_layer_to_layer`

**File:** `src/model.py`

The existing `forward_from_layer_to_final` runs through all remaining blocks. We need a variant that stops at a specific layer (the second-to-last). Minor variation of the existing function (~20 lines).

```python
def forward_from_layer_to_layer(model, context, point, start_layer, end_layer):
    """Forward from start_layer to end_layer (inclusive). Returns last-token residual stream.
    Runs blocks start_layer+1 through end_layer."""
```

**Layer numbering note:** Qwen3-0.6B has `num_hidden_layers` blocks (0-indexed). We'll compute `end_layer = n_layers - 2` at runtime (second-to-last block). If the model has 28 layers (0-27), that's layer 26. If it has more, it adjusts automatically.

### 2b. Differentiable Jacobian norm — reuse existing pattern

The `optimization_step` function in `scripts/run_delta_optimization.py` (lines 108-156) already implements the exact pattern we need:
- `forward_from_layer_to_final` for forward pass
- `torch.autograd.grad(s, variable, create_graph=True, retain_graph=True)` for differentiable VJP
- Hutchinson probe accumulation
- Returns differentiable loss

We'll adapt this pattern directly in the new script (swapping `forward_from_layer_to_final` for `forward_from_layer_to_layer`, and differentiating w.r.t. knot positions instead of a delta vector). **No new function in `src/analysis.py` needed.**

### 2c. Path optimization script

**File:** `scripts/path_optimization_experiment.py` (new)

#### Path parameterization
- `N_knots = 10` free knot points in layer 2 activation space (d_model-dimensional each)
- Plus 2 fixed endpoints (extracted from the two prompts at layer 2)
- 12 waypoints total, 11 segments
- Spherical interpolation (slerp with linear magnitude) between consecutive waypoints
- Initialize knots along the slerp path between endpoints (our baseline)

#### Sampling points along the path
- For each optimization step, sample random points along the path
- Each sample: pick a random segment, pick a random t in [0, 1] within that segment, slerp to get the point
- ~2-3 points per segment for Jacobian computation = ~22-33 points per step
- Start with `n_jac_samples = 15` per step (subsample for speed)

#### Loss function
```
L = softmax_jac_loss + (path_length / L_baseline)
```
Where:
- `softmax_jac_loss`: `(1/alpha) * log(mean(exp(alpha * ||J||_F^2_i / J_baseline)))` — a soft-max over sampled Jacobian norms
- `alpha`: soft-max temperature parameter (default: 3). Controls interpolation between mean (alpha→0) and max (alpha→∞)
- `J_baseline`: mean ||J||_F^2 along the slerp path (computed once before optimization)
- `path_length`: sum of ||k_{i+1} - k_i|| for consecutive waypoints (chord-length approx)
- `L_baseline`: chord-length of the slerp path (computed once)
- Both terms normalized to ~1 at initialization, giving equal effective weight

**Why soft-max instead of mean?** The first run (v1) used `mean(||J||_F^2)` as the Jacobian loss. This successfully reduced the *average* Jacobian norm (~120 vs ~400 for slerp) but left a sharp spike at the boundary crossing (||J||_F ≈ 1011) — the optimizer doesn't mind one bad point if everything else is tiny. What we actually want is *stability*: keeping the worst-case Jacobian norm low everywhere. The log-sum-exp soft-max approximates `max(||J||_F^2)` while remaining smooth and differentiable, so it penalizes the worst points much more heavily than the mean does.

#### Optimization
- Optimizer: Adam, lr=0.01
- Steps: 300 (with convergence monitoring)
- Boundary conditions: endpoints fixed (not optimized)
- SDPA disabled during Jacobian computation (per existing pattern in delta optimization)
- Gradient clipping if needed (second-order gradients can be noisy)

#### Checkpointing
- Save every 50 steps (knot positions, loss history, baselines)
- Can resume from checkpoint

#### Script structure
```
1. Load model (qwen3-0.6b via HuggingFace, src/model.py)
2. Extract layer 2 activations for both prompts
   - Use extract_activations or just run model with output_hidden_states
3. Compute slerp baseline:
   a. Slerp path between endpoints (use src/paths.py slerp_path)
   b. Measure Jacobian norm at ~20 points along slerp path -> J_baseline
   c. Compute slerp path chord-length -> L_baseline
4. Initialize 10 knots along slerp path
5. Optimization loop (300 steps):
   a. Sample random points along current path (slerp between consecutive knots)
   b. Compute differentiable Jacobian norm at sampled points (Hutchinson VJP, 5 probes)
   c. Compute path length (chord-length between consecutive waypoints)
   d. Combined normalized loss, backward, Adam step
   e. Fix endpoints
   f. Log & checkpoint
6. Save final results
```

### 2d. Hyperparameters (all configurable via argparse)
- `--n_knots`: 10
- `--n_probes`: 5 (Hutchinson probes per Jacobian estimate during optimization)
- `--n_jac_samples`: 15 (points sampled per step for Jacobian)
- `--lr`: 0.01
- `--n_steps`: 300
- `--start_layer`: 2
- `--prompt_a`, `--prompt_b`: "The house was big", "The house was in"
- `--n_probes_eval`: 50 (Hutchinson probes for post-optimization analysis)
- `--results_dir`: results/path_optimization

## Step 3: Analysis of optimized path

After optimization, compute and plot (all in the same script):

1. **Jacobian norm along the path** (dense, ~100 points, 50 Hutchinson probes)
   - Optimized path vs slerp baseline, side by side
2. **Path length comparison** — optimized vs slerp
3. **Logit entropy along the path** (via existing `compute_logit_entropy`)
4. **Top predicted tokens** at evenly-spaced points (via existing `get_top_k_predictions`)
5. **Optimization loss curve** — convergence, relative contributions of the two terms

All plots saved to `results/path_optimization/`.

## Files to create/modify
- **Modify** `src/model.py`: Add `forward_from_layer_to_layer` (~20 lines, minor variant of `forward_from_layer_to_final`)
- **Create** `scripts/path_optimization_experiment.py`: Main experiment script (~350 lines)

## Existing code to reuse
- `src/model.py`: `load_model`, `extract_activations`, `forward_from_layer` (for logits in analysis)
- `src/paths.py`: `slerp_path` (initialization + baseline), `compute_path_length`
- `src/analysis.py`: `compute_logit_entropy`, `get_top_k_predictions`, `compute_jacobian_norm_vjp` (for post-optimization eval)
- `scripts/run_delta_optimization.py`: Differentiable Jacobian VJP pattern (lines 108-156), checkpoint/resume pattern
- `scripts/interpolation_experiment.py`: `slerp_with_linear_magnitude` (for inter-knot interpolation)

## Verification
1. Step 1: visual inspection of interpolation plot — expect S-curve at layer 2
2. Step 2: loss should decrease during optimization; both terms should decrease or trade off
3. Step 3: optimized path should show lower Jacobian norm than slerp; path should be modestly longer; predictions should remain sensible

## Things to watch
- **Computational cost**: Each step has ~15 Jacobian estimates x 5 probes = 75 VJP+grad operations through ~24 transformer blocks with second-order gradients. Should be feasible on A100 but may take a few minutes total. If too slow, reduce `n_jac_samples` first.
- **Gradient stability**: Second-order gradients through many transformer blocks can be noisy. May need gradient clipping or lower lr.
- **Layer count**: Will verify Qwen3-0.6B layer count at runtime and use `n_layers - 2` for second-to-last.
