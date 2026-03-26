# Plan: Reconstruct Activation from MELBO Directions via Jacobian Minimization

## Context

We found 27 highly reproducible sensitive directions at layer 7 of Qwen3-1.7B via iterative MELBO at 0.1° perturbation. These directions are identical across seeds and across additive/spherical optimization — they appear to be genuine structural features of the model at this activation.

The question: can we reconstruct a "realistic" activation using just these 27 directions? If they are active features, a linear combination should produce an activation the model treats as on-distribution. We operationalize "realistic" as having low Jacobian norm (= being on the plateau = insensitive to small perturbations), since that's the signature of real activations.

## Approach

**Optimize 27 coefficients** $c_1, \ldots, c_{27}$ to minimize the Jacobian Frobenius norm at $a = Dc$, with a norm penalty.

**Objective**: $\min_c \|J_f(Dc)\|_F^2 + \lambda(\|Dc\| - R)^2$

where $f$ maps layer 7 → final layer, $D$ is the $[d_\text{model}, 27]$ matrix of MELBO directions, $R$ is the target norm (~62).

**Hutchinson estimator via VJP** (same approach as `run_delta_optimization_v8.py`):
1. Construct $a = Dc$
2. Forward pass: `final_act = forward_from_layer_to_final(model, ctx, a, layer)`
3. For each of $K$ probes: sample $u \sim \mathcal{N}(0, I)$, compute $g_a = \nabla_a(f(a) \cdot u)$ via `autograd.grad(s, a, create_graph=True)`
4. $\|J\|_F^2 \approx \frac{1}{K}\sum_k \|g_{a,k}\|^2$
5. `loss.backward()` flows second-order gradients back to $c$ through $a = Dc$

## New script

**File**: `scripts/jacobian/reconstruct_min_jacobian.py`

**Args**:
- `--direction_pkl`: path to MELBO results.pkl (required)
- `--model`, `--layer`: model and injection layer (default from pkl args)
- `--n_probes`: Hutchinson probes per step (default 10)
- `--n_steps`: optimization steps (default 300)
- `--n_restarts`: parallel random initializations of c (default 5)
- `--lr`: learning rate (default 0.01)
- `--norm_lambda`: weight of norm penalty (default 1.0)
- `--results_dir`: output directory

**Reuse from existing code**:
- `src/model.py`: `load_model`, `forward_from_layer_to_final`, `_get_blocks`
- `src/data.py`: `load_fineweb_fixed_length`
- `src/analysis.py`: `compute_jacobian_norm_vjp` (for evaluation, not optimization)
- Pattern from `scripts/jacobian/run_delta_optimization_v8.py` lines 131-148: VJP Hutchinson with `create_graph=True`
- Pattern from `scripts/melbo/run_iterative_melbo_hf.py`: activation extraction, context setup

**Algorithm**:
```
1. Load model, extract activation + context from FineWeb (same act_idx=0 as MELBO)
2. Load 27 MELBO directions, build D [d_model, 27]
3. Compute R_target = mean activation norm at this layer
4. Compute reference Jacobian norm at the real activation (for comparison)
5. Initialize c as [n_restarts, 27] random, normalized so ||Dc|| ≈ R_target
6. For each step:
   a. a = D @ c  → [n_restarts, d_model]  (process one restart at a time for memory)
   b. For each restart:
      - Forward: final_act = forward_from_layer_to_final(model, ctx, a_i, layer)
      - For k in n_probes:
        u = randn_like(final_act)
        s = (final_act * u).sum()
        g_a = autograd.grad(s, a_i, create_graph=True, retain_graph=True)
        frob_sq += ||g_a||²
      - frob_sq /= n_probes
      - norm_loss = (||a_i|| - R_target)²
      - loss = frob_sq + norm_lambda * norm_loss
      - loss.backward()
   c. optimizer.step()
   d. Log: step, frob_norm, activation_norm, loss
7. Pick best restart (lowest final Jacobian norm)
8. Evaluate: Jacobian norm, perturbation sweep, logit entropy, comparison to real activation
```

**Evaluation / output**:
- Print Jacobian norm of: real activation, reconstructed activation, random activation of same norm
- Save reconstructed activation, coefficients, optimization trajectory
- Run perturbation sweep on reconstructed activation (reuse `sweep_dir` from MELBO script or inline)
- Save plot comparing perturbation profiles

## Key files to modify/create
- **Create**: `scripts/jacobian/reconstruct_min_jacobian.py`
- **Read only**: `src/model.py`, `src/analysis.py`, `scripts/jacobian/run_delta_optimization_v8.py` (pattern reference)

## Verification
1. Run on pod with: `python scripts/jacobian/reconstruct_min_jacobian.py --direction_pkl results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_angle0.1_act0/results.pkl`
2. Check that Jacobian norm decreases during optimization
3. Compare reconstructed vs real vs random Jacobian norms
4. Inspect perturbation sweep profile of reconstructed activation
