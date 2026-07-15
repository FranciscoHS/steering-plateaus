# Interpolation neuron-flip experiment

## Question
Does a single activation plateau correspond to a **concept**, or is it just a region
of activation space **without neuron flips**? If a plateau is "merely" a flip-free
region, then interpolating between two *synonymous* prompts (same concept) might still
cross many neuron flips; conversely a flip-free stretch might span multiple concepts.

## Setup (merges two existing scripts)
- `scripts/paths/interpolation_experiment.py` — slerp path between two prompts'
  last-token activations at inject layer `L` (slerp direction + linear magnitude,
  prefix held fixed), measuring relative distance in final-logit space.
- `scripts/jacobian/run_gate_switches.py` — counts MLP gate active↔inactive flips at
  downstream layers (cumulative vs start, and per-step vs previous point).

New script: `scripts/paths/interpolation_neuron_flips.py`.

## Design decisions (all confirmed with Francisco)
1. **Flip = sign of MLP pre-activation** (`blocks.{l}.mlp.hook_pre`, the input to the
   nonlinearity / SwiGLU gate). Threshold τ, default 0, plus small-τ sweep (0.1).
2. **Layers:** all intervening MLP layers `L+1 … measure_layer`, reported per-layer
   and summed.
3. **Cumulative + per-step** both. Per-step is the headline ("is this sub-region
   flip-free?"), overlaid on the logit S-curve.
4. **Model:** qwen3-0.6b, inject at a middle layer (~14 of 28).
5. **Prompts:** prefix "The house was ___": `big` (anchor, t=0), `large` (synonym arm),
   `in` (known-boundary control). Two interpolations: big→large and big→in. Assert
   each final word is a single token (lengths must match) and bail clearly otherwise.
6. **Resolution:** ~300 steps in t.

## Outputs (results/interpolation_flips/)
- Per pair: `flips_{pair}_{model}_L{L}.pt` with t_values, rel_distances (logit space),
  per-step & cumulative flip fractions per layer + summed, for each τ.
- Plot: per-step flip rate vs t (total + per-layer) overlaid with logit rel-distance.

## Interpretation
- big→in expected: a flip "wall" near the logit-space boundary.
- big→large is the real test: flip-free plateau (→ plateau ≈ concept) or flips anyway
  (→ plateau is just a flip-free region, not concept-aligned). Cross-check the boundary
  location against `scripts/boundary/run_boundary_interpolation.py` if useful.

## Run (on RunPod GPU, not locally)
    python scripts/paths/interpolation_neuron_flips.py \
        --model qwen3-0.6b --layer 14 --n_steps 300
