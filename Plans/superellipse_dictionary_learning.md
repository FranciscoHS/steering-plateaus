# Learning a superellipse direction dictionary

Goal: discover pairs (and eventually a dictionary) of unit directions in
activation space whose response geometry is super-additive in the
superellipse sense — the model is *less* responsive along the 50/50 mixture
of two directions than along either direction alone (the p>2 regime of
`R(v) = F(Σ w_i |d_i·v|^n)`). We do this by directly optimizing the
directions to minimize a proxy loss built from measured downstream
L² responses.

## What already exists (reuse, don't rebuild)

- `src/model.py`
  - `load_model(name, dtype)` → model, tokenizer, device.
  - `forward_from_layer_to_final(model, ctx, point, layer)` — differentiable
    forward from `layer` to the final residual stream, last token. `point` is
    `[n, 1, d]`, returns `[n, d]`. This is our response map f(·).
  - `forward_from_layer_to_layer(...)` — same but stops at an intermediate
    `measure_layer` (optional, for measuring response earlier).
  - we are going to measure at 2nd-to-last layer, so this is the one we want
  - `_get_blocks`, `_get_final_norm`, `_get_lm_head`.
- `scripts/delta_optimization/per_activation_opt_v9.py`
  - `spherical_perturb(a, direction, angle_rad)` — **exactly the building
    block we need.** Differentiable w.r.t. `direction`; orthogonalizes
    `direction` against `a`, renormalizes, returns `cos·a + sin·R·v_perp`.
    This keeps the perturbation norm-matched on the constant-norm sphere.
  - The Adam + renormalize-each-step pattern = Riemannian SGD on the sphere
    (renormalization is the retraction; explicit tangent projection is
    optional since Adam step + renorm already keeps us on S^{d-1}).
- `scripts/perturbation/run_perturbation_sweep_spherical.py`
  - `extract_context_and_activation(model, tokenizer, token_ids_list, layer, ...)`
    → contexts (prefix resid) + final-token activations. Use to get the ~10
    activations.
- `src/data.py`: `load_fineweb_fixed_length(n, tokenizer, seq_len, seed, ...)`
  for the prompt source.
- `superellipse_derivation.tex` — the theory the proxy is meant to track.

## New code

New dir `scripts/superellipse/`, script `learn_superellipse_dict.py`.
Results → `results/superellipse_dict/<model>_L<layer>_a<angle>/`.

### Core response measurement (per activation, per direction)

```
R_i(a) = || f(spherical_perturb(a, d_i, α)) − f(a) ||_2
```
where `f = forward_from_layer_to_final` (or `..._to_layer` if measure_layer set).
- Reference `f(a)` does **not** depend on d₁,d₂ → precompute once per activation
  (no grad).
- For one activation, stack the perturbed points for the 3 conditions
  (d₁, d₂, mix) — and ×|α| if multiple angles — into `point [n_cond, 1, d]`
  and call `f` once → `[n_cond, d]`. Loop over the ~10 activations.
- First-order grad only (no Jacobian/double-backward), so no `create_graph`
  and no need for the SDPA-MATH kernel guard that v9 uses.

### The mix direction

Per activation a, in the tangent space at a:
1. orthogonalize d₁ against a, normalize → d₁⊥
2. orthogonalize d₂ against a, normalize → d₂⊥
3. mix = normalize(d₁⊥ + d₂⊥)  (the bisector of the two tangent directions)

`spherical_perturb` does the orthogonalize-against-a step internally, so in
practice we can pass d₁, d₂, and `normalize(d₁+d₂)` straight in; the bisector
is cleanest when d₁⊥d₂ (see open question on the global orthogonality
constraint).

### Loss

Per activation: `L_a = R_mix / (R₁ + R₂)`.
Total: `L = mean_a L_a`, also averaged over α if multiple angles.
Minimize L. (L=0.5 is the no-interference / d₁≈d₂ baseline; L<0.5 is the
superellipse/p>2 regime we want; L>0.5 is sub-additive.)

### Stage 1 — learn the first pair (d₁, d₂)

- Params: d₁, d₂ ∈ R^d as `requires_grad` unit vectors (random init).
- Adam (lr≈3e-3, matching v9), N steps (~100–300), renormalize both after
  each `optimizer.step()`.
- Optional: a few random restarts, keep lowest L.
- Log per step: L, mean R₁, R₂, R_mix, and d₁·d₂.

### Stage 2 — grow the dictionary (d₃, d₄, …)

To add direction d_k with d₁..d_{k-1} **frozen**:
- Loss = mean over existing j<k of pair-loss L(d_k, d_j)
  (each existing direction paired against the new one).
- Same Adam+renorm optimization.
- Repeat until ~10 directions.

### Outputs (save to files)

- `results.pkl`: learned directions `[k, d]`, per-step loss curves, final
  per-activation R₁/R₂/R_mix arrays, pairwise Gram matrix d_i·d_j, and the
  per-direction individual responsiveness R_i.
- Plots (`*.png`, paths reported to Francisco — no inline interpretation):
  loss curve; histogram of L_a across activations; bar of mean R₁/R₂/R_mix;
  Gram-matrix heatmap of learned directions.
- Console + saved scalars only; never stdout-only.

## Defaults

- model `Qwen 3-1.7B`, layer 2, FineWeb, seq_len 10 (matches existing sweeps).
- n_act = 10, α = [30°] (single, per Francisco; list-capable).
- Checkpoint directions + loss curve every K steps (resumable like v9).

# extra detail

1. **Global orthogonality of d₁,d₂. We implement soft penalty
   λ(d₁·d₂)² to encourage approximate orthogonalization during training

