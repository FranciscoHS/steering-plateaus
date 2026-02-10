# Delta Optimization Experiment

## Goal

Find a single vector `delta` in R^1280 such that for a large set of real activations `a_i`,
the shifted points `a_i + delta` have low Jacobian norms (similar to real activations).

This tests whether we can find "feature directions" — directions along which you can
move in activation space and land on another plateau. If the optimization succeeds
even at large `||delta||`, that's strong evidence that delta encodes a meaningful
feature direction.

**Proposed by Stefan Heimersheim.**

## Setup

- **Model:** GPT-2 Large (d_model=1280, 36 layers)
- **Layer:** 6 (activation injection point)
- **Jacobian:** Maps layer 6 last-token activation → layer 35 last-token activation
  (pre-ln_f, pre-unembedding — pure residual stream after final transformer block)
- **Activation dimension:** 1280 (last token only)
- **Sequence length:** 5 tokens (for context)

## Approach: Projected Gradient Descent on a Sphere

Fix `||delta|| = r` and optimize only the direction. After each gradient step,
project delta back to the sphere of radius r. This eliminates the delta=0
degeneracy (trivial solution where nothing changes).

Sweep r across values to map out when optimization succeeds vs fails.
Natural scale for r: the typical L2 distance between real activations.

## Loss Function

Minimize mean Jacobian Frobenius norm at shifted activations:

```
L(delta) = (1/N) sum_i ||J(a_i + delta)||_F^2
```

Estimated via Hutchinson:

```
L(delta) ≈ (1/(N*K)) sum_i sum_k ||J^T u_k||^2
```

where u_k ~ N(0, I) in R^1280 are random probe vectors.

## Cross-Context Evaluation

We compare real activations against random activations (and shifted activations).
Random activations don't have a "natural" context — they were never produced by
processing a real sequence. If we evaluate real activations with their own context
but random/shifted ones with an arbitrary context, the comparison is unfair: real
activations get a coherence bonus (all positions at layer 6 were produced together
by the same forward pass) that other categories can't have. Any observed difference
could be this coherence bonus rather than a property of the point's location.

Cross-context levels the playing field: all categories are evaluated with a context
that isn't "theirs." If real activations still show lower Jacobian norms, it's
about the points themselves, not about one category getting a more favorable setup.

Concretely:
- Sample 200 FineWeb sequences, split into two disjoint sets of 100
- Set A (sequences 1-100): extract last-token activations at layer 6 → a_1..a_100
- Set B (sequences 101-200): extract prefix activations at layer 6 → contexts c_1..c_100
- When evaluating J(a_i + delta), patch `a_i + delta` into context c_i

The prefix is positions 0..3 (4 tokens) of the residual stream at layer 6.
The patched point is at position 4 (last token).

## Technical Approach: Double Backward

We need to optimize delta through the Jacobian norm. This requires differentiating
through a derivative — "double backward."

### How it works

1. **Forward pass:** Patch `x_i = a_i + delta` at layer 6 last position, run through
   transformer blocks 7-35 (30 blocks). Get `final_act = hidden[:, -1, :]`.

2. **First backward (create_graph=True):** For random probe `u ~ N(0, I)` in R^1280:
   - Compute `s = final_act · u` (dot product, scalar)
   - `g = torch.autograd.grad(s, delta, create_graph=True, retain_graph=True)[0]`
   - This gives `g = J^T u` (shape [1280])
   - Because of `create_graph=True`, g is itself a differentiable tensor (not just numbers)

3. **Accumulate:** `loss += ||g||^2`. This is our Hutchinson estimate of `||J||_F^2`.

4. **Second backward:** `loss.backward()` differentiates `||J^T u||^2` w.r.t. delta,
   flowing gradients through the first backward. This gives `d(||J^T u||^2)/d(delta)`.

### Why create_graph=True?

Normally, `torch.autograd.grad` computes gradients and then discards the computation
graph. The returned gradient is just a tensor of numbers — you can't differentiate
through it. `create_graph=True` tells PyTorch to record HOW the gradient was computed,
so the gradient itself becomes a differentiable function that you can backprop through.

### SDPA Compatibility

Must use `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` for the forward pass,
because the efficient SDPA backends don't support higher-order gradients.

### Memory Management

Process one activation at a time. For each activation, compute all K probes,
accumulate the loss, call `.backward()` (which frees that activation's graph),
then move to the next activation. This bounds memory to a single activation's
graph (with K probe subgraphs).

```python
optimizer.zero_grad()
for i in mini_batch_indices:
    # Forward for activation i (builds computation graph)
    point = (activations[i] + delta).reshape(1, 1, d_model)
    hidden = torch.cat([contexts[i], point], dim=1)
    with sdpa_kernel(SDPBackend.MATH):
        for block in model.transformer.h[layer + 1:]:
            hidden = block(hidden)[0]
    final_act = hidden[:, -1, :]

    # Probes (each adds a subgraph via create_graph)
    act_loss = 0
    for k in range(n_probes):
        u = torch.randn(d_model, device=device)
        s = (final_act.squeeze() * u).sum()
        g = torch.autograd.grad(s, delta, create_graph=True, retain_graph=True)[0]
        act_loss = act_loss + (g ** 2).sum()

    act_loss = act_loss / (B * K)
    act_loss.backward()  # accumulates into delta.grad, frees graph

# Update and project
optimizer.step()
with torch.no_grad():
    delta.data.mul_(r / delta.data.norm())
```

## Hyperparameters

- **B (mini-batch size):** 10 activations per step (out of 100)
- **K (probes per activation):** 5
- **Optimizer:** Adam, lr=1e-2
- **Steps:** 1000 (with early stopping if loss plateaus)
- **r (delta norm):** Start with mean pairwise distance between real activations

## Estimation Quality

Hutchinson with K=5 probes: ~10-15% relative error on ||J||_F^2 per activation.
Averaging over B=10 activations reduces this further. For SGD, unbiased noisy
gradients are fine — Adam handles noise well. We use 100 probes only for the
final evaluation (not during optimization).

## Cost Estimate (A100)

**Per optimization step (B=10, K=5):**
- 10 activations × (1 forward + 5 first-backwards + 1 second-backward)
- Forward from layer 6 through 30 blocks, seq_len=5: ~10ms
- First backward with create_graph: ~10-20ms per probe
- Second backward: ~10-20ms per activation
- **Total: ~1-3 seconds per step**

**Full optimization (1000 steps):** ~15-50 minutes

**Reference norm computation (100 activations, 100 probes each):** ~10-15 minutes

## Implementation Plan

### Files to create/modify

1. **`src/model.py`** — Add `forward_from_layer_to_final()`
2. **`src/analysis.py`** — Add `compute_jacobian_norm_vjp()`
3. **`scripts/run_delta_optimization.py`** — Main experiment script (NEW)

### src/model.py: New function

```python
def forward_from_layer_to_final(model, context, point, start_layer):
    """
    Forward from layer L to final transformer block output (pre-ln_f, pre-unembedding).

    Returns the residual stream at the final layer, last token only.
    Differentiable w.r.t. point.

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [1, 1, hidden_dim] - Last-token activation (requires grad through delta)
        start_layer: Layer index (0-indexed)

    Returns:
        final_act: [1, hidden_dim] - Last-token residual stream after final block
    """
```

Key difference from `forward_from_layer`: NO `ln_f`, NO `lm_head`.
Same concatenation pattern: `hidden = torch.cat([context, point], dim=1)`.
This is already differentiable — `torch.cat` creates a new tensor, no in-place ops.

### src/analysis.py: New function

```python
def compute_jacobian_norm_vjp(model, context, point, layer, n_probes=100, device='cuda'):
    """
    Estimate ||J||_F for J mapping layer L last-token → final layer last-token.

    Uses VJP (backward-mode AD) with Hutchinson estimator.
    Does NOT use torch.func — just standard autograd.

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim] - Prefix at layer L (detached)
        point: [hidden_dim] tensor - Last-token activation (detached)
        layer: Start layer index
        n_probes: Number of Hutchinson probes

    Returns:
        frobenius_norm: estimated ||J||_F
    """
```

For each probe:
1. Create `point_var = point.clone().requires_grad_(True)`
2. Forward: concat with context, run blocks, extract last-token
3. Random u ~ N(0, I), compute s = final_act · u
4. g = grad(s, point_var) — this is J^T u
5. Accumulate ||g||^2

No create_graph needed here (we're just measuring, not optimizing through it).

### scripts/run_delta_optimization.py: Staged pipeline

**Stage 0: Setup & distance estimation**
1. Load GPT-2 Large
2. Load 200 FineWeb sequences (seq_len=5)
3. Extract activations at layer 6 for all 200 sequences (batch forward pass)
4. Split: last-token activations a_1..a_100, prefix contexts c_1..c_100
5. Compute pairwise L2 distances among a_1..a_100
6. Report distances, set r = mean distance
7. Checkpoint: activations, contexts, distances

**Stage 1: Reference Jacobian norms**
1. For each a_i (i=1..100), compute ||J(a_i)||_F using context c_i
2. Use 100 probes via compute_jacobian_norm_vjp
3. Report mean, std
4. Checkpoint reference norms

**Stage 2: Optimization**
1. Initialize: delta = random_unit_vector * r
2. Adam optimizer, lr=1e-2
3. Loop for n_steps:
   a. Sample B=10 random indices from {0..99}
   b. For each activation i:
      - Forward: patch a_i + delta at layer 6 last position, run to final layer
      - For K=5 probes: compute ||J^T u||^2 via create_graph=True
      - Accumulate into loss, call backward (free graph per activation)
   c. Adam step
   d. Project: delta = delta * (r / ||delta||)
   e. Log every 10 steps: loss, grad norm, direction cosine vs previous
   f. Checkpoint every 100 steps
4. Save final delta

**Stage 3: Evaluation**
1. Compute ||J(a_i + delta_opt)||_F for all 100 activations (100 probes)
2. Compare with Stage 1 reference norms
3. Report mean, std for real vs shifted
4. Save histogram plot

**Stage 4: r sweep (manual)**
Re-run Stage 2-3 with different --r values.
Suggested: r = 0.1, 0.3, 1, 3, 10 times mean_distance.

### Command-line arguments

```
--model         gpt2-large (default)
--layer         6 (default)
--seq_len       5 (default)
--n_act         100 (number of activations)
--n_ctx         100 (number of contexts)
--n_probes_opt  5 (probes during optimization)
--n_probes_eval 100 (probes during evaluation)
--batch_size    10 (activations per optimization step)
--lr            0.01
--n_steps       1000
--r             0 (auto = mean pairwise distance)
--results_dir   results/delta_opt
--resume
--seed          42
```

## What Could Go Wrong

1. **create_graph=True through HuggingFace transformer blocks**: Might fail
   if some internal op doesn't support higher-order gradients. Test early
   (Stage 1 can be modified to test this). Fallback: implement manual forward pass.

2. **SDPA issues**: Must use MATH backend. Already known from existing code.

3. **Learning rate too high/low**: Start with 1e-2, adjust based on loss curve.
   If loss oscillates wildly → decrease lr. If loss barely moves → increase lr.

4. **Optimization finds trivial solution**: Can't happen with fixed-norm constraint
   (delta=0 is not on the sphere for r > 0).

5. **Loss doesn't converge**: Might need more steps, different lr schedule,
   or the optimization genuinely can't find a good direction at this r.

## Interpretation (After Optimization Succeeds)

If we find a delta that keeps Jacobian norms low:
1. **Project real activations onto delta**: `proj_i = (a_i · delta_hat)` where delta_hat = delta/||delta||
2. **Sort by projection magnitude**: Which activations have the largest component along delta?
3. **Inspect top-activating sequences**: What do they have in common?
4. **Compare with known features**: Does delta align with any SAE features or known directions?
