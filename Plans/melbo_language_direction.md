# Finding the Language Direction via Constrained MELBO Optimization

**Date:** 2026-02-11 (updated)

## Motivation

The contrastive mean steering vector (mean of en-zh activation differences) successfully steers language but behaves like a random direction in the perturbation sweep. The SVD analysis showed this vector is a superposition of many features (~30 PCs for 90% of variance).

We want to find a *cleaner* language direction — one that:
1. Steers language (changes what language the model speaks)
2. Shows a different perturbation response than random directions

The MELBO principle gives us a tool: directions that cause disproportionately large downstream changes relative to their magnitude are structurally important. We apply this within the language subspace to find which component of the steering vector does the most "work."

## Method: Constrained MELBO

**What:** Find the direction *within the span of the difference matrix D* that maximizes KL divergence when used as a spherical perturbation. This asks: "which component of the language steering vector causes the most behavioral change?"

**Setup:**
- Model: Qwen3-0.6B
- Source layer: 14 (strong language signal, cos(PC1,mean)≈0.98, 13 layers remaining)
- Activations: 50 FineWeb activations (seq_len=5)
- Optimization angle: 15° (in the plateau regime for random directions)

**Optimization:**
- Difference matrix D = en_acts - zh_acts at layer 14, shape [50, 1024]
- SVD: D = U S V^T. The row space of D (= column span of V) is the language subspace.
- Parameterize u = V (α/||α||) where α ∈ R^r. Since V has orthonormal columns, ||u|| = 1.
- For each FineWeb activation a_i, apply spherical perturbation: a_i' = cos(θ)·a_i + sin(θ)·||a_i||·u_⊥ where u_⊥ is u projected orthogonal to a_i and renormalized.
- Objective: maximize mean KL divergence across activations: L(α) = (1/N) Σ_i D_KL(p(a_i) || p(a_i'))
- Optimizer: Adam on α, lr=0.01, 300 steps, 5 random restarts per direction
- Find top-3 orthogonal directions: enforce α_k ⊥ α_1,...,α_{k-1} via Gram-Schmidt + projection after each step
- All N=50 activations batched in a single forward pass per step (efficient)

**Evaluation: Perturbation sweep comparison**

After optimization, run full angle sweeps (0°–180°, 60 points) comparing:
- Random directions (10 per activation, baseline)
- Contrastive mean direction
- SVD PC1 direction
- MELBO directions 1–3

Metrics: KL divergence on logits + L2 distance on final-layer hidden states.

**Key question:** Does the max-impact direction within the language subspace show a qualitatively different perturbation curve than the contrastive mean / random directions?

## Practical considerations

**Compute:** Each optimization step = 1 batched forward pass (N=50, seq_len=5) through 13 layers. With 300 steps × 5 restarts × 3 directions = 4500 forward+backward passes. Should take ~10-20 minutes on GPU. The evaluation sweep adds ~60 × 6 = 360 forward passes per direction type.

**Gradient flow:** Model parameters are frozen (requires_grad=False), but gradients flow through the forward pass w.r.t. the input activations (through α). Causal masking means only the last token position is affected by the perturbation.

## Expected outcomes

- **If MELBO directions break the plateau:** The plateau is about direction choice, not a universal property. Structurally important directions behave differently from random ones. Key finding.
- **If MELBO directions also plateau:** The plateau is a genuine geometric property — all directions are equally "flat" near real activations. Surprising but important.
- **If MELBO direction ≈ contrastive mean:** The mean was already near-optimal within the language subspace.
- **If MELBO direction ≠ mean but steers language:** There's a cleaner language feature than the contrastive mean.
