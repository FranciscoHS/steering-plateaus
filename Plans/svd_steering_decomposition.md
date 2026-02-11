# SVD Decomposition of Steering Vector

**Date:** 2026-02-11

## Motivation

The current English-Mandarin steering vector is a difference-of-means: $v = \bar{a}^{en} - \bar{a}^{zh}$. Under the linear representation hypothesis, this is a linear combination of multiple features — the language direction plus content variation across pairs. This may explain why steering along it looks similar to random perturbations in the perturbation sweep.

**Goal:** Determine whether the language signal in the contrastive differences is essentially 1D (mean already captures it) or multi-dimensional (individual components might behave differently).

## Method

### Matrix of differences

Collect N=50 paired activations and form:

$$D = \begin{pmatrix} a(x_1^{en}) - a(x_1^{zh}) \\ \vdots \\ a(x_N^{en}) - a(x_N^{zh}) \end{pmatrix} \in \mathbb{R}^{N \times d}$$

### Uncentered SVD (not PCA)

Standard PCA centers by subtracting the mean — but the mean of D *is* the steering vector. We don't want to remove it; we want to know whether it's a clean summary of D. So we do SVD on D directly.

Key diagnostics:
- **Singular value spectrum:** $\sigma_1^2 / \sum_k \sigma_k^2$ = fraction of $\|D\|_F^2$ explained by PC1
- **Cosine(PC1, mean direction):** If close to 1, the dominant mode is the mean and the steering vector is already clean. If not, there's a dominant mode the mean doesn't fully capture.
- **Variance decay:** How many PCs needed to explain 90%+ of the variance?

### Layer sweep

There's a trade-off in choosing which layer to analyze/perturb at:
- **Too early** (e.g. layer 2): Language may not be well-defined yet, so the differences are noisy
- **Too late** (e.g. layer 20+): Not enough layers between perturbation and output for clean perturbation response
- **Sweet spot:** Layer where language direction is clean AND enough depth remains

**Step 1** resolves this empirically by running the SVD analysis at multiple layers.

## Plan

### Step 1: SVD analysis across layers (diagnostic, cheap) — DO THIS NOW
- Model: Qwen3-0.6B
- Layers: all layers (0 through 27). This is cheap since we extract all hidden states in a single forward pass per prompt.
- For each layer:
  - Extract 50 paired activations (English + Chinese prompts)
  - Form difference matrix D
  - Compute SVD
  - Record: singular value spectrum, cosine(PC1, mean), cumulative variance explained
- Output: plots showing how the language direction crystallizes across depth

### Step 2: Perturbation sweep along PC directions (informed by Step 1) — LATER, pending Step 1 results
- Pick layer where language signal is strong (from Step 1)
- Run perturbation sweep for:
  - Mean direction (current steering vector)
  - PC1, PC2, PC3 from SVD
  - Random directions (baseline)
- Compare perturbation curves: does PC1 break the plateau differently from random?

## Expected outcomes

- **If PC1 dominates and $\approx$ mean direction:** The steering vector is already clean; the plateau-like behavior isn't because the direction is messy — it's a real property of how the model responds to language perturbations.
- **If variance is spread across many PCs:** The mean direction is a noisy superposition. Individual PCs (especially PC1) might show qualitatively different perturbation behavior.
- **If PC1 dominates but $\neq$ mean direction:** There's a cleaner language direction that the simple mean misses. Worth testing whether this direction shows different plateau structure.
