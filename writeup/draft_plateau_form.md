---
title: "The plateau response function is quadratic in perturbation"
geometry: margin=0.75in
fontsize: 11pt
---

**Setup.** We take as a working hypothesis the linear representation hypothesis (LRH): features are directions in activation space and downstream computation reads each feature *i* via an inner product d_i·h, with the model more sensitive along feature directions than along non-feature ones. We further assume superposition: M > D features in D-dimensional activation space. Under superposition, pairs of features are generically not orthogonal, so feature read-offs carry interference from other active features. Reliable readout in this regime plausibly requires the model to ignore sub-threshold components of *h* along feature directions (argument due to Hänni; not a proof, but motivation).

**Per-feature thresholding is the simplest error-correction scheme.** Feature *i* only drives downstream computation when |d_i·h| exceeds some threshold c_i; the plateau — the set of activations with no downstream response — is the box {h : |d_i·h| < c_i for every *i*}, aligned to the feature directions.

**Empirical evidence (Figure 1).** We measure downstream response to on-sphere perturbations *v* via three metrics at a later layer: cosine distance, L2 distance, and KL divergence. Two things hold: (i) responses are small up to a plateau-breaking angle θ*; (ii) θ* is smaller along contrastive "best-guess feature" directions than along random directions, across all three metrics. Figure 1 shows this on Qwen3-1.7b layer 14 for 10 language-contrastive directions vs 10 random directions, median over 30 FineWeb activations. Consistent with feature-aligned plateaus, but does not yet distinguish shapes.

![Contrastive (solid) vs random (dashed) single-direction sweeps, median over 30 activations. Qwen3-1.7b L14 → penultimate. Contrastive directions cross threshold (dotted) at smaller angles.](../results/pair_slices_10contrastive/qwen3-1.7b_L14/phase_a_sensitivity.png){width=85%}

**Prediction from per-feature thresholding.** Perturb along v = α_1 d_1 + α_2 d_2. The plateau holds iff |α_1| < θ_1* and |α_2| < θ_2*. In normalized coordinates (divide each axis by its own θ_i*) the iso-plateau boundary is therefore a **square**.

**Observation: it's a circle (Figure 2).** For Gemma-2-9b L20, contrastive Mandarin–Haskell pair, L2 metric: the normalized contour traces a circle almost perfectly (fitted superellipse exponent p = 2.01, radial residual 0.05° mean / 0.12° max). The p→∞ square (per-feature thresholding) and p=1 diamond are visibly wrong.

![Iso-plateau shape, Gemma-2-9b L20, Mandarin–Haskell, L2=50. Data (blue) with reference p=1 diamond, p=2 circle, p→∞ square. Fit: p=2.01 [2.0, 2.1].](../results/mandarin_haskell/gemma-2-9b/isoplateau_shape_mandarin_haskell_l2.png){width=45%}

**Proposed functional form.** A one-parameter family,  R(v) = F(Σ_i w_i |d_i·v|^n), with F convex monotonic and w_i ≥ 0. Per-feature thresholds are the n → ∞ limit; a circular iso-plateau is n = 2.

**Fitted *n* clusters near 2.** Multiple Gemma-2-9b L20 contrastive pairs (Mandarin/Haskell, French/C++, French/Rust, French/random, …): per-pair superellipse fits at p ≈ 2. On Qwen3-1.7b L14, 45 pair slices drawn from 10 language-contrastive directions, median over 30 activations: median fitted p = 2.17 / 2.12 / 2.04 for cos / L2 / KL at representative thresholds, stable across a ~30× threshold range, 5–95% per-pair spread roughly [1.6, 2.7], per-pair radial residual ~1–2% of contour radius (RMS 0.2–0.4° on ~20° contours). Axis-aligned superellipse is biased upward when the direction basis is correlated; mutually Gram–Schmidting the 10 directions first lowers the median from ~2.3–2.7 to ~2.0–2.2, consistent with this explanation. Across two models, two layers, multiple contrastive-direction sets, and three response metrics, the fitted *n* is compatible with 2.

**Corollary.** For R of this form with n = 2, no choice of weights or threshold can make the downstream landscape strictly more sensitive along every feature direction than along every non-feature unit direction (the "universal" reading of LRH), under superposition M > D. The obstruction is a one-line Rayleigh argument: Σ_i w_i d_i d_i^T is PSD, its top eigenvector beats every feature it is built from, and M > D features cannot all lie in a single eigenspace. Companion note gives the proof.

So the empirical *n* = 2 response function rules out strict-universal-LRH + superposition as a joint picture. Either superposition is weaker than assumed, or readout is less strictly feature-aligned, or the strong reading of LRH should be softened to "features beat the typical random direction, not every one."
