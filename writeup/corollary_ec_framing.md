§8 — Corollary (EC framing)

Error correction, if the model performs it on the activation stream, exists to tell signal from noise: perturbations along meaningful (feature) directions should be let through, perturbations orthogonal to meaning should be suppressed. Under superposition ($M > D$) there is no "non-feature subspace" to discard — the features span $\mathbb{R}^D$, so every perturbation $v$ decomposes as $v = \sum_i c_i d_i$ with some nonzero coefficients. EC therefore has to act on these coefficients. Two natural schemes:

1. **Per-feature thresholds.** Each feature has its own tolerance $\tau_i$; $v$ triggers a response iff $|c_i| > \tau_i$ for some $i$. This corresponds to the $n \to \infty$ limit of our functional form and predicts square iso-plateaus.

2. **Aggregate threshold.** The model tolerates perturbations with small total feature content and rejects those with large total. Response governed by $\sum_i w_i c_i^2$ crossing a single threshold. Circular iso-plateau, $n = 2$.

Empirically we find $n \approx 2$, ruling out per-feature thresholds. The aggregate scheme is not by itself falsified.

However, the aggregate scheme combined with superposition has an uncomfortable consequence. Writing the response as $R(v) = F(v^\top A v)$ with $A = \sum_i w_i d_i d_i^\top \succeq 0$, the direction that crosses the threshold at the smallest perturbation magnitude is the top eigenvector $u^*$ of $A$. By Rayleigh, $u^{*\top} A u^* = \lambda_{\max}(A) \geq d_j^\top A d_j$ for every feature $j$. Under $M > D$ with generic feature geometry, $u^*$ does not coincide with any $d_j$. So the model's *easiest* breaking direction is not a feature.

That is the opposite of what EC is for. EC is supposed to pick out features as the directions of signal, making them the directions along which perturbations matter. Our results force features to be *less* sensitive than a specific non-feature direction — the model's output changes faster along a direction nobody would call a feature than along any feature it is built from.

So: the narrow claim our results support cleanly is that the model is not doing per-feature threshold error correction. The broader claim — that the model is not doing *any* error-correcting scheme that privileges features as the maximally-sensitive directions — requires only that one accepts "EC privileges features" as the definition of what EC is supposed to do. That avoids any universal quantifier over features or any strong reading of LRH; it just says EC's job is to make features signal, and our data shows features aren't the most sensitive.
