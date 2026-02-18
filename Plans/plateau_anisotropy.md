## Plateau Anisotropy Experiments

### Motivation

Real activations sit on plateaus in activation space — regions where perturbations don't change model behavior. We've observed that these plateaus are *anisotropic*: perturbation along a feature direction (e.g., diff-of-means between English and Mandarin) breaks the plateau at a smaller angle than perturbation along random/orthogonal directions. We want to characterize this anisotropy systematically.

### Key definitions

- **Feature direction** u_feat: diff-of-means between two contrastive sets of activations (e.g., English vs Mandarin prompts)
- **Orthogonal direction** u_orth: random direction orthogonal to u_feat (and to the activation). In ~1000 dimensions, random directions are nearly orthogonal anyway, but we Gram-Schmidt for cleanliness
- **2D perturbation**: given angles θ_y (along feature) and θ_x (along orthogonal):
  a' = ||a|| * normalize(â + tan(θ_y) * u_feat + tan(θ_x) * u_orth)
  This reduces to standard spherical perturbation when one angle is zero.
- **Breaking angle** r: angle at which median KL divergence (over activations) first exceeds 1 nat
- **r_feat**: breaking angle along the feature direction
- **r_plateau**: breaking angle along orthogonal directions (averaged over multiple random orthogonal directions)
- **Anisotropy ratio**: r_plateau / r_feat — how much more "room" there is in non-feature directions. Ratio > 1 means the plateau is elongated perpendicular to the feature.

### Experiment 1: 2D heatmap (EN-ZH)

For the English-Mandarin pair, compute the 2D heatmap of KL divergence as a function of (θ_x, θ_y). This directly visualizes the shape of the plateau in the plane spanned by the feature and orthogonal directions. We expect elliptical iso-KL contours, narrow along the feature axis and wide along the orthogonal axis. Average over multiple random orthogonal directions and take median over activations.

### Experiment 2: Anisotropy ratio across contrastive pairs

For ~6 language pairs spanning different linguistic distances (EN-NL, EN-ES, EN-ZH, EN-JA, EN-RU, ZH-JA), compute r_feat, r_plateau, and the anisotropy ratio. Same layer, same model. Questions: Is the anisotropy ratio constant across pairs? Does it correlate with linguistic distance? Does r_feat vary while r_plateau stays fixed (or vice versa)?

### Experiment 3: Anisotropy ratio across layers

For a single contrastive pair (EN-ZH), compute the anisotropy ratio at each layer. Does the plateau become more anisotropic at later layers (features sharpen)? Does r_feat shrink while r_plateau grows?
