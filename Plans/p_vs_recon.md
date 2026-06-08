# Reconstruction quality vs superellipse-p of the directions used

**Goal.** Plot y = reconstruction quality vs x = superellipse-p of the
directions used, with **3 points per direction type** (sparsity K = 16, 32, 64).

**Axes.**
- y = mean relative L2 `||a-â||²/||a||²` over held-out activations (n=2000) at K.
- x = pooled superellipse-p, estimated from `n_pairs=100` pairs sampled from the
  directions actually USED in the reconstruction at that K. p is pairwise
  (iso-L2 boundary in the 2D plane of perturbing along dᵢ,dⱼ at L14, measured at
  penultimate layer), so we fit p per pair and pool (bootstrap CI over pairs),
  mirroring `fit_pairs_dictionary.py`.

**Types / points.**
- **dict** (OMP on first-20k snapshot): 3 pts. Per K, OMP picks K atoms/act;
  co-occurrence pair sampling (pick a held act, then 2 of its K atoms).
- **pca** (top-K projection): 3 pts. Directions = top-K PCs (fixed subspace);
  uniform pairs among those K.
- **random** (OMP on random unit dict, N=20k): 3 pts in y, **one shared x**
  (random-pair p is K-independent → estimate once, reuse).
- **sae** (Qwen-Scope TopK, d_sae=32768): **1 pt** at K=`sae_K` (default 32);
  co-occurrence pairs among active latents.

≈ (3 dict + 3 pca + 1 random + 1 sae) × 100 = **800 p-fits**. Each fit is a
31×31 grid × 20 anchors sweep. Rate measured by `--smoke` first.

**Inputs (network volume).**
- dict snapshot: `.../qwen3-1.7b_L14_K8000_ratio30_actresample/snap_dict_first20k.pt`
  (first 20k atoms frozen from the still-growing `dictionary.pt`).
- held acts: `.../qwen3-1.7b_L14_K8000/held_acts.pt` (n=2000).
- pca pool: `.../qwen3-1.7b_L14_K8000/pca_pool_acts.pt`.
- sae ckpt: `.../qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt` (d_sae=32768).

**Script.** `scripts/superellipse/p_vs_recon.py`. Run on pod `dict-eval`
(build runs separately on `dict-build-8k`).

**Open choices made (flag to change):** SAE = 1 pt at K=32; co-occurrence pair
sampling; n_pairs=100; n_anchors=20; max_angle=60; grid 31; L2 (not KL).
