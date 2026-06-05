# Perturbation experiment on grok-adversarial MNIST MLP

## Goal
Replicate the perturbation-along-random-directions experiments (à la our transformer
scripts) on the MLP trained in https://github.com/AhmedImtiazPrio/grok-adversarial
("Deep Networks Always Grok and Here is Why", ICML 2024).

Extract a representation at the **first latent layer**, perturb in random directions,
and measure the downstream effect at the **last hidden layer** (still high-dim, just
before projecting down to logits) via **L2 distance** perturbed vs unperturbed.

## Target model (MNIST MLP, depth 4)
`Flatten → Linear(784,200) → GeLU → Linear(200,200) → GeLU → Linear(200,200) → GeLU → Linear(200,10)`
- h1 = output of first Linear+GeLU (inject here, "first latent layer")
- h3 = output of third Linear+GeLU (measure here, before final Linear → 10 logits)

## Training recipe (faithful to config_base_mnist_mlp)
float64, seed 0, 1000 random MNIST train points, MSE on one-hot, AdamW lr 1e-3
wd 0.01, batch 200, 1e6 steps, init scale 1.0, GeLU.
**Difference from upstream:** dropped local-complexity diagnostic logging (slow,
orthogonal to training). Periodic checkpoints saved so we can pick pre/post-grok.

## Perturbation (spherical convention, from run_perturbation_sweep_spherical.py)
`a' = ||a|| (cos θ · â + sin θ · v_perp)`, v_perp random unit ⊥ a.
Sweep θ ∈ (0, 60°]. Conditions: real h1 vs norm-matched random h1.
Metrics: L2 @ last hidden (primary), cosine @ last hidden, KL @ logits.

## Files
- `scripts/grok_mlp/train_mlp_grok.py` — training
- `scripts/grok_mlp/perturb_mlp_spherical.py` — perturbation sweep

## Run location
RunPod (213.173.103.50:41420), RTX 5090.

## Pod environment note
RTX 5090 is Blackwell (sm_120); pod's pre-installed torch 2.4.1/cu124 has no
kernels for it. Built `/root/venv` with `torch torchvision --index-url
https://download.pytorch.org/whl/cu128` (torch 2.11/cu128) — runs fine.
Project on pod: `/workspace/grok-mlp/` (scripts, results, data, venv at /root/venv).

## Status
- [x] Read both repos, wrote scripts
- [x] Train MLP on pod — running (1e6 steps, ~80 min); train acc 100% by ~7k
      steps, test acc ~91%. Checkpoints in results/grok_mlp/checkpoints/.
- [x] Run spherical sweep on the final grokked checkpoint (step 1M, test acc 90.8%).
      Results: results/grok_mlp/perturb_spherical/ckpt_s1000000_final/{sweep_median.png,results.pkl}
- [x] Port superellipse direction-dictionary optimization (R_mix/(R1+R2) + λ·orth, L2)
      to the MLP: scripts/grok_mlp/learn_superellipse_dict_mlp.py. Ran 15 directions,
      n_act=10, θ=30°. Results: results/grok_mlp/superellipse_dict/ckpt_s1000000_final_a30/
      {summary.png,results.pkl}. Pair L: min 0.004, median 0.28, max 0.64.
- [x] Port canonical superellipse-p fit (exp-map 2D sweep + iso-contour fit of
      |x|^p+|y|^p=1, bootstrap robustness) to the MLP:
      scripts/grok_mlp/fit_superellipse_pair_mlp.py. Reuses canonical/parametrize.py
      and canonical/superellipse.py verbatim; only the 2D sweep is MLP-specific.
      Smoke test on pair (0,1): fitted p ~0.85-1.2 (NB: <2, i.e. sub-additive in
      the contour fit) despite the dict optimization's strong super-additivity
      (L~0.02) — tension flagged to Francisco, not yet investigated.
- [x] Contrastive directions for the MLP = diff-of-class-means in h1 (true labels).
      Optimize ONE free direction against a fixed contrastive leg, same superellipse
      ratio objective, averaged over 10 randomly-chosen class pairs (of 45).
      Script: scripts/grok_mlp/learn_against_contrastive_mlp.py. Full run (n_act=10,
      n_c=10, theta=30): mean L over contrastive set = 0.297 (range 0.25-0.39).
      Results: results/grok_mlp/against_contrastive/ckpt_s1000000_final_a30_nc10/.
      Saved directions = [free, *contrastive] so fit/boundary scripts work on pairs.
- [x] Superellipse-p fits for direction pairs (scripts/grok_mlp/fit_free_pairs_mlp.py,
      modes free_free / free_contrastive / contrastive_contrastive; spread across |cos|).
      free-free p ~1.2-2.4; free-contrastive p ~0.5-3.8 (wide bootstrap ranges);
      contrastive-pair p ~1.2-4.3.
- [x] One-vs-rest class dirs (one per digit, d_c = norm(mean_c - grand_mean)), fit p
      for ALL 45 pairs: scripts/grok_mlp/fit_onevsrest_classes_mlp.py. p over all pairs:
      min 2.12, median 3.58, max 6.73. Results: .../fit_onevsrest_classes/
      {p_matrix.png, boundary_panels.png, p_vs_cos.png, fit_onevsrest.pkl}.
