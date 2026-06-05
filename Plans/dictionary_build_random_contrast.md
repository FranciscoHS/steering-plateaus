# Plan: Dictionary build with random-direction "contrast" (Qwen3-1.7B, L14)

## Motivation

The contrastive-vs-contrastive pair fits (Qwen3-1.7B, L14, FineWeb anchors)
gave median p = **2.03**, pooled p = **2.03**, 95% CI [2.00, 2.06] — i.e. the
contrastive directions don't show super-additive iso-contours against each
other. The 1D sweep tells the same story (contrastive vs random direction
ratio ~1.2–1.4 in qwen, slightly higher in gemma, but still small).

Implication for our dictionary-build proxy:
- The proxy `R_mix / (R_f + R_c)` averaged over a small **contrastive**
  scaffold rewards directions that interfere super-additively with those
  specific contrastive dirs.
- If contrastive dirs themselves sit near p≈2, then *random* unit directions
  satisfy the same proxy roughly as well as *features* would. Random
  directions outnumber features by many orders of magnitude in R^d, so the
  optimizer is much more likely to land on "random plateau-breaking against
  this scaffold subset" than on a real feature.

This is consistent with the actual results we observed: pairwise p among the
8 k dictionary atoms came out at p≈2.1, and reconstruction quality on
held-out FineWeb was barely above random (PCA crushed it, SAE crushed it).

**Fix:** stop optimizing against the contrastive scaffold. Optimize directly
against **random unit directions**, with a fresh sample per atom. A feature
direction is then one that interferes super-additively with a *typical*
random direction in residual-stream space — that's a much stronger
constraint, and the only way to satisfy it is to actually find a privileged
direction (since random partners are interchangeable, super-additivity
against them is a robust property, not a coincidence of the scaffold).

## What stays the same

- Model: **Qwen3-1.7B**, layer **14**, measure_layer 26 (n_layers - 2).
- bfloat16, n_steps=150, lr=0.003.
- One activation per atom, sampled fresh from a 8 k FineWeb pool (seq_len=10,
  seed=42).
- 8 k atoms total (resume to 16 k optional, as before).
- Soft-orthogonality against all previously-found atoms (lam_prev=1.0).
- Same checkpoint cadence (every 100 atoms), same metadata bookkeeping.

## What changes

Inside `build_dictionary.optimize_one_atom`:

1. **Replace the contrastive scaffold with a fresh random sample.**
   Drop `load_or_build_scaffold` from the per-atom path. Instead, sample
   `n_contrast` (default 10) unit vectors from `N(0, I_d) / ||.||` using a
   per-atom RNG.
2. **Drop `lam_scaffold`.** With random contrast dirs in 2048-d, the
   expected `(d · d_c)²` is ~1/d_model ≈ 5×10⁻⁴; the penalty term carries no
   signal and effectively adds noise. Set it to 0 by default.
3. **Keep `lam_prev`** — still need diversity across the dictionary.
4. **Loss becomes:**
   ```
   L(d_k) = mean_c [R_mix / (R_f + R_c)]                # over 10 fresh random d_c
          + lam_prev * mean_prev (d_k · d_prev)²        # diversity
   ```
   Same Adam, same n_steps, same lr.

Other knobs (n_contrast, contrast_seed strategy, optional re-sampling
*within* a single atom's 150 steps) are CLI-exposed but default to: 10
random dirs, sampled once per atom (deterministic per-atom seed).

## Files

- **Modify** `scripts/superellipse/build_dictionary.py`:
  - Add `--contrast_source {scaffold, random}` (default `random`).
  - Add `--n_contrast` (default 10, replaces `--scaffold_sample_size`).
  - Add `--contrast_seed_base` for per-atom random contrast RNG.
  - When `contrast_source=random`, skip `load_or_build_scaffold`; build a
    per-atom `[n_contrast, d_model]` random unit tensor in `main`'s loop
    using `atom_rng(k)` (already exists).
  - Default `lam_scaffold` to 0 when `contrast_source=random`; keep
    user-overrideable.
  - Output dir naming: include the contrast mode, e.g.
    `results/dictionary/qwen3-1.7b_L14_K8000_randcontrast/`.

- **No change** to `optimize_one_atom`'s internals beyond accepting the new
  scaffold tensor (it already does, since it takes `scaffold` as an arg).

- **No change** to `eval_dictionary_omp.py`, `eval_dictionary_kl.py`,
  `fit_pairs_dictionary.py`, `principal_angles.py` — they take the
  dictionary file and don't care how it was built. Just point them at the
  new `dictionary_K8000.pt`.

## Compute estimate

Per atom: same per-step cost as before (10 forward passes × n_act × 3
perturbations = 30 points/batch in the proxy forward). ~1.94 s/atom on RTX
5090 → **~4.3 h for 8000 atoms**.

## Eval pipeline (re-run as-is on the new dict)

1. **Sparse-recon (OMP) on held-out FineWeb**, K ∈ {8, 16, 32, 64, 100, 256},
   compared to:
   - random unit dict same width,
   - random unit dict at SAE width (32 k),
   - PCA top-K (using the existing PCA pool),
   - Qwen-Scope SAE (W=32 k, TopK=50, 100).
2. **KL reconstruction** on the same activations.
3. **Pair-fits for p**: 100 random pairs from the new dict (dict vs dict).
   Hypothesis: if the change is doing what we want, median p should rise
   meaningfully above 2.0 here, indicating the atoms are super-additive
   against each other (a stronger property than the previous build).
4. **Cross-build cosine**: cosine between new-dict atoms and old-dict atoms
   — are these qualitatively different sets, or did random-contrast just
   shuffle the same atoms?

## What we'd learn

- **Does random contrast help reconstruction?** If recon error drops vs the
  old dict at the same K, we found a better proxy.
- **Does random contrast produce more strongly super-additive directions?**
  If pair-fit p moves up vs old (and ideally past 2.0–2.5), the proxy is
  identifying real privileged directions.
- **Or: does it make no difference?** That would suggest the proxy +
  greedy-orthogonal structure is too weak in either form on this model /
  layer / proxy — and we'd want to revisit the proxy itself, not just the
  contrast pool.

## Open questions

- **Per-step vs per-atom random contrast.** Per-atom (default) is cheap and
  reproducible; per-step (fresh 10 random dirs every gradient step) would be
  noisier but harder for the optimizer to game against a specific direction
  sample. Worth a small ablation (~500 atoms each) before committing 8 k.
- **n_contrast.** 10 is the same as our prior runs. Could try larger (30,
  100) for more stable averaging; cheap to do since it's only the per-step
  cost that grows.
- **Layer choice.** Qwen3-1.7B L14 is what we have all the prior comparisons
  on. Worth re-running at L7 (fsec's default for qwen) if the L14 result is
  unconvincing; the previous 1D plateau plot showed earlier layers might be
  more plateau-rich.

## Risks

- **Same compute as before, same conclusion possible.** If the random
  contrast also gives p ≈ 2 pairwise and similar recon, that's a bigger
  message: the proxy may simply not identify features on this
  model/layer. Plan accordingly — don't auto-extend to 16 k atoms.
- **Pod fragility.** Memory was tight on 24 GB cards (we needed a 32 GB 5090
  for the qwen build). Same constraint here.
