# Plan: build an 8k-direction "feature dictionary" via privileged-direction optimization

## Motivation

Greedy decomposition of a single activation gave a small set of plateau-breaking + reconstruction-aligned directions, but reproducibility across seeds was weak past d_1–d_2 (mean rank-wise cos ≈ 0.08–0.5). One hypothesis: features are a discrete-ish set, and our procedure is undersampling them — each per-activation run finds 5-6 of them, but the full set is much larger.

Instead of decomposing a single activation into a few features, build a **dictionary** of ~8k unit directions by repeated single-direction optimization with soft orthogonality, then evaluate whether activations decompose well over that dictionary.

This is closer in spirit to a manually-trained SAE: a fixed dictionary that any activation should be sparsely representable in.

## Model

**Qwen2.5-1.5B.** Smaller than gemma-2-9b → ~10× faster per direction. We'd lose the existing scaffold (need to rebuild) and the existing Gemma Scope SAEs (need to either find or train one for Qwen).

## Compute estimate

- Per-direction time on a 1.5B model: ~10 s with n_act=10; **~1.5-2 s with n_act=1** (kernel underutilization eats the full 10× scaling).
- 8000 directions × 2 s ≈ **3-4 hours** of GPU time.
- Fits in a single 100h pod runtime with margin.

## Scaffold setup

Extract contrastive directions for Qwen2.5-1.5B at **layer 14** (middle of the 28-layer stack — TBD whether mid-layer is best; can sweep later).

**Semantic-only direction names (17 total)** — exclude language (10 en→lang) and code (9 en→code) families since they all bias toward English:

Gender, Sentiment, Age, Era, Tense, TensePresent, Number, Person, Status, Honesty, HonestyShort, Refusal, Formal, Literary, Wealth, Health, Certainty

Built via `fsec-anonymous`'s `scripts.lib.directions.extract_all`, same path as the existing gemma scaffold. A few minutes of setup + a few minutes of extraction.

## Optimization procedure

For each new direction k = 1, ..., 8000:

1. **Sample scaffold subset**: 10 of the 17 semantic directions, sampled uniformly at random.
2. **Sample activation**: 1 activation from FineWeb (seq_len 10, last-token Layer-14 residual), fresh per direction. Different RNG seed per direction.
3. **Optimize d_k via Adam** with the existing privileged-direction loss, *minus* the overlap-with-residual term (no reconstruction target):

   ```
   L(d_k) = mean_c [R_mix / (R_f + R_c)]          # plateau-breaking proxy
          + λ_scaffold · mean_c (d_k · d_c)²       # stay away from scaffold
          + λ_prev     · mean_prev (d_k · d_prev)² # stay away from prev features
   ```

   - λ_scaffold = 1.0, λ_prev = 1.0 (match prior runs)
   - lr = 0.003, n_steps = **150**
   - n_act = 1, n_perturbation_angles = 1 at 30°
   - measure_layer = 2nd-to-last (Qwen2.5-1.5B has 28 layers → measure_layer=26)
4. **Renormalize to unit, append to dictionary pool, checkpoint** every 100 directions.

## Files / structure

- New script: `scripts/superellipse/build_dictionary.py` — orchestrates the 8k-direction loop. Loads scaffold once, holds prev-dictionary in memory, samples activation + scaffold subset per direction, calls a single-direction optimizer (reuses `optimize_one_feature`-style core from `decompose_plateau_breaking.py` with `lam_overlap=0`).
- Output: `results/dictionary/qwen2.5-1.5b_L14_K8000/`
  - `dictionary.pt` — `[8000, d_model]` unit directions, saved incrementally.
  - `metadata.json` — sampling seeds per direction, scaffold subset used, etc.
  - `progress.log` — checkpoint markers.

## Evaluation

Once the dictionary is built:

1. **Reconstruction on held-out FineWeb**:
   - Held-out set: 1000 FineWeb activations (different seed from training).
   - For each activation, compute K-sparse reconstruction in dictionary via OMP (orthogonal matching pursuit) at K = 16, 64, 256.
   - Metric: mean `||a − â||² / ||a||²`.

2. **Comparison to SAE**:
   - **If a Qwen2.5-1.5B SAE at L14 with ~8k latents exists** → use directly.
   - **Else** (likely): train a small JumpReLU SAE on Qwen2.5-1.5B L14 with 8k latents. Quick to train (hours, not days). Standard SAE-lens config.
   - Eval same K-sparse reconstruction.

3. **Feature inspection (qualitative)**:
   - For an activation `a` whose reconstruction picks dictionary atoms `d_{i_1}, ..., d_{i_K}`, are those atoms "expected" features for the context? E.g., for a French sentence: does the top atom resemble a French-related direction? (Bear in mind we excluded en→lang scaffolds, so any such atom would be emergent, not induced.)

## What we'd learn

- **Does the dictionary cover well?** If mean reconstruction error with K=64 atoms is comparable to a same-sized SAE, that's evidence the privileged-direction proxy is identifying genuine features (despite not optimizing for reconstruction).
- **Is the dictionary redundant?** Pairwise cosine distribution among the 8k atoms — if heavy-tailed toward 1, soft orth wasn't strong enough; if uniformly low, we've spanned a diverse set.
- **Are individual atoms interpretable?** Inspecting top-activating examples for the highest-coefficient atoms in held-out reconstructions gives a feel for whether they correspond to recognizable features.

## Open questions

- **Qwen layer**: layer 14 is a reasonable middle, but no principled choice yet.
- **Per-direction noise**: n_act=1 introduces variance per direction. If the proxy is sensitive to activation identity, the optimizer could converge to "this activation's plateau-breaking dir" rather than a generic feature. Should sanity-check by re-running a few directions with the same RNG and varying activations to see how much the result drifts.
- **OMP vs LASSO vs top-K**: OMP is the simplest sparse decoder; LASSO is closer to SAE-style sparse coding. May matter at the eval step.
- **SAE for comparison**: if we train one, choose width carefully (matching 8k exactly vs nearest available SAE-lens preset).
- **Stopping criterion**: do we just hard-cap at 8000, or do we add a "saturation" check (e.g., new directions found have high overlap with existing ones → we've covered the easy ones)?

## Risks

- **GPU availability**: pod provisioning has been flaky. Worth requesting `--runtime 6000` (100h) up front and ideally checkpointing every ~100 directions so a mid-run pod death is recoverable.
- **No corresponding SAE**: comparison to SAE is the headline eval. If we can't find one and don't want to train, the absolute reconstruction number is informative but harder to contextualize.
- **OOM on whatever pod we get**: the earlier sweep OOM'd on 24 GB cards even at chunk_k=2. With n_act=1 and a 1.5B model, memory should be a non-issue (gemma-2-9b at n_act=10 was the problem). But worth confirming on first launch.
