# Are our found directions used for computation? (causal ablation → KL)

**Question.** Take a found direction `d` (a 40k-dictionary atom) at its layer. During a
forward pass on a large natural corpus, project it out of the residual stream:
`h'(x) = h(x) − (d·h(x)) d`. Measure `KL(p_baseline ‖ p_ablated)` per token, aggregate.
Compare against **contrastive** (diff-of-means) directions and **random** matched-norm
directions as controls. If our found directions carry computation, ablating them should
move the next-token distribution more than random directions of the same norm.

## Setup
- Model: `qwen3-1.7b`, layer **14** (where the dict atoms live; residual stream after
  block 14, i.e. `hidden_states[15]`). `d_model = 2048`.
- 40k dictionary: `results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_align1.0_k100_40k/dictionary.pt`
  → `blob['dictionary']` `[K, 2048]`, rows already unit-norm (on the **network volume** —
  the local checkpoint is only a 300-atom partial).
- Contrastive: `results/allpairs_contrastive/qwen3-1.7b/directions.pkl` → `features`
  (10 lang en→X diff-of-means) and `pairwise_means` (45). All L14, qwen3-1.7b.
- Corpus: FineWeb-Edu via `src.data.load_fineweb_fixed_length` (cache keyed by tokenizer).

## Method
1. All directions **unit-normalized**. Note: the rank-1 projector `h − (d·h)d` only removes
   the component along the unit direction and is **scale-invariant** in `d`, so "matched
   norm" is automatically satisfied by making every direction (found / contrastive / random)
   unit-norm. The random control is random Gaussian unit vectors.
2. Ablate at **all token positions** at layer 14 (stronger, more natural causal test than
   last-token-only — the dict was *built* on last-token scaffold acts, but the use-question
   is general).
3. Per corpus batch: one full forward → cache `resid@L14`. Then per direction, ablate the
   cached resid and re-run only **blocks 15→end + norm + lm_head** (≈half the net; ~2× cheaper
   than full forwards across hundreds of directions). Baseline log-probs come from the **same
   truncated path** on the unablated resid (guarantees `KL≥0`, `KL=0` at zero projection).
4. `KL = Σ_v p_base·(log p_base − log p_abl)` per token; log-softmax in fp32.
5. Also record **captured projection** `mean_token (h·d)²` per direction — a "used" direction
   should be both *present* (high projection) and *causal* (high KL); random dirs ≈ low/low.

## Families & counts (first run)
- **found**: 256 random dict atoms (seed 0).
- **contrastive**: 10 language + 45 pairwise = 55 diff-of-means directions (tagged separately).
- **random**: 256 Gaussian unit vectors (seed 0).

## Corpus size (first run)
- 512 sequences × 64 tokens = 32768 token positions. Batch = 16 seqs.

## Outputs (`results/ablate_kl/qwen3-1.7b_L14_40k/`)
- `kl_per_token.pt` — `[N_dir, n_tokens]` fp32 (≈74 MB).
- `summary.pkl` — per-direction: family, label, mean/median/p90/max KL, captured-proj.
- `hist_meankl.png` — overlaid distributions of per-direction mean-KL (found vs contrastive
  vs random), log-x.

## Notes / knobs to revisit after first look
- Subsampling 256/40k atoms characterizes the family distribution; can scale to all 40k later.
- Position 0 (BOS) included; can re-aggregate excluding it from the saved per-token matrix.
- KL direction is `KL(base‖ablated)` exactly as specified.
