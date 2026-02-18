# Plan: Multi-Contrastive Scaling Verification (k=2..10)

## Context

We've established multi-contrastive feature estimation with k=3 languages (EN, ZH, NL) on Qwen3-0.6B layer 14. Theory predicts the estimation error scales as sin(theta) = 1/sqrt(k) for orthogonal unit-norm features. We want to verify this empirically by scaling to k=10 languages, but we have **no ground truth** for the true feature directions.

**Key insight:** We can test self-consistency instead. If f_hat^(k) is the English estimator using k languages, then for orthogonal unit-norm features:

```
cos angle(f_hat^(k), f_hat^(k')) = sqrt(k'(k-1) / (k(k'-1)))
```

Using k'=10 as reference, the predicted angles for each k are:
- k=2: cos = 0.745 (41.8 deg)
- k=3: cos = 0.913 (24.1 deg)
- k=5: cos = 0.972 (13.5 deg)
- k=7: cos = 0.989 (8.5 deg)
- k=9: cos = 0.998 (3.6 deg)

If the measured angles match these predictions, the 1/sqrt(k) scaling is confirmed.

## Step 1: `data/multilingual_prompts.py` — Prompt data

**10 languages:** EN, ZH, NL, ES, FR, DE, JA, IT, RU, PT
- 35+ semantically matched prompts per language (same meaning across all 10)
- Short prompts (3-8 tokens each), similar to existing bilingual prompt style
- Single file with dict structure: `PROMPTS[lang] = [(prompt1, prompt2, ...), ...]`
- Can reuse/adapt existing EN-ZH and EN-NL pairs as a starting point

**Why these 10:** All well-supported by Qwen3-0.6B, mix of language families (Germanic, Romance, CJK, Slavic), and Francisco can personally verify ~6 of them (EN, ES, FR, PT, IT, DE).

**Token length matching:** This is the trickiest part. For each prompt set (same meaning across 10 languages), tokenize with Qwen3 tokenizer and compute max token length spread. Discard prompt sets where `max_tokens - min_tokens > 3` across any language. Start with ~35 candidates to have ~30 survive filtering. If too many are discarded, regenerate replacements targeting the problematic lengths.

**LLM-as-judge translation verification:** Before running the experiment, use Claude (via subagent) to verify all translations:
- For each prompt set, check that all 10 translations convey the same meaning
- Flag any semantic drift, mistranslations, or unnatural phrasing
- Especially important for JA/ZH/RU where we can't easily verify manually
- Fix flagged prompts before proceeding

## Step 2: `scripts/run_scaling_test.py` — Main experiment

### Phase A — Token length filtering (CPU, in script)
- Tokenize all prompts with Qwen3 tokenizer
- For each prompt set, compute token counts across all 10 languages
- Discard sets where `max - min > 3`
- Print summary: how many survived, which were dropped and why
- Assert at least 30 prompt sets remain

### Phase B — Language verification (~1 min)
- For each language, generate 20 tokens from 3 prompts with temp=0.7
- Print outputs for manual inspection
- Ensures Qwen3 actually generates coherent text in all 10 languages

### Phase C — Activation extraction (~1 min)
- For each language, extract final-token activations at layer 14 for all 30 surviving prompts
- Reuse `get_acts()` from `scripts/run_multi_contrastive.py`
- Result: `acts[lang]` tensor of shape [30, d_model] for each of 10 languages
- Total: 300 forward passes (trivial compute)

### Phase D — Pairwise diff-of-means directions
- For each pair (EN, lang_j) where j != EN, compute:
  - `d_Ej = normalize(mean(acts['en']) - mean(acts[lang_j]))`
- Result: 9 pairwise directions d_E1, ..., d_E9

### Phase E — Multi-contrastive estimation at each k
- For each k from 2 to 10:
  - Enumerate all C(9, k-1) subsets of the 9 non-English languages
  - For each subset S of size k-1:
    - Compute mean estimator: `f_hat_E = normalize(sum(d_Ej for j in S) / (k-1))`
    - (This is the mean-based multi-contrastive estimator for English using k languages)
  - Store all estimators for analysis
- Subset counts: C(9,1)=9, C(9,2)=36, C(9,3)=84, C(9,4)=126, C(9,5)=126, C(9,6)=84, C(9,7)=36, C(9,8)=9, C(9,9)=1 → 511 total estimators

### Phase F — Self-consistency analysis
- **Reference:** f_hat^(10) (the single k=10 estimator using all 9 contrasts)
- For each k=2..9, for each subset:
  - Compute cos angle(f_hat^(k, S), f_hat^(10))
- Compare distribution of measured angles against predicted:
  - `cos_predicted(k) = sqrt(10*(k-1) / (9*k))`
- Report: mean, std, and full distribution of cosines at each k

### Phase G — SVD-based estimator (comparison)
- For each k, also compute the SVD-based estimator:
  - Stack difference matrices D_E1, ..., D_E(k-1) → D_joint
  - SVD → leading right singular vector u_1
  - This should converge to the same direction (and potentially faster)
- Compare convergence rates: mean-based vs SVD-based

### Phase H — Save results
- Save to `results/scaling_test/qwen3-0.6b/`
- `scaling_results.pkl`: all directions, cosine measurements, predictions

## Step 3: `scripts/plot_scaling_test.py` — Visualization

**Plot 1: Scaling curve** (main result)
- x-axis: k (number of languages)
- y-axis: angle to k=10 reference (degrees)
- Points: mean +/- std across subsets at each k
- Line: theoretical prediction arccos(sqrt(10(k-1)/(9k)))
- This is the money plot — if points track the line, 1/sqrt(k) is confirmed

**Plot 2: Cosine similarity heatmap**
- All 9 pairwise d_Ej directions + f_hat^(10) + a few representative f_hat^(k) directions
- Shows the geometry of the language subspace

**Plot 3: Distribution of angles at each k**
- Violin or box plot showing spread across subsets
- Helps assess whether deviations from prediction are systematic or random

## Key files to reuse
- `scripts/run_multi_contrastive.py` — `get_acts()` for activation extraction (line ~220)
- `src/model.py` — `load_model()`
- Existing bilingual prompts in `data/` as starting point for multilingual prompts

## Compute estimate
- Activation extraction: 300 forward passes (~30 seconds)
- Multi-contrastive estimation: 511 normalize operations (instant, CPU)
- SVD: 511 small SVDs (instant)
- **Total GPU time: ~2 minutes** (dominated by language verification generation)
- Prompt creation is the real bottleneck (human effort)

## Execution order
1. Create `data/multilingual_prompts.py` with ~35 prompt sets in 10 languages (local, no GPU)
2. LLM-as-judge verification of translations (Claude subagent reviews all prompts, flags issues)
3. Fix flagged prompts, regenerate if needed
4. Create `scripts/run_scaling_test.py` and `scripts/plot_scaling_test.py` (local)
5. Sync to RunPod, run experiment (~2 min)
6. Sync results back, generate plots locally
7. Present results

## Verification
- f_hat^(10) should be close to existing f_E from the k=3 experiment (cos > 0.9)
- All 9 pairwise d_Ej should have positive mutual cosines (all "point toward English")
- At k=10, there's only 1 estimator, so no variance — the curve should end at 0 degrees
- The sum f_E + f_M + ... for any k languages should be approximately zero (by construction)
