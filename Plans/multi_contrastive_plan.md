# Plan: Multi-Contrastive Language Direction Isolation

## Context
We've been studying how different steering directions (MELBO_1, -mean, MELBO_1⊥mean) affect Mandarin CJK output in Qwen3-0.6B at layer 14. The current diff-of-means direction is computed from English-Mandarin pairs only, so it conflates "toward Mandarin" with "away from English." By adding Dutch as a third language, we can use multi-contrastive geometry to isolate individual language features more cleanly.

**Key prediction:** The multi-contrastive English estimator f_E = (d_EM + d_ED) / 2 should steer toward English better *from Dutch prompts* than d_EM alone, because d_EM contains Mandarin contamination irrelevant to Dutch.

## What we'll build

### Step 0: Data — New prompt pairs

**Extend `data/bilingual_prompts_dutch.py`:** Add 20 more EN-NL pairs (currently 30, need 50 to match EN-ZH).

**Create `data/bilingual_prompts_mandarin_dutch.py`:** 50 semantically matched ZH-NL prompt pairs. These need to be direct ZH-NL pairs (not bridged through English). **Important:** Match token lengths as closely as possible between the ZH and NL prompts in each pair (similar to the existing EN-ZH pairs where both sides are ~3-8 tokens).

### Step 1: `scripts/run_multi_contrastive.py` — Main experiment script

**Phase A — Collect activation differences**
- Load Qwen3-0.6B, layer 14
- Extract activations for all prompt sets (EN, ZH, NL) at final token position
- Reuse `extract_bilingual_diffs()` pattern from `run_melbo_constrained.py` (lines 37-57)
- Compute three difference matrices:
  - `D_EM = en_acts - zh_acts` [50 x d_model] from EN-ZH pairs
  - `D_ED = en_acts - nl_acts` [50 x d_model] from EN-NL pairs
  - `D_MD = zh_acts - nl_acts` [50 x d_model] from ZH-NL pairs
- Compute three diff-of-means directions: `d_EM`, `d_ED`, `d_MD` (each normalized)

**Phase B — Multi-contrastive mean estimators**
- `f_E_mean = normalize((d_EM + d_ED) / 2)` — English feature
- `f_M_mean = normalize((-d_EM + d_MD) / 2)` — Mandarin feature
- `f_D_mean = normalize((-d_ED - d_MD) / 2)` — Dutch feature
- Note: these are averages of the *mean directions*, not the raw difference matrices

**Phase C — MELBO on joint span**
- Stack all three difference matrices: `D_joint = cat([D_EM, D_ED, D_MD])` → [150 x d_model]
- SVD of D_joint → V_joint (joint language subspace)
- Run constrained MELBO on V_joint to find top-3 directions (reuse `optimize_direction()` from `run_melbo_constrained.py` lines 116-196)
- Also run MELBO on each single-pair span (V_EM, V_ED, V_MD) separately for comparison
- Same optimization settings: n_steps=300, lr=0.01, n_restarts=5, opt_angle=15°

**Phase D — Load FineWeb activations + reference logits**
- `load_fineweb_fixed_length(50, tokenizer, seq_len=5, seed=42)`
- `extract_context_and_activation()` at layer 14
- Compute reference log-probs

**Phase E — Perturbation sweeps**
- Sweep all directions: d_EM, d_ED, d_MD, f_E_mean, f_M_mean, f_D_mean, MELBO_joint_1/2/3, MELBO_EM_1, MELBO_ED_1, MELBO_MD_1, random
- Same angles as existing experiments: 60 points from 3° to 180°
- Reuse `sweep_single_direction()` and `sweep_random_directions()` from `run_melbo_constrained.py`

**Phase F — Cosine similarity matrix**
- Full pairwise cosines between all directions (pairwise diffs, multi-contrastive, MELBO variants)
- Print as formatted table

**Phase G — Save results**
- Save to `results/multi_contrastive/qwen3-0.6b/`
- `optimization_results.pkl`: all directions, MELBO results, cosine matrix
- `sweep_results.pkl`: all perturbation sweep data

### Step 2: `scripts/run_steering_multi_contrastive.py` — Steering evaluation on held-out language

**Key idea:** Test on Spanish (not used in training the directions) to see whether multi-contrastive estimators isolate *language features* rather than *contrastive-pair features*.

**Create `data/spanish_prompts.py`:** ~30 Spanish prompts for evaluation.

**Test:** Start from Spanish prompts, steer with each direction, measure what language comes out.
- Directions to test: d_EM, d_ED, d_MD, f_E_mean, f_M_mean, f_D_mean, MELBO_joint_1
- Prompts: 30 Spanish prompts
- Angles: [0, 5, 10, 15, 20, 25, 30, 45]
- Metrics: CJK fraction, Dutch-ness (e.g., common Dutch words/morphemes), English-ness, Spanish-ness — or just use a simple language detection heuristic
- 50 tokens per generation, temp=0.7

**Key prediction:** f_E_mean should steer Spanish→English more cleanly than d_EM (which carries Mandarin contamination irrelevant from Spanish's perspective). Similarly, f_M_mean should steer Spanish→Mandarin better than -d_EM.

### Step 3: `scripts/plot_multi_contrastive.py` — Visualization

- KL sweep comparison: pairwise diffs vs multi-contrastive estimators vs MELBO
- Cosine similarity heatmap
- Steering results comparison (CJK/Dutch fractions)

## Key files to reuse
- `scripts/run_melbo_constrained.py` — `extract_bilingual_diffs`, `optimize_direction`, `sweep_single_direction`, `sweep_random_directions`, `build_perturbed_hidden`, `extract_context_and_activation`, `forward_to_hidden_and_logits`
- `src/model.py` — `load_model`, `forward_from_layer_allpos`, `_get_blocks`, etc.
- `src/data.py` — `load_fineweb_fixed_length`
- `data/bilingual_prompts.py` — EN-ZH pairs (50)
- `data/bilingual_prompts_dutch.py` — EN-NL pairs (30, extending to 50)
- `scripts/run_steering_mandarin_fraction.py` — steering + CJK fraction infrastructure

## Execution order
1. Create/extend prompt pair files (local, no GPU needed)
2. Sync to RunPod
3. Run `scripts/run_multi_contrastive.py` (GPU, ~15-20 min estimate: activation extraction + 4x MELBO optimization + sweeps)
4. Sync results back
5. Run `scripts/run_steering_multi_contrastive.py` on RunPod (~15 min)
6. Sync results, run plotting script locally
7. Present results

## Verification
- Sanity check: d_EM from this script should match existing mean_dir from `optimization_results.pkl` (cosine ~1.0)
- f_E + f_M + f_D should sum to ~0 (they're defined to)
- Cosine(f_E_mean, d_EM) should be high but not 1.0 (the Dutch contrast adds information)
