# Fig 4 v3: denser robustness beeswarm, 6 models, 4 direction families

## What changes vs. v2
1. Drop the standalone "concept pair" column. Pairs become the *replicates* inside every other column.
2. Each remaining column is filled with all ~30 cross-family contrastive pairs (instead of 5 representative pairs).
3. Show **mean ± 2σ** (95% CI) as a translucent band per column, plus **median** as a tick (distinct color from mean).
4. Add 3 more models for diversity → 6 total.
5. **Direction family column**: {contrastive DoM, SAE latent, MELBO, random}, ~30 fits each.

## Models (6)
| Model | Org | Params | Layers | Perturb→Measure | Family / multilingual |
|---|---|---|---|---|---|
| Gemma-2-9B | Google | 9B | 42 | L2 → L40 | base |
| Qwen3-1.7B | Alibaba | 1.7B | 28 | L7 → L26 | existing |
| Llama-3.1-8B | Meta | 8B | 32 | L4 → L30 | existing |
| **Mistral-7B-v0.3** | Mistral | 7B | 32 | L4 → L30 | new, EU multi |
| **Aya-Expanse-8B** | Cohere | 8B | 32 | L4 → L30 | new, designed for 23 langs |
| **Yi-1.5-9B** | 01.AI | 9B | 48 | L6 → L46 | new, Chinese-leaning |

OLMo-2-7B skipped (mostly English). Phi-3.5-mini skipped (3.8B, size-out-of-band).

## Direction families (Gemma-2-9B at L=2, 30 directions each)
- **Contrastive DoM** — existing 25 named directions (Portuguese, Gender, Formal, Python, ...). Pair as in current Phase A (cross-family).
- **SAE latents** — Gemma Scope `gemma-scope-9b-pt-res` width-16k at L2. Pick top-30 by mean activation on the 30 FineWeb contexts (Option B). Use `decoder.W_dec[idx]` as the direction. Pair at random.
- **MELBO** — `run_melbo_diverse_capped.py` with `--n_directions 30 --target_kl 1.0 --opt_angle 15`. Strictly orthogonal by construction. Pair at random.
- **Random** — Gaussian unit vectors. Pair at random.

## Columns (in plot order, left → right)
1. **Direction family** — {DoM, SAE, MELBO, random} on Gemma at L2: 4 × 30 = 120 fits.
2. **Metric** (L² / cos / KL) — 30 Gemma DoM pairs × 3 metrics = 90 fits.
3. **Perturb layer** (L=2, 5, 10, 20 on Gemma) — 30 pairs × 4 = 120 fits.
4. **Measure layer** (L=30, 35, 40 on Gemma) — 30 pairs × 3 = 90 fits.
5. **Perturb method** (additive vs. spherical) — 30 Gemma pairs additive = 30 fits.
6. **Model** — 30 pairs × 6 models = 180 fits.
7. **Threshold** — 30 Gemma pairs × ~10 thresholds = 300 fits (post-hoc, free).

Total ≈ 930 fits vs. ~75 in v2.

## Compute (RTX 4090; ~4–5 min/sweep on Gemma-9B; Yi-48L ~50% slower)
| Pod name | Work | Hours |
|---|---|---|
| fig4-prep | MELBO 30 dirs (~1h) + SAE picks (~5min) + sweeps for SAE/MELBO direction-family columns (60 sweeps, ~5h) | ~6h |
| fig4-gemma-A | A v2 (with KL) 30 + G additive 30 | ~5h |
| fig4-gemma-E | E perturb-layer × 4 × 30 | ~10h |
| fig4-gemma-F | F measure-layer × 3 × 30 | ~7h |
| fig4-qll | Qwen 30 + Llama 30 | ~5h |
| fig4-mistral | Mistral 30 | ~3h |
| fig4-aya | Aya 30 | ~3h |
| fig4-yi | Yi 30 (48 layers) | ~5h |

8 pods. Could collapse to 6: merge Mistral+Aya sequentially (~6h) and merge prep into Gemma-A pod (~11h, push runtime to 12h).

Estimated total spend: ~$40–80 spot.

## Code changes
1. **`src/model.py`** — add aliases:
   - `mistral-7b-v0.3` → `mistralai/Mistral-7B-v0.3`
   - `aya-expanse-8b` → `CohereForAI/aya-expanse-8b`
   - `yi-1.5-9b` → `01-ai/Yi-1.5-9B`
2. **`scripts/perturbation/fig4_robustness_v2.py`** (fork of v1):
   - TARGETS includes 6 models with per-model perturb layer
   - All phases expand `REP_PAIRS` → `ALL_PAIRS_30`
   - Phase A always emits KL (re-run for Gemma)
   - **New phase I**: direction-family sweeps using a `directions` dict that's loaded from disk (DoM / SAE / MELBO / random sources)
   - Filename prefix `v2_` so existing pkls aren't clobbered
3. **`scripts/sae/extract_gemma_scope_l2_top30.py`** (new): load Gemma Scope L2, score on FineWeb contexts, save top-30 latent directions.
4. **`scripts/melbo/run_melbo_diverse_capped.py`** (existing, no changes) — invoke with `--model gemma-2-9b --layer 2 --n_directions 30 --target_kl 1.0`.
5. **`scripts/perturbation/plot_fig4_v3.py`** (new):
   - 7 columns
   - Per column: scatter dots + mean line + ±2σ translucent band + median tick (orange median, blue mean)
   - Auto-fit y-range
   - Same Inter font/palette

## Sanity / risks
- Some pairs will give p=None on weaker-multilingual models — filter, count, report.
- Aya is instruction-tuned; if its hidden states behave oddly, swap for Mistral-Nemo-12B base.
- Yi 48 layers means longer sweeps; budget 5h, may stretch.
- MELBO `target_kl=1.0` may give directions whose θ_50 differs systematically from DoM's. If so, tune target_kl to match median θ_50 (~15°). Quick check after first 5 directions.
- SAE latents: `decoder.W_dec[idx]` is at residual stream norm scale (~unit), should be directly usable. Verify the unembed direction matches what we expect by sanity-steering.

## New repo
Defer until v3 is rendered and we're happy with all figures. Then extract a clean repo `error-correction-perturbations` with just the code that produces the paper's figures.

## Open questions for Francisco
- 8 pods OK, or collapse to 6?
- MELBO `target_kl=1.0` default OK to start, or pick something else?
- New repo: defer until after v3 (my lean), or do it now?
