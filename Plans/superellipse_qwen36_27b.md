# Plan: Superellipse p-fitting on Qwen3.6-27B — J-Lens + contrastive directions

**Date:** 2026-07-22
**Goal:** Repeat the superellipse p-fitting on Qwen3.6-27B, the open model where
Neel Nanda's group replicated the Anthropic global-workspace / J-Lens results.
Two direction families on the same model, same machinery:
1. **J-Lens token-direction pairs** (EN vs ZH), as done on qwen3-8b L20.
2. **Contrastive direction pairs**, using the error-correction-paper prompt
   inventory (already in this repo under `scripts/exp_map/` + `data/`).
Question: does p ≈ 2 hold here, or do we see p > 2 (feature privilege)?

## Facts gathered
- **Model:** `Qwen/Qwen3.6-27B`. Config: 64 text layers, d_model 5120, vocab
  248,320. Hybrid architecture: 3 linear-attention (mamba-style) layers per 1
  full-attention layer. Multimodal wrapper class
  `Qwen3_5ForConditionalGeneration` (text model inside; vision tower small,
  irrelevant for us). Needs transformers ≥ 4.57.
- **Pre-fitted lens exists:** HF `neuronpedia/jacobian-lens`, folder
  `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`
  (3.3 GB, presumably bf16 [d,d] per layer like the qwen3-8b one — verify
  format on pod with adapted `download_inspect_lens.py`).
- **Layer band (Neel's review, LW post "A Review of Anthropic's Global
  Workspace Paper"):** replication by Nanda + MATS scholars (Camila Blank,
  Agam Bhatia); their J-Lens used Jacobians to the penultimate layer, 25 Pile
  prompts × 128 tokens. Workspace band ≈ 55–80% depth → L35–51 of 64.
  Interpretative meta-tokens salient ~L31–39, genre tokens ~L40–50.
  CKA on their fig shows blocks ~L18–33 and ~L34–51 (less clean than paper's).
- **Contrastive inventory (error-correction paper, App. B):** 33 directions —
  14 binary semantic + 10 natural languages + 9 programming languages, 30–50
  matched prompt pairs each; diff-of-means of last-token activations at the
  perturbation layer, unit-normalised; 528 pairs → 318 kept at |cos| < 0.1.
  All prompt data + registry live in `scripts/exp_map/lib/registry.py` and
  `data/*.py` — nothing new to author.

## Decisions (Francisco, 2026-07-22)
1. **Perturbation layer: L36** (56% depth, matches qwen3-8b convention;
   inside Neel's workspace band L35–51). Measure layer: n_layers−2 = **L62**.
2. **Contrastive pair budget: ~30 cross-family pairs** (fig4-robustness
   style), 30 anchors each. Expand only if results warrant.
3. **GPU:** single 80 GB card (H100 / H200 / A100-80G), bf16.

## Runtime estimate
- Per J-Lens pair: 30 anchors × 41×41 grid ≈ 50k forwards through L36→L62
  (26/64 blocks, ~10B params traversed), 5-token seqs, batch 64 →
  ~1–3 min/pair on H100 (memory/overhead-bound at this seq length).
- J-Lens leg (~20 pairs): ~0.5–1 h. Contrastive leg (30 pairs, 31×31 grid):
  ~0.5–1.5 h. Direction extraction: minutes.
- Fixed costs: 54 GB model download (unless on network volume), 3.3 GB lens,
  venv with transformers ≥ 4.57, forward-consistency check, threshold
  diagnostic.
- **Total: ~1–3 h GPU compute; ~half a day pod wall-clock end-to-end.**
  Main uncertainty is the qwen3_5 architecture adaptation, not sweep time.
  Time the first pair explicitly and report before running the full queue.
  Cost ~$15–30 at H100/H200 rates.

## Steps
1. **Model support (local code, verify on pod):**
   - Add `qwen3.6-27b` → `Qwen/Qwen3.6-27B` to `src/model.py` registry.
   - Block access: text layers likely at `model.model.language_model.layers`;
     extend `_get_blocks` / `_get_final_norm` accordingly.
   - `forward_from_layer_to_layer` must run hybrid qwen3_5 blocks: linear
     attention layers may need different kwargs than full-attention ones.
     **Mandatory sanity check on pod:** manual forward from L0 ≡ full forward
     (hidden-state match) before any sweep.
   - Pin/install transformers ≥ 4.57 in pod venv.
2. **J-Lens leg** (mirror `scripts/jlens/`, new scripts, don't overwrite):
   - Adapt `download_inspect_lens.py` → qwen3.6-27b folder; verify format.
   - `build_jlens_directions.py` variant: J_L^T @ W_U rows at chosen layer;
     same EN/ZH candidate lists (incl. Neel's interpretative meta-tokens
     什么意思 / 是什么意思 / 这句话 / 是何含义 — verify single-token in the
     248k vocab); |cos| ≤ 0.2 matching, ~10 pairs per category.
   - Sweep script variant: 30 FineWeb 5-token base prompts (cache keyed by
     tokenizer!), 41×41 grid to 20°, measure L2 at n_layers−2 = **L62**,
     checkpoint per pair.
   - **Threshold recalibration:** run `inspect_sweep_range.py`-style
     diagnostic first (noise floor / axis max at L62 response), then set
     L2_THRESHOLDS; don't reuse qwen3-8b values.
3. **Contrastive leg** (exp_map machinery):
   - Add a `qwen36` target to `registry.TARGETS` with the chosen layer.
   - Extract 33 directions at that layer (cheap), compute 33×33 overlap,
     apply |cos| < 0.1 filter, pick pair subset per decision 2.
   - Run `sweep_2d.py`-style sweeps at 30 FineWeb anchors + superellipse fits
     with recalibrated thresholds (same threshold diagnostic as J-Lens leg).
4. **Analysis/plots:** beeswarm p by family (J-Lens EN, J-Lens ZH,
   Contrastive), per-pair shape panels — same style as
   `results/jlens/qwen3-8b/` and the paper beeswarms.
5. **Outputs:** `results/jlens/qwen3.6-27b/` and
   `results/exp_map/.../qwen3.6-27b/`; everything checkpointed per pair.

## Status
- [x] Code: `src/model.py` qwen3_5 support (alias `qwen3.6-27b`, blocks at
      `model.model.language_model.layers`, mrope rotary, uniform block call)
- [x] Code: J-Lens leg scripts (`scripts/jlens/*_qwen36.py` + generic range
      inspector + `verify_qwen36_forward.py`)
- [x] Code: `qwen36` target in exp_map registry + `select_qwen36_pairs.py`
- [x] Pod: B200 (183 GB) `runpod-qwen36-sup`, cu128 venv `~/venv-q36`,
      transformers 5.14.1
- [x] Verify forward consistency (cos=1.0 all legs) + timing (~1.6 min/pair)
- [x] J-Lens leg: lens (format = qwen3-8b), directions, sweeps, ranges, fits.
      EN=0 pairs (see below); ZH=10 pairs done.
- [x] Contrastive leg: 33 dirs, 30 cross-family pairs, sweeps, thresh=84.9
      (f=0.4×median axis-max), fits done
- [x] Plots + combined summary

## Results (2026-07-22) — COMPLETE, pod terminated
Superellipse p (perturb L36, measure L62):
- **Contrastive: median p = 2.45** (n=30, mean 2.53, IQR [2.38, 2.60], 97%>2).
  Replicates the paper's contrastive FSEC result on the 27B, slightly higher.
- **J-Lens ZH-10: median p = 1.88**; **J-Lens ZH-100: median p = 1.88**
  (n=100, IQR [1.76, 2.01], 28%>2) — scale-up confirms it robustly.
- **J-Lens EN concrete (|cos|≤0.25, n=10): median p = 2.04** (IQR [1.85, 2.40]).
- → All three J-Lens families sit AT p≈2, not privileged; contrastive is the
  only family clearly above. Consistent with qwen3-8b J-Lens (~1.6–2.0).

1D responsiveness (single-direction L2 response vs angle, 20 dirs × 30 anchors):
- Median L2 at 10/20/40/60°: J-Lens ZH 35/71/170/268, Contrastive 26/55/144/261,
  Random 16/31/73/141. J-Lens ZH and Contrastive both ~2× Random; J-Lens ZH
  slightly ABOVE contrastive at small/mid angles, converging by 60°.
- → J-Lens directions ARE privileged in raw responsiveness (like contrastive,
  unlike random) even though their 2D iso-response contour is p≈2, not p>2.
  The two measures decouple.

Files (all synced local):
- `results/jlens/qwen3.6-27b/`: p_summary_combined.txt, jlens_p_summary_L36{,_zh100,_en_concrete}.*,
  qwen36_p_beeswarm.png (4 families), response_1d_comparison_L36.{png,pkl}
- `results/exp_map/data/fits/fits_qwen36_L36.pkl` (contrastive)
- Scripts: scripts/jlens/*_qwen36.py, response_1d_comparison_qwen36.py,
  build_jlens_zh_pairs_qwen36.py, concrete_en_pairs_qwen36.py

## EN J-Lens pairs issue (documented null → expanded)
- Original 30 concrete-noun EN candidates: 0 pairs under |cos|<=0.2. Not a
  bug: EN J-Lens vectors at L36 share a dominant direction (mean-vec norm
  0.68–0.71, median pairwise |cos| ~0.48), while raw W_U rows are near-
  orthogonal (median 0.056) → the Jacobian injects the shared component.
  ZH far less affected (mean-vec 0.49, median 0.22) → ZH pairs form fine.
- Vocab scan (3000 EN word-tokens): 227 pairs under 0.2, but dominated by
  code/technical tokens (` nullptr`, ` argc`, ` charset`…) that escape the
  shared direction; concrete nouns cluster on it.
- Curated 305 concrete nouns: only 5 pairs at |cos|<=0.2, 18 at 0.25.
  → concrete-noun EN leg needs threshold 0.25 for 10 pairs. AWAITING
  Francisco's call (5 @ 0.2 vs 10 @ 0.25). Pkls: `jlens_en_pairs_expanded_L36.pkl`
  (code-token, 0.2), `jlens_en_concrete_pairs_L36.pkl` (not written — <10 @ 0.2).

## Risks
- Hybrid linear-attention blocks break the manual block-iteration forward →
  caught by the L0 consistency check; fallback is hook-based full forward
  (slower but architecture-agnostic).
- 27B sweep cost: per grid point ~3× qwen3-8b (d² and layer-count scaling),
  offset by faster GPU. Time one pair first; scale pair count to fit budget.
- Lens .pt format may differ from qwen3-8b (n1000 naming) → inspect first.
