# MWE-derived contrastive concept directions

## Goal
Add ~30 new contrastive directions extracted from Anthropic's Model-Written
Evaluations (MWE) — a much broader concept set than our current 15 sem +
11 lang + 9 code directions. Run pairwise iso-perturbation sweeps with
the new directions on Gemma-2-9b at L=2 and update the beeswarm to see
whether the n≈2 result holds across this broader concept set.

## Source data
- Repo: `nrimsky/CAA` provides 7 ready-packaged concepts in
  `datasets/generate/<concept>/generate_dataset.json` with format
  `(question, answer_matching_behavior, answer_not_matching_behavior)`.
  Refusal is already in our set; the other 6 are new:
  *coordinate-other-ais, corrigible-neutral-HHH, hallucination,
  myopic-reward, survival-instinct, sycophancy*.
- Upstream: `Anthropic/model-written-evals` on HF, ~155 jsonl files in
  `advanced-ai-risk/`, `persona/`, `sycophancy/`. Same triple format,
  500–1000 items per file.

## Concept selection
Extract DoMs for **all ~155 MWE files** (DoM extraction is cheap,
~1.5 GPU-hr total). Then **post-hoc filter to ~1000 pairs with
smallest raw overlap** (or all pairs with |⟨d_i, d_j⟩| < 0.15,
whichever is smaller). The methodology filter we already use does
the concept selection for us — no manual curation needed.

After filtering we'll have 33 existing + 155 new = 188 directions,
C(188, 2) = 17,578 candidate pairs, of which ~1000 survive the
overlap cut. Sweeps run only on those.

## Pipeline integration

1. **Adapter `data/mwe_prompts.py`**: takes a path to a MWE-style
   jsonl/json file and emits two parallel prompt lists `(positive,
   negative)` of equal length. Each item in the source becomes:
   - `pos = question + " " + answer_matching_behavior`
   - `neg = question + " " + answer_not_matching_behavior`
   The two strings are minimal pairs (only the trailing `(A)` vs `(B)`
   differs), as in CAA's recipe.

2. **Registry**: register each concept in
   `scripts/exp_map/lib/registry.py:DIRECTIONS` with a fixed sign
   convention (`matching_behavior → not_matching_behavior`). Keep
   short Python names (e.g. `Sycophancy`, `Power`, `Agreeable`,
   `Openness`, `PolLiberal`, `MemoryEraseNo`, …).

3. **Extraction**: call existing `dirlib.extract_all` on the new names
   to populate `dirs_gemma_L2.pkl`. No code change needed — extract_all
   reads the registry. Extraction is cheap (~minutes).

4. **Pair selection + sweeps**: compute all C(188, 2) = 17,578
   pairwise raw-DoM overlaps (cheap — ~10 ms). Take the
   intersection of (a) overlap < 0.15 and (b) bottom-1000 by
   overlap; run `sweep_2d.py --target gemma --layer 2 --pairs ...`
   on just that filtered set. ~1000 pairs × ~21 s/pair ≈ 6 GPU-hr.
   The existing 528 pairs from the canonical fits stay as-is; only
   the new pairs (where at least one direction is from MWE) get
   swept now.

5. **Fits**: `fit_pairs.py` over the extended sweep set →
   regenerates `fits_gemma_L2.pkl` with all 1953 pairs.

6. **Plotting**: re-run the existing beeswarm. New optional
   pair-kind classifier `_pair_kind` to add a "concept" or "MWE"
   bucket alongside the existing sem/lang/code so the new
   directions show up as their own group. Keep the overlap<0.15
   filter applied throughout.

## Run plan

**Phase 1 — Pilot (1 concept, 1 GPU)**
Add `Sycophancy` from CAA. Extract DoM. Inspect:
- ‖d_sycophancy‖, sign sanity (does pos − neg point in the
  expected direction on a held-out probe?).
- Pairwise overlap of `Sycophancy` with the existing 33 DoMs
  — flag any with |⟨·,·⟩| > 0.3 (would tell us this concept is
  not orthogonal to existing axes).
- Run sweep_2d for the 33 Sycophancy×<existing> pairs (~10 min);
  fit; spot-check 3 pairs visually.

**Phase 2 — Full extraction**
Same RunPod setup as before (A100/H100 80GB, `--runtime 600`,
`HF_HOME=/network/hf_cache`). Extract DoMs for all ~155 MWE
concepts (~1.5 GPU-hr). Save extended `dirs_gemma_L2.pkl`. Skip
files that fail prompt-format sanity (multi-turn, no clean A/B at
end, etc.) — log skipped concepts.

**Phase 3 — Filter + sweep**
Compute the 17,578 pairwise overlaps; pick the bottom-1000 by
overlap that also satisfy < 0.15. Run `sweep_2d.py --pairs ...`
restricted to those (~6 GPU-hr). `--shard` for parallel pods if
useful.

**Phase 4 — Fits + plots**
Local: `fit_pairs.py`, regenerate beeswarm with overlap<0.15
filter. Add a "concept group" pair-kind column showing
sem×concept, lang×concept, code×concept, concept×concept buckets.

## Decisions wanted before kickoff
- **Prompt format**: CAA's full-question-plus-letter recipe
  is the default. Revisit if DoMs come out heavily aligned
  with persona-prefix artifacts in the pilot.
- **Pair budget**: 1000 lowest-overlap (~6 GPU-hr)? Bigger
  (~2000)? Smaller?

## Output / deliverables
- `data/mwe_prompts.py` adapter module.
- New entries in `registry.DIRECTIONS` with associated prompt
  modules.
- Extended `dirs_gemma_L2.pkl` with ~63 directions.
- `sweep2d_gemma_L2_<a>__<b>_fineweb_60deg.pkl` files for the
  1425 new pairs.
- Extended `fits_gemma_L2.pkl`.
- Updated `robustness_beeswarm_gemma_L2_ov0p15.{png,pdf}` with
  the new concepts visible.
- A short methods note for the writeup describing concept
  selection and the prompt-construction choice.
