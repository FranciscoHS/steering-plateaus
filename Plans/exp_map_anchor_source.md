# Anchor-source ablation

## Goal
Add an "anchor source" column to the robustness beeswarm. Currently every fit
in `fits_gemma_L2.pkl` uses the same 30 FineWeb-edu prompts as anchors. We
want to verify the fitted superellipse exponent `n` is robust to that choice
by re-running all 528 pairs with anchors drawn from three additional sources:

1. **Wikipedia (English)** — curated long-form text, very different from
   FineWeb's web-crawl distribution.
2. **Wikipedia (Mandarin/Chinese)** — same domain as English but a different
   *language*; tests whether the boundary geometry depends on which "language
   subspace" the anchors live in.
3. **Code** — a syntactic / token-sparse distribution that lives elsewhere in
   residual space.

Plus the existing FineWeb-edu as the baseline → 4 sub-groups in the new
column.

## Data sources

| label | HF dataset | split / config | filter |
|---|---|---|---|
| fineweb (existing) | `HuggingFaceFW/fineweb-edu` | `train` | first 30 prompts (existing seed) |
| wiki-en | `wikimedia/wikipedia` | `20231101.en` | random 30, seeded |
| wiki-zh | `wikimedia/wikipedia` | `20231101.zh` | random 30, seeded |
| code | `bigcode/the-stack-smol` | `train` | random 30 from `python` subset, seeded |

Per-anchor protocol must match `fineweb_acts` exactly (same token-window
length, last-token residual at perturb layer, same dtype handling). Anchor
caches written to `fineweb_cache_<target>` analogues:
`wiki_en_cache_<target>`, `wiki_zh_cache_<target>`, `code_cache_<target>`.

Sanity checks before launching the full sweep:
- `||a||` distribution per source — if Mandarin or code anchors have
  drastically different norms, that's a real distribution shift to flag.
- Tokenizer round-trip on a few examples (esp. zh) — confirm Gemma-2's
  tokenizer doesn't fragment Chinese into byte-fallback noise.

## Code changes

1. **`scripts/exp_map/lib/activations.py`** — add three functions matching
   `fineweb_acts`'s signature:
   - `wiki_acts(model, tokenizer, device, target, layer, n=30, language="en")`
   - `code_acts(model, tokenizer, device, target, layer, n=30)`
   Refactor common logic (load → tokenize → forward to layer → cache last-token
   residual) into `_anchor_acts(loader_fn, cache_dir, ...)`. Each new function
   provides its own `loader_fn` returning `(contexts, prompts)` and the
   cache-dir path.

2. **`scripts/exp_map/sweep_2d.py`** — add CLI flag:
   ```
   --anchor_source {fineweb, wiki_en, wiki_zh, code}    default fineweb
   ```
   Dispatches to the right `*_acts` function. Filename gets a `_src<label>`
   suffix when `anchor_source != fineweb` (default stays the existing
   filename for back-compat). Encoded in the PKL header as
   `anchor_source: <label>`.

3. **`scripts/exp_map/fit_pairs.py`** — needs to know which sweep files to
   read for which source. Add an `--anchor_source` flag mirroring sweep_2d's,
   so fits land in `fits_gemma_L2_src<label>.pkl`. Default fits filename
   unchanged.

4. **`scripts/exp_map/plotting/plot_robustness_beeswarm.py`** — add a new
   column loader:
   ```python
   def col_anchor_source(target="gemma", layer=2):
       out = {}
       default = load_default_fits(target, layer)
       if default: out["FineWeb-edu"] = list(default.values())
       for label, suffix in [("Wikipedia (en)", "_srcwiki_en"),
                             ("Wikipedia (zh)", "_srcwiki_zh"),
                             ("Code",           "_srccode")]:
           fp = f"{FITS_DIR}/fits_{target}_L{layer}{suffix}.pkl"
           ... load + collect ...
   ```
   Wire into `main()` so `--columns ...,anchor_source,...` adds it.

## Compute / runtime estimate

Per-source full run on Gemma-2-9b, L=2, all 528 pairs (= `all_pairs()` Tier-1
plus extensions), 30 anchors, 31² grid:

- Existing FineWeb run took ~3 hr on an RTX 4090 / A100-class GPU per the
  prior `fits_gemma_L2.pkl` audit.
- 3 new sources × ~3 hr ≈ **9–10 GPU-hours** total for the sweep step.
- Fit step (no GPU, runs in ~minutes) on top.

Comfortably fits in one pod-day with `--runtime 600`.

## Run plan

Order chosen so we get a result early even if the pod dies:

1. **Pilot, 1 pair (Gender, Tense), 3 sources sequentially** — ~10 min total.
   Verify `||a||`, tokenization, and that the L² grids look sensible (no
   nan, similar order of magnitude across sources). Print summary inline.
2. **Full sweep, wiki-en** — 528 pairs, ~3 hr. Write
   `sweep2d_gemma_L2_*_fineweb_60deg_srcwiki_en.pkl` per pair. The literal
   `fineweb` token stays in the filename (kept for back-compat with
   existing tooling); only the `_src<label>` suffix marks the actual anchor
   source.
3. **Full sweep, wiki-zh** — ~3 hr.
4. **Full sweep, code** — ~3 hr.
5. **Fit step**: `fit_pairs.py --anchor_source <label>` for each of the 3
   new sources → `fits_gemma_L2_src<label>.pkl`.
6. **Replot beeswarm** — adds the new column. Compare medians/means to the
   existing FineWeb-edu values.

Pod hygiene: existing `runpod-lowthresh` pod is fine to reuse if still up.
Otherwise spin a new `--name anchorabl` A100/H100 pod, transfer code via
`scp`, leave HF cache pointing at `/network/hf_cache` (Gemma already there).

## Settled decisions

- **Filename token**: `fineweb` literal stays in sweep filenames (back-compat
  with existing tooling); the `_src<label>` suffix is what actually
  distinguishes anchor sources.
- **Code dataset**: `bigcode/the-stack-smol`, default (Python) subset.
- **Directions**: fixed — DoM directions are re-used from the existing
  FineWeb-derived extraction. We vary *only* the evaluation-anchor
  distribution.

## Output / deliverables

- `results/exp_map/data/sweeps_2d/sweep2d_gemma_L2_*_fineweb_60deg_src<label>.pkl`
  for each of `wiki_en`, `wiki_zh`, `code`, and each of 528 pairs.
- `results/exp_map/data/fits/fits_gemma_L2_src<label>.pkl` for each new
  source.
- Updated `results/exp_map/figures/robustness_beeswarm_gemma_L2.{png,pdf}`
  with the "Anchor source" column.
- A short log of `||a||` distributions per source for the writeup methods
  section.
