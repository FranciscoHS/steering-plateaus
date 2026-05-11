# Plan: extending the robustness beeswarm with Wealth/Health/Certainty

## Goal
Add the three new contrastive directions — `Wealth`, `Health`, `Certainty`
— to every column of the canonical robustness beeswarm
(`scripts/exp_map/plotting/plot_robustness_beeswarm.py`). The pruned
33-direction set (drop `HonestyShort`, `TensePresent`, `Formal`; keep
the 11 retained semantic + 10 lang + 9 code; add the 3 new) gives
**318 pairs** at `--max_overlap 0.10`, matched in size to the existing
baseline (318 pairs at the same cutoff). Pruning is filter-only at
plot time — all existing sweep PKLs remain valid; the only NEW sweeps
needed are pairs that *touch a new direction*.

## Pruning policy (no rerun, plot-time only)
- Drop from beeswarm: `HonestyShort` (overlap 0.887 with `Honesty`),
  `TensePresent` (0.557 with `Tense`), `Formal` (0.287 with `Literary`,
  user-chose to keep `Literary`).
- Existing sweep PKLs that *involve* these three directions stay on
  disk; the beeswarm filter just drops them from `_pair_kind` /
  `col_*`. Implement by filtering `pair` keys against an
  excluded-names set in `_filter_by_overlap` (or a sibling).

## What needs sweeping for each new direction
For each `(target, layer, variant)` config used by the beeswarm, the
new pairs are:

```
new_pairs = {(W, x), (H, x), (C, x) for x in kept_dirs} ∪ {(W,H), (W,C), (H,C)}
```

with `kept_dirs = 11 sem + 10 lang + 9 code = 30`. That's
`3*30 + 3 = 93` new pairs per config. After `overlap < 0.10` plot-time
filter, ≈ 70 land in the figure (Health × Sentiment, etc. drop).

## Configs to rerun
Each row of this table is one independent run that lands one set of
new sweep PKLs.

| # | Target  | Perturb L | Variant flags                       | New sweeps | Why                          |
|---|---------|-----------|--------------------------------------|------------|------------------------------|
| 1 | gemma   | 2         | (canonical: geodesic, l2, fineweb, M=−2) | 93         | Model column (gemma), default cells |
| 2 | gemma   | 2         | `--measure_offset -7`               | 93         | Measure layer (penult-5)     |
| 3 | gemma   | 2         | `--measure_offset -12`              | 93         | Measure layer (penult-10)    |
| 4 | gemma   | 2         | `--metrics cos+kl+l2`               | 93         | Metric column (cos, KL)      |
| 5 | gemma   | 2         | `--mode additive`                   | 93         | Method column                |
| 6 | gemma   | 2         | `--anchor_source wiki_en`           | 93         | Anchor source                |
| 7 | gemma   | 2         | `--anchor_source wiki_zh`           | 93         | Anchor source                |
| 8 | gemma   | 2         | `--anchor_source code`              | 93         | Anchor source                |
| 9 | gemma   | 5         | (canonical)                         | 93         | Perturb layer column (L=5)   |
|10 | gemma   | 10        | (canonical)                         | 93         | Perturb layer column (L=10)  |
|11 | gemma   | 20        | (canonical)                         | 93         | Perturb layer column (L=20)  |

Cross-model column (qwen / llama / mistral / aya / yi) — **decided
not to add W/H/C** in those models. The Model column keeps its
existing 528-pair non-gemma slots; only the gemma slot picks up the
new 3 directions through config #1.

Total: **11 configs × 93 = 1023 new sweeps.** At ~24 s/sweep that's
**~6.8 GPU-hr** cumulative.

Threshold column (col_threshold) re-fits the canonical sweeps at
25/50/75% of per-pair max — no new sweeps required.

Direction-family column (col_direction_type) — only DoM gains the new
dirs. SAE / MELBO / random use their own 33-dir caches (not these
contrastive sets) and do not need updating.

## Extraction prerequisites
For each config the per-(target, layer) directions cache must contain
`Wealth`, `Health`, `Certainty`. Currently only `gemma L=2` has them.

| Cache pkl                          | Currently has W/H/C? | Action |
|------------------------------------|----------------------|--------|
| dirs_gemma_L2.pkl                  | yes                  | none   |
| dirs_gemma_L5.pkl                  | no                   | extract |
| dirs_gemma_L10.pkl                 | no                   | extract |
| dirs_gemma_L20.pkl                 | no                   | extract |

Extraction is cheap: 3 dirs × 30 prompts × 2 sides × ~50 ms ≈ 9 s of
forwards plus the per-(target, layer) model load. ~1–2 min wall per
cache. `dirlib.extract_all` is already idempotent — pass
`names=["Wealth", "Health", "Certainty"]` and it extends the existing
cache.

## Pod plan (parallelized, target wall ≈ 3 hr from now)

`mwepilot` continues Phase 3b independently — none of these new
configs depend on it. We allocate three pods alongside it and don't
wait. (User direction: "I wouldn't wait on mwepilot at all.")

```
                   t=0       t=1h      t=2h      t=3h
mwe-sweep1d | #1 canonical | #2 M-7  | #3 M-12 | #4 cos+kl | #5 additive
pod-layer   | extract L=5  | #9 (L=5) | extract L=10 | #10 (L=10) | extract L=20 | #11 (L=20)
pod-anchor  | #6 wiki_en   | #7 wiki_zh | #8 code
```

- `mwe-sweep1d` (existing): linear queue of gemma L=2 variants:
  canonical → M-7 → M-12 → cos+kl → additive (~3 hr total at
  ~38 min/run).
- `pod-layer` (new): gemma at L=5 / L=10 / L=20. Each layer needs a
  ~2 min DoM extraction first because dirs are layer-specific
  (`dirs_gemma_L<L>.pkl` does not yet have W/H/C). ~2.5 hr.
- `pod-anchor` (new): wiki_en / wiki_zh / code anchor sources. Same
  gemma L=2 dirs cache as `mwe-sweep1d`, so no extra extraction.
  ~1.9 hr — finishes early; pod can be torn down or held as buffer.
- All pods write to the shared network volume, so PKLs collect into
  `results/exp_map/data/sweeps_2d/` automatically.

If only one new pod is available we degrade gracefully:
- 1 new pod (so 2 active) → ~5 hr wall.
- 0 new pods (just `mwe-sweep1d`) → ~7 hr wall.

## Implementation pieces (small, all already partially in place)
1. **One-shot launcher script**: `scripts/exp_map/_pod_run_new3.sh`
   takes `(config_name)` as arg, dispatches the right
   extraction + sweep_2d invocation. One pod runs one config.
2. **Pair list helper**: a small `select_new3_pairs.py` that emits the
   93-pair CSV per `(target, layer)` once dirs are extracted. Pruned
   set of 30 + 3 new is hard-coded so we don't depend on registry
   pruning.
3. **Beeswarm patch**: extend `plot_robustness_beeswarm.py` to (a)
   accept an `--exclude_dirs` flag (drops pairs touching any name in
   the list — applies the prune at plot time), (b) ensure the new 3
   names get the right `_pair_kind` (sem, since they are
   contrastive-DoM directions, not MWE).
4. **fit_pairs.py invocations**: re-run for each `(target, layer,
   suffix)` once all sweeps for that config are on disk. 16 invocations
   total. Each takes ~2 min (200 bootstraps × ~93 new pairs).
5. **Final beeswarm**: once all fits land, regenerate
   `robustness_beeswarm_gemma_L2_ov0p1*.{png,pdf}` at 0.10, then a
   second version excluding HonestyShort/TensePresent/Formal.

## Decisions wanted before kickoff
- **Pair count for the prune**: confirm the 11-keep semantic list
  (Age, Era, Gender, Honesty, Literary, Number, Person, Refusal,
  Sentiment, Status, Tense). Anything else to drop or add?
- **Cross-model extra-direction strategy**: extract W/H/C in *all 6
  models* (current plan) vs. only gemma. The cross-model column is
  meant to test "n=2 across models," so adding the new dirs there
  matches the spirit of the figure.
- **GPU budget**: ~10 GPU-hr cumulative across all pods. Acceptable?

## Out of scope for this plan
- Re-running the 1000 MWE-touching sweeps under all conditions. The
  MWE-touching pairs are on `gemma L=2 / fineweb / l2 / penult /
  geodesic` only. Extending MWE to all 16 conditions × ~1000 pairs
  ≈ 16k sweeps would be a separate, much larger project.
- Re-running existing 528 canonical sweeps. They're done.
- Direction-family column for the new 3 (DoM only — SAE/MELBO/random
  use unrelated dir sets).
