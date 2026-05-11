# Plan: extend Wealth/Health/Certainty to the cross-model column

## Background
The first new3 plan (`exp_map_new3_directions.md`) only added W/H/C
DoMs to gemma. That was a misinterpretation: the user's earlier
"gemma only" referred to anchor sources, not directions. The Model
column in the canonical robustness beeswarm therefore shows
asymmetric direction sets — gemma 33 dirs (incl. W/H/C), other
models 30 dirs (no W/H/C). Same fix as the gemma rerun, applied to
the other 5 models.

## Scope (deliberately narrow)
Only the canonical L=2 condition needs the new directions for the
non-gemma models — the variant axes (perturb-layer, measure-layer,
metric, threshold, method, anchor-source) are **gemma-only columns**
in the beeswarm, so the existing fits for those still apply.

| target  | layer | what's missing                            |
|---------|-------|-------------------------------------------|
| qwen    | 2     | W/H/C DoMs + 93 new sweep pkls + refit    |
| llama   | 2     | W/H/C DoMs + 93 new sweep pkls + refit    |
| mistral | 2     | W/H/C DoMs + 93 new sweep pkls + refit    |
| aya     | 2     | W/H/C DoMs + 93 new sweep pkls + refit    |
| yi      | 2     | W/H/C DoMs + 93 new sweep pkls + refit    |

5 conditions × 93 candidate pairs each = up to **465 new sweep pkls**,
plus 15 DoM extractions (3 dirs × 5 models). After the
**|⟨d_i, d_j⟩| < 0.10 pre-filter** (see below), the actual sweep
count drops to ~280–320 (rough estimate, ≈ 60–70% survival, varying
per model). The Pruned-33 (drop HonestyShort/TensePresent/Formal) is
plot-time-only and doesn't require new sweeps.

### Overlap pre-filter
The beeswarm filters pairs with `|⟨d_i, d_j⟩| ≥ 0.10` at plot time
(Gram-Schmidt artifact concern). No reason to sweep pairs we'll
discard. After DoM extraction step but before the sweep step:

1. Compute the 93 candidate-pair overlaps using the just-extended
   `dirs_<tgt>_L2.pkl` (cheap, ~10 ms per pair).
2. Drop any pair with overlap ≥ 0.10.
3. Write the surviving pair list to a CSV that `sweep_2d.py
   --pairs_file` can consume.

Concretely this is a small extension to `select_new3_pairs.py` —
add an `--overlap_max` flag that filters using the dirs cache.

## Work breakdown

### Per model on a pod (each takes ~55 min)
1. **DoM extraction** (≈ 1–2 min after model load):
   - `python scripts/exp_map/_extract_new3.py --target <tgt> --layer 2`
   - Idempotent — extends `results/exp_map/data/directions/dirs_<tgt>_L2.pkl`
2. **Pair-list emission** (instant):
   - `python scripts/exp_map/select_new3_pairs.py --target <tgt> --layer 2`
3. **2D sweeps** (≈ 38 min: 93 pairs × ~24s):
   - `python scripts/exp_map/sweep_2d.py --target <tgt> --layer 2 \
       --pairs_file results/exp_map/data/pair_lists/new3_pairs_<tgt>_L2.csv \
       --skip_existing`

The same orchestrator from the gemma run works as-is — drop into one
of:
```
bash scripts/exp_map/_pod_run_new3.sh gemma_L2_canonical
```
…with the launcher generalised to non-gemma. The script currently
only knows gemma configs; need a small generalisation step (next
section).

### Code change needed
`scripts/exp_map/_pod_run_new3.sh` currently only handles
gemma-specific configs. Add a `<target>_L2_canonical` family:

```bash
case "$CONFIG" in
  gemma_L2_canonical|qwen_L2_canonical|llama_L2_canonical|...)
    TARGET=${CONFIG%_L2_canonical}; LAYER=2;;
  ...
esac
```

Or write a thin wrapper `_pod_run_new3_canonical.sh <target> <layer>`
that calls the three steps directly. Wrapper is cleaner.

### Local refit (after pkls are synced down)
For each non-gemma model:
1. **Canonical fits** (median over 6 cells) — refit so the 93 new
   pairs appear in the fits pkl:
   - `python scripts/exp_map/fit_pairs.py --target <tgt> --layer 2 --n_bootstrap 0`
2. **thrfixed fits** — rerun the per-condition recommended threshold:
   - `python scripts/exp_map/refit_thrfixed_all.py --only <tgt>`
   - The runner already has the canonical row for each model; with
     the new sweeps in place the recommended T may shift slightly
     (median axis-max changes) — that's expected.

### Re-render
- `python scripts/exp_map/plotting/plot_robustness_beeswarm.py \
    --use_thrfixed --max_overlap 0.10 \
    --exclude_dirs Formal,HonestyShort,TensePresent --exclude_mwe \
    --columns model,perturb,measure,metric,threshold,method,anchor_source`
- Cross-model column will now show ≈ same n per model (within
  overlap-filter variation).

## Pod plan

GPU work is ~55 min per non-gemma model.

| pods | parallelism | wall |
|------|-------------|------|
| 5    | one model per pod  | ≈ 1 h         |
| 3    | (qwen+llama), (mistral+aya), (yi) | ≈ 2 h |
| 2    | (qwen+llama+mistral), (aya+yi) | ≈ 2.7 h |
| 1    | sequential               | ≈ 4.5 h         |

Recommended: 3 pods. Each pod runs a small queue.

Pod-A (queue): qwen → llama
Pod-B (queue): mistral → aya
Pod-C (queue): yi
(Pod-C frees up first → can be used for fits sync or re-render.)

GPU type retry list (per `~/.claude/runpod.md`): RTX 4090, RTX A5000,
RTX A4000, RTX 4000 Ada, L40, L40S, RTX 3090, RTX A6000, A40, A100
PCIe, A100 SXM, H100 PCIe, H100 SXM, H200 SXM, RTX 6000 Ada, RTX 2000
Ada. Always `--runtime 600` and the older CUDA image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

## Sync checklist
Before launching pods (lessons from prior run):
- All script edits in `scripts/exp_map/` synced to the network
  volume **before** launching multi-pod queues — `scp` to one pod
  overwrites the file across all pods. (See
  `feedback_no_scp_during_run.md`.)
- Verify each pod has `transformers` + `accelerate` installed and HF
  token loaded (`/workspace/hf_cache/token`).

## Risk register / what could go wrong
- **Some model's tokenizer mishandles the new prompts** — unlikely,
  the Wealth/Health/Certainty prompts are plain English, but flag if
  any extraction returns ‖d‖ ≠ 1 or large overlap (>0.30) with
  existing dirs.
- **GPU availability** — if only 1–2 pods provision, fall back to
  sequential.
- **Pod auto-termination at 10 h cap** — each pod's queue is short
  enough (<2 h) that this shouldn't bite, but monitor and resume
  with `--skip_existing` if needed.
- **Model-specific overlap distributions** differ; the canonical
  `min(axis_max)` denominator for thrfixed should still give a
  sensible per-model T but worth eyeballing the recommended values
  before kicking off the refit.

## Out of scope
- Re-running the variant axes (perturb-layer, measure-layer, metric,
  threshold, method, anchor-source) for non-gemma models. Those
  columns of the beeswarm are gemma-specific by design.
- Running MWE pairs on non-gemma models. (Same reason.)
