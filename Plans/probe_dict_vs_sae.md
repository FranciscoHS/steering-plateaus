# Sparse-probing: found dictionaries vs Qwen SAE

**Goal:** measure how well our found directions probe for sentiment vs the Qwen
SAE, in-distribution and OOD, as a function of sparsity k. Discriminating axis is
**k-sparse probing** (a dense probe can't separate these — all linear re-encodings
of the same activation).

## Setup
- Model: `qwen3-1.7b`, layer **14** (where the dicts live).
- Representation: last-token residual at L14.
- Task: train probe on **SST-2**, ID-test SST-2, OOD-test **Amazon polarity**.
- Probe: `LogisticRegression` on the reconstruction at each k.
- k grid: 1,5,10,20,40,80,160.

## Methods (same k grid, same probe)
- **raw** — probe on raw activation (ceiling line).
- **sae_native** — real Qwen SAE: encoder → top-k latents → decode (+b_dec).
- **sae_omp** — Qwen SAE *decoder* as a plain dictionary, OMP-k (fair atom-vs-atom).
- **dict:lam100 / lam1000 / lam10000** — our 3 dicts (`lam_prev` = diversity penalty
  = 100 / 1000 / 10000; 40000 / 38700 / 32600 atoms), OMP-k.
- **pca / random** — OMP-k baselines (best linear subspace / hard null).

## Key methodological choice: mean as free bias
Every OMP method reconstructs the deviation and adds the mean back:
`recon = abar + OMP_k(a - abar, D)`, with `abar = train_acts.mean(0)` (fixed, fit on
train, applied to all splits). This mirrors the SAE's decoder bias `b_dec`, so the
k atoms are only spent on the deviation from the mean — otherwise the dicts waste
atoms re-deriving the mean and the comparison is unfair. (Francisco's point.)

## Artifacts (on network volume)
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` → `.../layer14.sae.pt`
  (W_enc/W_dec/b_enc/b_dec, 32768 latents). This is the "Qwen Scope" SAE used in
  the repo's other L14 comparisons — NOT adamkarvonen's.
- Dicts: `results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_align1.0_k100_40k`,
  `..._lamprev1000_40k`, `..._lamprev10000_40k`.

## Implementation
- New script `scripts/probes/run_probe_dict_vs_sae.py` (does NOT clobber
  `run_probe_melbo_comparison.py`).
- GPU OMP (batched greedy, copied from `run_probe_melbo_only.py`) since OMP over
  40k atoms is too slow on sklearn/CPU.
- Caches activations to skip model reload on re-runs.

## Expectations (Francisco's priors)
- Sentiment is near-linear → ID accuracy may saturate near the ceiling for most
  methods; the **OOD drop** and the **very-low-k** regime are where differences show.
- Don't expect the SAE to do especially well OOD.
