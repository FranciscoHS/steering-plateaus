# 10-direction fit: estimate $n$ on MELBO directions

## Goal
Fit the exponent $n$ in $R(v) = F(\sum_i w_i |d_i^\top v|^n)$ from many-direction data, as an extension of the pair iso-plateau result. Target: page-2 figure for the short Google Doc. Test: does a single $n$ (matched to the pair-experiment value, ideally $\approx 2$) fit combinations of 10 MELBO directions?

## Setup (confirmed)

- **Model:** `qwen3-1.7b`
- **Perturb layer:** 7
- **Measure layer:** penultimate (`n_layers - 2`, i.e. layer 26 for qwen3-1.7b)
- **Directions:** 10 raw MELBO directions from `results/melbo_multiseed/qwen3-1.7b_L7_meas-2_angle5.0_N20/results.pkl` → key `raw_directions` (list of 20, take first 10)
- **Activation:** same as the MELBO run — FineWeb seed=42, `act_idx=0`, `seq_len=5`. Single activation. (MELBO directions are tuned to this specific activation; using them on other activations would compare apples to oranges.)
- **Response metrics:** cosine distance, L2 distance, and KL divergence at measure layer.
- **Thresholds:** calibrated via Phase 0 below, not hard-coded.
- **On-sphere parametrization:** same as `sweep_isoplateau.py` / `sweep_dir`:
  $$p = R \cdot (\cos\alpha \cdot \hat{a} + \sin\alpha \cdot \hat{u})$$
  where $\hat{u} = \sum_i c_i \hat{d}_i^\perp / \|\cdot\|$ is the combination unit vector (single direction for Phase A, mixed for Phase B), and $\hat{d}_i^\perp$ are each MELBO direction Gram-Schmidt'd against $\hat{a}$.

## Phase 0: threshold calibration

Sweep one direction (e.g., the first MELBO direction) across angles, plot cosine/L2/KL response curves, pick thresholds at "the plateau has clearly broken but not saturated" — roughly mid-rise on each curve. Record and reuse across Phases A/B.

## Phase A: individual plateau-breaking angles

For each of 10 directions $d_i$:
- Sweep $\alpha \in [0, \alpha_{\max}]$ along $d_i^\perp$ only (other directions at 0). Use fine grid (~40 steps).
- For each holdout activation, find breaking angle $\alpha_i^*$ where median response (across activations) crosses threshold.
- Do this separately for L2 and cosine metrics.

Output: $\{\alpha_i^*\}_{i=1}^{10}$ for each metric.

Cost estimate: 10 directions × N_ACT activations × ~40 angle steps per direction = ~12k forward passes. Small.

## Phase B: combination sweeps

Generate $N = 20$ random dense unit combinations $u^{(j)} = \sum_i c_i^{(j)} \hat{d}_i^\perp / \|\cdot\|$ with $\{c_i^{(j)}\}$ drawn i.i.d. standard normal then normalized. Fixed seed for reproducibility.

For each $u^{(j)}$:
- Sweep $\alpha \in [0, \alpha_{\max}]$ along $u^{(j)}$.
- Find breaking angle $\alpha^{(j)}$ at threshold.
- Record $(\{c_i^{(j)}\}, \alpha^{(j)})$.

Cost: 20 × 30 × 40 = ~24k forward passes. Still small.

## Phase C: fit $n$

For each combination point $(c^{(j)}, \alpha^{(j)})$, the iso-plateau condition is
$$\sum_i |c_i^{(j)} \sin\alpha^{(j)} / \sin\alpha_i^*|^n = 1.$$
(For small angles, $\sin\alpha \approx \alpha$, and this simplifies further.)

Fit a single scalar $n$ to minimize summed squared residuals across all $N$ combinations:
$$\hat{n} = \arg\min_n \sum_j \left( \sum_i |c_i^{(j)} \sin\alpha^{(j)} / \sin\alpha_i^*|^n - 1 \right)^2.$$

Run the fit independently for L2 and cosine metrics. Compare the two $\hat{n}$s and compare to the pair-experiment $n$.

**Optional robustness:** also fit by bootstrap resampling over the 30 holdout activations to get uncertainty on $\hat{n}$.

## Outputs

- `results/ten_direction_fit/qwen3-1.7b_L7/` (new directory)
- `individual_sweeps.pkl` — Phase A data
- `combination_sweeps.pkl` — Phase B data
- `fit_results.pkl` — Phase C fitted $n$ for each metric, with per-combination residuals
- `ten_direction_fit.png` — summary plot: observed vs predicted breaking angles under fitted $n$, and residuals; clean pairing of (predicted, measured) per combination

## Code organization

- Put main script at `scripts/perturbation/sweep_ten_direction_fit.py`.
- Reuse `sweep_isoplateau.py`'s Gram-Schmidt + on-sphere parametrization. Likely simplest to import helper functions rather than duplicate.
- Plot script at `scripts/perturbation/plot_ten_direction_fit.py` (or merge into main script, since it's a one-off).

## Decisions (locked)

1. Raw MELBO directions (first 10 of the 20 in `raw_directions`).
2. Calibrate thresholds via Phase 0.
3. Use the same FineWeb activation the MELBO run used (seed=42, act_idx=0, seq_len=5).
4. Three metrics: cosine, L2, KL.
5. 20 random dense unit combinations.

Not checkpointed — single-GPU run should complete in well under an hour.

## Deliverable for the doc

One figure showing measured breaking angle vs form-predicted breaking angle under the fitted $n$, for ~20 combinations. If $n$ (and the data) matches: tight scatter around $y=x$. Report the fitted $n$ alongside the pair-experiment $n$.
