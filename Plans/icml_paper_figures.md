# ICML paper figures — plan

Target: `~/Research/error-correction-paper/icml2026/paper.tex`.

**Core decision (new):** all flagship figures at **Gemma-2-9B, perturb L2 → measure penultimate (L40)**. This makes Fig 2's plateau-then-break shape visually dramatic and keeps the whole paper on a single (perturb, measure) pair. Prior existing material at L20 (e.g. `results/mandarin_haskell/gemma-2-9b/sweep_isoplateau.pkl`, the 4-quadrant composition figure, French × Rust, etc.) becomes legacy — not thrown away, but not used.

Directions at L2 that work (confirmed, `results/paper_L2_gemma2_9b/sweep_1d_L2.pkl`):
- Portuguese (EN prose → PT prose DoM)
- Gender (Male → Female DoM)
- Python (EN prose → Python code DoM)

Directions at L2 that do **not** meaningfully separate from random: Rust, Sentiment. Dropped from flagship use.

All output figures live in `writeup/figures/` (already created).

---

## Figure 2 — single-direction sensitivity  ✅ DONE

File: `writeup/figures/fig2_sensitivity_L2.{png,pdf}`.
Script: `scripts/perturbation/plot_fig2_final.py`.
Data: `results/paper_L2_gemma2_9b/sweep_1d_L2.pkl`.

Plateau-breaking angles at threshold $L^2 = 50$:
- Portuguese 15.0°, Gender 10.7°, Python 21.3°, Random 34.6°.

Nothing more to do unless we decide to polish further.

---

## Composition table (Section: Results)

**Goal:** 4-row LaTeX table replacing the existing 4-quadrant figure. Starting prompt: masculine English ("They elected a new chairman."). Rows:
1. Baseline (no steering)
2. +PT direction only
3. +Gender direction only
4. PT + Gender at $\varphi = \pi/4$ (equal mix)

Text is color-coded: blue for Portuguese tokens, pink for feminine-gendered English tokens, purple for feminine-gendered Portuguese tokens (overlap).

**Needs re-running at L2:** existing generations (`results/mandarin_haskell/gemma-2-9b/generations_portuguese_gender_L20*.{txt,json}`) are L20. We need the L2 analogue.

**Steering setup:**
- Spherical rotation at L2, angle TBD — `composition_pt_gender_highlight.png` used 20°. At L2 the breaking angles are larger (PT 15°, Gender 10.7°), so the equivalent "past-threshold but not saturated" angle is probably ~20–25° along each axis. Pick by inspection from the 1D sweep curves.
- For PT + Gender at $\varphi = \pi/4$: total angle such that the mix is past threshold. If individual breaking angles are 15° and 10.7°, a combined $\theta$ around 22° at $\varphi = \pi/4$ is a reasonable starting guess (roughly the superellipse radius for $n=2$).
- Use Gemma-2-9B, hooking into the residual stream at L2, generating ~80 tokens per row.
- Multiple generations per row (say 5), pick the cleanest for the table — but save all to file.

**Script to write:** `scripts/perturbation/generate_composition_L2.py`. Saves `results/paper_L2_gemma2_9b/generations_composition_L2.{json,txt}`.

**LaTeX:** build the table inside the ICML tex file (`icml2026/paper.tex`) directly, or as a standalone `composition_table.tex` include. Colors defined via `\usepackage{xcolor}` with named colors (`ptblue`, `femred`, `ptfem`).

**Check point:** Francisco reviews generations, decides on one per row, then we lock the table.

---

## Figure 3 — combination geometry (PT × Gender boundary at L2)

**Goal:** the definitive Portuguese × Gender superellipse plot. Data + $p=2$ circle (dashed) + $p=\infty$ square (dashed) in normalized coords. $L^2$ only. No $p=1$ diamond. Fit $p$ reported in legend or caption; residual in caption.

**Needs:** new 2D sweep at L2. Existing `sweep_isoplateau.pkl` is L20.

**Setup:**
- Reuse the existing `sweep_isoplateau.py` structure but with `LAYER = 2` and PT/Gender directions loaded from `results/paper_L2_gemma2_9b/directions_L2.pkl`.
- **Grid extent:** at L2, PT breaks at 15° and Gender at 10.7° — so the 2D grid needs to span at least 0–20° per axis, ideally 0–25° with ~40 steps per axis.
- 30 holdout male prompts (same as before), N_STEPS ≈ 41, MAX_ANGLE = 25°.
- Cost: 41² × 30 ≈ 50k forward passes on Gemma-2-9B. Small.

**Output:**
- `results/paper_L2_gemma2_9b/sweep_isoplateau_L2.pkl`
- `writeup/figures/fig3_boundary_pt_gender_L2.{png,pdf}` — normalized, $p=2$ and $p=\infty$ references, data dots.
- Fit value $p_{L2}$ reported; compare to the L20 fit ($p_{L20} \approx 2.3$ in the existing data).

**Script:** `scripts/perturbation/sweep_isoplateau_L2.py` (modeled on `sweep_isoplateau.py`), plus the plot is a one-liner via the existing `plot_isoplateau_clean.py` pointed at the new pkl.

---

## Figure 1 — intro teaser (prettified Fig 3)

Same data as Fig 3, but:
- Bigger fonts, cleaner legend.
- Remove the data-fit superellipse curve (if any); just show data dots + $p=2$ and $p=\infty$ references.
- Possibly larger markers and axis labels.

Do this *after* Fig 3 is locked. Script: `scripts/perturbation/plot_fig1_intro.py` (based on `plot_isoplateau_clean.py` with tweaks).

Output: `writeup/figures/fig1_intro_boundary.{png,pdf}`.

---

## Figure 4 — $n$ is robust (beeswarm)

One beeswarm plot, horizontal reference line at $n = 2$. Each point is one fitted exponent. X-axis groups by "axis of variation". Fit residual reported in caption.

**Axes of variation to cover (no MELBO, per Francisco):**
1. **Direction-type** (feature+feature, feature+random, random+random) — on Gemma-2-9B at L2 if possible.
2. **Pair identity** — different contrastive pairs (PT × Gender, PT × Python, Gender × Python, PT × French, Gender × Rust, etc.) on Gemma-2-9B L2.
3. **Many-direction fit** (10-language pairs). Existing: `results/pair_slices_10contrastive/qwen3-1.7b_L14/fitted_p_summary.pkl` — 45 fitted $p$'s. Do we re-run at L2 on Gemma? Probably yes for consistency, but not cheap. **Decision point.**
4. **Models** — qwen3-0.6b, qwen3-1.7b, qwen3-4b, Gemma-2-9B. Existing `allpairs_contrastive/` has qwen3 runs; need to check what perturb/measure layer. Probably mix L2/L3 depending on model.
5. **Perturbation layer** — vary perturb layer (L2, L5, L10, L20) on one model, fix measure to penultimate. Gives points on the "layer" axis. Mostly new runs.
6. **Measurement layer** — vary measure layer (penultimate ± few) for a fixed perturb layer. Likely small effect — quick runs.
7. **Metric** — L2, cosine, KL. Existing 45-pair run has all three. Can reuse.
8. **Threshold** — sweep threshold by ±half-decade around the calibrated value, refit $p$. Existing: `isoplateau_p_vs_cosine_threshold.pdf` shows one run. Reuse.

**Audit step first** (before any new runs): for each axis above, list what already exists in `results/`, what can be re-fit from existing pkls, and what requires new GPU time. Then decide what to run.

**Script structure:**
- `scripts/analysis/fig4_audit.py` — walks `results/` and reports existing fitted-$p$ values per category. Produces a stub pkl with what we have.
- `scripts/analysis/fig4_collect.py` — orchestrates new runs as needed.
- `scripts/analysis/plot_fig4_beeswarm.py` — produces `writeup/figures/fig4_n_robust_beeswarm.{png,pdf}`.

**Don't start runs for Fig 4 until the audit is reviewed.** Present the audit table to Francisco, decide which new runs are worth the compute.

---

## Execution order

1. **Composition table**: write the L2 generation script, run, save generations to file. Francisco reviews, picks per-row completions, we lock the LaTeX table.
2. **Fig 3**: 2D PT × Gender sweep at L2, plot. Francisco reviews, iterate on thresholds/axes.
3. **Fig 1**: derived from Fig 3 once locked. Mostly styling.
4. **Fig 4**: audit existing results, present to Francisco, decide new runs, aggregate, plot.

Throughout: all Gemma-2-9B L2 data/pkls go under `results/paper_L2_gemma2_9b/` (maybe rename this dir to `results/paper_L2_gemma2_9b/` once we're sure). Figures go under `writeup/figures/`.

## Pair exploration (overnight autonomous run, 2026-04-23)

Running `scripts/perturbation/pair_exploration_L2.py` on RunPod pod `pair-explore` to find pairs that (i) individually break the plateau early, (ii) have low mutual cos sim (orthogonal), (iii) show the smallest threshold-sweep variation in fitted $p$. Comparison set for "PT × Gender has p ranging 1.8–2.85 across thresholds — can we do better?"

Candidate directions at Gemma-2-9B L2: PT, Gender, Python + 9 more languages (zh/es/fr/de/nl/it/ja/ru/ar). All 10 languages + Gender pass the 1D breaking-angle filter (break < 0.6 × random pooled). Python just misses (ratio 0.62).

Pair selection: 24 pairs with $|\cos| < 0.5$. Top 10 (ranked by feature-type diversity then low cos) all Gender × Language; lang × lang pairs have cos > 0.4 and were ranked below the 10-pair cutoff.

Top 10 pairs going to 2D sweep, MAX_ANGLE = 30° (wider than prior 25° so high thresholds don't hit the ceiling):
- Gender × German (cos 0.000)
- Gender × Dutch (cos -0.011)
- Gender × Japanese (cos 0.031)
- Gender × French (cos 0.035)
- Gender × Spanish (cos 0.039)
- Gender × Portuguese (cos 0.043) — this replaces the narrower prior sweep
- Gender × Italian (cos 0.046)
- Gender × Chinese (cos 0.051)
- Gender × Arabic (cos 0.071)
- Gender × Russian (cos 0.086)

Analysis after pipeline completes: `scripts/perturbation/analyze_pair_exploration.py` walks the 2D pkls, runs a threshold sweep (thr $\in$ {10, 20, ..., 400}), fits $p$ each time, reports `(median, std, range)` per pair, ranks by smallest std. Produces `writeup/figures/pair_explore_ranking.png` + `pair_explore_thr_panel.png`.

Est. 2h for the 2D sweeps; will report back when done.

## Open questions

- **Composition table:** what angle along each axis for rows 2–4? Use 20°/15°/22° as rough guesses and iterate, or pick more systematically?
- **Fig 4 scope:** how completist do we want to be on layer/model ablations? Full sweep across 4 models × 3 perturb layers is ~12 new runs each for a handful of direction pairs. Could be an afternoon of GPU time. OK to spend?
- **Data dir rename:** OK to move everything to `results/paper_L2_gemma2_9b/`? Keeps the analysis dirs clean, but requires path updates in scripts.
