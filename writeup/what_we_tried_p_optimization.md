# What we tried: optimizing directions for superellipse p

Draft v1, 2026-08-02. Compiled by Claude from the daily logs (2026-06-01 →
2026-08-02, Obsidian vault), `objective_attempts_summary.md`, and the repo.
All verdicts are Francisco's own, quoted or closely paraphrased from the logs,
with dates. Numbers are copied from the logs and not re-verified against the
result files — spot-check before publishing. Items marked [check] are
ambiguous in the sources.

## TL;DR

- The superellipse exponent p separates candidate feature directions (contrastive,
  MELBO, SAE latents: p > 2) from controls (random, PCA, neuron basis, averaged
  activation-diffs: p ≈ 2). This is the FSEC-paper result and it is robust.
- We inverted the test: optimize directions to have high p, hoping to recover
  features without needing candidates first.
- The optimization succeeds in the narrow sense — we can reliably produce
  directions with (held-out) p ≈ 3–5, well above contrastive (~2.3–2.4).
- Every independent featureness test fails or is explained by a confound:
  reconstruction barely beats random, SAE/contrastive alignment is weak,
  dashboards are not monosemantic, probing is not better than random, AxBench
  detection underperforms SAE and contrastive, ablation impact is at random
  level, and synthetic activations built from the found directions never show
  plateaus.
- Adding an activation-alignment term makes the directions look feature-like by
  most tests — but the ablation (alignment term alone, no p term) gets the same
  featureness with p ≈ 2. The p term contributes nothing measurable to
  featureness. "Have we just done SAEs with extra steps?" (06-17)
- Bottom line, in Francisco's words: "Optimizing for p alone does not recover
  the features. Settled." "p > 2 is necessary but not sufficient." (06-25);
  "i am fairly convinced that optimizing for p does not find the features,
  having tried many variants" (07-16).
- The residual mystery is what the high-p non-feature directions *are*. Partial
  structure findings: there appear to be whole high-p subspaces; p is largely a
  pair property, not a direction property (06-11; confirmed quantitatively by
  the additivity fit, 08-02); mixing optimized directions kills p toward 2.

## 1. Background: the p measurement

- Setup (FSEC paper, Ferreira da Silva & Heimersheim): perturb a residual
  activation along a normalized direction pair, measure the downstream response
  contour, fit |x|^p + |y|^p = 1. p > 2 = super-additive interference
  ("feature-like"), p = 2 = sphere (generic).
- Established positives: contrastive (diff-of-means), MELBO, SAE latents.
  Established controls at p ≈ 2: random, PCA, neuron basis in the LLM (06-17),
  single random-activation diffs (2.08, slightly above 2, expected since a diff
  is a sparse feature combination), averaged random diffs (95% CI 1.98–2.03,
  the cleaner baseline; 06-17).
- Also holds outside LLMs: 4-layer MNIST-like classifier (arXiv 2402.15555),
  class-mean contrastive pairs give median p ≈ 3.5, CI [3.3, 3.8] (06-02).
- Toy (denoising / CiS) model: feature–feature p ≈ 3.9, feature–random ≈ 2.7,
  random–random ≈ 2 (06-03; a first estimate of ≈2 for feature-vs-random "was
  wrong"). Note: feature-vs-random > 2 is what justified optimizing against
  random partners.

## 2. The idea being tested

If p > 2 is a signature of feature directions, search for features directly by
maximizing p (or a differentiable proxy). Interesting either way: success gives
a candidate-free feature finder; failure means high p is necessary but not
sufficient, and the test loses its "featureness" reading.

## 3. Objectives / proxies tried

All on Qwen3-1.7B, perturb L14, measure ~L26 (2nd-to-last), unless noted.

- **Ratio proxy** (the workhorse): minimize R_mix / (R_1 + R_2), spherical
  perturbation at fixed 30° input angle. Cheap, differentiable, optimizes
  cleanly. But it only weakly tracks eval-p (Spearman ≈ −0.33 in one
  calibration, 06-03 / `objective_attempts_summary.md`) — "we're not directly
  maximizing p — which I suppose we knew."
- **Multi-magnitude ratio + per-step activation resampling (+ minibatch 5
  anchors/step)**: the fix that made it work. "This appears to work! Over 100
  pairs we get p vs random median ~3.06, p vs other found directions ~3.9"
  (06-04). ~10 s/direction. This became the dictionary-build objective.
- **Interpolation-based p estimate as objective**: calibrates well as a
  *measurement* (Spearman 0.86 vs eval-p) but "gradient descent just doesn't
  work so well. It's very choppy" (06-04). Same conclusion later as **soft_p**
  (Gauss–Newton soft-intercept fit): good estimator, poor objective — p̂
  wanders, doesn't climb (`objective_attempts_summary.md`).
- **Intercept-at-45° variant**: "a little bit better, but still very noisy,"
  and blows past the square / non-monotonic (06-04). Smoothed version =
  **diag_rho** (maximize normalized 45° intercept ρ, monotone in p): climbs,
  but noisy and overshoots ρ past √2 into a degenerate corner
  (`objective_attempts_summary.md`).
- [check] Tension between sources: `objective_attempts_summary.md` reports
  eval-p pinned at ~2.0–2.2 for ratio, soft_p, AND diag_rho — but those runs
  used a *single fixed anchor* per direction and random partners. The June logs
  get p ≈ 3–4 once activations are resampled per step and multiple magnitudes
  are used. So the summary documents the pre-fix phase [verify ordering — the
  file is undated].
- **Known failure modes found along the way**: overfitting to the training
  anchor (eval-on-train shows points "way beyond the square", one fit p = 0.79,
  06-04); overfitting to fixed partners (p 2.6 vs training partners, 2.14
  among themselves, 06-03); joint 2-direction optimization finds *cancellation*
  pairs instead — response non-monotonic, superellipse model breaks — even
  with 10 fresh random activations per step ("there are many, many directions
  that cancel out in this way", 06-03, retried 06-08).

## 4. Partner (contrast) schemes tried

Who the candidate direction is paired against in the loss:

- **Fixed 10 random directions** → overfits to them (06-03).
- **Random, resampled per step** (`randcontrast_stepresample`) → works, p ≈ 2.3
  among found dirs (06-03), higher with the multi-magnitude objective.
- **Fixed contrastive references** → first attempt failed for a confusing
  reason: found-vs-found p ≈ 2, and even contrastive-vs-contrastive came out
  ≈ 2 in that setting ("That is no bueno", 06-03; same in Gemma; never fully
  resolved [check]). Retried with resampling on 06-08: "Hmm this is looking
  pretty good.. High-p both against contrastive and against each other."
- **Contrastive scaffold, subsampled per step** (`--contrast_source scaffold`,
  17 semantic diff-of-means dirs from the fsec registry): the main scheme for
  the later dictionaries.
- **Scaffold + previously-found atoms** (`--contrast_source scaffold_grow` in
  `scripts/superellipse/build_dictionary.py`): partner pool starts as the 17
  contrastive dirs and grows with every optimized atom. One build ran:
  `results/dictionary/qwen3-1.7b_L14_scaffoldgrowcontrast_lam100_40k` (~7500
  atoms at analysis time; only the max-cos-alignment analysis is synced
  locally; the logs never discuss its results by name [check]).
- **Previously-found directions only** (`learn_superellipse_dict.py`): grow a
  dictionary from a jointly-learned first pair. Also
  `learn_feature_dir_vs_random.py` / `learn_feature_dir_vs_contrastive.py`
  (fixed scaffolds, push-apart penalty against previous finds).

## 5. Precursor: single-activation decomposition (abandoned)

- Sequential decomposition of one activation into high-p directions +
  reconstruction + soft orthogonality (Gemma L15, "Paris is the capital of";
  `results/decomp_plateau/gemma_L15_paris_capital_of`).
- Cross-prompt sanity was encouraging (matched directions cos 0.29–0.75, ~10×
  random baseline; 06-01).
- Seed reproducibility failed: across 5 seeds, 4–6 directions, only the first
  direction stable (rank-wise mean cos 0.86, then 0.49, rest ≤ 0.14); one
  shared principal axis. "So our method is not reproducible!" (06-01).
  Diagnosis: greedy-sequential bias. → switched to corpus-scale dictionaries
  (06-02).

## 6. Dictionary builds (chronological)

- `qwen3-1.7b_L14_K8000_randcontrast_stepresample` (06-03): reconstruction
  "barely beats random"; SAE alignment ≤ ~0.2; contrastive ≤ 0.112 ("barely
  higher than random"). "Generally quite damning."
- `qwen3-1.7b_L14_K8000_ratio30_actresample` (06-04/05): the multi-magnitude
  objective. p vs random ~3.2, vs self ~3.4 at 1100 dirs, no decay. OMP
  reconstruction (`eval_dictionary_omp.py`): L2 barely better than random,
  clearly worse than PCA and SAE; KL worse than random. p_vs_recon plot:
  ours median ~3–3.5, SAE slightly > 2, random/PCA ≈ 2.
- `qwen3-1.7b_L14_scaffoldcontrast_resample_lam100_20k` (06-08→10): contrastive
  scaffold partners + strong diversity penalty. λ-sweep surprise: λ×10 dropped
  mean |cos| 0.2 → 0.02 AND raised median p 3.6 → 5.1 ("I did not see that
  coming", 06-09). Loss-recovered beats random and PCA for < 64 directions;
  VE unchanged; KL beats random, not PCA. Max SAE alignment ~0.3. Dashboards
  not monosemantic; kurtosis-ordered ones show token-y themes (dates/numbers,
  pronouns) — "definitely a theme... not really conceptual" (06-10).
- `qwen3-1.7b_L14_scaffoldcontrast_align1.0_k100_40k` (06-11/12): added the
  third term — top-k corpus-alignment (presence in activations). "Well, these
  are definitely aligned lol." p ~4 per pair. Dashboards "All definitely
  structured!" SAE hits up to cos 0.762; 75 dirs with > 0.5 match. Rotating
  each dir onto its matched SAE latent decays p 3.8 → 2.4 monotonically.
- **Mahalanobis-penalty builds** (maha 0.1 / 1 / 10; 06-18/19): stay-on-manifold
  term. Clean trade-off: strength 10 → p = 2, strength 1 → p = 2.7, strength
  0.1 → p = 4. Maha-10 probes best of all our dicts (at 20k it slightly
  outperforms a size-matched random SAE subset) — but it has p ≈ 2, so the win
  is not the p term. Dashboards uninterpretable at all strengths.
- `qwen3-1.7b_L14_scaffoldgrowcontrast_lam100_40k`: see §4; results not
  written up [check].

## 7. The featureness battery (what we measured, what it said)

Ordered roughly by how damning the outcome was.

- **Synthetic-activation plateau test** (build an activation from k found
  directions, norm-matched, sweep): consistently fails — no plateau, and the
  synthetic points are *more* responsive than real ones ("It's actually way
  more responsive lol that's weird!", 06-03; "Still does not pass our test!"
  for the 3-term dict, 06-11). SAE-latent bases look random-like without the
  encoder bias, "close-ish to real" with it (06-09). Per LRH any sparse
  feature combination should be a valid activation, "so either LRH not true or
  our method is not recovering real features or both" (06-03).
- **AxBench-style concept detection** (06-12, ~500 concepts protocol,
  best-direction AUROC): ours mean/median 0.84/0.94 vs SAE 0.94/0.97 vs
  contrastive 0.96/0.98. "We underperform!" Size-matched SAE subset doesn't
  change it. Mean≪median: "we have some stinkers in there!"
- **Sparse probing suite** (13 concepts, 06-16→19): our dicts slightly better
  than SAE at k = 1–2, equivalent above; response-only dict worse than random
  across the board; alignment-only equal or better than the full objective.
  Side-finding: the SAE itself probes *worse than random directions* here.
  "Makes me not-confident that we're finding the features!"
- **Ablation / KL importance** (project direction out of the stream, 06-16/17):
  without the alignment term, at-or-below random ("these directions are indeed
  not being written significantly"); with it, SAE-order but clearly below
  contrastive.
- **Reconstruction** (OMP L2/KL, VE, loss-recovered): see §6 — never better
  than marginal over random, except loss-recovered < 64 dirs for lam100.
- **Alignment histograms** vs SAE decoder and contrastive registry: weak
  throughout pre-alignment-term (max ~0.2–0.3); alignment term produces real
  SAE hits (0.762 max) — but see the confound in §8.
- **Dashboards / max-activating examples** (neuronpedia-light): pre-alignment
  "I find it hard to assign monosemanticity" (06-10) while contrastive
  Gender/Refusal/Python dashboards are clearly monosemantic; with alignment
  term, structured but token-level (acronyms, 'new', species names, commas,
  past tenses...) (06-11).
- **Steering** (06-16): high-p dirs at 15° "a bit too strong", not
  interpretable; the contrastive vectors in the same harness also didn't
  visibly steer, so "I don't conclude much from this steering test."
- **Mahalanobis on-manifold check** (06-17): optimized dirs go off-manifold
  faster than contrastive under equal rotation; alignment term makes them
  SAE-like.
- **Misc nulls**: token-boost search finds no gender/France direction (06-10);
  no MWE persona-vector overlap (06-10); rotation-derivative histograms show
  no family difference once normalized (06-16); response-direction-cosine
  metric separates in the toy model but is Goodhartable there and shows only
  "small separation" in the LLM (06-22).

## 8. The alignment-term confound (key control)

- Full objective = response(p) + diversity + corpus-alignment. The featureness
  gains all arrived with the alignment term (06-11).
- Ablation, alignment + diversity only, NO response term (100 dirs, then 20k):
  median p 2.05 (IQR 1.9–2.3) — and *strongly* SAE-aligned, equal-or-better
  probing, "Apparently more alignment without the response" (06-12, 06-15).
- So the p term adds high p and nothing else measurable. "which points us
  towards p on its own not being enough for featureness" (06-17). Concern
  recorded: alignment-only ≈ "SAEs with extra steps", inheriting the standard
  criticism of SAEs learning dataset structure (06-11).

## 9. What ARE the high-p directions? (structure findings)

- **There are a lot of them.** p shows no decay over build order up to ≥9k
  atoms (06-03, 06-10, 06-12; `fit_p_vs_buildorder.py`,
  `p_and_overlap_vs_order.py`). "I suspect there are many directions with
  p > 2, of which the features are only a subset" (06-11).
- **They look like subspaces, not isolated directions.** Rotating an optimized
  dir toward random: p ≈ 5 holds until ~40% of the way, then decays to 2 by
  ~80%. "Pretty concerning... there's kind of like a whole subspace that has
  very high p" (06-11).
- **Mixtures kill p.** Mixing k optimized directions drives p monotonically
  toward 2 (06-11, 06-15/16; `mixture_k_sweep_p.py`). Contrastive mixtures did
  NOT clearly decay (2.39 vs 2.34, n = 45 pairs, "we don't have enough
  contrastive directions to conclude", 06-11) [open].
- **p is (mostly) a pair property.** Fixed-direction beeswarm: within-direction
  variance of p exceeds between-direction variance — "p is truly a measure of
  a pair, not of a given direction" (06-11; `fit_p_fixed_direction.py`).
  Quantified 08-02 (`p_additivity_analysis.py`, log-additive model
  log p_ij = mu + a_i + a_j, permutation null):
  - optimized ratio-K20: R² 0.310, null 0.101 ± 0.030, excess 0.210
  - optimized softp-K20: R² 0.461, null 0.101, excess 0.360
  - scaffold-contrast K10: R² 0.180, null 0.206 — no excess (consistent with 0)
  - contrastive-17: R² 0.310, null 0.117, excess 0.193
  - SAE latents: not yet run (`p_additivity_sae/` empty).
- **Toy-model account exists but doesn't transfer.** In the CiS toy, the neuron
  basis has p ≈ 5 (above the features!) because the decoder couples
  independent neurons; optimized dirs avoid this coupling even better — "I am
  decently satisfied that I understand this" (06-08). In the LLM the neuron
  basis has p ≈ 2, "does not transfer, as expected" (06-17).
- **Toy model also has super-feature p directions**, so "even though the
  features have high p, there exist directions with higher p still that are
  not the features" (06-11) — weak-but-real prior for the LLM situation.
- **Cancellation directions are abundant** (joint 2-dir optimization, §3).
- Curve-shape note: contrastive response curves saturate, random don't;
  distinguishable even after 2-point affine collapse (Gemma L2, 06-24) —
  optimized-dirs version wanted but not done [open].

## 10. p as a classifier elsewhere (the measurement kept paying rent)

- **High-p SAE** (07-06→09): add a p term to the SAE loss. Raises median p
  2.15 → 2.6 (λ=10) → 2.75 (λ=100) at ~no reconstruction cost; probing
  *identical*; no gain in contrastive alignment; KL unchanged. Culling +
  reinit of low-p latents does nothing. Early "2.83 vs 2.13" estimate was from
  a small biased sample (confessed 07-08). Verdict: dead end.
- **J-Lens directions, Qwen3-8B L20** (07-14): medians ≈ 2 or lower, EN and ZH.
  Caveat noted: small model for J-Lens.
- **J-Lens directions, Qwen3.6-27B L36** (07-23): ZH pairs p ≈ 2 ("consistent
  with these directions not being aligned with the model's features");
  contrastive comparison p > 2. EN pairs blocked by a shared direction
  (needed |cos| < 0.25 to even get n = 10). Explicitly NOT claiming J-Lens
  dirs aren't privileged in other ways.
- **Small 4-layer classifier** (06-02): contrastive p ≈ 3.5 — p > 2 without
  (presumed) superposition [worth revisiting for the story].
- **Baselines sharpened** (06-17): averaged random-activation-diffs sit at
  p ≈ 2.00 — the honest non-semantic twin of contrastive directions.

## 11. Open threads / ideas not yet run

- **k > 2 additivity test** (07-31 math note): equal mixture of k legs
  predicts intercept r·k^(1/2−1/p); protocol needs only k+1 1D sweeps.
  Motivation: pairwise condition may be too weak ("MELBO is n=1, we do n=2,
  go higher"). Flagged blocker, in his words: "this assumes the same p for
  every (pair of) direction(s)! This is a strong assumption and not what we
  observe... I could have noticed this earlier." The additivity result (§9)
  quantifies exactly how wrong that assumption is.
- **Additivity vs SAE latents** (§9, last bullet) — directory created, not run.
- **scaffold_grow build**: built but never evaluated in the logs [check §6].
- **Contrastive mixture test at larger n** — blocked on having more contrastive
  directions (06-11).
- **Response-curve shape collapse for optimized dirs** (06-24).
- **Layer dependence of p**; **difference-of-activations perturbations** went
  to a separate thread (06-25/26, "Not convinced I am looking at anything
  meaningful here") and later connected to Lee & Heimersheim's sensitive
  directions.

## 12. Possible framing for a LessWrong negative-result section

One-paragraph story the logs support: *p > 2 reliably separates known feature
directions from controls, so we tried to run it backwards as a feature finder.
The optimization works; the featureness never arrives; a battery of ~10
independent tests says the found directions are not features; the one
modification that produces feature-looking directions (corpus alignment) works
just as well without the p term. Conclusion: high p is a necessary signature
with a large non-feature kernel — there appear to be whole subspaces of
high-p directions, p lives on pairs more than on directions, and mixing
high-p directions destroys it. What these directions are remains open; the
most concrete characterization to date is the additive-fit decomposition.*

## Appendix: artifact index

- Builders: `scripts/superellipse/build_dictionary.py` (objectives: ratio /
  soft_p / diag_rho; contrast: random / scaffold / scaffold_grow; penalties:
  lam_prev, lam_scaffold, lam_align, lam_maha),
  `learn_superellipse_dict{,_resample}.py`,
  `learn_feature_dir_vs_{random,contrastive}.py`.
- Evals: `eval_dictionary_omp{,_ve}.py`, `p_vs_recon.py`,
  `fit_pairs_dictionary.py`, `fit_p_fixed_direction.py`,
  `fit_p_vs_buildorder.py`, `p_and_overlap_vs_order.py`,
  `mixture_{test,k_sweep,k_sweep_contrast}_p.py`, `p_additivity_analysis.py`,
  `sweep_p_dict_to_sae_fullfit.py`, probing under `scripts/probes/`,
  AxBench under `scripts/axbench_*.py`, dashboards via
  `build_neuronpedia_dashboard.py` / `build_sae_latent_dashboard.py`.
- Dictionaries (results/dictionary/): `qwen3-1.7b_L14_K8000_randcontrast_stepresample`,
  `..._K8000_ratio30_actresample`, `..._scaffoldcontrast_resample_lam100_20k`,
  `..._scaffoldcontrast_align1.0_k100_40k`, maha builds,
  `..._scaffoldgrowcontrast_lam100_40k` (analysis only, local),
  `..._K10_scaffoldcontrast_resample`.
- Prior partial write-up of the objective phase: `objective_attempts_summary.md`
  (repo root; covers ratio/soft_p/diag_rho with the fixed-anchor caveat).
