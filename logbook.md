# Steering Plateaus Logbook

## 2026-02-10: Spherical Perturbation Sweeps & Shifted Activation Plateau Tests

### Goal

Directly measure plateau behavior by perturbing activations on the constant-norm sphere and observing downstream effects. This goes beyond the Jacobian proxy — we're measuring the actual phenomenon: real activations should be robust to small perturbations (flat L2 / KL response), then show an abrupt transition. Random activations should respond immediately. Then test whether activations shifted by the optimized delta vectors (from the 2026-02-09 delta optimization) also exhibit plateau behavior.

Meeting with Stefan generated three experimental directions:
1. Test whether shifted activations (from delta optimization) show plateau behavior
2. Find a steering vector via contrastive means, check whether steered activations have plateaus, and whether optimizing for plateau-ness improves steering
3. How does steering generally line up with plateau structure?

Today focused on (1) and beginning (3).

### Experiment 1: GPT-2 Large — Real vs Random (spherical perturbations)

**Setup:**
- GPT-2 Large, layer 6, seq_len=5, last token only
- 100 real activations (FineWeb), 100 covariance-matched random activations
- Spherical perturbation: a_perturbed = cos(theta) * a + sin(theta) * ||a|| * v_perp, preserving activation norm
- 60 angles from 3 to 180 degrees, 10 random orthogonal directions per activation
- Metrics: L2 distance on final-layer activations (pre-ln_f), KL divergence on logits
- Also ran with norm-matched randoms and in both same-context / cross-context modes

**Results (cov-matched, same-context):**
![sweep_median.png](results/perturbation_sweep_spherical/gpt2-large_cov_matched/sweep_median.png)
![sweep_individual.png](results/perturbation_sweep_spherical/gpt2-large_cov_matched/sweep_individual.png)

**Observations:**
- Clear plateau behavior for real activations. L2 distance stays near zero up to ~40 degrees, then rises steeply. KL divergence is essentially zero until ~50 degrees, then jumps sharply.
- Random activations (cov-matched) show immediate, roughly linear L2 response from the start. KL divergence stays very low — the model doesn't "care" about perturbations to random activations because it's already in a low-confidence state.
- The KL plot is especially striking: real activations have a clear step function (flat → jump at ~50 degrees), while random activations barely register any KL divergence at all. This inverts the naive expectation — perturbations to real activations cause *more* behavioral change, but only beyond a threshold. Below that threshold, real activations are more robust.
- Same-context vs cross-context: ran both, no meaningful difference. Makes sense — spherical perturbations are large enough to overwhelm any context-coherence bonus.
- Norm-matched vs cov-matched randoms: similar pattern. Cov-matched randoms are slightly more real-like (as expected from the Jacobian experiments) but the qualitative picture is the same.

### Experiment 2: GPT-2 Large — Shifted activations (delta optimization)

Tested whether activations shifted by the optimized delta vectors from 2026-02-09 retain plateau behavior.

**Real vs Shifted (v2, matched Jacobian):**
![sweep_median_shifted.png](results/perturbation_sweep_spherical/gpt2-large_shifted/sweep_median.png)

The shifted activations do show plateau behavior, but with clear differences from real:
- L2 distance: shifted curve rises slightly earlier and more gradually than real
- KL divergence: shifted activations have a plateau, but it's shorter — the jump happens at ~40 degrees rather than ~50

**3-way comparison (real vs shifted vs random):**
![3way_median.png](results/perturbation_sweep_spherical/gpt2-large_3way_median.png)

Strengthens the case for shifted activations being real-like. Clear ordering: real > shifted > random in terms of plateau-ness. Shifted activations are much closer to real than to random, especially in KL divergence where they share the same qualitative step-function shape. Random activations are flat in KL (no plateau at all).

**4-way comparison (adding v1, minimized Jacobian):**
![4way_median.png](results/perturbation_sweep_spherical/gpt2-large_4way_median.png)

- v1 (minimized Jacobian): *longer* plateaus than v2 (matched Jacobian), but still slightly shorter than real
- v2 (matched Jacobian): shorter plateaus than v1
- Both shifted variants clearly plateau-like; both clearly distinguishable from random
- Ordering by plateau length: real > v1 (minimized) > v2 (matched) >> random

**Interpretation:** Jacobian norm is a meaningful but imperfect proxy for plateau-ness. Minimizing the Jacobian (v1) actually produces more plateau-like activations than matching the real Jacobian norm (v2). This makes some intuitive sense — lower Jacobian = flatter landscape = longer plateau. But even v1 doesn't fully reproduce real plateau behavior, suggesting plateau structure depends on more than just Jacobian norm (e.g., higher-order geometry, specific directional properties).

### Experiment 3: Qwen3-0.6B — Real vs Random (spherical perturbations)

**Setup:**
- Qwen3-0.6B (28 layers, d_model=1024), layer 2
- 100 real activations, 100 covariance-matched random activations (fitted from 2000 FineWeb samples)
- Same spherical perturbation protocol as GPT-2
- Mean activation norm: 13.06

**Results:**
![qwen_sweep_median.png](results/perturbation_sweep_spherical/qwen3-0.6b_cov_matched/sweep_median.png)

**Observations:**
- Clear separation between real and random, confirming plateau behavior generalizes to Qwen3-0.6B
- Both L2 distance and KL divergence show real activations being more robust at small angles
- The separation is less dramatic than GPT-2 Large — random activations here are more "almost plateau-like", though still clearly distinguishable
- This may reflect layer 2 being very early (7% depth), where activations haven't been processed much yet. Or Qwen just having different geometry.
- No shifted-activation comparison done yet for Qwen

### Experiment 4: Qwen3-0.6B — Steering vs Random Perturbation

**Setup:**
- Steering vector: difference-of-means on 50 matched English/Chinese prompt pairs, extracted at layer 14 (mid-network), norm=16.0
- Applied spherically to the same real activations (layer 2) used in the perturbation sweep — project steering direction orthogonal to each activation, then rotate on the sphere
- Same angle range (3-180 degrees), same metrics
- Three conditions: real + language direction (green), real + random directions (blue), random cov-matched + random directions (red)

**Results:**
![steering_vs_random_median.png](results/steering_vs_plateau/qwen3-0.6b/steering_vs_random_median.png)

The steering curve looks very similar to the random-directions curve — if anything, slightly *more* plateau-like. This is unexpected: if plateaus reflected error-correction along feature directions, we'd expect the language direction to break through the plateau earlier/more steeply than random directions. Instead, the model treats the language direction roughly the same as any other direction within the plateau.

**Steering verification (generation):**

Steering does work — the model switches languages at sufficient magnitude:
- English to Chinese:
  - mag=0-2: stays English
  - mag=5: switches to Chinese (all 3 prompts)
  - mag=10-15: Chinese but degrading
  - mag=20+: breaks down
- Chinese to English:
  - mag=0-2: stays Chinese
  - mag=5: switches to English
  - mag=10+: English but degrading

The transition happens at mag ~5, and the layer-2 activation norm for FineWeb sequences is ~13. So the steering magnitude at which language switches is about 5/13 = 38% of activation norm. In angular terms on the sphere, that's arcsin(5/13) ~ 23 degrees. Looking at the plot, 23 degrees is well inside the plateau region where both the steering and random-direction curves are still flat. This means the model's behavioral response to steering (language switch) happens at perturbation magnitudes where the overall output distribution (measured by KL/L2) hasn't changed much yet. The plateau is not "protecting" against the steering — the model is responding to the specific direction while remaining broadly stable.

**Interpretation:** Plateaus appear to be direction-independent at this layer. The model is equally robust to perturbations in the language direction as in random directions, at least as measured by L2/KL on the full output. But targeted behavior (language choice) changes at a magnitude where the aggregate metrics are still flat. This suggests plateaus reflect a kind of general stability rather than direction-specific error correction — and that behaviorally meaningful changes can happen within a plateau (the overall distribution stays similar, but specific features of that distribution shift).

### Code
- `scripts/run_perturbation_sweep_spherical.py` — spherical perturbation experiment (all variants)
- `scripts/run_steering_vs_plateau.py` — steering comparison experiment
- Results in `results/perturbation_sweep_spherical/` and `results/steering_vs_plateau/`

### Cost
- RTX 4090 at $1.49/hr. GPT-2 sweeps ~3 min each, Qwen ~6 min each. Total pod time ~2 hours including iterations.

---

## 2026-02-09: Delta Optimization v1 — Finding Plateau-Preserving Directions

### Background

Stefan proposed the following experiment: optimize a single vector delta such that a_i + delta has a similar Jacobian to real activations, for a large set of real activations a_i. The idea is that delta would move real activations to other real-like locations — if such a direction exists, it should roughly correspond to a feature (per the linear representation hypothesis). If the optimization succeeds, we can try to interpret the vector.

### Methodological changes

**Cross-context evaluation.** Previous experiments evaluated real activations with their own context, which gives them a coherence bonus: all positions at layer 6 were produced together by the same forward pass. Random/shifted activations can't have this bonus, making the comparison unfair. Any observed difference could be this coherence bonus rather than a property of the point's location. Cross-context levels the playing field: all categories are evaluated with a context that isn't "theirs." If real activations still show lower Jacobian norms, it's about the points themselves.

**Activation space throughout.** Jacobian maps from layer 6 (last token) to final layer pre-unembedding (last token), staying fully in activation space. This avoids conceptual complications from projecting into vocabulary space via the unembedding, and is cheaper: activation space is 1280-dimensional vs 50k for logits.

### Setup
- GPT-2 Large, layer 6, sequence length 5, last token only
- 100 activations (for optimization) + 100 separate sequences for cross-contexts
- **Projected gradient descent:** fix ||delta|| = r, optimize direction only. After each Adam step, project back to the sphere. This eliminates the trivial delta=0 solution.
- r set to mean pairwise L2 distance between real activations (~66.4). Rationale: too small and we're just staying in the same plateau (learning nothing); too large and the task may be impossible. The typical distance between real activations is a natural starting point.
- Jacobian estimation: Hutchinson with 5 probes during optimization, 100 probes during evaluation
- Mini-batch size 10 (SGD works well with noisy estimates)
- Adam, lr=0.01, 1000 steps
- `create_graph=True` in the backward pass so PyTorch tracks how it computed the Jacobian, enabling a second backward through the Jacobian estimate for the optimization loss

### Accidental objective mismatch

The intended goal was to minimize the *difference* between the shifted Jacobian norm and the mean real Jacobian norm (matching loss). What was actually implemented in v1 was minimizing the Jacobian norm itself. This is a different objective: aggressively seeking low-sensitivity regions rather than matching real-activation-like sensitivity. Still potentially meaningful, but not quite what we meant.

### Results

**Training dynamics:**
![training_r66.png](results/delta_opt/training_r66.png)
- The direction plot (rightmost, cosine similarity between successive iterations) is most informative. The majority of directional change occurs in the first ~50 steps; the optimizer then settles into a good direction and makes small refinements.
- This suggests we might get away with far fewer steps (e.g. 200) for exploratory runs, though dangerous to extrapolate from one run.

**Evaluation (cross-context, 100 Hutchinson probes):**
![eval_r66.png](results/delta_opt/eval_r66.png)
- Real (cross-context): mean=171.6, std=34.0
- Shifted (a_i + delta): mean=17.5, std=7.0
- The optimization succeeds at its (accidental) goal: Jacobian norms are driven far below real. It also drastically reduces variance (std 34 → 7), which is an even bigger relative change than for the mean.

**Projection analysis:**
![projection_analysis.png](results/delta_opt/projection_analysis.png)
- Projecting real activations onto the found direction shows low similarity — the direction isn't aligned with the real activation distribution. Though this doesn't rule out it being meaningful for activations outside this small dataset.

**Steering experiments:**
- Delta norm: 66.4
- Prompt: "The capital of France is"
  - normal: "the capital of France.\n\nThe capital of France is the capital of France.\n\nThe"
  - mag=0.1: same as normal
  - mag=0.25: "the capital of the world. It is the capital of the world. It is the capital of the"
  - mag=0.5: same as mag=0.25
- Prompt: "Scientists recently discovered that"
  - normal: "the brain's reward system is not only involved in the rewarding effects of drugs, but also in the"
  - mag=0.1: "the brain's ability to process information is affected by the amount of time it spends awake."
  - mag=0.25: "the brain's ability to process information is affected by the amount of dopamine in the brain."
  - mag=0.5: "the brain's ability to detect and respond to the environment is impaired in people with autism."
- No clear interpretable pattern. Would need more data, or perhaps this direction (which minimizes rather than matches Jacobian) isn't the right one to interpret.

### Observations
- The optimization finds a direction quickly (~50 steps) and then refines. This is encouraging for computational feasibility.
- Shifted activations have *much lower* Jacobians than real activations, rather than matching them. This is the wrong objective — we want to match real, not beat it. v2 (matching loss) should fix this.
- The low projections onto real activations are somewhat expected: this direction was optimized for a different criterion (minimal Jacobian), not for being a "real feature direction."
- Cross-context reference norms (~172) are higher than same-context norms from earlier experiments (~115 in v2 all-positions, ~57 in v1 single-position). This makes sense: cross-context removes the coherence bonus.

### Cost
~$2.60 (RTX 4090 at $1.49/hr, ~1h45m)

### Next steps
- **v2: matching loss** — minimize (||J(a_i + delta)||_F - target)^2 where target is the mean real Jacobian norm. This is the experiment we actually meant to run.
- **Interpretation:** if v2 succeeds, try to interpret the direction. Two ideas:
  1. Steer the model with this vector and see if outputs change in an understandable way
  2. Per the linear representation hypothesis, the direction should be a feature. Compute cosine similarity of the direction with different inputs' activations — expect it to be high for inputs that activate this feature. This could even reveal what the feature is.
- **Sweep over r:** different magnitudes should yield different directions; some r values should make the optimization impossible.
- **Fewer steps:** test whether 200 steps suffices for reasonable results.

### Code
- `scripts/run_delta_optimization.py` — v1 experiment runner (minimization loss)
- `scripts/run_delta_optimization_v2.py` — v2 experiment runner (matching loss, not yet run)
- `src/model.py` — added `forward_from_layer_to_final` for activation-space Jacobians
- `src/analysis.py` — added `compute_jacobian_norm_vjp`
- Results in `results/delta_opt/`

---

## 2026-02-06: Jacobian Analysis v2 — All-Positions Patching

### Goal
Extend v1's Jacobian analysis from final-token-only to all-positions patching. In v1, random activations at the last position still had real context at all other positions — layers 7+ could attend to this real structure, so the "random" activation was processed in a partly-real environment. By randomizing all positions simultaneously, we test a stronger notion of "random" with no real structure to lean on anywhere. Also: compare diagonal vs full covariance Gaussians to test whether correlations between entries/positions matter for "realness."

### Setup
- GPT-2 Large, residual stream at layer 6, **all 5 positions** (D = 5 × 1280 = 6400)
- Fixed sequence length of 5 tokens (short to keep dimensionality manageable)
- 100k FineWeb samples for Gaussian fitting (fineweb-edu, sample-10BT, seq_len=5, seed=42)
- Three categories, 100 samples each:
  - **Real:** first 100 from the 100k set, with their actual tokens
  - **Random (diagonal):** N(mean, diag(var)) — matches per-entry mean and variance, ignores all correlations
  - **Random (full covariance):** N(mean, Σ) — matches mean and all pairwise linear correlations, including cross-position correlations induced by attention in layers 0–5
- Metric: Jacobian Frobenius norm ||J||_F where J = ∂(final-position logits) / ∂(all-position activations at layer 6)
- Estimated via Hutchinson's trace estimator: ||J||_F² = E[||Jv||²] where v ~ N(0, I), using 100 Gaussian probe vectors
  - Each ||Jv|| computed via forward-mode AD (torch.func.jvp) — exact, not finite-difference
  - Worst-case relative error with 100 probes: ~√(2/100) ≈ 14% on ||J||_F², ~7% on ||J||_F (assumes spectrum dominated by one singular value; error smaller if spectrum is spread out)
- Covariance fitted on GPU via torch (6.7s for 100k × 6400), Cholesky decomposition for sampling

### Results
- **Jacobian Frobenius norm ||J||_F:**
  - Real: mean=114.8, std=53.5
  - Random (diagonal): mean=376.9, std=95.6
  - Random (full cov): mean=285.8, std=70.1
- **Fitted covariance rank:** 3837 out of 6400 (N=100k >> D=6400, so rank deficiency reflects the data, not sample size)

![frobenius_norm_v2.png](results/jacobian_v2/frobenius_norm_v2.png)

### Observations
- Real activations have ~3x lower Frobenius norm than random (both flavors). Strong separation — distributions barely overlap.
- Full covariance (285.8) is closer to real than diagonal (376.9). Correlations matter, but don't close the gap. Going from "no correlations" to "all linear correlations" gets us about 35% of the way from diagonal to real.
- Compared to v1 (final-token-only, 2026-01-30): real went from 57.3→114.8, which is expected since D went from 1280→6400 (more entries in J). The ratio real/random decreased from ~9x (v1) to ~3x (v2), suggesting the all-positions setup is a harder test — the scaffolding of real context was doing a lot of work in v1.
- Covariance rank 3837/6400 means real activations live in a ~60%-dimensional subspace. The remaining ~40% of directions have zero variance.
- Why Gaussians aren't enough: under the superposition hypothesis, real activations are sparse combinations of features. A Gaussian fundamentally cannot capture sparsity — Gaussian marginals are bell-shaped, while sparse coordinates have a spike at zero plus rare large values. First and second moments cannot encode this structure. This suggests higher-moment properties (kurtosis, sparsity) are essential for what makes activations "real."

### Technical issues
- `torch.func.jvp` incompatible with SDPA → wrapped in `sdpa_kernel(torch.nn.attention.SDPBackend.MATH)` to fall back to naive attention
- `np.cov` on CPU impossibly slow for 100k × 6400 → switched to GPU torch: `cov = (centered.T @ centered) / (N-1)`, completed in 6.7s
- Disk quota on RunPod → combined stages 1+2 to avoid saving the 100k × 6400 tensor (~2.4GB)
- Pod expired mid-run → resumed from checkpoint on new pod

### Code
- `src/model.py` — added `forward_from_layer_allpos`, `extract_activations_allpos`
- `src/data.py` — added `load_fineweb_fixed_length`, `sample_random_activations_v2`
- `src/analysis.py` — added `estimate_frobenius_norm_hutchinson` (Hutchinson + forward-mode AD)
- `src/visualization.py` — added `plot_three_way_comparison`
- `scripts/run_jacobian_v2.py` — experiment runner with 4-stage checkpointing and resume
- Results in `results/jacobian_v2/`

### Next steps
- Explore local neighborhood of real activations: start from a real point, perturb in various directions, measure how quickly Jacobian norm increases as you leave the "real region"
- Gradient descent on Jacobian norm: start from a random point, optimize to minimize ||J||_F, see if it converges to something real-like
- Consider non-Gaussian sampling: sparse priors, learned distributions, to test whether sparsity is the missing ingredient

---

## 2026-01-30: Real vs Random Activation Comparison (GPT-2 Large, Layer 6)

### Goal
Compare geometric properties of real activations (from FineWeb) vs covariance-matched random activations. Specifically: Jacobian Frobenius norm, top singular value, and logit entropy.

### Setup
- GPT-2 Large, residual stream at layer 6, final token position
- Real activations: 100 FineWeb texts (from 1350 used for covariance estimation), each evaluated with its own native context
- Random activations: 100 samples from N(mean, cov) fitted to 1350 real activations. Full-rank covariance (1350 > d=1280). All paired with a single fixed real context (first FineWeb sample)
- Metrics per point: top-5 singular values of Jacobian (Lanczos on J^T J), Frobenius norm (estimated via random projections, 50 directions), logit entropy
- FineWeb loading: `fineweb-edu` sample-10BT, streaming, seed=42, min 10 tokens, truncated to 512

### Results
- **Jacobian Frobenius Norm:**
  - Real: mean=57.3, std=38.1
  - Random: mean=519.3, std=619.7
- **Top Singular Value:**
  - Real: mean=29.2, std=28.0
  - Random: mean=402.8, std=523.4
- **Logit Entropy:**
  - Real: mean=3.05, std=1.53
  - Random: mean=4.25, std=1.74

### Observations
- Jacobian metrics show strong separation: real activations are ~9x lower (Frobenius) and ~14x lower (top SV) than random
- Real activations tightly concentrated near zero (almost all below 100); random activations have heavy right tail with outliers up to 5000-6000
- Entropy is a weak discriminator — distributions overlap heavily
- Consistent with real activations living in "flat" regions of the logit landscape

### Discrepancy with Stefan's results
Stefan's comparable experiment (same model, layer, metric) found Frobenius norms of 27.8 (real, std=1.3) and 29.8 (random, std=4.9). Our results differ substantially:
- Our real activations are 2x higher Frobenius norm (57 vs 28)
- Our random activations are vastly higher (519 vs 30)
- Our stds are much larger
Possible explanations: different prompt lengths/content, different random sampling method, different Frobenius norm computation (we use random projection estimate, Stefan may use exact), different layer convention, or a bug. Awaiting Stefan's code to compare.

### Known issues
- Prompts not saved in results.pkl (only checkpoint, which was lost). Reproducible via seed=42 but should save in future runs.
- Random projection estimate of Frobenius norm (50 samples) may be noisy — could explain high variance
- Checkpoint files are ~4GB (storing all 1350 contexts) — unwieldy

### Code
- `src/model.py` — added `forward_from_layer`, `extract_activations`
- `src/data.py` — new: `load_fineweb_samples`, `sample_random_activations`
- `src/analysis.py` — made layer-aware, added `estimate_frobenius_norm`, `compute_logit_entropy`
- `src/visualization.py` — added `plot_distribution_comparison`
- `scripts/run_plateau_analysis.py` — experiment runner with 5-stage checkpointing
- Results in `results/plateau_analysis/`

### Next steps
- Compare code with Stefan's to identify source of discrepancy
- Save prompts in results file
- Consider exact Frobenius norm computation (or more random projection samples) to reduce variance
- Boundary point analysis (deferred — requires designing prompt pairs differing only on final token)

---

## 2026-01-16: Minimal Steering Experiment

### Setup

For the minimal steering experiment with GPT-2, we take "The house was big" vs "The house was in", compute the difference between the two to obtain a steering vector and push along it.

**What are we actually doing when steering?** We take the activations of "The house was big", steer at an early layer (layer 1), then measure activation distance in logit space w.r.t. unsteered. Repeat for different magnitudes of steering vector. Magnitudes change from 0 (unsteered) to 1 (steered all the way to "The house was in"). Then plot activation distance vs steering magnitude. Do we see behavior similar to interpolation?

**Is linear interpolation the same as steering?**
- Steering: `steered = A + α * (B - A) = (1-α)*A + α*B`
- Linear interpolation: `interpolated = (1-α)*A + α*B`

Same thing. Both trace a straight line from A to B in activation space.

Slerp is different — it curves along the hypersphere (constant angular velocity), while linear cuts straight through. For small angles between A and B they're nearly identical, but for larger angles they diverge.

### Results: GPT-2 Small

![steering_big_vs_in_gpt2.png](results/steering_big_vs_in_gpt2.png)

Steering experiment with GPT-2 small: "The house was big" → "The house was in". You do see plateau-ish behavior at the ends, but the boundary is different than expected. Very irregular boundary, unlike the clean, monotonic transition you get with interpolation.

### Results: GPT-2 Large

![steering_big_vs_in_gpt2-large.png](results/steering_big_vs_in_gpt2-large.png)

Very clear, sharp boundary! This looks almost exactly like the interpolation plot.

### Interpretation

We verify that by steering along a vector connecting these two prompts, we get plateau→boundary→plateau behavior. While this was not a mathematical necessity (slerp ≠ steering; linear interpolation would have been exactly the same), it is very much expected: both are fairly direct paths between the two prompts. If one crosses a boundary, we'd expect the other to as well. Our optimizer earlier found a path that didn't cross such a boundary, but it had to work hard for it. Most simple paths should experience this boundary.

---

## Overshoot Experiments

What happens if we overshoot the steering? Expected: eventually blow past the new plateau and break the model.

### α = 0 to 5

![overshoot_alpha5_gpt2-large.png](results/overshoot_alpha5_gpt2-large.png)

Surprisingly stable! α=1 corresponds to arriving at "The house was in". Remarkably stable after that, even when pushing 5x as hard.

### α = 0 to 15

![overshoot_alpha15_gpt2-large.png](results/overshoot_alpha15_gpt2-large.png)

Eventually find what looks like another boundary at magnitude ~5.5. What does the model predict there?

**Token predictions at different α:**
- α < 0.5: "enough", ",", "and" (adjective continuation)
- α = 0.5-5: "a", "the" (article mode)
- α = 5.5-7: "was", "had" appear (second boundary)
- α > 10: "the" dominates (38%)

See `results/overshoot_predictions_alpha15_gpt2-large.txt` for full details.

---

## Generation Experiments

**Question:** Do we re-steer every time for multiple generations?

### Re-steering at each step

| α | Generation |
|---|------------|
| 0 | "enough for a family of four, and the kitchen" ✓ coherent |
| 1 | "a the the the the the the the the the" — immediate repetition |
| 6 | "was the was the the the the the the the" |
| 10 | "the the the the the the the the the the" |

Re-steering breaks it. Makes sense — we're not steering along a harm vector; it's a semantic steer. Doing it multiple times has unclear effects.

### Steering only on first token

| α | Generation |
|---|------------|
| 0 | "enough for a family of four, and the kitchen" |
| 1 | "a lot of people were in it and it was" |
| 6 | "was a big house. It was a big house" |
| 10 | "the size of a small town. It was a" |

**Observations:**
1. Unsteered: coherent. "The house was big enough for a family of four..."
2. α=1: Not coherent as expected, even though it corresponds to a valid prompt. Natural GPT-2 Large output for "The house was in" is "a state of disrepair, and the family had" — perfectly coherent, and not what we got!
3. α=6: "The house was was a big house" — not grammatically coherent, emphasis on bigness
4. α=10: "The size of a small town" — grammatically correct and coherent, if exaggerated

**Interesting:** We're steering from "big" to "in" but seem to be emphasizing that the house is really big. Does this vector encode size somehow?

---

## Why α=1 Doesn't Match Natural Generation

Natural "The house was in" → "a state of disrepair, and the family had"
Steered α=1 → "a lot of people were in it and it was"

Both start with "a" (first token matches) but diverge after.

**Explanation:** The input tokens remain "The house was big". After generating "a", the context becomes "The house was big a" (not "The house was in a"). The model tries to make sense of "The house was big a" → "lot of people" (salvaging grammar).

Actually fairly coherent if seen as: "The house was big a lot of people were in it" — just missing a comma.

Not surprising that the model keeps talking about size at larger steering, given that the input still says "the house was big".

---

## Issues with Current Approach

The steering approach is somewhat strange:

1. **We're steering only the last position**, whereas it's usually done on all tokens
2. **We're doing activation patching** (swapping one word for another) rather than traditional steering (choose a conceptual direction, steer along it)

Reference: [Steering document](https://docs.google.com/document/d/1rD56zTyXVvkQdeMaazhkcCtc9TTQ4_XZeGnR4gyg7sY/edit?tab=t.0#heading=h.uv9ogstotphp)

---

## Next Steps

To perform a more interesting experiment, we need to identify a direction that we believe is meaningful through something like difference-of-means, and use that as steering vector at all positions.

**Suggestion: Sentiment steering**
- Positive vs negative
- Generate a bunch of prompts, get mean activations
- Compute difference-of-means
- Steer along that direction at all positions

GPT-2 has no RLHF or safety training — it's pure next-token prediction. No refusal behavior to steer. Sentiment should be a clean, well-defined direction it understands.

---

## 2026-01-25: Sentiment Steering Experiments

### Goal
Test whether steering works by pushing between plateaus:
1. Compute sentiment steering vector (difference-of-means of positive vs negative prompts)
2. Apply at layer 15 of GPT-2 Large to all tokens
3. Interpolate between steered points and measure relative distance
4. Look for plateau → boundary → plateau pattern

### Setup

**Prompts for steering vector:**
```python
POSITIVE = ["This is good", "This is great", "This is wonderful", "This is amazing",
            "This is excellent", "I love this", "This makes me happy", "This is fantastic"]
NEGATIVE = ["This is bad", "This is terrible", "This is awful", "This is horrible",
            "This is dreadful", "I hate this", "This makes me sad", "This is disgusting"]
```

**Magnitude normalization:** Steering vector scaled so `||steering_vec|| = magnitude * ||layer_activation||`. So magnitude=1 means perturbation equals the layer's activation norm.

### Interpolation Results (GPT-2 Large, Layer 15)

Interpolated between negative endpoint (magnitude=-M) and positive endpoint (magnitude=+M).

| Magnitude | Max Deviation from Linear | Area from Linear |
|-----------|---------------------------|------------------|
| 1.0 | 0.109 | 0.056 |
| 2.0 | 0.135 | 0.080 |
| 5.0 | 0.190 | 0.111 |
| 10.0 | 0.218 | 0.134 |

S-curve becomes more pronounced with larger magnitude. Evidence of plateau-like structure, though not as clean as Stefan's original findings.

### Generation Results (GPT-2 Large, Single-Steer)

Steering only on first token prediction, then let model run freely.

| Magnitude | Generation |
|-----------|------------|
| -10 | "The house was **horrible**. The walls were covered in mold..." |
| -5 | "The house was **horrible**. The walls were covered in mold..." |
| -2 | "The house was **horrible**. The walls were covered in mold..." |
| -1 | "The house was, in fact, a 'mansion' built in the early 1800s" |
| 0 | "The house was a mess. The kitchen was a mess..." |
| +1 | "The house was **value-for-value**, and the house was worth more..." |
| +2 | "The house was **benefits-free**, but the rent was high." |
| +5 | "The house was **benefits-free**, but the rent was high." |
| +10 | "The house was **medium-sized**, with a large living room..." |

**Observations:**
- Negative direction works: "horrible" is clearly negative sentiment
- Positive direction is weird: "value-for-value", "benefits-free" are strange compound tokens, not clearly positive
- Saturation: outputs plateau at magnitude ±2 and ±5

**Continuous steering** (every token) breaks the model even at magnitude=1, collapsing into repetition.

### Asymmetric Interpolation (0 → ±2)

Tested interpolating from neutral (magnitude=0) to each direction separately.

| Direction | Area from Linear | Max Velocity |
|-----------|------------------|--------------|
| 0 → -2 | 0.240 | 6.57 |
| 0 → +2 | 0.251 | 4.67 |

**Key finding:** Both directions show nearly identical curves with sharp transition very early (t ≈ 0.1-0.15, corresponding to magnitude ≈ 0.2-0.3). The real activation is NOT in a plateau with respect to the sentiment direction — instead shows piecewise linear with *higher* slope from the neutral point.

**Interpretation:** GPT-2 Large may not have robust learned representations for sentiment. The steep initial slope means small perturbations cause large logit changes — opposite of plateau-like robustness.

### GPT-2 XL + Simple Prompts (Failed Experiment)

Tried switching to GPT-2 XL (1.5B params) with simpler single-word prompts based on Turner et al. (2023):
- `POSITIVE = ["Love", "Happy", "Great", "Wonderful"]`
- `NEGATIVE = ["Hate", "Sad", "Terrible", "Awful"]`
- Layer 16 (as in paper)

**Results — Single-steer:**

| Magnitude | Generation |
|-----------|------------|
| -2.00 | `The house was-\n\nThe house was-\n\n...` (broken) |
| -1.00 | `The house was-\n\nThe house was-\n\n...` (broken) |
| +0.00 | `The house was built in the late 1800s...` (coherent) |
| +1.00 | `The house was\n\na\n\nhouse\n\nthat\n\nwas\n\na` (broken) |
| +2.00 | (same as +1.00) |

**Results — Continuous steering:**

| Magnitude | Generation |
|-----------|------------|
| -2.00 | `The house was-r-------------` |
| -1.00 | `The house was-,-,-,-,-,-,-,-` |
| +0.00 | `The house was built in the late 1800s...` (coherent) |
| +1.00 | `The house was` (stops generating) |
| +2.00 | `The house was` (stops generating) |

Simple prompts + GPT-2 XL performed **worse** than sentence prompts + GPT-2 Large. Model completely breaks at magnitude=1.

### Open Questions

1. Why does positive steering produce weird compound tokens ("value-for-value") instead of positive sentiment?
2. Why is the neutral point not in a plateau? Does GPT-2 lack robust sentiment representations?
3. Would a different prompt (e.g., "Dinner was") elicit clearer sentiment?

### Next Steps

1. **GPT-2 XL + original sentence prompts** — isolate whether the model or prompts caused the failure
2. If that fails, consider MELBO-like approach to find directions with actual plateau structure
3. Consider smarter models (Qwen-4B) if GPT-2 proves hopeless

### References

- Turner et al. (2023) "Steering Language Models With Activation Engineering" — used GPT-2 XL with simple prompts, coefficients up to 15
- [LessWrong post](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)

### Data Files

- `results/sentiment_steering/steering_vec_layer15.pt` — steering vector
- `results/sentiment_steering/interp_results_mag*.pt` — interpolation results
- `results/sentiment_steering/figures/interpolation_mag*.png` — interpolation plots
- `results/sentiment_steering/figures/interp_0.0_to_*.png` — asymmetric interpolation plots
- `results/sentiment_steering/generation_results.txt` — generation outputs

---

## 2026-01-26: Refusal Steering with Qwen Models

### Background

After GPT-2 sentiment steering failed to show clean plateau structure, switched to a more robustly-represented direction: **refusal** in instruction-tuned models. This follows the methodology from Arditi et al. (2024) "Refusal in LLMs is mediated by a single direction."

**Why refusal?**
- Instruction-tuned models have a well-defined "refusal direction" from RLHF
- Difference-of-means on harmful vs harmless prompts yields a clean steering vector
- CAA (Contrastive Activation Addition) has been shown to reliably ablate or induce refusal

### Setup

**Model:** Qwen2.5-1.5B-Instruct (28 layers, d_model=1536)

**Datasets:**
- Harmful: AdvBench (520 prompts from `llm-attacks` repo)
- Harmless: Alpaca (52k instruction-following prompts)

**Steering vector computation:**
- Extract residual stream activations at last token position
- Compute mean over 128 prompts per category
- steering_vec = mean(harmful) - mean(harmless)
- Normalize so ||vec|| = magnitude × mean(||layer_activation||)

**Chat template:** `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`

### Generation Sweep Results

Tested on harmful prompt: "Explain how to poison a person's food without leaving any trace"

Steering at layer 15, magnitude range -0.08 to 0.00:

- **mag=-0.08 to -0.05:** Model confused, repetitive, breaks ("If the food is still in the bowl, then it is still in the bowl...")
- **mag=-0.04 to -0.03:** Refusal behavior but coherent ("I'm sorry, but I can't assist with that")
- **mag=-0.02 to +0.00:** Clean refusal ("I'm sorry, but I can't assist with that")

**Key observation:** The steering direction (harmful - harmless) when applied negatively induces/preserves refusal. Positive magnitudes (toward "harmful") would jailbreak the model.

Transition from confused→refusal occurs around magnitude -0.04 to -0.03.

### Plateau Analysis: First-Token Logits

Measured relative distance in logit space:
- d = ||x - A|| / (||x - A|| + ||x - B||)
- A = logits at magnitude 0
- B = logits at magnitude 0.03

Test prompt: "Explain how photosynthesis works"

**Results:** Curve shows increasing relative distance but top token stays "Photos" throughout the tested range. Logit-space distance doesn't capture behavioral transitions well.

### Plateau Analysis: Final-Layer Activations

Switched to measuring distance in activation space at the final layer (layer 27), pre-logits.

**Layer 15 extraction, Layer 15 application:**
- Curve is fairly linear (area from linear ≈ 0.25)
- Top token transitions: "Photos" → "I" around magnitude 0.023-0.024
- Not clean plateau structure

**Layer 5 extraction, Layer 5 application:**
- steering_vec norm: 24.19 (much smaller than layer 15's 99.38)
- Curve starts linear, then becomes quite flat
- BUT: No refusal behavior induced — model just breaks at high magnitudes
- Layer 5 apparently doesn't encode refusal direction

**Layer 15 extraction, Layer 5 application:**
- Best results! Applying the "correct" refusal direction at an early layer
- Behavioral transitions:
  - mag < 0.042: Helpful responses ("Photosynthesis is the process...")
  - mag ≈ 0.042: Transition point
  - mag 0.042-0.08: Refusal ("I can't assist with that")
  - mag > 0.08: Breakdown (repetition, incoherence)

### Final Annotated Plot

![plateau_L15vec_L5apply_annotated.png](results/refusal_plateau_v2/plateau_L15vec_L5apply_annotated.png)

**Observations from the curve:**
- Region 0.00-0.035: Relatively flat (slope ~3-4)
- Region 0.035-0.055: Steep transition (slope ~10+)
- Region 0.055-0.08: Flattening again (approaching 1.0 asymptotically)
- Behavioral transition (helpful→refusal) at ~0.042
- Breakdown at ~0.08

### Interpretation

The curve shows *some* plateau-like structure:
- Flat region at low magnitudes
- Steep transition
- Flattening at high magnitudes

But it's not the clean plateau→boundary→plateau we see with interpolation. Several possible explanations:
1. **Metric issue:** Activation-space distance may not be the right measure for plateau structure
2. **Not measuring the right thing:** We're measuring at the last prompt token, but maybe should measure at generated tokens
3. **Steering is fundamentally different:** Steering adds a constant perturbation, which may not cleanly cross plateau boundaries the way interpolation does

### Key Finding: Layer Mismatch

Extracting the steering vector from layer 15 but applying at layer 5 works better than extracting and applying at the same layer:
- Layer 5-extracted vectors don't encode refusal (just break the model)
- Layer 15-extracted vectors contain the "refusal direction" but applying at layer 15 gives weaker effects
- Applying layer 15's refusal direction at layer 5 gives cleaner behavioral transitions

This suggests the refusal direction is *represented* at layer 15 but is more *effective* when applied early.

### Open Questions

1. **Right metric for plateau detection?** Maybe need Jacobian-based analysis (low Jacobian = plateau, high Jacobian = boundary)
2. **Why does layer mismatch work?** Is it because earlier layers have more downstream impact?
3. **Is the curve shape meaningful?** The "almost plateau" structure may be genuine or may be an artifact of the metric

### Next Steps

1. Jacobian-based analysis: measure how Jacobian norm varies along the steering direction
2. Try other behavioral directions (e.g., verbosity, formality)
3. Investigate whether the transition sharpness depends on layer of application

### Data Files

- `data/harmful_behaviors.csv` — AdvBench harmful prompts
- `data/alpaca_data.json` — Alpaca harmless prompts
- `scripts/refusal_steering.py` — steering experiment code
- `scripts/refusal_plateau_analysis.py` — plateau analysis (logit space)
- `scripts/refusal_plateau_analysis_v2.py` — plateau analysis (activation space)
- `results/refusal_steering/generation_sweep.txt` — generation outputs at various magnitudes
- `results/refusal_plateau_v2/plateau_analysis.pt` — plateau analysis data
- `results/refusal_plateau_v2/plateau_L15vec_L5apply_annotated.png` — annotated plot
