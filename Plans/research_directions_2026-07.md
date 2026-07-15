# Research Directions — July 2026

Status-aware update of `research_directions.md` (2026-06-23 meeting notes, kept as-is for the record).
Incorporates the 09/07 meeting notes and the p-optimization post-mortem.
Attribution: Directions 1–5 below trace to the Francisco+Stefan June meeting; the stacked-CIS
thread is Francisco's; items marked (C) are Claude suggestions, weigh accordingly.

## Framing (unchanged, restated)

- p = superellipse exponent of the plateau-breaking boundary under two-direction mixtures,
  after per-axis normalization. p>2 means pure directions are privileged over mixtures.
- Theory (FSEC paper): error correction under superposition requires p>2 for features;
  p=2 is provably insufficient (quadratic form has ≤ d_model eigendirections).
- Empirics: candidate feature directions (contrastive, MELBO, SAE latents) have p>2;
  controls (random, random-difference, PCA) have p≈2. Replicates across six models.
- **p>2 is necessary but not sufficient** — this is now an empirical conclusion, not a caveat.

## Closed threads (post-mortem)

- **Optimizing for p does not recover features.** Many variants run; none yield
  feature-like directions. Includes training an SAE with a gradient p-term in the loss:
  raises p (2.13 → 2.83) at no reconstruction cost, but the latents still don't look
  like features. Culling low-p latents does nothing. Dead end at least for now — no
  promising modification currently in view.
- **Neuron-basis confound checked in the LLM: negative.** In Dmitry's toy model the
  neuron basis has *higher* p than the ground-truth features (plausibly because the
  nonlinearity acts elementwise in neuron space). Francisco checked the analogous thing
  in the LLM: neuron directions do *not* have high p there. So the toy-model confound
  does not explain the LLM optimization failure, and **what the LLM's high-p non-feature
  directions actually are remains an open question** (see Open questions).

## Main thread — Stacked CIS model (Francisco; in progress)

Take the reverse-engineered compressed-computation model (L⁴ paper: binary-code encoder,
pinv decoder, no error correction — only one layer) and stack layers.

- Core question: does something like error correction *emerge* with depth, as required
  for multi-step computation in superposition (Hänni et al.)?
- **Current status: started. Error correction is required by the task, and later layers
  do observably perform something like it — but apparently not the kind that gives rise
  to plateaus.** So the question sharpens: what kind of error correction is this, why
  doesn't it produce plateaus, and is there a regime (depth, bottleneck placement,
  sparsity, task structure) where plateau-producing error correction appears? (FSEC
  paper offers p>2 as the benchmark a good error-correction toy model should reproduce.)
- If plateaus/p>2 do appear in some regime: this becomes the missing testbed for
  feature-finding — a *trained* model with ground-truth features and emergent (not
  hand-built, not tanh³) error correction, on which optimization objectives can be
  honestly evaluated. (C: this is what Dmitry's model couldn't provide; it's the
  strongest downstream payoff.)
- Either way the current observation cuts at a live assumption — see the plateaus-as-
  signature question under Open questions.
- Fold-in (June Direction 2, ReLU flips): *if* plateaus appear, flip-pattern analysis is
  far better posed here than in an LLM — flips are enumerable and the ground-truth code
  is known ("are plateau boundaries codeword-level or single-neuron events?"). If no
  plateaus, this fold-in dies with them.

## Second thread — Fix the optimization in the toy model (June Direction 4)

- Dmitry's model has ground-truth features but p-maximization finds the neuron basis
  instead (its p is higher). Task: design an objective/constraint set under which the
  optimum *is* the features, then diagnose what property it exploits.
- Multi-anchor p (high p consistently across many anchor activations) is already part
  of the existing setup — not a new lever. Remaining candidate modifications (C,
  speculative): conjoin p with responsiveness or with sparse-activation criteria; use
  n>2 mixtures.
- Iteration loop (June meeting): once something works in the toy model, ask which of
  its properties the method exploits, build a variant model where that property fails
  (known unrealistic aspects: tanh³ activation, single layer, hand-constructed weights),
  and iterate toward LLM-likeness. Alternatively port the working method to the LLM
  directly and learn from its failure.
- Connects to the main thread: a stacked CIS model with emergent error correction would
  be the natural next rung on this ladder — trained, more LLM-like, still ground-truth.

## Bounded side quest — Linebreak features' p (09/07 meeting)

- Measure p for the linebreak features (transformer-circuits linebreaks post) — features
  regarded as high-confidence by people we trust.
- Not a project; a supporting-evidence measurement. Both outcomes informative: p>2 in
  a new model strengthens the FSEC story; p≈2 for genuinely trusted features would
  challenge the necessity claim — and be the more interesting result.
- Gate before committing: check whether their feature directions are actually
  released/reconstructable, and what model they live in (do we have the weights?).
- Good candidate for a low-energy stretch — bounded, definite answer either way.

## Set aside (recorded, not planned)

- **June Direction 1 (shape of response curves):** prior (loosely held) is no shape
  difference after plateau-breaking-angle rescaling; exploratory with likely-null
  outcome. Parked.
- **June Direction 3 (plateaus ↔ concepts):** interesting question, no sharp handle yet.
  MNIST-ResNet plateau-mapping remains the most concrete version if revived.
- **June Direction 5 (layer dependence of p):** informative but descriptive; run
  opportunistically if it becomes cheap during other work.

## Open questions

- What *are* the high-p non-feature directions the LLM optimization finds? Not the
  neuron basis (checked). Substantial characterization effort already spent (Francisco):
  it's hard, and they don't seem to be anything identifiable. Likely (boring)
  explanation: high-p directions are simply plentiful — features have high p, but so do
  at least as many non-features, and optimization lands on those.
- What property, beyond p>2 and responsiveness, should feature directions have that
  could serve as an optimization target? (June meeting open question, still open.)
- Are plateaus really a signature of error correction? Quite a lot of evidence in
  favor, but not a slam dunk — and the stacked-CIS observation of error correction that
  does *not* give rise to plateaus is evidence to weigh here.
