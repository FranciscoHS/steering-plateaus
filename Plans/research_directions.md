# Research Directions — Activation Plateaus

Ideas from Francisco + collaborator (meeting), 2026-06-23.

## Framing

- Unlikely that optimizing for p alone recovers the features
- p = a property of the perturbation response curve. More specifically, p>2 means that directions are more responsive than their combinations, p=2 means they are not. We have argument from theory that features should have p>2 and non-features should have p=2. This is further supported by empirical evidence showing that candidate feature directions have p>2 and controls do not. However, it appears that p>2 is necessary but not sufficient to identify the features
- Plateau = a behavioral observation: when you perturb an activation, you get a region of no response. Mechanistically tied to ReLUs not switching — if no ReLUs flip, there's no change in downstream behavior.

## Direction 1 — Shape of the response curves

- Do the different perturbation response curves look the same up to rescaling, or is there structure we can exploit?
- Specifically: does the curve along a contrastive direction differ in shape from a random direction, or is the contrastive one just more responsive? Broadly, feature vs non-feature; contrastive vs random is stand-in for this
- "Shape of the curve" is not well-defined yet — fine to start loose. By curve we mean: plot downstream response vs amount of perturbation, and look at what that plot looks like.
- Candidate directions are definitely more responsive. The question is whether some rescaling would make a shape difference obvious, if there is one.
- Prior (loosely held, no proof): there is no difference in shape. Don't yet know how to prove it either way.
- Rescaling idea: we define the plateau-breaking angle as the amount of perturbation at which we first see a response. This varies — candidate feature directions need less perturbation than random directions. So normalize so they're equal at that point, then check whether the shape of the curve matches. If not, there's maybe something to pull on.

## Direction 2 — How ReLUs flip

- Just interesting to look at — we should do this. No strong prior right now.
- Naive expectation: random directions are more spread out, since they're randomly sampled. But that's only true in the feature basis. The neuron basis is not the feature basis, so in principle there's no reason random directions should be more spread out in the neuron basis.
- Worth looking at: does the pattern of flipping differ for candidate feature directions vs non-feature directions — more clustered, more concentrated, different speed of change?

## Direction 3 — Plateaus ↔ concepts

- Does each plateau correspond to a distinct concept / combination of feature values?
- One view: you leave a plateau whenever a single ReLU switches — which means a huge number of plateaus. Don't think this is true.
- Guess instead that it's at a higher level of abstraction — related to values of features, or whether features are active or not. Loosely held.
- Based on Matthew's results: when interpolating from the activation of one token to another between semantically-matched prompts (different tokens), there doesn't seem to be a plateau crossing.
- Image model: map all plateaus. Does every plateau contain an image, or do images we (or the model) recognize as "the same" land in the same plateau? Feasible on a big ResNet trained on MNIST.
- LLMs: could start by checking Matthew's example in an LLM — do any ReLUs flip across that interpolation? Seems worthwhile.

## Direction 4 — Get the optimization working in the toy model

- Applying the LLM optimization method to the toy model didn't recover the features — it converged to different directions.
- Toy model has ground-truth features, so we can check whether optimization is working. Can we modify the optimization so it does yield the feature? If so, diagnose why.
- Is there some property of the toy model we're exploiting that naively won't hold in the LLM? If so, build a new toy model where the trick fails, and iterate.
- Alternatively, port the working toy-model method directly to the LLM — learn from its failure, or it might just work.
- Known unrealistic aspects of the toy model to fix toward LLM-likeness: the odd activation function, single layer.

### What property to optimize for?

- Context: there's a line of work where optimizing for responsiveness finds things that look like features, but mostly finds refusal directions.
- Our paper's approach: beyond responsiveness, features should be more responsive than combinations of features — we have an argument for why that follows from superposition plus our correction.
- As a classifier this seems to hold: candidate feature directions satisfy it, random directions don't, other controls don't.
- But when we optimize for the corresponding quantity, we get things that are not features.
- Open question: is there a different property that feature directions should have that we could optimize for? So far: responsiveness; being more responsive in mixtures. What else?

## Direction 5 — Layer dependence

- How does p vary if you measure at different layers? Compare contrastive directions vs our found directions.
- For the directions we found: we don't necessarily expect the same feature directions to be privileged at different layers, since the feature direction might change across layers. But it may be informative to look at how they change.
