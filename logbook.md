# Steering Plateaus Logbook

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
