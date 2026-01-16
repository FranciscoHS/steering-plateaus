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
