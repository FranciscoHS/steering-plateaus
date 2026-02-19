# Plan: Multi-Contrastive Steering for Language Registers

## Context
We want to apply the multi-contrastive projection method (from `multi_contrastive_projection.pdf`, especially Section 8's all-pairwise approach) to **language registers** instead of languages. The goal is to isolate register-specific feature directions in Qwen3-0.6B's residual stream — e.g., a "formal" direction, a "casual" direction, etc.

The existing codebase has strong infrastructure for multi-contrastive direction discovery with languages (`run_multi_contrastive.py`, `multilingual_prompts.py`). We adapt it for registers.

## Step 1: Generate candidate register prompts + test Qwen3-0.6B completions

**File:** `scripts/registers/test_register_completions.py`

Candidate registers (starting broad, will prune):
- **formal** — professional/business tone
- **casual** — everyday conversational
- **academic** — scholarly, technical vocabulary
- **simple** — plain, basic vocabulary (like explaining to a child)
- **literary** — poetic, elaborate prose
- **slang** — very informal, internet-speak / colloquial

The script will:
1. Define ~10 hand-crafted test prompt sets, each with all 6 register variants (same semantic content, different style)
2. Load Qwen3-0.6B and generate 20-token completions for each prompt
3. Print completions side-by-side so we can visually inspect which registers the model "gets" (i.e., continues in the same register)

This is explicitly a human-in-the-loop step: we look at the results, decide which registers to keep based on whether the model reliably continues in that register.

**Example prompt set:**
```
formal:   "The quarterly financial results indicate that"
casual:   "So basically the money stuff shows that"
academic: "The empirical fiscal data suggest that"
simple:   "The money numbers show that"
literary: "The ledger's tale whispers of fortunes that"
slang:    "Yo the cash report is saying that"
```

## Step 2: Create full register prompt dataset

**File:** `data/register_prompts.py`

After deciding which registers Qwen3-0.6B handles well (from Step 1), hand-craft ~50 matched prompt sets across the surviving registers. Format matches `multilingual_prompts.py`:

```python
REGISTERS = ['formal', 'casual', 'academic', ...]  # whichever survive Step 1

PROMPT_SETS = [
    {
        'formal': "The committee has determined that",
        'casual': "So the group figured out that",
        'academic': "The committee's deliberations concluded that",
        ...
    },
    ...
]
```

Short prompts (5-10 tokens) to keep token counts manageable.

## Step 3: Run all-pairwise multi-contrastive direction discovery (Section 8)

**File:** `scripts/directions/run_register_contrastive.py`

Implements the Section 8 procedure adapted for registers:

1. **Extract activations**: For each prompt set, extract final-token activations at layer L for each register variant
2. **Compute pairwise differences**: For every pair (i, j) of registers, compute D_ij = activations_i - activations_j (matched prompts)
3. **Mean differences**: v_ij = mean(D_ij) for each pair
4. **Least-squares recovery** (Section 8): f̂_i = (1/k) * Σ_{j≠i} v_ij
5. **SVD of stacked difference matrix** (Section 5): Stack all individual differences as rows, compute leading singular vector
6. **Diagnostics**: σ₂/σ₁ ratio, left singular vector weights, cosine similarities between estimators
7. **Perturbation sweeps**: Test directions on FineWeb data (KL divergence, L2 distance)

Reuses from existing codebase:
- `src/model.py`: `load_model`, `forward_from_layer_allpos`, architecture helpers
- `src/data.py`: `load_fineweb_fixed_length`
- `run_multi_contrastive.py`: `get_acts`, `extract_context_and_activation`, `forward_to_hidden_and_logits`, sweep functions, MELBO optimization

Key difference from existing `run_multi_contrastive.py`: that script uses star topology (3 pairs for 3 languages). This script uses ALL k(k-1)/2 pairs (Section 8), which is the main methodological point.

## Step 4: Steering evaluation

**File:** `scripts/steering/run_steering_registers.py`

Steer Qwen3-0.6B using the discovered register directions:
- Generate text with spherical perturbations along each register direction
- Sweep angles (0-90 degrees)
- Evaluate: does steering toward "formal" actually make output more formal? (Qualitative + could use simple heuristics like average word length, sentence complexity)

Reuses steering infrastructure from `run_steering_multi_contrastive.py`.

## Verification
- Step 1 output: printed completions for visual inspection
- Step 3 output: pickled directions + diagnostic plots (σ₂/σ₁, cosine matrix, sweep curves)
- Step 4 output: steered text samples at various angles

## Files to create
- `scripts/registers/test_register_completions.py` (Step 1)
- `data/register_prompts.py` (Step 2)
- `scripts/directions/run_register_contrastive.py` (Step 3)
- `scripts/steering/run_steering_registers.py` (Step 4)

## Files to reuse (not modify)
- `src/model.py` — model loading, forward passes, architecture helpers
- `src/data.py` — FineWeb loading
- `scripts/directions/run_multi_contrastive.py` — reference for pipeline structure
