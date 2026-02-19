# Auto-Interpretability Pipeline for 100 MELBO Features

## Context
We have 100 MELBO feature directions (Qwen3 0.6B, layer 14) and want to generate natural-language descriptions + evaluate how well those descriptions predict feature activation on held-out data. This follows the approach in Anthropic's "Towards Monosemanticity" paper, adapted for MELBO directions instead of SAE features.

## Pipeline Overview
Three scripts, each independent and checkpointed:
1. **Compute projections** (GPU) — extract 50k activations, project onto 100 directions
2. **Explain + Simulate** (API) — call Claude Opus 4.6 to describe features and predict activations
3. **Score** (local) — compute Spearman correlations, summarize results

## File Structure
```
scripts/autointerp/
    01_compute_projections.py
    02_explain_and_simulate.py
    03_score.py

results/autointerp/qwen3-0.6b_layer14_k100/
    projections.pkl          # from step 1
    checkpoints/             # per-feature JSON files from step 2
        feature_000.json
        ...
    results.pkl              # final aggregated results from step 3
```

## Script 1: `01_compute_projections.py` (GPU, RunPod)
- Load Qwen3 0.6B via `load_model('qwen3-0.6b')` from `src/model.py`
- Load 50k FineWeb 5-token prompts via `load_fineweb_fixed_length(50000, tokenizer, seq_len=5)` from `src/data.py`
- Extract layer-14 last-token activations in batches (reuse `extract_context_and_activation` pattern from `scripts/melbo/run_melbo_diverse_capped.py`)
- Load 100 directions from `results/melbo_diverse_capped/qwen3-0.6b_target1.0_k100/results.pkl`
- Compute projections: `projections = activations @ torch.stack(directions).T` → shape [50000, 100]
- Decode all prompts to text: `[tokenizer.decode(tids) for tids in token_ids_list]`
- Save `projections.pkl` with: `prompts_text`, `projections` (signed), `abs_projections`, metadata

Note: we only need projections (50k × 100 = 20MB), not the full activations (50k × 1024 = 200MB).

## Script 2: `02_explain_and_simulate.py` (API, no GPU needed)
Uses `anthropic` async client with concurrency limit of 5.

### Per-feature pipeline
For each of 100 features:
**Explain step:**
- Select top 30 prompts by |projection|
- Send explain prompt to Claude Opus 4.6 → get description (1-3 sentences)

**Simulate step:**
- From remaining ~49,970 prompts, select:
  - 50 randomly sampled
  - 50 evenly spaced across the |projection| range (from 0 to max)
- Send simulate prompt with description + 100 prompts → get 0-10 predictions
- Parse numeric predictions via regex

**Checkpoint:** Save per-feature JSON after each feature completes. On restart, skip features that already have checkpoints.

### Explain prompt
```
You are helping to interpret a feature direction found in the residual stream
of a language model (Qwen3 0.6B, layer 14).

Below are the 30 text fragments where this feature activates most strongly,
ranked from strongest to weakest. Each is a 5-token prompt; activation is
the absolute projection of the last-token hidden state onto the feature direction.

1. "The cat sat on" (activation: 12.34)
2. ...

Based on these examples, provide a concise description (1-3 sentences) of what
this feature responds to. Be specific. If no clear pattern emerges, say so.
```

### Simulate prompt
```
You are helping to evaluate a feature direction in a language model (Qwen3 0.6B,
layer 14).

This feature has been described as: "{description}"

For each prompt below, predict how strongly this feature activates on a scale
from 0 (not at all) to 10 (very strongly, comparable to top activating examples).

Respond with one prediction per line in the format "N: score".

1. "JavaScript is a"
2. ...
```

### Error handling
- API errors: exponential backoff (2s, 8s, 32s), 3 retries max
- Parse errors: retry once with clarifying follow-up, then mark feature as failed
- Failed features saved with `"status": "failed"` in checkpoint

## Script 3: `03_score.py` (local)
- Load all per-feature checkpoints
- For each feature: Spearman rank correlation between predicted (0-10) and actual |projection|
- Print summary: mean, median, std, top/bottom 5 features with descriptions
- Save `results.pkl` with all correlations and descriptions

## Key design decisions
- **Absolute value projections** for ranking (both directions of a feature count)
- **Exclude explain examples** from simulate pool (not a separate held-out set)
- **Spearman correlation** — rank-based, so no need to normalize projections to match 0-10 scale
- **Three scripts** because GPU step and API step have different runtime environments, and scoring can run independently on partial results
- **Per-feature JSON checkpoints** (not a growing pickle) to avoid corruption from interrupted writes

## Dependencies
- Add `anthropic` to `requirements.txt`
- `ANTHROPIC_API_KEY` must be set in environment

## Critical files to reuse
- `src/model.py` — `load_model('qwen3-0.6b')`
- `src/data.py` — `load_fineweb_fixed_length(n, tokenizer, seq_len, seed)`
- `scripts/melbo/run_melbo_diverse_capped.py` — `extract_context_and_activation` pattern
- `results/melbo_diverse_capped/qwen3-0.6b_target1.0_k100/results.pkl` — 100 directions

## Verification
1. Run script 1 on RunPod, check `projections.pkl` has expected shapes
2. Run script 2 on just 2-3 features first (`--features 0,1,2`), inspect descriptions and predictions
3. Run script 3 on the partial results to verify scoring works
4. Then run script 2 on all 100 features
