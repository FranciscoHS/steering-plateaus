# Plan: Large-scale batch decomposition (2000 activations, 20 pods)

## Context
We want to decompose 2000 FineWeb activations at layer 2 of Llama-3.2-1B into 20 plateau-breaking directions each, using the response decomposition method. This is a scale-up from the case studies we ran (single activations on qwen3-1.7b). 20 RunPod pods run in parallel, each handling 100 activations. Sub-agents manage groups of pods.

## New script: `scripts/jacobian/batch_decomp.py`

Self-contained script (duplicates core functions from response_decomp.py, following case_study_decomp.py precedent). Interface:

```bash
python -u scripts/jacobian/batch_decomp.py \
    --model meta-llama/Llama-3.2-1B --layer 2 --k 20 \
    --n_total 2000 --start_idx 0 --end_idx 100 \
    --response_self --n_restarts 3 --n_steps 100 \
    --seq_len 10 --seed 42 \
    --results_dir results/batch_decomp/llama-3.2-1b_layer2
```

### Flow
1. Load model once
2. Load all 2000 FineWeb samples (same seed=42 on every pod, cached on network volume)
3. Loop over activations in `[start_idx, end_idx)`:
   - Skip if `act_XXXX/results.pkl` exists with `complete: True`
   - Resume from `act_XXXX/checkpoint.pkl` if partial
   - Extract single activation (1 forward pass)
   - Calibrate lambda_orth
   - Run greedy decomposition (k=20 directions, 3 restarts × 100 steps)
   - Checkpoint after each direction
   - Save final results.pkl with `complete: True`

### Output structure
```
results/batch_decomp/llama-3.2-1b_layer2/
  act_0000/results.pkl     # directions, magnitudes, responses, target_activation, ...
  act_0000/checkpoint.pkl  # overwritten after each direction (for resume)
  act_0001/...
  chunk_XX.log             # stdout log per pod
```

### Checkpointing
- After each direction: save checkpoint.pkl (directions so far, residual state)
- After all directions: save results.pkl with `complete: True`
- On startup: skip completed activations, resume partial ones from checkpoint

### Timing estimate
- Case study baseline: qwen3-1.7b, 5×200 steps, 19 blocks → 24s/direction
- Here: Llama-3.2-1B, 3×100 steps, 12 blocks → ~5s/direction (~5x faster)
- 20 directions ≈ 100s ≈ 1.5 min per activation
- 100 activations ≈ 2.5 hours per pod
- 10-hour pod runtime gives very comfortable margin

## Pre-caching FineWeb data

`load_fineweb_fixed_length` reads from cache but doesn't write to it. Before launching the 20 pods, we need to:
1. On one pod, load and cache the 2000 samples to `/workspace/fineweb_cache/fineweb_n2000_seq10_seed42.pt`
2. All other pods will load from this cached file instantly

This avoids 20 simultaneous HuggingFace streaming requests.

## Pod orchestration

### Naming: `decomp-00` through `decomp-19`
SSH aliases: `runpod-decomp-00` through `runpod-decomp-19`

### Sub-agent structure: 4 agents × 5 pods each

| Agent | Pods | Activations |
|-------|------|-------------|
| A | decomp-00 to decomp-04 | 0–499 |
| B | decomp-05 to decomp-09 | 500–999 |
| C | decomp-10 to decomp-14 | 1000–1499 |
| D | decomp-15 to decomp-19 | 1500–1999 |

### Each sub-agent's responsibilities
1. Create 5 pods (try RTX 4090, fall back to other GPUs)
2. Sync code + install deps (pip install transformers accelerate, HF login)
3. Launch experiments (nohup, redirect to log file)
4. Monitor periodically: count completed act_*/results.pkl, check process alive
5. Handle failures: if pod died, create new one, re-sync, re-launch (checkpointing handles resume)
6. Report completion

### Monitoring one-liner per pod
```bash
ssh runpod-decomp-XX "ls /workspace/steering-plateaus/results/batch_decomp/llama-3.2-1b_layer2/act_*/results.pkl 2>/dev/null | wc -l && pgrep -f batch_decomp | wc -l"
```
Returns: completed_count running_processes (e.g. "47 1" = 47 done, still running)

### Failure recovery
- Pod dies → checkpoint exists for partial activations → create new pod, re-launch same command → script skips completed, resumes partial

## Phases

1. **Write** `batch_decomp.py`
2. **Add** `'llama-3.2-1b': 'meta-llama/Llama-3.2-1B'` to MODEL_ALIASES in src/model.py (convenience)
3. **Pre-cache**: Use existing pod to cache FineWeb data on network volume
4. **Launch 4 sub-agents** in parallel, each creates+manages 5 pods
5. **Monitor** until all 2000 activations complete
6. **Download** results: `scp -r runpod-decomp-XX:.../act_*/ ./results/...`
7. **Aggregate** (separate analysis script, not part of this plan)

## Files to modify/create
- **Create**: `scripts/jacobian/batch_decomp.py` (~200 lines)
- **Edit**: `src/model.py` — add one line to MODEL_ALIASES
- **Edit**: `src/data.py` — add cache-saving to `load_fineweb_fixed_length` (it currently only reads from cache, never writes)

## Verification
- Run batch_decomp.py locally on 2-3 activations with k=2 to verify checkpointing and resume work
- Kill and restart to confirm checkpoint resume
- Verify results.pkl format matches expectations
