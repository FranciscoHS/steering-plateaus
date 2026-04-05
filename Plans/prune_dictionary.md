# Dictionary Pruning via Matching Pursuit

## Context

`batch_decomp.py` produces ~40k unit-direction vectors (2000 activations x up to 20 directions each) stored in `results/batch_decomp/llama-3.2-1b_layer2/act_XXXX/results.pkl`. We want to prune this down to a 20k-atom dictionary by iteratively removing the least-used atoms, where "usage" is measured by GPU-accelerated matching pursuit on 5000 held-out activations.

## Script

`scripts/jacobian/prune_dictionary.py`

## Output

`results/pruned_dictionary/llama-3.2-1b_layer2/` containing:
- `pruned_dictionary.pkl` — final dictionary tensor + provenance + pruning log

## Args

- `--results_dir` (default `results/batch_decomp/llama-3.2-1b_layer2`) — where to find `act_XXXX/results.pkl`
- `--model`, `--layer`, `--seed`, `--seq_len` — must match batch_decomp settings
- `--n_total` (default 7000) — total FineWeb samples to load
- `--val_start` (default 2000) — first index for held-out set
- `--val_end` (default 7000) — last index (exclusive) for held-out set
- `--target_size` (default 20000) — final dictionary size
- `--n_nonzero_coefs` (default 20) — MP sparsity level
- `--prune_frac` (default 0.25) — fraction to remove per round
- `--output_dir` (default `results/pruned_dictionary/llama-3.2-1b_layer2`)

## Implementation Steps

### 1. Load dictionary atoms
- Glob `results_dir/act_*/results.pkl`
- For each, check `complete: True`, then collect:
  - All direction vectors → stack into `(N_atoms, d_model)` tensor
  - Provenance array: list of `(act_idx, dir_idx)` tuples, one per atom
- Print: total atoms loaded, number of activations, atoms per activation stats

### 2. Extract held-out activations
- Load model via `load_model()` from `src/model.py`
- Load 7000 FineWeb samples via `load_fineweb_fixed_length()` from `src/data.py`
- Extract last-token hidden states at `args.layer` for indices [val_start, val_end)
  - Use batched forward pass pattern from `compute_activation_pcs.py`: stack sequences, run `model(batch, output_hidden_states=True)`, grab `hidden_states[layer+1][:, -1, :]`
  - Process in batches of 64
- Stack into `(5000, d_model)` float32 tensor
- Save to `output_dir/val_activations.pt` for potential reuse (skip extraction if file exists and `--reuse_val` flag set)

### 3. Iterative matching pursuit pruning
```
while len(dictionary) > target_size:
    # Matching pursuit on GPU
    residuals = val_activations.clone()           # (5000, d_model)
    usage_counts = zeros(len(dictionary))         # on GPU

    for step in range(n_nonzero_coefs):           # 20 iterations
        corr = residuals @ dictionary.T           # (5000, N_atoms)
        best = corr.abs().argmax(dim=1)           # (5000,)
        usage_counts.scatter_add_(0, best, ones)
        coefs = corr[arange(5000), best]          # (5000,)
        residuals -= coefs.unsqueeze(1) * dictionary[best]  # subtract projection

    # Determine how many to remove
    n_remove = min(len(dictionary) // 4, len(dictionary) - target_size)
    keep_mask = argsort(usage_counts)[n_remove:]
    dictionary = dictionary[keep_mask]
    provenance = provenance[keep_mask]

    # Log: round number, dict size, usage stats (min/median/max/zero-count)
```

Key detail: on the last round, `n_remove` is clamped so we land exactly at `target_size` (or stop if already there).

### 4. Save results
```python
{
    'dictionary': tensor (target_size, d_model),
    'provenance': list of (act_idx, dir_idx) tuples,
    'pruning_log': [
        {'round': i, 'size_before': ..., 'size_after': ...,
         'usage_min': ..., 'usage_median': ..., 'usage_max': ...,
         'n_zero_usage': ...},
        ...
    ],
    'args': vars(args),
}
```

## Key files to reuse
- `src/model.py`: `load_model()`, `_get_blocks()`
- `src/data.py`: `load_fineweb_fixed_length()`
- Batched activation extraction pattern from `scripts/jacobian/compute_activation_pcs.py` (lines 48-108)

## Verification
- Run on a small subset first: `--n_total 100 --val_start 50 --val_end 100 --target_size 100` with a small batch_decomp results dir
- Check that dictionary size decreases as expected each round
- Check that provenance array stays consistent (lengths match, indices valid)
- Confirm output pkl loads cleanly and contains expected shapes
