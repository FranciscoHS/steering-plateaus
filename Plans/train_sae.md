# Train Residual Stream SAE

## Context

Train an 8x TopK SAE on resid_post at layer 2 of Llama-3.2-1B for an apples-to-apples comparison with our pruned dictionary. The EleutherAI SAE operates on mlp_out, so we need our own resid_post SAE for a fair comparison.

## Script

`scripts/jacobian/train_sae.py`

## Output

`results/trained_sae/llama-3.2-1b_layer2_8x/`
- `sae_weights.pt` — state dict + config
- `training_log.pkl` — per-step loss, dead feature counts
- `training_plot.png` — loss curve + dead feature count over training

## Architecture: TopK SAE

```
d_in = 2048, d_sae = 16384 (8x expansion)
k = configurable (train both k=32 and k=20)

encode(x):
    x_centered = x - b_dec
    pre_acts = ReLU(x_centered @ W_enc.T + b_enc)  # (batch, 16384)
    top_vals, top_idx = pre_acts.topk(k)             # (batch, k)
    return top_vals, top_idx, pre_acts

decode(top_vals, top_idx):
    recon = scatter_sum(top_vals * W_dec[top_idx]) + b_dec
    return recon

Loss = MSE(x, recon)  # no L1 penalty — sparsity is structural
```

Decoder columns normalized to unit norm after each optimizer step.

## Data index layout

All use same seed=42, seq_len=10:
- [0, 2000): dictionary training (batch_decomp)
- [2000, 7000): pruning validation
- [7000, 8000): benchmark eval
- **[8000, 8000+n_cache): SAE training cache** ← no overlap with eval

## Args

- `--model`, `--layer`, `--seed`, `--seq_len` — defaults: Llama-3.2-1B, 2, 42, 10
- `--d_sae` (default 16384)
- `--k` (default 32)
- `--n_total` — total FineWeb sequences to load (must be >= cache_start + n_cache)
- `--cache_start` (default 8000) — first index for SAE training data
- `--n_cache` (default 2000000) — number of activations to pre-cache
- `--n_steps` (default 50000)
- `--batch_size` (default 4096)
- `--lr` (default 3e-4)
- `--resample_interval` (default 25000)
- `--output_dir`
- `--reuse_cache` — skip extraction if cache file exists

## Implementation

### Step 1: Pre-cache activations

- Load `cache_start + n_cache` FineWeb sequences via `load_fineweb_fixed_length`
- Extract last-token resid_post at layer 2 for indices [cache_start, cache_start + n_cache)
- Save to `output_dir/cached_activations.pt`
- Skip if `--reuse_cache` and file exists
- 2M activations × 2048 × 4 bytes = 16GB on disk

Compute: 2M / 64 = 31k batches × ~5ms ≈ 2.5 minutes.

After caching, free the model from GPU.

### Step 2: Train SAE

```python
activations = torch.load("cached_activations.pt").to(device)  # 16GB → GPU

sae = TopKSAE(d_in=2048, d_sae=16384, k=32).to(device)
optimizer = Adam(sae.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)

# Track dead features from step 0
feature_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)

for step in range(n_steps):
    idx = torch.randint(0, n_cache, (batch_size,))
    batch = activations[idx]

    recon, top_vals, top_idx = sae(batch)
    loss = F.mse_loss(recon, batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Normalize decoder columns
    with torch.no_grad():
        sae.W_dec.data = F.normalize(sae.W_dec.data, dim=-1)

    # Track which features fired
    feature_fired[top_idx.flatten().unique()] = True

    # Dead feature resampling
    if step > 0 and step % resample_interval == 0:
        resample_dead_features(sae, optimizer, activations, feature_fired)
        feature_fired.zero_()  # reset tracking for next window
```

At 50k steps × batch_size 4096 / 2M cache = ~100 epochs. If loss curves look like overfitting, reduce n_steps or increase n_cache.

### Step 3: Dead feature resampling

At each resampling checkpoint:
1. Identify features where `feature_fired == False` (never activated since last resample / start)
2. For dead features:
   - Sample a batch of activations from cache
   - Compute per-activation MSE, find the highest-loss examples
   - Reinitialize dead encoder rows toward high-loss inputs (with some noise)
   - Reinitialize dead decoder columns to normalized high-loss inputs
   - Reset encoder bias for dead features to 0
   - Reset Adam optimizer state for those parameters

### Step 4: Evaluation

After training, quick eval on a sample from the cache (not the held-out set, which isn't loaded here):
- Reconstruction MSE, relative L2, cosine sim at native k
- Dead feature count
- Log the numbers; full benchmark comparison done separately via benchmark_dictionary.py

### Step 5: Save

```python
torch.save({
    'state_dict': sae.state_dict(),
    'config': {'d_in': 2048, 'd_sae': 16384, 'k': 32},
    'training_log': training_log,
    'args': vars(args),
}, output_path)
```

## Training plan

Train two SAEs (both cheap enough to do sequentially):
1. `k=32` — matches EleutherAI SAE's sparsity, standard comparison
2. `k=20` — matches our dictionary's 20 directions per activation

## Compute estimates

- **Pre-caching 2M activations**: ~2.5 minutes
- **SAE training (50k steps)**: ~50-100 seconds
- **Total: ~5 minutes per SAE** (not counting model load)

Far less than 40 GPU-hours. Report compute costs transparently.

## Memory

- 2M cached activations in float32: 16GB. Fits on a 24GB GPU with room for the SAE (~130MB).
- If GPU memory is tight: keep cache on CPU, transfer batches per step (slower but works on smaller GPUs).

## Integration with benchmark

The TopKSAE class will expose `encode(x) → (top_vals, top_idx, pre_acts)` and `decode(top_vals, top_idx) → recon`, matching the interface that `benchmark_dictionary.py:sae_reconstruct_topn` expects. Add a third method to the benchmark: "Our SAE (top-n on resid_post)".

## Key files to reuse

- `src/model.py`: `load_model()`
- `src/data.py`: `load_fineweb_fixed_length()`
- Batched extraction pattern from `scripts/jacobian/compute_activation_pcs.py`
