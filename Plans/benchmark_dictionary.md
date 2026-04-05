# Benchmark Dictionary vs SAE

## Context

We want to measure how well our pruned dictionary reconstructs activations compared to an SAE (EleutherAI/sae-Llama-3.2-1B-131k), as a function of the number of active features n. This is an apples-to-oranges comparison: our dictionary reconstructs resid_post[2], the SAE reconstructs mlp_out[2]. KL-div is the metric that makes the comparison most meaningful since both ultimately affect the same downstream logits.

## Script

`scripts/jacobian/benchmark_dictionary.py`

## Output

`results/benchmark_dictionary/llama-3.2-1b_layer2/`
- `benchmark_results.pkl` — all metrics
- `benchmark_plot.png` — 3-panel figure (relative L2, cosine sim, KL-div)

## Dependencies

- `sparsify-sae` (`pip install sparsify-sae`) — for loading the EleutherAI SAE (`from sparsify import Sae`)

## Args

- `--model` (default `meta-llama/Llama-3.2-1B`)
- `--layer` (default 2)
- `--dictionary_path` — path to pruned_dictionary.pkl
- `--sae_repo` (default `EleutherAI/sae-Llama-3.2-1B-131k`)
- `--seed`, `--seq_len` (defaults 42, 10) — must match data pipeline
- `--n_total` (default 8000)
- `--eval_start` (default 7000), `--eval_end` (default 8000) — held-out indices
- `--n_values` (default `1,2,5,10,15,20,30,50`)
- `--output_dir` (default `results/benchmark_dictionary/llama-3.2-1b_layer2`)
- `--batch_size` (default 64)

## Implementation

### Phase 1: Extract activations

Load model, load 8000 FineWeb samples (same seed), process indices [7000, 8000):

For each batch:
1. Run `model(tokens, output_hidden_states=True)` to get `resid_post[2]` from `hidden_states[3]`
2. Use a forward hook on `model.model.layers[2].mlp` to capture `mlp_out[2]`
3. Capture clean logits from `outputs.logits[:, -1, :]`
4. Capture context at layer 2: `hidden_states[3][:, :-1, :]` (needed for KL-div forward passes)

Store (all last-token, float32, CPU):
- `resid_post`: (1000, 2048) — dictionary reconstruction target
- `mlp_out`: (1000, 2048) — SAE reconstruction target
- `clean_logits`: (1000, vocab_size) — KL-div reference
- `contexts`: (1000, seq_len-1, 2048) — prefix states for forward_from_layer

### Phase 2: Reconstruct for each n

**Dictionary (MP on resid_post):**
Reuse `matching_pursuit_usage` logic but return reconstructions instead of counts:
```python
def matching_pursuit_reconstruct(dictionary, activations, n_nonzero_coefs):
    """Returns (n_val, d_model) reconstructed activations."""
    residuals = activations.clone()
    recon = torch.zeros_like(activations)
    for step in range(n_nonzero_coefs):
        corr = residuals @ dictionary.T
        best = corr.abs().argmax(dim=1)
        coefs = corr[arange(n_val), best]
        contribution = coefs.unsqueeze(1) * dictionary[best]
        recon += contribution
        residuals -= contribution
    return recon
```

**SAE (top-n from encoder on mlp_out):**
```python
sae = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k",
                         hookpoint="layers.2.mlp", device=device)
top_acts, top_indices, pre_acts = sae.encode(mlp_out)
# pre_acts: (1000, 131072) — full ReLU'd activations before top-k
# For each n, take top-n from pre_acts:
topn_vals, topn_idx = pre_acts.topk(n, dim=-1)
recon_mlp = sae.decode(topn_vals, topn_idx)
```

### Phase 3: Compute metrics

For each n, for each method:

**Relative L2 error:** `||original - recon|| / ||original||`, averaged over 1000 activations
- Dictionary: compare recon vs resid_post
- SAE: compare recon vs mlp_out

**Cosine similarity:** `cos_sim(original, recon)`, averaged over 1000 activations

**KL-divergence:** requires forward passes from layer 2 onward
- Dictionary: `patched = recon_resid_post`, pass directly to `forward_from_layer`
- SAE: `patched = resid_post + (recon_mlp_out - clean_mlp_out)`, then `forward_from_layer`
- Both use: `forward_from_layer(model, context, patched.unsqueeze(1), start_layer=2)` → logits
- KL = mean over activations of `KL(clean_probs || patched_probs)` where probs = softmax(logits)

The patching trick for the SAE: since `resid_post = post_attn + mlp_out`, replacing mlp_out is equivalent to `resid_post + (recon_mlp - mlp)`. No need to split block 2.

KL-div forward passes: process in batches. ~10 n-values × 2 methods × 1000 activations = 20k partial forward passes (layer 2 → end). Should take a few minutes on GPU.

### Phase 4: Plot

3-panel figure (1 row × 3 cols):
- Panel 1: Relative L2 error vs n
- Panel 2: Cosine similarity vs n
- Panel 3: KL-divergence vs n

Each panel has two lines: "Dictionary (MP)" and "SAE (top-n)". Log scale on y-axis for KL-div and relative L2. x-axis: n values.

## Key files to reuse

- `src/model.py`: `load_model()`, `forward_from_layer()`, `_get_blocks()`
- `src/data.py`: `load_fineweb_fixed_length()`
- Matching pursuit pattern from `scripts/jacobian/prune_dictionary.py`
- Batched activation extraction pattern from `scripts/jacobian/compute_activation_pcs.py`

## Notes

- This is an apples-to-oranges comparison (resid_post vs mlp_out). KL-div is the most meaningful cross-comparison metric. Relative L2 and cosine sim are within-method only (each measured on its own target space).
- SAE's native sparsity is top-32. Our dictionary was trained with k=20 directions per activation. Results beyond n=20 are extrapolation for our dictionary.
- Will train our own resid_post SAE later for a proper apples-to-apples comparison.
