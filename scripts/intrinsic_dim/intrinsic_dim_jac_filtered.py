"""Compare TWO-NN ID: real activations vs Jacobian-norm-matched Gaussian samples.

1. Collect real layer-2 activations + contexts from FineWeb
2. Estimate ||J||_F (layer 2 → final layer) via Hutchinson's for real activations
3. Sample a large pool of cov-matched Gaussian points
4. Estimate ||J||_F for Gaussian points (using real contexts as carriers)
5. Filter Gaussian pool to match the real Jacobian norm distribution
6. Run TWO-NN on real, unfiltered Gaussian, and filtered Gaussian
"""

import sys
sys.path.insert(0, "/workspace/steering-plateaus")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import skdim

from src.model import load_model, _get_blocks, _is_rope_model, _get_position_embeddings, _run_block


def forward_batch_layer_to_final(model, hidden_states, start_layer):
    """Forward from start_layer to final block for a batch of full hidden states.

    Args:
        model: HF model
        hidden_states: [batch, seq_len, d] - full residual stream at start_layer
        start_layer: runs blocks [start_layer+1:]

    Returns:
        [batch, d] - last-token residual stream after final block
    """
    hidden = hidden_states
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    return hidden[:, -1, :]


def estimate_jac_norms_batched(model, contexts, activations, source_layer,
                               n_probes=10, eps=1e-3, batch_size=64):
    """Estimate ||J||_F for each activation via batched finite-difference Hutchinson's.

    Args:
        contexts: [N, seq_len-1, d] - prefix residual streams
        activations: [N, d] - last-token activations
        source_layer: layer index
        n_probes: number of random directions per point
        eps: finite difference step
        batch_size: points per batch

    Returns:
        [N] tensor of estimated ||J||_F
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    N, d = activations.shape
    all_norms_sq = torch.zeros(N)

    for i in range(0, N, batch_size):
        bs = min(batch_size, N - i)
        ctx = contexts[i:i + bs].to(device, dtype)    # [bs, seq-1, d]
        act = activations[i:i + bs].to(device, dtype)  # [bs, d]

        # Base forward
        hidden_base = torch.cat([ctx, act.unsqueeze(1)], dim=1)  # [bs, seq, d]
        with torch.no_grad():
            f_base = forward_batch_layer_to_final(model, hidden_base, source_layer)  # [bs, d]

        # Hutchinson probes
        batch_norm_sq = torch.zeros(bs, device=device)
        for _ in range(n_probes):
            v = torch.randn(bs, d, device=device, dtype=dtype)
            act_pert = act + eps * v
            hidden_pert = torch.cat([ctx, act_pert.unsqueeze(1)], dim=1)
            with torch.no_grad():
                f_pert = forward_batch_layer_to_final(model, hidden_pert, source_layer)
            Jv = (f_pert - f_base) / eps
            batch_norm_sq += (Jv.float() ** 2).sum(dim=-1)

        all_norms_sq[i:i + bs] = (batch_norm_sq / n_probes).cpu()

        if i % (batch_size * 50) == 0:
            print(f"  {i}/{N}")

    return all_norms_sq.sqrt()


def main():
    source_layer = 2
    n_real = 5000
    n_gauss_pool = 20000
    n_probes = 10
    seq_len = 10
    batch_size = 64

    print("Loading model...")
    model, tokenizer, device = load_model("Qwen/Qwen3-1.7B", dtype=torch.float16)

    # Load texts
    print("Loading texts...")
    ds = load_dataset("HuggingFaceFW/FineWeb-Edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex["text"]) >= seq_len * 3:
            texts.append(ex["text"])
        if len(texts) >= n_real:
            break
    print(f"Collected {len(texts)} texts")

    # Collect real activations and contexts
    print("Extracting real activations + contexts...")
    all_acts = []
    all_contexts = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        tokens = tokenizer(
            batch, return_tensors='pt', padding='max_length',
            truncation=True, max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        hidden = out.hidden_states[source_layer + 1]
        all_acts.append(hidden[:, -1, :].cpu())
        all_contexts.append(hidden[:, :-1, :].cpu())
    X_real = torch.cat(all_acts, dim=0)
    contexts = torch.cat(all_contexts, dim=0)
    print(f"Real activations: {X_real.shape}, contexts: {contexts.shape}")

    # TWO-NN on real
    est_real = skdim.id.TwoNN()
    est_real.fit(X_real.float().numpy())
    print(f"Real activations TWO-NN ID: {est_real.dimension_:.2f}")

    # Jacobian norms for real activations
    print(f"\nComputing Jacobian norms for real activations ({n_probes} probes, batch={batch_size})...")
    real_jac = estimate_jac_norms_batched(
        model, contexts, X_real, source_layer,
        n_probes=n_probes, batch_size=batch_size,
    )
    print(f"Real ||J||_F: mean={real_jac.mean():.2f}, std={real_jac.std():.2f}, "
          f"median={real_jac.median():.2f}, [p10={real_jac.quantile(0.1):.2f}, p90={real_jac.quantile(0.9):.2f}]")

    # Cov-matched Gaussian pool
    print(f"\nSampling {n_gauss_pool} cov-matched Gaussian points...")
    X_real_f32 = X_real.float().numpy()
    mean = X_real_f32.mean(axis=0)
    cov = np.cov(X_real_f32, rowvar=False)
    X_gauss = torch.from_numpy(
        np.random.RandomState(0).multivariate_normal(mean, cov, size=n_gauss_pool)
    ).float()

    # Cycle through real contexts for Gaussian points
    gauss_contexts = contexts[torch.arange(n_gauss_pool) % len(contexts)]

    # Jacobian norms for Gaussian points
    print(f"Computing Jacobian norms for {n_gauss_pool} Gaussian points...")
    gauss_jac = estimate_jac_norms_batched(
        model, gauss_contexts, X_gauss, source_layer,
        n_probes=n_probes, batch_size=batch_size,
    )
    print(f"Gaussian ||J||_F: mean={gauss_jac.mean():.2f}, std={gauss_jac.std():.2f}, "
          f"median={gauss_jac.median():.2f}, [p10={gauss_jac.quantile(0.1):.2f}, p90={gauss_jac.quantile(0.9):.2f}]")

    # Filter Gaussian to match real Jacobian norm range (10th-90th percentile)
    lo = real_jac.quantile(0.1).item()
    hi = real_jac.quantile(0.9).item()
    mask = (gauss_jac >= lo) & (gauss_jac <= hi)
    X_gauss_filtered = X_gauss[mask]
    print(f"\nFiltering to ||J||_F in [{lo:.2f}, {hi:.2f}] (real p10-p90)")
    print(f"Kept {mask.sum().item()}/{n_gauss_pool} Gaussian points")

    if mask.sum() < 100:
        print("Too few points survived filtering!")
        return

    # TWO-NN comparisons
    est_unfilt = skdim.id.TwoNN()
    est_unfilt.fit(X_gauss[:5000].float().numpy())

    est_filt = skdim.id.TwoNN()
    est_filt.fit(X_gauss_filtered.float().numpy())

    # Save results
    import os
    os.makedirs("results/intrinsic_dim", exist_ok=True)
    torch.save({
        'real_jac_norms': real_jac,
        'gauss_jac_norms': gauss_jac,
        'X_real': X_real,
        'X_gauss': X_gauss,
        'mask': mask,
        'real_id': est_real.dimension_,
        'unfilt_id': est_unfilt.dimension_,
        'filt_id': est_filt.dimension_,
        'source_layer': source_layer,
        'n_probes': n_probes,
    }, "results/intrinsic_dim/jac_filtered_data.pt")
    print("Saved data to results/intrinsic_dim/jac_filtered_data.pt")

    print(f"\n=== Summary ===")
    print(f"Real activations ID:           {est_real.dimension_:.2f}  (n={n_real})")
    print(f"Unfiltered Gaussian ID:        {est_unfilt.dimension_:.2f}  (n=5000)")
    print(f"Jacobian-filtered Gaussian ID: {est_filt.dimension_:.2f}  (n={mask.sum().item()})")

    # Histogram of Jacobian norms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(real_jac.max().item(), gauss_jac.max().item()) * 1.05, 80)
    ax.hist(real_jac.numpy(), bins=bins, alpha=0.6, label=f'Real (n={n_real})', density=True)
    ax.hist(gauss_jac.numpy(), bins=bins, alpha=0.6, label=f'Cov-matched Gaussian (n={n_gauss_pool})', density=True)
    ax.axvline(lo, color='k', ls='--', alpha=0.5, label=f'Real p10/p90')
    ax.axvline(hi, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('||J||_F')
    ax.set_ylabel('Density')
    ax.set_title('Jacobian norm distribution: real vs cov-matched Gaussian\n'
                 f'Qwen3-1.7B, layer {source_layer}')
    ax.legend()
    plt.tight_layout()
    outpath = 'results/intrinsic_dim/jacobian_norm_hist.png'
    plt.savefig(outpath, dpi=150)
    print(f"Saved histogram to {outpath}")


if __name__ == "__main__":
    main()
