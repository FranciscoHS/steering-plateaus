#!/usr/bin/env python3
"""
Optimize Gaussian activations to MATCH the Jacobian norm distribution of real
activations (mean + std), rather than minimizing it.

Usage:
    python -u scripts/jacobian/intrinsic_dim_jac_matched.py
"""

import os
import sys
import time

import numpy as np
import torch
import skdim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model import load_model, forward_from_layer_to_final



def main():
    source_layer = 2
    n_points = 200
    n_steps = 100
    lr = 0.1
    lam = 0.001  # movement penalty
    n_probes = 5

    print("Loading model...")
    model, tokenizer, device = load_model("Qwen/Qwen3-1.7B", dtype=torch.float32)

    # Load saved data
    data_path = "results/intrinsic_dim/jac_filtered_data.pt"
    print(f"Loading saved data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    X_real = data['X_real']
    X_gauss = data['X_gauss']
    real_jac_norms = data['real_jac_norms']

    # Re-extract contexts
    print("Re-extracting contexts from FineWeb...")
    from datasets import load_dataset
    seq_len = 10
    ds = load_dataset("HuggingFaceFW/FineWeb-Edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex["text"]) >= seq_len * 3:
            texts.append(ex["text"])
        if len(texts) >= 5000:
            break

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
        all_contexts.append(hidden[:, :-1, :].cpu())
    contexts = torch.cat(all_contexts, dim=0)

    # Subset
    gauss_subset = X_gauss[:n_points]
    ctx_subset = contexts[torch.arange(n_points) % len(contexts)]
    # Use first n_points real activations with matching contexts
    real_subset = X_real[:n_points]
    real_ctx_subset = contexts[:n_points]

    # Compute real Jacobian norms using the SAME VJP method used in optimization
    d = gauss_subset.shape[1]
    print(f"\nComputing real Jacobian norms via VJP ({n_points} points, {n_probes} probes)...")
    real_vjp_norms = []
    for i in range(n_points):
        ctx_i = real_ctx_subset[i:i+1].to(device).detach()
        point_i = real_subset[i:i+1].to(device).unsqueeze(1).detach().requires_grad_(True)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, ctx_i, point_i, source_layer)

        frob_sq = 0.0
        for k in range(n_probes):
            u = torch.randn(1, d, device=device)
            s = (final_act * u).sum()
            g = torch.autograd.grad(s, point_i, retain_graph=True)[0]
            frob_sq += (g.float() ** 2).sum().item()
        frob_sq /= n_probes

        norm_val = frob_sq ** 0.5
        real_vjp_norms.append(norm_val)
        if (i + 1) % 50 == 0:
            norms_so_far = torch.tensor(real_vjp_norms)
            print(f"  {i+1}/{n_points} done, running mean={norms_so_far.mean():.1f}, "
                  f"max={norms_so_far.max():.1f}, any inf={torch.isinf(norms_so_far).any()}")

    real_vjp_norms = torch.tensor(real_vjp_norms)
    # Filter out any inf/nan
    valid = torch.isfinite(real_vjp_norms)
    if not valid.all():
        print(f"  WARNING: {(~valid).sum()} inf/nan norms, filtering them out")
        real_vjp_norms = real_vjp_norms[valid]
    print(f"  Real VJP Jacobian norms: mean={real_vjp_norms.mean():.1f}, std={real_vjp_norms.std():.1f}")
    print(f"  (cf. saved finite-diff norms: mean={real_jac_norms.mean():.1f}, std={real_jac_norms.std():.1f})")

    # TWO-NN before
    est_before = skdim.id.TwoNN()
    est_before.fit(gauss_subset.float().numpy() + np.random.randn(*gauss_subset.shape) * 1e-7)
    print(f"\nBefore optimization: TWO-NN ID = {est_before.dimension_:.2f} (n={n_points})")

    # Optimize
    # Assign each point a target Jacobian norm sampled from the real VJP distribution.
    # This matches both mean and std while allowing per-point backward (no OOM).
    x0 = gauss_subset.clone().to(device)
    x = gauss_subset.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    # Sample per-point targets from the real VJP Jacobian norm distribution
    target_norms = real_vjp_norms[torch.randperm(n_points)].float().to(device)
    print(f"\nPer-point targets: mean={target_norms.mean():.1f}, std={target_norms.std():.1f}")

    jac_history = []
    dist_history = []
    match_history = []

    print(f"Optimizing {n_points} points ({n_steps} steps, lr={lr}, lam={lam})...")

    for step in range(n_steps):
        optimizer.zero_grad()

        total_match_loss = 0.0
        total_dist_loss = 0.0
        step_jac_norms = []

        for i in range(n_points):
            ctx_i = ctx_subset[i:i+1].to(device).detach()
            point_i = x[i:i+1].unsqueeze(1)

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                final_act = forward_from_layer_to_final(model, ctx_i, point_i, source_layer)

            frob_sq = torch.tensor(0.0, device=device)
            for k in range(n_probes):
                u = torch.randn(1, d, device=device)
                s = (final_act * u).sum()
                g = torch.autograd.grad(s, x, create_graph=True, retain_graph=True)[0]
                frob_sq = frob_sq + (g[i].float() ** 2).sum()
            frob_sq = frob_sq / n_probes
            jac_norm = torch.sqrt(frob_sq + 1e-8)

            # Per-point matching loss: push toward sampled target
            match_loss_i = (jac_norm - target_norms[i]) ** 2
            dist_loss_i = lam * ((x[i] - x0[i]) ** 2).sum()

            loss_i = (match_loss_i + dist_loss_i) / n_points
            loss_i.backward()

            total_match_loss += match_loss_i.item()
            total_dist_loss += dist_loss_i.item()
            step_jac_norms.append(jac_norm.item())

        optimizer.step()

        step_jac_norms = np.array(step_jac_norms)
        jac_history.append(float(step_jac_norms.mean()))
        dist_history.append(total_dist_loss / n_points)
        match_history.append(total_match_loss / n_points)

        if step % 10 == 0:
            avg_dist = ((x.detach() - x0) ** 2).sum(dim=-1).sqrt().mean().item()
            print(f"  step {step}: jac_mean={step_jac_norms.mean():.1f}, jac_std={step_jac_norms.std():.1f}, "
                  f"match_loss={total_match_loss/n_points:.1f}, avg_move={avg_dist:.2f}")

        # Checkpoint every 50 steps
        if (step + 1) % 50 == 0:
            os.makedirs("results/intrinsic_dim", exist_ok=True)
            torch.save({
                'X_opt': x.detach().cpu(),
                'history': {'jac_mean': jac_history, 'dist': dist_history, 'match': match_history},
                'step': step + 1,
            }, "results/intrinsic_dim/jac_matched_checkpoint.pt")
            print(f"  [checkpoint saved at step {step + 1}]")

    X_opt = x.detach().cpu()

    # Checkpoint final
    os.makedirs("results/intrinsic_dim", exist_ok=True)
    torch.save({
        'X_opt': X_opt,
        'X_gauss_subset': gauss_subset,
        'history': {'jac_mean': jac_history, 'dist': dist_history, 'match': match_history},
        'n_steps': n_steps, 'lr': lr, 'lam': lam, 'n_probes': n_probes,
        'target_mean': target_mean, 'target_std': target_std,
    }, "results/intrinsic_dim/jac_matched_checkpoint.pt")
    print("Checkpoint saved")

    # TWO-NN after
    est_after = skdim.id.TwoNN()
    est_after.fit(X_opt.float().numpy() + np.random.randn(*X_opt.shape) * 1e-7)
    print(f"\nAfter optimization: TWO-NN ID = {est_after.dimension_:.2f} (n={n_points})")

    # Distance to nearest real activation
    dists_before = torch.cdist(gauss_subset.float().unsqueeze(0), X_real.float().unsqueeze(0)).squeeze(0)
    dists_after = torch.cdist(X_opt.float().unsqueeze(0), X_real.float().unsqueeze(0)).squeeze(0)
    nn_before = dists_before.min(dim=1).values
    nn_after = dists_after.min(dim=1).values
    print(f"\nDist to nearest real activation:")
    print(f"  Before: mean={nn_before.mean():.2f}, median={nn_before.median():.2f}")
    print(f"  After:  mean={nn_after.mean():.2f}, median={nn_after.median():.2f}")

    move_dist = ((X_opt - gauss_subset) ** 2).sum(dim=-1).sqrt()
    print(f"\nMovement from start: mean={move_dist.mean():.2f}, median={move_dist.median():.2f}")

    # Real ID for reference
    est_real = skdim.id.TwoNN()
    est_real.fit(X_real.float().numpy() + np.random.randn(*X_real.shape) * 1e-7)
    print(f"\nReal activations TWO-NN ID: {est_real.dimension_:.2f} (n={X_real.shape[0]})")

    # Save
    torch.save({
        'X_opt': X_opt,
        'X_gauss_subset': gauss_subset,
        'history': {'jac_mean': jac_history, 'dist': dist_history, 'match': match_history},
        'id_before': est_before.dimension_,
        'id_after': est_after.dimension_,
        'nn_before': nn_before,
        'nn_after': nn_after,
        'n_steps': n_steps, 'lr': lr, 'lam': lam, 'n_probes': n_probes,
        'target_mean': target_mean, 'target_std': target_std,
    }, "results/intrinsic_dim/jac_matched_data.pt")
    print("Saved to results/intrinsic_dim/jac_matched_data.pt")

    # Loss curves
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(jac_history, label='Batch mean ||J||_F')
    ax1.axhline(target_mean, color='red', linestyle='--', label=f'Target mean={target_mean:.0f}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('||J||_F')
    ax1.set_title('Jacobian norm mean')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(match_history, label='Match loss')
    ax2.plot(dist_history, label='Distance penalty')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.savefig("results/intrinsic_dim/jac_matched_loss_curves.png", dpi=150, bbox_inches='tight')
    print("Saved loss curves to results/intrinsic_dim/jac_matched_loss_curves.png")


if __name__ == '__main__':
    main()
