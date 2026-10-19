"""Optimize Gaussian-sampled activations to minimize Jacobian norm, then check TWO-NN ID.

Starting from cov-matched Gaussian points, run gradient descent to minimize ||J||_F
with a penalty lambda * ||x - x_start||^2 to prevent collapse. Then check if
the optimized points have lower intrinsic dimensionality.

Uses saved data from 1_sample_real_and_gaussian.py (originally
intrinsic_dim_jac_optimized.py).

Note: the carrier contexts are re-extracted by streaming FineWeb again, and
they are not saved with the optimized points. The stream order is not
guaranteed to be reproducible, so later scripts cannot recover exactly which
context each point was optimized in.

Usage (from the repo root):
    python scripts/2_minimize_jacobian.py

Output: results/intrinsic_dim/jac_optimized_data.pt (key 'X_opt': [200, 2048]),
jac_opt_loss_curves.png
"""

import sys
sys.path.insert(0, ".")  # run from the repo root

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import skdim

from src.model import load_model, forward_from_layer_to_final


def optimize_points(model, points, contexts, layer, n_steps=200, lr=0.1,
                    lam=1.0, n_probes=5, device='cuda'):
    """Optimize points to minimize ||J||_F + lam * ||x - x0||^2.

    Args:
        points: [N, d] starting points (Gaussian samples)
        contexts: [N, seq-1, d] carrier contexts
        layer: source layer
        n_steps: optimization steps
        lr: learning rate
        lam: penalty weight for distance from start
        n_probes: Hutchinson probes per step

    Returns:
        optimized: [N, d] optimized points
        history: dict with loss curves
    """
    N, d = points.shape
    x0 = points.clone().to(device)  # anchor
    x = points.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    jac_history = []
    dist_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        total_jac_loss = 0.0
        total_dist_loss = 0.0

        # Process one at a time (each has its own context)
        for i in range(N):
            ctx_i = contexts[i:i+1].to(device).detach()
            point_i = x[i:i+1].unsqueeze(1)  # [1, 1, d]

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                final_act = forward_from_layer_to_final(model, ctx_i, point_i, layer)

            # Hutchinson estimate of ||J||_F^2
            frob_sq = torch.tensor(0.0, device=device)
            for k in range(n_probes):
                u = torch.randn(1, d, device=device)
                s = (final_act * u).sum()
                g = torch.autograd.grad(s, x, create_graph=True, retain_graph=True)[0]
                frob_sq = frob_sq + (g[i] ** 2).sum()
            frob_sq = frob_sq / n_probes

            jac_loss = torch.sqrt(frob_sq + 1e-8)
            dist_loss = lam * ((x[i] - x0[i]) ** 2).sum()

            loss_i = (jac_loss + dist_loss) / N
            loss_i.backward()

            total_jac_loss += jac_loss.item()
            total_dist_loss += dist_loss.item()

        optimizer.step()

        jac_history.append(total_jac_loss / N)
        dist_history.append(total_dist_loss / N)

        if step % 20 == 0:
            avg_dist = ((x.detach() - x0) ** 2).sum(dim=-1).sqrt().mean().item()
            print(f"  step {step}: jac={total_jac_loss/N:.1f}, dist_loss={total_dist_loss/N:.1f}, "
                  f"avg_move={avg_dist:.2f}")

        # Checkpoint every 50 steps
        if (step + 1) % 50 == 0:
            import os
            os.makedirs("results/intrinsic_dim", exist_ok=True)
            torch.save({
                'X_opt': x.detach().cpu(),
                'history': {'jac': jac_history, 'dist': dist_history},
                'step': step + 1,
            }, "results/intrinsic_dim/jac_optimized_checkpoint.pt")
            print(f"  [checkpoint saved at step {step + 1}]")

    return x.detach().cpu(), {'jac': jac_history, 'dist': dist_history}


def main():
    source_layer = 2
    n_points = 200
    n_steps = 200
    lr = 0.1
    lam = 0.001  # light penalty — let them move
    n_probes = 5

    print("Loading model...")
    model, tokenizer, device = load_model("Qwen/Qwen3-1.7B", dtype=torch.float32)

    # Load saved data
    data_path = "results/intrinsic_dim/jac_filtered_data.pt"
    print(f"Loading saved data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    X_real = data['X_real']
    X_gauss = data['X_gauss']

    # We need contexts — re-extract them (they weren't saved)
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

    # Pick subset of Gaussian points + their carrier contexts
    gauss_subset = X_gauss[:n_points]
    ctx_subset = contexts[torch.arange(n_points) % len(contexts)]

    # TWO-NN before optimization (on same subset for comparison)
    est_before = skdim.id.TwoNN()
    est_before.fit(gauss_subset.float().numpy())
    print(f"\nBefore optimization: TWO-NN ID = {est_before.dimension_:.2f} (n={n_points})")

    # Optimize
    print(f"\nOptimizing {n_points} points ({n_steps} steps, lr={lr}, lam={lam})...")
    X_opt, history = optimize_points(
        model, gauss_subset, ctx_subset, source_layer,
        n_steps=n_steps, lr=lr, lam=lam, n_probes=n_probes, device=device,
    )

    # Checkpoint immediately so we don't lose optimization results
    os.makedirs("results/intrinsic_dim", exist_ok=True)
    torch.save({
        'X_opt': X_opt,
        'X_gauss_subset': gauss_subset,
        'history': history,
        'n_steps': n_steps, 'lr': lr, 'lam': lam, 'n_probes': n_probes,
    }, "results/intrinsic_dim/jac_optimized_checkpoint.pt")
    print("Checkpoint saved to results/intrinsic_dim/jac_optimized_checkpoint.pt")

    # TWO-NN after optimization
    est_after = skdim.id.TwoNN()
    est_after.fit(X_opt.float().numpy())
    print(f"\nAfter optimization: TWO-NN ID = {est_after.dimension_:.2f} (n={n_points})")

    # Distance to nearest real activation
    dists_before = torch.cdist(gauss_subset.float().unsqueeze(0), X_real.float().unsqueeze(0)).squeeze(0)
    dists_after = torch.cdist(X_opt.float().unsqueeze(0), X_real.float().unsqueeze(0)).squeeze(0)
    nn_before = dists_before.min(dim=1).values
    nn_after = dists_after.min(dim=1).values
    print(f"\nDist to nearest real activation:")
    print(f"  Before: mean={nn_before.mean():.2f}, median={nn_before.median():.2f}")
    print(f"  After:  mean={nn_after.mean():.2f}, median={nn_after.median():.2f}")

    # How far did points move?
    move_dist = ((X_opt - gauss_subset) ** 2).sum(dim=-1).sqrt()
    print(f"\nMovement from start: mean={move_dist.mean():.2f}, median={move_dist.median():.2f}")

    # Real activation ID for reference
    est_real = skdim.id.TwoNN()
    est_real.fit(X_real.float().numpy())
    print(f"\nReal activations TWO-NN ID: {est_real.dimension_:.2f} (n={X_real.shape[0]})")

    # Save
    os.makedirs("results/intrinsic_dim", exist_ok=True)
    torch.save({
        'X_opt': X_opt,
        'X_gauss_subset': gauss_subset,
        'history': history,
        'id_before': est_before.dimension_,
        'id_after': est_after.dimension_,
        'nn_before': nn_before,
        'nn_after': nn_after,
        'n_steps': n_steps, 'lr': lr, 'lam': lam, 'n_probes': n_probes,
    }, "results/intrinsic_dim/jac_optimized_data.pt")
    print("Saved to results/intrinsic_dim/jac_optimized_data.pt")

    # Loss curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['jac'], label='||J||_F')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean ||J||_F')
    ax1.set_title('Jacobian norm')
    ax1.legend()

    ax2.plot(history['dist'], label='dist penalty')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Mean dist loss')
    ax2.set_title('Distance penalty')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/intrinsic_dim/jac_opt_loss_curves.png", dpi=150)
    print("Saved loss curves to results/intrinsic_dim/jac_opt_loss_curves.png")


if __name__ == "__main__":
    main()
