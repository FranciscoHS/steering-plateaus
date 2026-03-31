"""
Measure how much of real activations' L2 norm is captured by the MELBO
direction subspace vs random subspaces of the same dimensionality.

For each activation a:
    fraction = ||proj_S(a)|| / ||a||
where S is the subspace spanned by the directions.

Usage:
    python scripts/analysis/subspace_projection.py
    python scripts/analysis/subspace_projection.py --n_acts 1000 --seq_len 64
"""

import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model import load_model
from src.data import load_fineweb_fixed_length


def project_fractions(activations, directions):
    """
    Compute ||proj_S(a)|| / ||a|| for each activation.

    Args:
        activations: [n_acts, d_model]
        directions: [n_dirs, d_model] (orthonormalized internally via QR)

    Returns:
        fractions: [n_acts]
    """
    # Orthonormalize via QR
    Q, _ = torch.linalg.qr(directions.T)  # [d_model, n_dirs] orthonormal columns
    # coeffs[i,j] = Q[:,i] . activations[j]
    coeffs = Q.T @ activations.T  # [n_dirs, n_acts]
    proj_norms = coeffs.norm(dim=0)  # [n_acts]
    act_norms = activations.norm(dim=1)  # [n_acts]
    return proj_norms / act_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', type=str,
                        default='results/iterative_melbo_hf/qwen3-1.7b_L7_meas-2_angle0.1_act0/results.pkl')
    parser.add_argument('--n_acts', type=int, default=500,
                        help='Number of real activations to collect')
    parser.add_argument('--n_random_trials', type=int, default=20,
                        help='Number of random subspace trials')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length for FineWeb samples')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/subspace_projection')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load MELBO directions ---
    print(f"Loading MELBO directions from {args.results_pkl}")
    with open(args.results_pkl, 'rb') as f:
        results = pickle.load(f)

    melbo_dirs = torch.stack(results['directions']).float()  # [n_dirs, d_model]
    n_dirs = melbo_dirs.shape[0]
    d_model = melbo_dirs.shape[1]
    melbo_args = results['args']
    model_name = melbo_args['model']
    layer = melbo_args['inject_layer']

    print(f"  {n_dirs} directions, d_model={d_model}")
    print(f"  Model: {model_name}, layer: {layer}")

    # --- Load model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model {model_name} on {device}...")
    model, tokenizer, device = load_model(model_name, device=device)

    # --- Collect real activations ---
    print(f"\nLoading {args.n_acts} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_acts, tokenizer, seq_len=args.seq_len, seed=args.seed)

    print(f"Extracting activations at layer {layer}...")
    all_activations = []
    for i in range(0, len(token_ids_list), args.batch_size):
        batch = torch.stack(token_ids_list[i:i + args.batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]  # after block `layer`
        acts = hidden[:, -1, :].cpu()  # last token position
        all_activations.append(acts)
        print(f"  {min(i + args.batch_size, len(token_ids_list))}/{len(token_ids_list)}", end='\r')
    print()

    activations = torch.cat(all_activations, dim=0).float()  # [n_acts, d_model]
    act_norms = activations.norm(dim=1)
    print(f"Collected {activations.shape[0]} activations")
    print(f"  Norm: mean={act_norms.mean():.2f}, std={act_norms.std():.2f}")

    # --- MELBO projection ---
    melbo_fracs = project_fractions(activations, melbo_dirs)

    # --- Uniform random projections ---
    torch.manual_seed(args.seed)
    random_fracs_all = []
    for trial in range(args.n_random_trials):
        rand_dirs = torch.randn(n_dirs, d_model)
        fracs = project_fractions(activations, rand_dirs)
        random_fracs_all.append(fracs)
    random_fracs_all = torch.stack(random_fracs_all)  # [n_trials, n_acts]

    # --- Covariance-matched random projections ---
    # Sample directions from N(0, Sigma) where Sigma is the activation covariance,
    # so the random subspace is biased toward high-variance directions.
    print("Computing activation covariance for cov-matched baseline...")
    centered = activations - activations.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (activations.shape[0] - 1)  # [d_model, d_model]
    # Cholesky for sampling: v = L @ z gives v ~ N(0, Sigma)
    # Cov is likely rank-deficient (n_acts < d_model), so regularize
    eye = torch.eye(d_model)
    trace = torch.trace(cov) / d_model
    L_cov = None
    for scale in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            L_cov = torch.linalg.cholesky(cov + scale * trace * eye)
            if scale > 0:
                print(f"  Cholesky succeeded with regularization scale={scale}")
            break
        except torch.linalg.LinAlgError:
            continue
    assert L_cov is not None, "Cholesky failed even with large regularization"

    torch.manual_seed(args.seed + 1000)  # different seed from uniform
    covmatch_fracs_all = []
    for trial in range(args.n_random_trials):
        z = torch.randn(n_dirs, d_model)
        cov_dirs = z @ L_cov.T  # [n_dirs, d_model], each row ~ N(0, Sigma)
        fracs = project_fractions(activations, cov_dirs)
        covmatch_fracs_all.append(fracs)
    covmatch_fracs_all = torch.stack(covmatch_fracs_all)  # [n_trials, n_acts]

    # --- Theoretical baseline ---
    # For isotropic activations and random n_dirs-dim subspace in d_model dims:
    # E[||proj||^2 / ||a||^2] = n_dirs / d_model
    # So E[||proj|| / ||a||] ≈ sqrt(n_dirs / d_model)
    theoretical = np.sqrt(n_dirs / d_model)

    # --- Print results ---
    rand_means = random_fracs_all.mean(dim=1)
    rand_medians = random_fracs_all.median(dim=1).values
    cov_means = covmatch_fracs_all.mean(dim=1)
    cov_medians = covmatch_fracs_all.median(dim=1).values

    print(f"\n=== Subspace projection: {n_dirs} directions, {activations.shape[0]} activations ===")
    print(f"MELBO subspace:")
    print(f"  mean  = {melbo_fracs.mean():.4f}")
    print(f"  median = {melbo_fracs.median():.4f}")
    print(f"  std   = {melbo_fracs.std():.4f}")
    print(f"Cov-matched random ({args.n_random_trials} trials):")
    print(f"  mean  = {cov_means.mean():.4f} +/- {cov_means.std():.4f}")
    print(f"  median = {cov_medians.mean():.4f} +/- {cov_medians.std():.4f}")
    print(f"Uniform random ({args.n_random_trials} trials):")
    print(f"  mean  = {rand_means.mean():.4f} +/- {rand_means.std():.4f}")
    print(f"  median = {rand_medians.mean():.4f} +/- {rand_medians.std():.4f}")
    print(f"Isotropic theoretical (sqrt(n/d)): {theoretical:.4f}")
    print(f"Ratios (mean):")
    print(f"  MELBO / uniform:     {melbo_fracs.mean() / rand_means.mean():.2f}x")
    print(f"  MELBO / cov-matched: {melbo_fracs.mean() / cov_means.mean():.2f}x")

    # --- Save results ---
    save_data = {
        'melbo_fractions': melbo_fracs,
        'random_fractions_all': random_fracs_all,
        'covmatch_fractions_all': covmatch_fracs_all,
        'n_dirs': n_dirs,
        'd_model': d_model,
        'n_acts': activations.shape[0],
        'activation_norms': act_norms,
        'theoretical_baseline': theoretical,
        'args': vars(args),
        'melbo_args': melbo_args,
    }
    save_path = os.path.join(args.output_dir, 'projection_fractions.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {save_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'melbo': 'C0', 'cov': 'C1', 'uniform': 'C2'}

    # Histogram
    ax = axes[0]
    ax.hist(melbo_fracs.numpy(), bins=50, alpha=0.6, density=True,
            color=colors['melbo'], label=f'MELBO')
    ax.hist(covmatch_fracs_all.flatten().numpy(), bins=50, alpha=0.6, density=True,
            color=colors['cov'], label=f'Cov-matched random')
    ax.hist(random_fracs_all.flatten().numpy(), bins=50, alpha=0.6, density=True,
            color=colors['uniform'], label=f'Uniform random')
    ax.axvline(theoretical, color='k', ls='--', alpha=0.5, label='Isotropic theory')
    ax.set_xlabel('||proj|| / ||act||')
    ax.set_ylabel('Density')
    ax.set_title(f'Fraction of L2 norm retained ({n_dirs} dirs)')
    ax.legend(fontsize=9)

    # CDF
    ax = axes[1]
    for data, label, color in [
        (melbo_fracs, 'MELBO', colors['melbo']),
        (covmatch_fracs_all.flatten(), 'Cov-matched random', colors['cov']),
        (random_fracs_all.flatten(), 'Uniform random', colors['uniform']),
    ]:
        sorted_vals = torch.sort(data).values.numpy()
        ax.plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)),
                label=label, color=color)
    ax.axvline(theoretical, color='k', ls='--', alpha=0.5, label='Isotropic theory')
    ax.set_xlabel('||proj|| / ||act||')
    ax.set_ylabel('CDF')
    ax.set_title(f'CDF of norm fraction ({n_dirs} dirs)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'projection_fractions.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()
