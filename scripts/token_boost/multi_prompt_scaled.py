#!/usr/bin/env python3
"""
Scaled-up version: optimize token-boosting directions for many FineWeb
activations, then plot cosine similarity (to reference direction) vs
L2 distance (to reference activation).

Usage:
    python scripts/token_boost/multi_prompt_scaled.py \
        --model gpt2 --layer 5 --opt_angle 10 --n_act 100
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


def extract_all_activations(model, token_ids_list, layer, device, batch_size=64):
    """Extract layer-L context and last-token activation for all sequences."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def build_perturbed_hidden(context, activation, direction, angle_rad):
    """Spherical perturbation: rotate activation toward direction by angle_rad.

    Args:
        context: [1, seq_len-1, d_model]
        activation: [d_model]
        direction: [d_model] unit vector (may require grad)
        angle_rad: float
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norm = activation.norm()
    a_unit = activation / a_norm
    d_normed = direction / (direction.norm() + 1e-10)
    proj = (d_normed * a_unit).sum()
    u_perp = d_normed - proj * a_unit
    u_perp = u_perp / (u_perp.norm() + 1e-10)
    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([context, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)


def optimize_token_direction(model, context, activation, target_token_id,
                             layer, angle_rad, device, d_model,
                             n_steps=300, lr=0.01, n_restarts=3):
    """Find unit direction that maximizes logit of target_token_id.

    Returns: (direction, logit_boost, ref_logit, final_logit)
    """
    with torch.no_grad():
        hidden_ref = torch.cat([context, activation.unsqueeze(0).unsqueeze(0)], dim=1)
        ref_logits = forward_from_layer_allpos(model, hidden_ref, layer)
        ref_target_logit = ref_logits[0, target_token_id].item()

    best_final_logit = -float('inf')
    best_direction = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device)
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            hidden = build_perturbed_hidden(context, activation, direction, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            loss = -logits[0, target_token_id]
            loss.backward()
            optimizer.step()

        final_logit = logits[0, target_token_id].item()
        if final_logit > best_final_logit:
            best_final_logit = final_logit
            with torch.no_grad():
                best_direction = F.normalize(direction, dim=0).clone()

    logit_boost = best_final_logit - ref_target_logit
    return best_direction, logit_boost, ref_target_logit, best_final_logit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_token', type=str, default=' book')
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--opt_angle', type=float, default=10.0)
    parser.add_argument('--n_act', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        safe_token = args.target_token.strip().replace(' ', '_')
        args.results_dir = (f'results/token_boost/scaled_{args.model}_L{args.layer}'
                            f'_{safe_token}_angle{int(args.opt_angle)}_n{args.n_act}')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")

    if args.layer < 0:
        args.layer = n_layers + args.layer

    # Verify target token
    target_token_id = tokenizer.encode(args.target_token, add_special_tokens=False)
    if len(target_token_id) != 1:
        print(f"WARNING: '{args.target_token}' tokenizes to {len(target_token_id)} tokens. Using first.")
    target_token_id = target_token_id[0]
    print(f"Target token: '{args.target_token}' -> id {target_token_id}")

    # Load FineWeb data
    print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)

    # Extract activations
    print("Extracting activations...")
    contexts, activations = extract_all_activations(
        model, token_ids_list, args.layer, device)
    print(f"Activations shape: {activations.shape}, mean norm: {activations.norm(dim=-1).mean():.2f}")

    # Optimize directions
    angle_rad = np.deg2rad(args.opt_angle)
    directions = []
    logit_boosts = []
    texts = [tokenizer.decode(tids) for tids in token_ids_list]

    print(f"\nOptimizing {args.n_act} directions (angle={args.opt_angle} deg, "
          f"{args.n_restarts} restarts x {args.n_steps} steps)...")
    t0 = time.time()

    for i in range(args.n_act):
        ctx = contexts[i:i+1].to(device)
        act = activations[i].to(device)

        direction, logit_boost, ref_logit, final_logit = optimize_token_direction(
            model, ctx, act, target_token_id,
            args.layer, angle_rad, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts)

        directions.append(direction.cpu())
        logit_boosts.append(logit_boost)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_act - i - 1) / rate
            print(f"  [{i+1:3d}/{args.n_act}] boost={logit_boost:.1f} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining) "
                  f"'{texts[i][:40]}'")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/args.n_act:.1f}s per prompt)")

    # Compute cosine sims to d_0 and L2 distances to act_0
    directions_t = torch.stack(directions)  # [n_act, d_model]
    ref_dir = directions_t[0]
    ref_act = activations[0]

    cos_sims = F.cosine_similarity(directions_t, ref_dir.unsqueeze(0), dim=-1).numpy()
    l2_dists = (activations - ref_act.unsqueeze(0)).norm(dim=-1).numpy()

    # Correlation
    # Exclude point 0 (trivially cos=1, l2=0)
    mask = np.arange(len(cos_sims)) > 0
    corr = np.corrcoef(l2_dists[mask], cos_sims[mask])[0, 1]
    print(f"\nPearson correlation (L2 dist vs cos sim): {corr:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(l2_dists[1:], cos_sims[1:], alpha=0.5, s=30, c='#2171b5', edgecolors='none')
    ax.scatter(l2_dists[0], cos_sims[0], c='gold', s=100, zorder=5,
              edgecolors='black', linewidths=1.5, label='Reference (act 0)')

    # Linear fit (excluding point 0)
    z = np.polyfit(l2_dists[mask], cos_sims[mask], 1)
    x_fit = np.linspace(l2_dists[mask].min(), l2_dists[mask].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), 'r--', alpha=0.7,
            label=f'Linear fit (r={corr:.3f})')

    ax.set_xlabel('L2 distance to reference activation', fontsize=12)
    ax.set_ylabel('Cosine similarity to reference direction', fontsize=12)
    ax.set_title(f'Direction similarity vs activation distance (n={args.n_act})\n'
                 f'GPT-2 L{args.layer}, target="{args.target_token.strip()}", '
                 f'angle={args.opt_angle}°', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'cosim_vs_l2dist.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # Save results
    results = {
        'args': vars(args),
        'texts': texts,
        'directions': directions_t,
        'activations': activations,
        'cos_sims': cos_sims,
        'l2_dists': l2_dists,
        'logit_boosts': np.array(logit_boosts),
        'correlation': corr,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {results_path}")


if __name__ == '__main__':
    main()
