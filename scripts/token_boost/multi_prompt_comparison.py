#!/usr/bin/env python3
"""
Optimize token-boosting directions across many prompts, then compare:
  - Cosine similarity of each direction to a reference direction (prompt 0)
  - L2 distance of each activation to the reference activation
  - Is there a relationship?

Usage:
    python scripts/token_boost/multi_prompt_comparison.py \
        --model gpt2 --layer 5 --opt_angle 10
"""

import argparse
import os
import pickle
import sys
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

# Prompts where "book" is a plausible continuation
BOOK_PROMPTS = [
    "My favorite object is a",            # reference (prompt 0)
    "I went to the library to borrow a",
    "She picked up the",
    "On the shelf there was a",
    "He was reading a",
    "The student opened her",
    "For her birthday she got a",
    "The professor recommended a",
]

# Prompts where "book" is NOT a plausible continuation
NON_BOOK_PROMPTS = [
    "The dog chased the",
    "I ate a delicious",
    "The temperature outside is",
    "She drove her car to the",
    "The chemical formula for water is",
    "The president of the United States is",
    "In the ocean you can find",
    "The score of the game was",
]


def extract_activation_at_layer(model, tokenizer, prompt, layer, device):
    """Extract residual stream at `layer` for a single prompt."""
    token_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(token_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]
    context = hidden[:, :-1, :]
    activation = hidden[0, -1, :]
    return context, activation, token_ids


def build_perturbed_hidden(context, activation, direction, angle_rad):
    """Spherical perturbation: rotate activation toward direction by angle_rad."""
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
                             layer, angle_deg, device, d_model,
                             n_steps=500, lr=0.01, n_restarts=5):
    """Find unit direction that maximizes logit of target_token_id."""
    angle_rad = np.deg2rad(angle_deg)

    with torch.no_grad():
        hidden_ref = torch.cat([context, activation.unsqueeze(0).unsqueeze(0)], dim=1)
        ref_logits = forward_from_layer_allpos(model, hidden_ref, layer)
        ref_target_logit = ref_logits[0, target_token_id].item()
        ref_target_prob = F.softmax(ref_logits, dim=-1)[0, target_token_id].item()

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
            target_logit = logits[0, target_token_id]
            loss = -target_logit
            loss.backward()
            optimizer.step()

        final_logit = logits[0, target_token_id].item()
        if final_logit > best_final_logit:
            best_final_logit = final_logit
            with torch.no_grad():
                best_direction = F.normalize(direction, dim=0).clone()

    logit_boost = best_final_logit - ref_target_logit
    final_prob = F.softmax(logits, dim=-1)[0, target_token_id].item()
    return best_direction, logit_boost, ref_target_logit, ref_target_prob, best_final_logit, final_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_token', type=str, default=' book')
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--opt_angle', type=float, default=10.0)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        safe_token = args.target_token.strip().replace(' ', '_')
        args.results_dir = f'results/token_boost/multi_prompt_{args.model}_L{args.layer}_{safe_token}_angle{int(args.opt_angle)}'
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

    all_prompts = BOOK_PROMPTS + NON_BOOK_PROMPTS
    is_book = [True] * len(BOOK_PROMPTS) + [False] * len(NON_BOOK_PROMPTS)
    n_prompts = len(all_prompts)

    # Extract activations
    print(f"\nExtracting activations for {n_prompts} prompts...")
    contexts = []
    activations = []
    for i, prompt in enumerate(all_prompts):
        context, activation, _ = extract_activation_at_layer(
            model, tokenizer, prompt, args.layer, device)
        contexts.append(context)
        activations.append(activation)
        cat = "book-plausible" if is_book[i] else "book-implausible"
        print(f"  [{i:2d}] ({cat}) '{prompt}' — norm={activation.norm().item():.2f}")

    # Optimize directions
    print(f"\nOptimizing directions (angle={args.opt_angle} deg)...")
    directions = []
    results_list = []
    for i, prompt in enumerate(all_prompts):
        cat = "book" if is_book[i] else "non-book"
        print(f"\n[{i:2d}/{n_prompts}] ({cat}) '{prompt}'")

        direction, logit_boost, ref_logit, ref_prob, final_logit, final_prob = \
            optimize_token_direction(
                model, contexts[i], activations[i], target_token_id,
                args.layer, args.opt_angle, device, d_model,
                n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts)

        directions.append(direction)
        results_list.append({
            'prompt': prompt,
            'is_book': is_book[i],
            'logit_boost': logit_boost,
            'ref_logit': ref_logit,
            'ref_prob': ref_prob,
            'final_logit': final_logit,
            'final_prob': final_prob,
        })
        print(f"  logit_boost={logit_boost:.2f}, prob: {ref_prob:.6f} -> {final_prob:.6f}")

    # Compute cosine similarities to d_0 and L2 distances to activation_0
    ref_dir = directions[0]
    ref_act = activations[0]

    cos_sims_to_ref = []
    l2_dists_to_ref = []
    for i in range(n_prompts):
        cs = F.cosine_similarity(directions[i].unsqueeze(0), ref_dir.unsqueeze(0)).item()
        l2 = (activations[i] - ref_act).norm().item()
        cos_sims_to_ref.append(cs)
        l2_dists_to_ref.append(l2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Reference: prompt 0 = '{all_prompts[0]}'")
    print(f"{'='*70}")
    for i in range(n_prompts):
        cat = "book" if is_book[i] else "NON-BOOK"
        r = results_list[i]
        print(f"  [{i:2d}] cos(d_i, d_0)={cos_sims_to_ref[i]:+.4f}  "
              f"L2(act_i, act_0)={l2_dists_to_ref[i]:.2f}  "
              f"boost={r['logit_boost']:.1f}  "
              f"prob={r['ref_prob']:.4f}->{r['final_prob']:.4f}  "
              f"({cat}) '{all_prompts[i]}'")

    # Plot: cosine sim to d_0 vs L2 distance to act_0
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_prompts):
        if i == 0:
            # Reference point
            ax.scatter(l2_dists_to_ref[i], cos_sims_to_ref[i],
                      c='gold', s=120, zorder=5, edgecolors='black', linewidths=1.5,
                      label='Reference (prompt 0)')
        elif is_book[i]:
            ax.scatter(l2_dists_to_ref[i], cos_sims_to_ref[i],
                      c='#2171b5', s=80, zorder=4, edgecolors='black', linewidths=0.5,
                      label='Book-plausible' if i == 1 else None)
        else:
            ax.scatter(l2_dists_to_ref[i], cos_sims_to_ref[i],
                      c='#cb181d', s=80, zorder=4, edgecolors='black', linewidths=0.5,
                      marker='s',
                      label='Book-implausible' if i == len(BOOK_PROMPTS) else None)
        # Annotate with prompt index
        ax.annotate(str(i), (l2_dists_to_ref[i], cos_sims_to_ref[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('L2 distance to reference activation', fontsize=12)
    ax.set_ylabel('Cosine similarity to reference direction', fontsize=12)
    ax.set_title(f'Direction similarity vs activation distance\n'
                 f'GPT-2 L{args.layer}, target=" book", angle={args.opt_angle}°', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'cosim_vs_l2dist.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot_path}")

    # Save full results
    results = {
        'args': vars(args),
        'prompts': all_prompts,
        'is_book': is_book,
        'directions': [d.cpu() for d in directions],
        'activations': [a.cpu() for a in activations],
        'cos_sims_to_ref': cos_sims_to_ref,
        'l2_dists_to_ref': l2_dists_to_ref,
        'per_prompt': results_list,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {results_path}")

    # Print prompt legend
    print(f"\nPrompt legend:")
    for i, p in enumerate(all_prompts):
        cat = "book" if is_book[i] else "NON-BOOK"
        print(f"  {i}: ({cat}) '{p}'")


if __name__ == '__main__':
    main()
