#!/usr/bin/env python3
"""
Optimize for directions that maximally boost a target token's logit,
then compare directions found from different prompts.

For each prompt:
  1. Extract activation at layer L (last token position)
  2. Find unit direction d that maximizes logit(target_token) when
     the activation is spherically perturbed toward d by a fixed angle
  3. Report optimized logit boost

Then:
  - Compute cosine similarity between directions from different prompts
  - Cross-test: apply each direction to the other prompt's activation
  - Sweep angles to show how target token prob changes

Usage:
    python scripts/token_boost/optimize_token_direction.py \
        --prompts "My favorite object is a" "The object I most hate is a" \
        --target_token " book" --model gpt2 --layer 5
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


def extract_activation_at_layer(model, tokenizer, prompt, layer, device):
    """Extract residual stream at `layer` for a single prompt.

    Returns:
        context: [1, seq_len-1, d_model] — prefix positions
        activation: [1, d_model] — last-token activation (no seq dim)
        token_ids: the tokenized prompt
    """
    token_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(token_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]  # [1, seq_len, d_model]
    context = hidden[:, :-1, :]   # [1, seq_len-1, d_model]
    activation = hidden[0, -1, :] # [d_model]
    return context, activation, token_ids


def build_perturbed_hidden(context, activation, direction, angle_rad):
    """Spherical perturbation: rotate activation toward direction by angle_rad.

    Args:
        context: [1, seq_len-1, d_model]
        activation: [d_model]
        direction: [d_model] unit vector (may require grad)
        angle_rad: float

    Returns:
        hidden: [1, seq_len, d_model] ready for forward_from_layer_allpos
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norm = activation.norm()
    a_unit = activation / a_norm
    # Component of direction perpendicular to activation
    d_normed = direction / (direction.norm() + 1e-10)
    proj = (d_normed * a_unit).sum()
    u_perp = d_normed - proj * a_unit
    u_perp = u_perp / (u_perp.norm() + 1e-10)
    a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([context, a_pert.unsqueeze(0).unsqueeze(0)], dim=1)


def optimize_token_direction(model, context, activation, target_token_id,
                             layer, angle_deg, device, d_model,
                             n_steps=500, lr=0.01, n_restarts=5):
    """Find unit direction that maximizes logit of target_token_id.

    Uses spherical perturbation at fixed angle.

    Returns:
        best_direction: [d_model] unit vector
        best_logit_boost: scalar (logit increase vs unperturbed)
        ref_target_logit, ref_target_prob: baseline values
        best_history: list of (target_logit, target_prob) per step
    """
    angle_rad = np.deg2rad(angle_deg)

    # Baseline (unperturbed)
    with torch.no_grad():
        hidden_ref = torch.cat([context, activation.unsqueeze(0).unsqueeze(0)], dim=1)
        ref_logits = forward_from_layer_allpos(model, hidden_ref, layer)
        ref_target_logit = ref_logits[0, target_token_id].item()
        ref_target_prob = F.softmax(ref_logits, dim=-1)[0, target_token_id].item()

    print(f"    Baseline: logit={ref_target_logit:.4f}, prob={ref_target_prob:.6f}")

    best_logit_boost = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device)
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            hidden = build_perturbed_hidden(context, activation, direction, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)

            target_logit = logits[0, target_token_id]
            loss = -target_logit
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prob = F.softmax(logits, dim=-1)[0, target_token_id].item()
            history.append((target_logit.item(), prob))

            if step % 100 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"logit={target_logit.item():.4f}, prob={prob:.6f}")

        final_logit = history[-1][0]
        logit_boost = final_logit - ref_target_logit
        if logit_boost > best_logit_boost:
            best_logit_boost = logit_boost
            with torch.no_grad():
                best_direction = F.normalize(direction, dim=0).clone()
            best_history = history

    return best_direction, best_logit_boost, ref_target_logit, ref_target_prob, best_history


def evaluate_direction(model, context, activation, direction, target_token_id,
                       layer, angle_deg, tokenizer):
    """Evaluate a direction on a given activation. Returns dict with logit, prob, top5."""
    angle_rad = np.deg2rad(angle_deg)
    with torch.no_grad():
        # Baseline
        hidden_ref = torch.cat([context, activation.unsqueeze(0).unsqueeze(0)], dim=1)
        ref_logits = forward_from_layer_allpos(model, hidden_ref, layer)
        ref_logit = ref_logits[0, target_token_id].item()

        # Perturbed
        hidden = build_perturbed_hidden(context, activation, direction, angle_rad)
        logits = forward_from_layer_allpos(model, hidden, layer)
        pert_logit = logits[0, target_token_id].item()
        pert_prob = F.softmax(logits, dim=-1)[0, target_token_id].item()

        # Top tokens
        probs = F.softmax(logits, dim=-1)[0]
        top_probs, top_ids = probs.topk(5)

    return {
        'ref_logit': ref_logit,
        'pert_logit': pert_logit,
        'pert_prob': pert_prob,
        'logit_boost': pert_logit - ref_logit,
        'top5_ids': top_ids.cpu(),
        'top5_probs': top_probs.cpu(),
    }


def sweep_angles(model, context, activation, direction, target_token_id,
                 layer, angles_deg):
    """Sweep perturbation angles, return target probs and logits."""
    probs = []
    logits_list = []
    with torch.no_grad():
        for angle in angles_deg:
            angle_rad = np.deg2rad(angle)
            hidden = build_perturbed_hidden(context, activation, direction, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            p = F.softmax(logits, dim=-1)[0, target_token_id].item()
            l = logits[0, target_token_id].item()
            probs.append(p)
            logits_list.append(l)
    return np.array(probs), np.array(logits_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', nargs='+',
                        default=["My favorite object is a",
                                 "The object I most hate is a"])
    parser.add_argument('--target_token', type=str, default=' book',
                        help='Target token (include leading space if needed)')
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--layer', type=int, default=5,
                        help='Layer to inject perturbation')
    parser.add_argument('--opt_angle', type=float, default=15.0,
                        help='Perturbation angle in degrees for optimization')
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
        args.results_dir = f'results/token_boost/{args.model}_L{args.layer}_{safe_token}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Load model =====
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}")

    if args.layer < 0:
        args.layer = n_layers + args.layer

    # ===== Verify target token =====
    target_token_id = tokenizer.encode(args.target_token)
    if len(target_token_id) != 1:
        target_token_id = tokenizer.encode(args.target_token, add_special_tokens=False)
    if len(target_token_id) != 1:
        print(f"WARNING: '{args.target_token}' tokenizes to {len(target_token_id)} tokens: "
              f"{target_token_id}. Using the first one.")
    target_token_id = target_token_id[0]
    print(f"Target token: '{args.target_token}' -> id {target_token_id} "
          f"(decodes to '{tokenizer.decode([target_token_id])}')")

    # ===== Extract activations for each prompt =====
    prompt_data = []
    for i, prompt in enumerate(args.prompts):
        print(f"\nPrompt {i}: '{prompt}'")
        context, activation, token_ids = extract_activation_at_layer(
            model, tokenizer, prompt, args.layer, device)
        a_norm = activation.norm().item()
        print(f"  Tokens: {token_ids.shape[1]}, activation norm: {a_norm:.2f}")
        prompt_data.append({
            'prompt': prompt,
            'context': context,
            'activation': activation,
            'a_norm': a_norm,
        })

    # ===== Optimize direction for each prompt =====
    print(f"\nOptimization angle: {args.opt_angle} deg")
    directions = []
    for i, pd in enumerate(prompt_data):
        print(f"\n{'='*60}")
        print(f"Optimizing direction for prompt {i}: '{pd['prompt']}'")
        print(f"{'='*60}")

        direction, logit_boost, ref_logit, ref_prob, history = optimize_token_direction(
            model, pd['context'], pd['activation'], target_token_id,
            args.layer, args.opt_angle, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
        )
        directions.append(direction)
        pd['direction'] = direction
        pd['logit_boost'] = logit_boost
        pd['ref_logit'] = ref_logit
        pd['ref_prob'] = ref_prob
        pd['history'] = history

        print(f"  => Logit boost: {logit_boost:.4f} "
              f"(from {ref_logit:.4f} to {ref_logit + logit_boost:.4f})")
        print(f"  => Prob: {ref_prob:.6f} -> {history[-1][1]:.6f}")

    # ===== Compare directions =====
    print(f"\n{'='*60}")
    print("Direction comparison")
    print(f"{'='*60}")

    n_prompts = len(args.prompts)
    cos_sims = torch.zeros(n_prompts, n_prompts)
    for i in range(n_prompts):
        for j in range(n_prompts):
            cos_sims[i, j] = F.cosine_similarity(
                directions[i].unsqueeze(0), directions[j].unsqueeze(0)).item()

    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            print(f"  cos_sim(d_{i}, d_{j}) = {cos_sims[i,j]:.6f}")

    # ===== Cross-evaluation =====
    print(f"\n{'='*60}")
    print("Cross-evaluation (apply direction from prompt X to prompt Y)")
    print(f"{'='*60}")

    cross_results = {}
    for i in range(n_prompts):
        for j in range(n_prompts):
            result = evaluate_direction(
                model, prompt_data[j]['context'], prompt_data[j]['activation'],
                directions[i], target_token_id, args.layer, args.opt_angle,
                tokenizer)

            label = f"d_{i} -> prompt_{j}"
            cross_results[(i, j)] = result
            top5_str = ', '.join(
                f"'{tokenizer.decode([tid])}' ({p:.4f})"
                for tid, p in zip(result['top5_ids'], result['top5_probs']))
            print(f"  {label}: logit_boost={result['logit_boost']:.4f}, "
                  f"prob={result['pert_prob']:.6f}")
            print(f"    top5: {top5_str}")

    # ===== Unembedding comparison =====
    print(f"\n{'='*60}")
    print("Comparison with unembedding vector")
    print(f"{'='*60}")

    with torch.no_grad():
        lm_head = _get_lm_head(model)
        W_U = lm_head.weight[target_token_id]  # [d_model]
        W_U_unit = F.normalize(W_U, dim=0)

    for i in range(n_prompts):
        cos_with_unembed = F.cosine_similarity(
            directions[i].unsqueeze(0), W_U_unit.unsqueeze(0)).item()
        print(f"  cos_sim(d_{i}, W_U['{args.target_token}']) = {cos_with_unembed:.6f}")

    # Also cross-eval the unembedding direction itself
    print(f"\n  Unembedding direction as steering vector:")
    for j in range(n_prompts):
        result = evaluate_direction(
            model, prompt_data[j]['context'], prompt_data[j]['activation'],
            W_U_unit, target_token_id, args.layer, args.opt_angle, tokenizer)
        top5_str = ', '.join(
            f"'{tokenizer.decode([tid])}' ({p:.4f})"
            for tid, p in zip(result['top5_ids'], result['top5_probs']))
        print(f"  W_U -> prompt_{j}: logit_boost={result['logit_boost']:.4f}, "
              f"prob={result['pert_prob']:.6f}")
        print(f"    top5: {top5_str}")

    # ===== Angle sweep =====
    print(f"\n{'='*60}")
    print("Angle sweep")
    print(f"{'='*60}")

    sweep_angles_deg = np.linspace(0, 60, 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n_prompts):
        for j in range(n_prompts):
            probs, logits = sweep_angles(
                model, prompt_data[j]['context'], prompt_data[j]['activation'],
                directions[i], target_token_id, args.layer, sweep_angles_deg)

            label = f"d_{i} -> prompt_{j}"
            linestyle = '-' if i == j else '--'
            axes[0].plot(sweep_angles_deg, probs, linestyle=linestyle, label=label)
            axes[1].plot(sweep_angles_deg, logits, linestyle=linestyle, label=label)

    # Also sweep the unembedding direction
    for j in range(n_prompts):
        probs, logits = sweep_angles(
            model, prompt_data[j]['context'], prompt_data[j]['activation'],
            W_U_unit, target_token_id, args.layer, sweep_angles_deg)
        label = f"W_U -> prompt_{j}"
        axes[0].plot(sweep_angles_deg, probs, linestyle=':', label=label)
        axes[1].plot(sweep_angles_deg, logits, linestyle=':', label=label)

    axes[0].axvline(args.opt_angle, color='gray', alpha=0.5, linestyle='--', label=f'opt angle ({args.opt_angle} deg)')
    axes[1].axvline(args.opt_angle, color='gray', alpha=0.5, linestyle='--', label=f'opt angle ({args.opt_angle} deg)')

    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel(f'P("{args.target_token.strip()}")')
    axes[0].set_title('Target token probability vs perturbation angle')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel(f'Logit("{args.target_token.strip()}")')
    axes[1].set_title('Target token logit vs perturbation angle')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'angle_sweep.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # ===== Save results =====
    results = {
        'args': vars(args),
        'prompts': args.prompts,
        'target_token': args.target_token,
        'target_token_id': target_token_id,
        'directions': [d.cpu() for d in directions],
        'cos_sims': cos_sims.numpy(),
        'cross_results': {k: {kk: vv if not isinstance(vv, torch.Tensor)
                               else vv.numpy() for kk, vv in v.items()}
                          for k, v in cross_results.items()},
        'prompt_data': [{
            'prompt': pd['prompt'],
            'logit_boost': pd['logit_boost'],
            'ref_logit': pd['ref_logit'],
            'ref_prob': pd['ref_prob'],
            'a_norm': pd['a_norm'],
        } for pd in prompt_data],
        'W_U_unit': W_U_unit.cpu(),
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved: {results_path}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}, Layer: {args.layer}, Target: '{args.target_token}' (id {target_token_id})")
    print(f"Optimization angle: {args.opt_angle} deg")
    for i, pd in enumerate(prompt_data):
        print(f"\nPrompt {i}: '{pd['prompt']}'")
        print(f"  Logit boost (own direction): {pd['logit_boost']:.4f}")
        print(f"  Prob: {pd['ref_prob']:.6f} -> {pd['history'][-1][1]:.6f}")
    print(f"\nDirection cosine similarities:")
    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            print(f"  cos(d_{i}, d_{j}) = {cos_sims[i,j]:.6f}")
    print(f"\nCross-evaluation (logit boosts):")
    for i in range(n_prompts):
        for j in range(n_prompts):
            r = cross_results[(i, j)]
            match = "same" if i == j else "cross"
            print(f"  d_{i} -> prompt_{j} ({match}): boost={r['logit_boost']:.4f}, "
                  f"prob={r['pert_prob']:.6f}")
    print(f"\nResults saved to: {args.results_dir}")


if __name__ == '__main__':
    main()
