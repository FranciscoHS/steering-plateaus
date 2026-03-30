#!/usr/bin/env python3
"""Analyze concentration of probability boost/suppression for each MELBO direction.

For each direction, steers at a given angle and computes:
- Effective N (exp of entropy) of the positive-delta distribution (boosted tokens)
- Effective N of the negative-delta distribution (suppressed tokens)
- Top boosted and suppressed tokens
- KL divergence from unsteered baseline

Can also generate bar chart plots sorted by effective N.

Usage:
    python scripts/melbo/boost_concentration.py \
        --results_pkl results/iterative_melbo_hf/.../results.pkl \
        --prompt "The old man sat by the" \
        --angle 15 \
        --plot
"""
import sys, pickle, argparse
import numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks


def steer_and_get_probs(model, tokenizer, device, prompt, direction, steer_layer, angle_deg):
    """Get steered probabilities for a single direction."""
    cos_t = np.cos(np.radians(angle_deg))
    sin_t = np.sin(np.radians(angle_deg))
    d = direction.to(device)
    d = d / d.norm()

    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        pos = hidden[0, -1:, :]
        R = pos.norm(dim=-1, keepdim=True)
        a_unit = pos / (R + 1e-10)
        proj = (a_unit * d).sum(dim=-1, keepdim=True)
        v_perp = d.unsqueeze(0) - proj * a_unit
        v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
        steered = R * (cos_t * a_unit + sin_t * v_perp)
        hidden[0, -1:, :] = steered
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layers = _get_blocks(model)
    handle = layers[steer_layer].register_forward_hook(steering_hook)
    with torch.no_grad():
        outputs = model(tokens, use_cache=False)
    handle.remove()

    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1).cpu()
    log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
    return probs, log_probs


def compute_concentration(deltas, tokenizer, positive=True):
    """Compute effective N and top tokens for positive or negative deltas."""
    if positive:
        mask = deltas > 0
        vals = deltas[mask]
    else:
        mask = deltas < 0
        vals = (-deltas[mask])

    total = vals.sum().item()
    dist = vals / vals.sum()
    entropy = -(dist * dist.log()).sum().item()
    eff_n = np.exp(entropy)

    # Top tokens by delta magnitude
    if positive:
        sorted_vals, sorted_idx = deltas.sort(descending=True)
    else:
        sorted_vals, sorted_idx = deltas.sort(descending=False)

    top_tokens = []
    for j in range(min(10, len(sorted_idx))):
        tok = tokenizer.decode([sorted_idx[j].item()]).strip()
        top_tokens.append((tok, sorted_vals[j].item()))

    return eff_n, total, top_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--angle', type=float, default=15.0)
    parser.add_argument('--outfile', type=str, default=None,
                        help='Text output file (default: auto-named)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate bar chart plots')
    parser.add_argument('--plot_file', type=str, default=None,
                        help='Plot output file (default: auto-named)')
    args = parser.parse_args()

    model, tokenizer, device = load_model('qwen3-1.7b')

    with open(args.results_pkl, 'rb') as f:
        data = pickle.load(f)

    directions = data['directions']
    steer_layer = data['args']['inject_layer']
    n_dirs = len(directions)

    # Unsteered baseline
    tokens = tokenizer(args.prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        base_logits = model(tokens, use_cache=False).logits[0, -1, :]
    base_probs = F.softmax(base_logits.float(), dim=-1).cpu()
    base_log_probs = F.log_softmax(base_logits.float(), dim=-1).cpu()

    # Analyze each direction
    results = []
    lines = []
    for di in range(n_dirs):
        probs, log_probs = steer_and_get_probs(
            model, tokenizer, device, args.prompt,
            directions[di], steer_layer, args.angle
        )
        deltas = probs - base_probs
        kl = (base_probs * (base_log_probs - log_probs)).sum().item()

        boost_eff_n, boost_total, boost_tops = compute_concentration(deltas, tokenizer, positive=True)
        suppress_eff_n, suppress_total, suppress_tops = compute_concentration(deltas, tokenizer, positive=False)

        results.append({
            'dir_idx': di, 'kl': kl,
            'boost_eff_n': boost_eff_n, 'boost_total': boost_total, 'boost_tops': boost_tops,
            'suppress_eff_n': suppress_eff_n, 'suppress_total': suppress_total, 'suppress_tops': suppress_tops,
        })

        n_boost_show = min(max(round(boost_eff_n), 1), 5)
        boost_label = ', '.join([t[0] for t in boost_tops[:n_boost_show]])
        suppress_label = ', '.join([t[0] for t in suppress_tops[:3]])
        line = (f'Dir {di:2d}: KL={kl:.3f}  '
                f'boost_effN={boost_eff_n:.1f} [{boost_label}]  '
                f'suppress_effN={suppress_eff_n:.1f} [{suppress_label}]')
        print(line)
        lines.append(line)

    # Save text output
    if args.outfile is None:
        import os
        results_dir = os.path.dirname(args.results_pkl)
        args.outfile = os.path.join(results_dir, f'boost_concentration_{args.angle:.0f}deg.txt')
    with open(args.outfile, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved to {args.outfile}")

    # Generate plots
    if args.plot:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        results_dir = os.path.dirname(args.results_pkl)
        kl_arr = np.array([r['kl'] for r in results])
        norm = plt.Normalize(vmin=0, vmax=max(kl_arr))
        cmap = plt.cm.viridis

        for key, title in [('boost', 'Boosted tokens'), ('suppress', 'Suppressed tokens')]:
            fig, ax = plt.subplots(figsize=(14, 6))

            eff_ns = [r[f'{key}_eff_n'] for r in results]
            tops = [r[f'{key}_tops'] for r in results]

            # Sort by eff_N
            order = np.argsort(eff_ns)
            eff_ns_sorted = [eff_ns[i] for i in order]
            tops_sorted = [tops[i] for i in order]
            kls_sorted = [kl_arr[i] for i in order]

            colors = cmap(norm(np.array(kls_sorted)))
            ax.bar(range(len(eff_ns_sorted)), eff_ns_sorted, color=colors,
                   edgecolor='black', linewidth=0.5)

            labels = []
            for eff_n, top in zip(eff_ns_sorted, tops_sorted):
                n_show = min(max(round(eff_n), 1), 8)
                labels.append(', '.join([t[0] for t in top[:n_show]]))

            ax.set_xticks(range(len(eff_ns_sorted)))
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylabel('Effective N', fontsize=11)
            ax.set_title(f'{title} ({args.angle:.0f}° steering)', fontsize=12)
            ax.set_ylim(0, max(eff_ns_sorted) * 1.15)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('KL divergence from baseline', fontsize=10)

            plt.tight_layout()
            outpath = os.path.join(results_dir, f'{key}_concentration_{args.angle:.0f}deg.png')
            plt.savefig(outpath, dpi=150)
            plt.close()
            print(f"Plot saved to {outpath}")


if __name__ == '__main__':
    main()
