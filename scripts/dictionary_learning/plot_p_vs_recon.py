#!/usr/bin/env python3
"""Plot superellipse-p and reconstruction (MSE, CE-increase) vs the p-term
weight lambda, for the popt SAEs against the baseline.

p_median comes from eval_sae_full_p.py; MSE/CE from eval_ce_loss.py JSONs.

Usage:
    python scripts/dictionary_learning/plot_p_vs_recon.py \
        --ce_dir results/ce_eval --out results/ce_eval/p_vs_recon.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, lambda, p_median)   p from full-p eval (320 pairs, median)
POINTS = [
    ('baseline', 0.0, 2.13),
    ('lam0p1', 0.1, 2.63),
    ('lam1', 1.0, 2.83),
    ('lam10', 10.0, 2.80),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ce_dir', default='results/ce_eval')
    ap.add_argument('--out', default='results/ce_eval/p_vs_recon.png')
    args = ap.parse_args()

    rows = []
    for label, lam, p in POINTS:
        ce = json.load(open(os.path.join(args.ce_dir, f'{label}.json')))
        rows.append({'label': label, 'lam': lam, 'p': p,
                     'mse': ce['mse_mean'], 'ce': ce['ce_increase_mean'],
                     'cos': ce['cos_sim_mean']})

    base = rows[0]
    popt = rows[1:]                       # lam = 0.1, 1, 10
    lams = [r['lam'] for r in popt]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [('p', 'superellipse p (median)', 'C0'),
             ('mse', 'reconstruction MSE', 'C1'),
             ('ce', 'CE increase (nats)', 'C3')]
    for ax, (key, title, color) in zip(axes, specs):
        ax.plot(lams, [r[key] for r in popt], 'o-', color=color, lw=2, ms=8)
        ax.axhline(base[key], color='0.4', ls='--', lw=1.5,
                   label=f'baseline (λ=0): {base[key]:.3f}')
        ax.set_xscale('log')
        ax.set_xticks(lams); ax.set_xticklabels(['0.1', '1', '10'])
        ax.set_xlabel('λ  (p-term weight)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    fig.suptitle('SAE p-optimization: p vs reconstruction across λ  '
                 '(Qwen3-1.7B L10, d_sae=32768, k=32)', fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved: {args.out}')
    print('\nlabel      λ     p     mse    ce_inc  cos')
    for r in rows:
        print(f"{r['label']:9s} {r['lam']:5g} {r['p']:.2f}  {r['mse']:.3f}  "
              f"{r['ce']:.4f}  {r['cos']:.4f}")


if __name__ == '__main__':
    main()
