#!/usr/bin/env python3
"""Add the Gaussian-start group to the per-context jac-opt measurements.

ctx_jacopt_perplexity.py saved the cov-matched Gaussian points each optimization
started from (`X_start`) but never measured them. This script loads each
ctx{i}.pt, rebuilds the carrier context from the saved token ids, and runs the
same three measurements on the starts:  ||J||_F, next-token perplexity exp(H),
and the 0..max_angle plateau sweep (final-hidden L2 + output KL).

The result is written back into the same ctx{i}.pt under groups['gauss'], and
summary.txt / ppl.png / sweep.png are regenerated with the group included.
"""

import argparse
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from ctx_jacopt_perplexity import jac_norm_fd, next_token_stats, plateau_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--n_dirs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='results/intrinsic_dim/ctx_jacopt_ppl')
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.outdir)
                   if f.startswith('ctx') and f.endswith('.pt'))
    print(f'{len(files)} context files in {args.outdir}', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.float32)

    blobs = []
    for f in files:
        path = os.path.join(args.outdir, f)
        b = torch.load(path, map_location='cpu', weights_only=False)
        ci = b['ctx_id']
        if 'gauss' in b['groups']:
            print(f'ctx {ci}: gauss already present, skipping', flush=True)
            blobs.append(b)
            continue
        print(f'\n=== ctx {ci}: {b["text"]!r} ===', flush=True)
        ctx_ids = b['ctx_ids'].to(device)
        with torch.no_grad():
            o = model(ctx_ids, output_hidden_states=True)
        ctx = o.hidden_states[args.layer + 1].float()[:, :-1, :]

        X = b['X_start'].to(device)
        angles_rad = np.deg2rad(b['angles_deg'])
        with torch.no_grad():
            jn = jac_norm_fd(model, ctx, X, args.layer)
            st = next_token_stats(model, ctx, X, args.layer)
            L2, KL = plateau_sweep(model, ctx, X, args.layer, angles_rad,
                                   args.n_dirs, args.seed + ci)
        b['groups']['gauss'] = {
            'jac': jn.numpy(), 'ppl': st['ppl'].numpy(),
            'p_max': st['p_max'].numpy(), 'top_ids': st['top_ids'].numpy(),
            'top_p': st['top_p'].numpy(), 'sweep_L2': L2, 'sweep_KL': KL,
            'norm': X.norm(dim=-1).cpu().numpy()}
        print(f'  gauss    n={X.shape[0]:3d}  '
              f'||x||={np.median(b["groups"]["gauss"]["norm"]):.2f}  '
              f'||J||={np.median(jn.numpy()):8.1f}  '
              f'ppl={np.median(st["ppl"].numpy()):8.2f}  '
              f'p_max={np.median(st["p_max"].numpy()):.3f}  '
              f'KLmax={np.median(KL[:, :, -1]):.3f}   (medians)', flush=True)
        torch.save(b, path)
        blobs.append(b)

    # Regenerate plots and summary with the gauss group included.
    order = ['jacopt', 'gauss', 'randtok', 'real']
    colors = {'jacopt': 'C0', 'gauss': 'C2', 'randtok': 'C1', 'real': 'k'}
    n = len(blobs)

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    for i, b in enumerate(blobs):
        ax = axes[0, i]
        for name in ['jacopt', 'gauss', 'randtok']:
            if name in b['groups']:
                ax.hist(b['groups'][name]['ppl'], bins=25, histtype='step',
                        color=colors[name], label=name, lw=1.5)
        ax.axvline(b['groups']['real']['ppl'][0], color='k', ls='--',
                   label='real token')
        ax.set_xscale('log')
        ax.set_xlabel('perplexity exp(H)')
        ax.set_title(f'ctx {b["ctx_id"]}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'ppl.png'), dpi=150)

    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.4), squeeze=False)
    for i, b in enumerate(blobs):
        for r, key in enumerate(['sweep_L2', 'sweep_KL']):
            ax = axes[r, i]
            for name in order:
                if name not in b['groups']:
                    continue
                v = b['groups'][name][key]
                med = np.median(v.reshape(-1, v.shape[-1]), axis=0)
                ax.plot(b['angles_deg'], med, color=colors[name], label=name)
            ax.set_xlabel('angle (deg)')
            ax.set_ylabel(key)
            ax.set_title(f'ctx {b["ctx_id"]} — {key} (median)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0 and r == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'sweep.png'), dpi=150)

    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'{args.model} L{args.layer}  n_ctx={n}\nmedians per group\n')
        for b in blobs:
            f.write(f'\nctx {b["ctx_id"]}: {b["text"]!r}\n')
            f.write(f'{"group":8s} {"n":>4s} {"||x||":>8s} {"||J||":>9s} '
                    f'{"ppl":>9s} {"p_max":>7s} {"KLmax":>8s} {"L2max":>9s}\n')
            for name in order:
                if name not in b['groups']:
                    continue
                gr = b['groups'][name]
                f.write(f'{name:8s} {len(gr["ppl"]):4d} '
                        f'{np.median(gr["norm"]):8.2f} '
                        f'{np.median(gr["jac"]):9.1f} '
                        f'{np.median(gr["ppl"]):9.2f} '
                        f'{np.median(gr["p_max"]):7.3f} '
                        f'{np.median(gr["sweep_KL"][:, :, -1]):8.3f} '
                        f'{np.median(gr["sweep_L2"][:, :, -1]):9.2f}\n')
    print(f'\nSaved {args.outdir}/summary.txt, ppl.png, sweep.png')


if __name__ == '__main__':
    main()
