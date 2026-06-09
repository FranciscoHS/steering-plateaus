#!/usr/bin/env python3
"""Are the optimized atoms aligned with the neuron basis of the toy model?

The atoms are input-space directions (R^d). The toy model's TwoLayerNet reads
the input through fc1.weight [H, d] (encoder): row h is the input direction that
hidden neuron h responds to. So the "neuron basis" in input space is the set of
encoder rows. We also check the decoder columns fc2.weight[:, h] [d] (the output
direction each neuron writes).

For each atom we report its cosine with every neuron, the best single-neuron
|cos| (how axis-aligned it is to one neuron), and compare against random unit
directions. A participation ratio over cos^2 summarizes how many neurons each
atom spreads across.

Usage:
    python scripts/superellipse/atom_neuron_alignment.py \\
        --ckpt /workspace/denoising/runs/phase1_tanh3/model.pt \\
        --atoms results/dictionary/toymodel_phase1_tanh3/optimize_features_randcontrast/atoms.pt
"""
import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def unit_rows(M, dim=1):
    return M / M.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--atoms', default='results/dictionary/toymodel_phase1_tanh3/'
                    'optimize_features_randcontrast/atoms.pt')
    ap.add_argument('--n_rand', type=int, default=2000, help='random baseline vecs')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/atom_neuron_alignment')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(os.path.expanduser(args.ckpt), map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    E = sd['fc1.weight'].float()          # [H, d] encoder rows (input neurons)
    Dec = sd['fc2.weight'].float().t()    # [H, d] decoder columns transposed -> rows
    H, d = E.shape
    A = torch.load(os.path.expanduser(args.atoms), map_location='cpu',
                   weights_only=False)['dictionary'].float()  # [K, d]
    K = A.shape[0]

    A_n = unit_rows(A)
    E_n = unit_rows(E)
    Dec_n = unit_rows(Dec)
    cos_enc = (A_n @ E_n.t()).numpy()     # [K, H]
    cos_dec = (A_n @ Dec_n.t()).numpy()   # [K, H]

    # random baseline: max |cos| of random unit vectors against encoder rows
    g = torch.Generator().manual_seed(args.seed)
    R = torch.randn(args.n_rand, d, generator=g)
    R_n = unit_rows(R)
    rand_cos_enc = (R_n @ E_n.t()).abs().numpy()
    rand_max_enc = rand_cos_enc.max(axis=1)

    def participation_ratio(c):  # over cos^2 across neurons
        w = c ** 2
        return float((w.sum() ** 2) / (w ** 2).sum())

    print(f'd={d} H={H} K={K} atoms', flush=True)
    print(f'random baseline max|cos|(enc): median={np.median(rand_max_enc):.3f} '
          f'95pct={np.quantile(rand_max_enc,0.95):.3f} '
          f'max={rand_max_enc.max():.3f}', flush=True)
    rows = []
    for k in range(K):
        ae = np.abs(cos_enc[k]); ad = np.abs(cos_dec[k])
        h_e = int(ae.argmax()); h_d = int(ad.argmax())
        pr_e = participation_ratio(cos_enc[k])
        rows.append({'atom': k, 'max_cos_enc': float(ae[h_e]), 'argmax_neuron_enc': h_e,
                     'max_cos_dec': float(ad[h_d]), 'argmax_neuron_dec': h_d,
                     'PR_enc': pr_e})
        print(f'  atom {k:2d}: max|cos|enc={ae[h_e]:.3f} (neuron {h_e:3d})  '
              f'max|cos|dec={ad[h_d]:.3f} (neuron {h_d:3d})  PR_enc={pr_e:.1f}', flush=True)
    atom_max_enc = np.array([r['max_cos_enc'] for r in rows])
    print(f'\natom max|cos|(enc): median={np.median(atom_max_enc):.3f} '
          f'min={atom_max_enc.min():.3f} max={atom_max_enc.max():.3f}', flush=True)
    print(f'fraction of atoms above random 95pct '
          f'({np.quantile(rand_max_enc,0.95):.3f}): '
          f'{np.mean(atom_max_enc > np.quantile(rand_max_enc,0.95)):.2f}', flush=True)

    # ---- plots -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(rand_max_enc, bins=40, density=True, color='#bbbbbb', alpha=0.8,
            label=f'random dirs (n={args.n_rand})')
    ax.axvline(np.quantile(rand_max_enc, 0.95), color='#888888', ls='--',
               label='random 95th pct')
    for x in atom_max_enc:
        ax.axvline(x, color='#3a7d3a', alpha=0.5, lw=1.2)
    ax.plot([], [], color='#3a7d3a', label=f'atoms (n={K})')
    ax.set_xlabel('best single-neuron alignment  max$_h$ |cos(atom, encoder row $h$)|')
    ax.set_ylabel('density')
    ax.set_title('Atom vs neuron (encoder) alignment')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    xs = np.arange(K)
    ax.bar(xs - 0.2, atom_max_enc, width=0.4, color='#3a7d3a', label='encoder')
    ax.bar(xs + 0.2, [r['max_cos_dec'] for r in rows], width=0.4,
           color='#D9822B', label='decoder')
    ax.axhline(np.quantile(rand_max_enc, 0.95), color='#888888', ls='--',
               label='random 95th pct (enc)')
    ax.set_xlabel('atom index'); ax.set_ylabel('best single-neuron |cos|')
    ax.set_title('Per-atom best-neuron alignment')
    ax.set_xticks(xs); ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    png = os.path.join(args.out, 'atom_neuron_alignment.png')
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    with open(os.path.join(args.out, 'alignment.pkl'), 'wb') as f:
        pickle.dump({'cos_enc': cos_enc, 'cos_dec': cos_dec, 'rows': rows,
                     'rand_max_enc': rand_max_enc, 'H': H, 'd': d, 'K': K}, f)
    print(f'\nsaved {png}', flush=True)
    print(f'saved {os.path.join(args.out, "alignment.pkl")}', flush=True)


if __name__ == '__main__':
    main()
