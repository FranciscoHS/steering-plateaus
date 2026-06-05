#!/usr/bin/env python3
"""Compare two greedy decompositions via cosine similarity of feature directions.

Loads two results.pkl files from decompose_plateau_breaking.py runs and emits:
  - cosine-similarity matrix between feature directions of A and B
  - signed-projection table (alpha_k for each side)
  - heatmap PNG + a short text summary

Usage:
    python scripts/superellipse/compare_decomps.py \\
        --a results/decomp_plateau/gemma_L15_paris_capital_of \\
        --b results/decomp_plateau/gemma_L15_lyon_city_in \\
        --out results/decomp_plateau/compare_paris_lyon
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(d):
    with open(os.path.join(d, 'results.pkl'), 'rb') as f:
        r = pickle.load(f)
    feats = r['features']
    if hasattr(feats, 'cpu'):
        feats = feats.cpu().numpy()
    feats = np.asarray(feats, dtype=np.float64)
    feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
    return {
        'features': feats,
        'alphas': np.asarray(r['alphas'], dtype=np.float64),
        'residual_norm_frac': np.asarray(r['residual_norm_frac'],
                                          dtype=np.float64),
        'target_prompt': r['args'].get('target_prompt'),
        'K_used': r['K_used'],
        'a_target': (r['a_target'].cpu().numpy()
                     if hasattr(r['a_target'], 'cpu') else
                     np.asarray(r['a_target'])),
        'a_target_norm': r['a_target_norm'],
        'scaffold_names': r['scaffold_names'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='dir with results.pkl (A)')
    ap.add_argument('--b', required=True, help='dir with results.pkl (B)')
    ap.add_argument('--out', required=True, help='output dir for plot+text')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    A = load_run(args.a)
    B = load_run(args.b)

    Ka, d_a = A['features'].shape
    Kb, d_b = B['features'].shape
    assert d_a == d_b, f'dim mismatch: {d_a} vs {d_b}'

    # Cosine sim matrix [Ka, Kb]
    cos = A['features'] @ B['features'].T

    # Also project A's target activation onto B's features (and vice versa)
    a_tgt_A = A['a_target'] / (np.linalg.norm(A['a_target']) + 1e-12)
    a_tgt_B = B['a_target'] / (np.linalg.norm(B['a_target']) + 1e-12)
    a_target_cos = float(a_tgt_A @ a_tgt_B)
    A_feat_dot_aB = A['features'] @ a_tgt_B  # how much A's features align with B's target
    B_feat_dot_aA = B['features'] @ a_tgt_A  # vice versa

    # Save raw numbers
    out_pkl = os.path.join(args.out, 'compare.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'cos': cos,
            'a_target_cos': a_target_cos,
            'A_feat_dot_aB_hat': A_feat_dot_aB,
            'B_feat_dot_aA_hat': B_feat_dot_aA,
            'A_alphas': A['alphas'],
            'B_alphas': B['alphas'],
            'A_residual_frac': A['residual_norm_frac'],
            'B_residual_frac': B['residual_norm_frac'],
            'A_target_prompt': A['target_prompt'],
            'B_target_prompt': B['target_prompt'],
        }, f)

    # Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(0.7 + 0.6 * Kb, 0.7 + 0.6 * Ka))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    for i in range(Ka):
        for j in range(Kb):
            v = cos[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    fontsize=8, color=('white' if abs(v) > 0.6 else 'black'))
    ax.set_xticks(range(Kb))
    ax.set_xticklabels([f'B:d{j+1}' for j in range(Kb)])
    ax.set_yticks(range(Ka))
    ax.set_yticklabels([f'A:d{i+1}' for i in range(Ka)])
    ax.set_xlabel(f"B = {B['target_prompt']!r}  (K={Kb})")
    ax.set_ylabel(f"A = {A['target_prompt']!r}  (K={Ka})")
    ax.set_title(f"cos(d_i^A, d_j^B)   |   cos(a_A_hat, a_B_hat)={a_target_cos:+.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'cosine_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Text summary
    txt = os.path.join(args.out, 'summary.txt')
    with open(txt, 'w') as f:
        f.write(f'A: {args.a}  target_prompt={A["target_prompt"]!r}  K={Ka}\n')
        f.write(f'B: {args.b}  target_prompt={B["target_prompt"]!r}  K={Kb}\n')
        f.write(f'cos(a_A_hat, a_B_hat) = {a_target_cos:+.4f}\n\n')
        f.write('A.alphas:           ' +
                ' '.join(f'{a:+.2f}' for a in A['alphas']) + '\n')
        f.write('B.alphas:           ' +
                ' '.join(f'{a:+.2f}' for a in B['alphas']) + '\n')
        f.write('A.residual_frac:    ' +
                ' '.join(f'{x:.3f}' for x in A['residual_norm_frac']) + '\n')
        f.write('B.residual_frac:    ' +
                ' '.join(f'{x:.3f}' for x in B['residual_norm_frac']) + '\n\n')
        f.write('Cosine similarity matrix cos(d_i^A, d_j^B):\n')
        header = '         ' + ' '.join(f'B:d{j+1:<2}' for j in range(Kb))
        f.write(header + '\n')
        for i in range(Ka):
            f.write(f'A:d{i+1:<2}     ' +
                    ' '.join(f'{cos[i,j]:+.3f}' for j in range(Kb)) + '\n')
        f.write('\n')
        # Best matches each way
        f.write('For each A:d_i, best matching B:d_j (by |cos|):\n')
        for i in range(Ka):
            j = int(np.argmax(np.abs(cos[i])))
            f.write(f'  A:d{i+1}  ->  B:d{j+1}   cos={cos[i,j]:+.3f}\n')
        f.write('\nFor each B:d_j, best matching A:d_i (by |cos|):\n')
        for j in range(Kb):
            i = int(np.argmax(np.abs(cos[:, j])))
            f.write(f'  B:d{j+1}  ->  A:d{i+1}   cos={cos[i,j]:+.3f}\n')
        f.write('\nA features projected onto B target (unit): cos(d_i^A, a_B_hat)\n')
        for i in range(Ka):
            f.write(f'  A:d{i+1}  {A_feat_dot_aB[i]:+.3f}\n')
        f.write('\nB features projected onto A target (unit): cos(d_j^B, a_A_hat)\n')
        for j in range(Kb):
            f.write(f'  B:d{j+1}  {B_feat_dot_aA[j]:+.3f}\n')

    print(f'wrote {out_pkl}')
    print(f'wrote {plot_path}')
    print(f'wrote {txt}')
    # Echo summary
    with open(txt) as f:
        print(f.read())


if __name__ == '__main__':
    main()
