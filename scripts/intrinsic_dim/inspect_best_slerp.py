#!/usr/bin/env python3
"""Show the most linear slerp path found, and say what its endpoints were.

Reads the results of slerp_jacopt_to_reals.py, finds the single path with the
lowest max_dev (distance from the linear ramp) for each condition, plots it,
and decodes both endpoints: the carrier context the path runs in, and the real
activation it runs to.

No model forward passes; everything comes from the saved curves plus the token
ids, decoded with the tokenizer.

Output in --outdir: best_paths.png, best_paths.txt
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from src.data import load_fineweb_fixed_length

STAT_NAMES = ['max_dev', 'rms_dev', 'max_step', 'width_25_75']


def describe(tok, ids):
    """Full text of a 5-token sequence, plus its final token on its own."""
    ids = [int(i) for i in ids]
    return tok.decode(ids), tok.decode(ids[-1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/slerp_jacopt_to_reals_gpt2xl')
    p.add_argument('--jac_pt', default='results/intrinsic_dim/'
                                       'ctx_sweep_oldconfig_gpt2xl/jac_optimized.pt')
    p.add_argument('--tokenizer', default='gpt2-xl')
    args = p.parse_args()

    with open(os.path.join(args.outdir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)
    a = res['args']
    ts = res['t']

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    blob = torch.load(args.jac_pt, map_location='cpu', weights_only=False)
    anchor_tokens = blob['tokens']
    cand_tokens = load_fineweb_fixed_length(a['n_real'], tok,
                                            seq_len=a['seq_len'],
                                            seed=a['real_seed'],
                                            cache_dir='/workspace/fineweb_cache')

    lines = []
    picks = {}
    for cname in ['jacopt', 'real']:
        s = res['stats'][cname][:, :, 0]              # max_dev
        i, j = np.unravel_index(int(np.argmin(s)), s.shape)
        picks[cname] = (i, j)
        stats = res['stats'][cname][i, j]
        ctx_text, ctx_last = describe(tok, anchor_tokens[i])
        cand_text, cand_last = describe(tok, cand_tokens[j])
        flat = np.sort(s.reshape(-1))
        lines += [
            f'=== {cname} -> real, most linear of '
            f'{s.shape[0]}x{s.shape[1]} paths ===',
            f'  anchor {i}, candidate {j}',
            '  ' + '  '.join(f'{n}={stats[k]:.4f}'
                             for k, n in enumerate(STAT_NAMES)),
            f'  next best max_dev in this condition: '
            f'{flat[1]:.4f}, {flat[2]:.4f}, {flat[3]:.4f}',
            '',
            f'  carrier context (anchor {i}, the sequence the jac-opt point '
            f'was optimized in):',
            f'    full 5 tokens : {ctx_text!r}',
            f'    final token   : {ctx_last!r}',
            f'    the path START is '
            + ('the jac-opt point at this final position'
               if cname == 'jacopt' else
               'the REAL activation of this final position'),
            '',
            f'  path END, the real activation it runs to (candidate {j}, from '
            f'its own sequence):',
            f'    full 5 tokens : {cand_text!r}',
            f'    final token   : {cand_last!r}',
            '',
        ]

    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.outdir, 'best_paths.txt'), 'w') as f:
        f.write(txt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (cname, color) in zip(axes, [('jacopt', 'C0'), ('real', 'k')]):
        i, j = picks[cname]
        ax.plot(ts, ts, color='r', ls='--', lw=1.2, label='linear ramp')
        ax.plot(ts, res['rel'][cname][i, j], color=color, lw=2.0, marker='o',
                ms=3, label=f'{cname} -> real, anchor {i}, cand {j}')
        md = res['stats'][cname][i, j, 0]
        ctx_text, _ = describe(tok, anchor_tokens[i])
        cand_text, _ = describe(tok, cand_tokens[j])
        ax.set_title(f'{cname}: max_dev={md:.3f}\nfrom {ctx_text!r}\n'
                     f'to {cand_text!r}', fontsize=8)
        ax.set_xlabel('interpolation t (slerp)')
        ax.set_ylabel('relative logit distance')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(args.outdir, 'best_paths.png')
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')
    print(f'Saved {os.path.join(args.outdir, "best_paths.txt")}')


if __name__ == '__main__':
    main()
