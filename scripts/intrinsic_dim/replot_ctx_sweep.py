#!/usr/bin/env python3
"""Replot the per-context sweep curves from the saved ctx{i}.pt files.

ctx_randseq_measure.py wrote sweep.png on log-log axes. This regenerates the
same curves on linear axes as sweep_linear.png, leaving sweep.png alone.
No measurement is repeated; everything comes from the saved arrays.
"""

import argparse
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['jacopt', 'gauss', 'randtok', 'randseq', 'real']
COLORS = {'jacopt': 'C0', 'gauss': 'C2', 'randtok': 'C1', 'randseq': 'C4',
          'real': 'k'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='results/intrinsic_dim/ctx_jacopt_ppl')
    p.add_argument('--name', default='sweep_linear.png')
    p.add_argument('--max_angle', type=float, default=None,
                   help='crop the x-axis to this angle (default: full range)')
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.outdir)
                   if f.startswith('ctx') and f.endswith('.pt'))
    blobs = [torch.load(os.path.join(args.outdir, f), map_location='cpu',
                        weights_only=False) for f in files]
    n = len(blobs)
    ang = blobs[0]['angles_deg']
    keep = slice(None) if args.max_angle is None else ang <= args.max_angle

    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.4), squeeze=False)
    for i, b in enumerate(blobs):
        for r, key in enumerate(['sweep_L2', 'sweep_KL']):
            ax = axes[r, i]
            for name in ORDER:
                if name not in b['groups']:
                    continue
                v = b['groups'][name][key]
                med = np.median(v.reshape(-1, v.shape[-1]), axis=0)
                ax.plot(ang[keep], med[keep], color=COLORS[name], label=name,
                        lw=1.6)
            ax.set_xlabel('angle (deg)')
            ax.set_ylabel(key)
            ax.set_title(f'ctx {b["ctx_id"]} — {key} (median)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0 and r == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(args.outdir, args.name)
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
