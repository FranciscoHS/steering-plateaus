#!/usr/bin/env python3
"""Shape-collapse plot for the concept-matched holdout sweep (run_holdout_sweep.py).

Three groups, all on matched holdout activations:
  - contrastive : one curve per concept (mean over that concept's activations of the
                  matched-direction response)
  - high-p      : one curve per soft_p atom (mean over ALL holdout activations)
  - random      : one curve per random dir (mean over ALL holdout activations)

Usage:
    python scripts/perturbation/plot_holdout_collapse.py \
        --pkl results/holdout_sweep/qwen3-1.7b_L14/holdout_sweep.pkl --metric l2
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.perturbation.plot_curve_shape_collapse import affine_normalize, group_mean_on_grid

C_HIGHP = '#16a085'; C_CONTRAST = '#c0392b'; C_RANDOM = '#2c3e50'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--metric', default='l2', choices=['l2', 'kl'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    theta = np.array(d['angles_deg'], dtype=float)
    concepts = np.array(d['concepts'])
    M = d[args.metric]

    # contrastive: per-concept mean over its own activations
    cm = np.array(M['contrast_matched'], dtype=float)   # [N, A]
    contrast = [cm[concepts == c].mean(0) for c in sorted(set(concepts)) if (concepts == c).any()]
    # high-p: per atom, mean over all acts
    hp = np.array(M['highp'], dtype=float)              # [N, n_atoms, A]
    highp = [hp[:, j, :].mean(0) for j in range(hp.shape[1])]
    # random: per dir, mean over all acts
    rd = np.array(M['random'], dtype=float)            # [N, n_rand, A]
    random = [rd[:, j, :].mean(0) for j in range(rd.shape[1])]

    groups = [('high-p', C_HIGHP, highp), ('contrastive', C_CONTRAST, contrast),
              ('random', C_RANDOM, random)]
    groups = [g for g in groups if len(g[2]) > 0]  # drop empty families (e.g. no high-p)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()
    xgrid = np.linspace(0, 1, 50)
    mlabel = {'l2': 'L2-distance', 'kl': 'KL'}[args.metric]

    for label, color, curves in groups:
        for r in curves:
            axA.plot(theta, r, color=color, alpha=0.25, lw=0.9)
        axA.plot([], [], color=color, lw=2, label=f'{label} (n={len(curves)})')
        norm = []
        for r in curves:
            res = affine_normalize(theta, r)
            if not res:
                continue
            x, y, tlo, _ = res
            axB.plot(x, y, color=color, alpha=0.16, lw=0.8)
            axC.plot(theta / tlo, r / r.max(), color=color, alpha=0.2, lw=0.8)
            norm.append((x, y))
        m, s = group_mean_on_grid(norm, xgrid)
        axB.plot(xgrid, m, color=color, lw=3, label=f'{label} mean')
        axB.fill_between(xgrid, m - s, m + s, color=color, alpha=0.12)

    axA.set_xlabel('perturbation angle (deg)'); axA.set_ylabel(f'{mlabel} response')
    axA.set_title('(A) raw response curves (matched holdout activations)'); axA.legend(fontsize=8)
    axB.scatter([0, 1], [0, 1], color='k', zorder=5, s=20)
    axB.set_xlabel('normalized angle  (theta_lo->0, theta_hi->1)')
    axB.set_ylabel('normalized response')
    axB.set_title('(B) 2-point affine collapse'); axB.legend(fontsize=8)
    axC.set_xlabel('angle / theta_break'); axC.set_ylabel('response / own max')
    axC.set_title('(C) onset-only normalization'); axC.set_xlim(0, 8)

    rng = np.random.RandomState(0)
    for i, (label, color, curves) in enumerate(groups):
        ys = []
        for r in curves:
            res = affine_normalize(theta, r)
            if res:
                x, y, *_ = res
                ys.append(float(np.interp(0.5, x, y)))
        axD.scatter(i + rng.uniform(-.07, .07, len(ys)), ys, color=color, alpha=0.7,
                    label=f'{label} (med={np.median(ys):.2f})')
    axD.axhline(0.5, color='gray', ls=':', lw=1)
    axD.set_xticks(range(len(groups))); axD.set_xticklabels([g[0] for g in groups])
    axD.set_ylabel('normalized response at midpoint'); axD.set_title('(D) shape scalar at x_norm=0.5')
    axD.legend(fontsize=8)

    pl, ml = d['args']['layer'], d['args'].get('measure_layer')
    fig.suptitle(f'Holdout shape collapse — qwen3-1.7b, perturb layer {pl}, '
                 f'measure {"final" if ml is None else f"L{ml}"} (metric: {mlabel})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(os.path.dirname(args.pkl), f'holdout_collapse_{args.metric}.png')
    fig.savefig(out, dpi=140)
    print(f'saved: {out}')

    # quick numeric separation readout
    def gm(curves):
        return np.array(curves).mean(0)
    cmean, rmean = gm(contrast), gm(random)
    hmean = gm(highp) if len(highp) else None
    for nm, t in [('~7.5deg', 7.5), ('~15deg', 15), ('~30deg', 30), ('~60deg', 60)]:
        i = int(np.argmin(np.abs(theta - t)))
        extra = f"  highp/random={hmean[i]/rmean[i]:.2f}" if hmean is not None else ""
        print(f"  {nm}: contrast/random={cmean[i]/rmean[i]:.2f}{extra}")


if __name__ == '__main__':
    main()
