#!/usr/bin/env python3
"""Summary figure for the v2 popt ladder: p rises with lambda while
reconstruction quality (variance explained), functional effect (KL of patched
vs clean logits), and native sparse-probing accuracy stay flat.

Reads existing artifacts only (no model):
  - results/sae_v2_eval/{name}_deadfrac.json   (val_mse, val_cos)
  - results/sae_v2_eval/{name}_fullp.pkl       (p_median, quartiles)
  - results/ce_eval_v2/{name}.json             (KL mean/std/median, n_eval)
  - results/probe_native_v2/results.pkl        (baseline_v2, lam10_v2)
  - results/probe_native_v2_lam100/results.pkl (lam100_v2)

Variance explained = 1 - val_mse / elementwise_var(val rows); the variance is
computed once from the cache val split (rows 1.8M:1.81M) and passed via
--val_var.

Usage:
    python scripts/dictionary_learning/plot_popt_flatness.py \
        --val_var 4.366134 --out results/sae_v2_eval/popt_flatness.png
"""
import argparse
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = ['baseline', 'lam0p1', 'lam1', 'lam10', 'lam100']
XLABELS = ['baseline', r'$\lambda$=0.1', r'$\lambda$=1', r'$\lambda$=10', r'$\lambda$=100']
# fixed series colors (colorblind-safe, assigned by entity)
C_MAIN = '#4053d3'    # ladder metrics
C_P = '#7a3a7d'       # p (matches the rotation-sweep plots)
PROBE_COLORS = {'baseline_v2': '#4053d3', 'lam10_v2': '#dd9f40', 'lam100_v2': '#b51d14'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_var', type=float, required=True,
                    help='elementwise variance of the val rows (see docstring)')
    ap.add_argument('--raw_ceiling', type=float, default=0.8459,
                    help='raw-activation probe ceiling; corrected logreg value '
                         'from results/probe_native/raw_ceiling.pkl (the suite '
                         'pkls store a miscalibrated unstandardized-ridge value)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ve, val_cos, p_med, p_lo, p_hi = [], [], [], [], []
    kl_mean, kl_sem, kl_med = [], [], []
    for name in RUNS:
        d = json.load(open(f'results/sae_v2_eval/{name}_deadfrac.json'))
        ve.append(1 - d['val_mse'] / args.val_var)
        val_cos.append(d['val_cos'])
        s = pickle.load(open(f'results/sae_v2_eval/{name}_fullp.pkl', 'rb'))['summary']
        p_med.append(s['p_median']); p_lo.append(s['p_p25']); p_hi.append(s['p_p75'])
        c = json.load(open(f'results/ce_eval_v2/{name}.json'))
        kl_mean.append(c['kl_div_mean'])
        kl_sem.append(c['kl_div_std'] / np.sqrt(c['n_eval']))
        kl_med.append(c['kl_div_median'])

    x = np.arange(len(RUNS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.errorbar(x, p_med, yerr=[np.array(p_med) - p_lo, np.array(p_hi) - p_med],
                fmt='-o', color=C_P, capsize=3, lw=2, label='median (bars: IQR)')
    ax.axhline(2.0, color='gray', ls='--', alpha=0.6, label='p=2 (Euclidean)')
    ax.set_ylabel('fitted superellipse p')
    ax.set_title('superellipse p (what the term optimizes)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, ve, '-o', color=C_MAIN, lw=2)
    ax.set_ylabel('variance explained (val)')
    ax.set_ylim(0, 1)
    ax.set_title('reconstruction: variance explained')
    for xi, v in zip(x, ve):
        ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='#555')

    ax = axes[1, 0]
    ax.errorbar(x, kl_mean, yerr=kl_sem, fmt='-o', color=C_MAIN, lw=2,
                capsize=3, label='mean ± SEM')
    ax.plot(x, kl_med, '-s', color=C_MAIN, lw=1.2, alpha=0.5, ms=4, label='median')
    ax.set_ylabel('KL(clean ‖ patched), last token')
    ax.set_ylim(bottom=0)
    ax.set_title('functional effect of recon patched at L10')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    probe_srcs = {'baseline_v2': 'results/probe_native_v2/results.pkl',
                  'lam100_v2': 'results/probe_native_v2_lam100/results.pkl'}
    for lab, path in probe_srcs.items():
        r = pickle.load(open(path, 'rb'))
        ks = r['k_values']
        mean_acc = [np.mean([e[f'sae:{lab}'][k] for e in r['concepts'].values()])
                    for k in ks]
        ax.plot(ks, mean_acc, '-o', color=PROBE_COLORS[lab], lw=2, ms=4, label=lab)
    ax.axhline(args.raw_ceiling, color='gray', ls=':', alpha=0.8)
    ax.annotate('raw-activation probe', (ks[-1], args.raw_ceiling),
                ha='right', va='bottom', fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('k (top-k latents probed)')
    ax.set_ylabel('mean probe accuracy (13 concepts)')
    ax.set_title('native sparse-code probing')
    ax.legend(fontsize=8)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(x)
        ax.set_xticklabels(XLABELS)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('p-optimized SAEs (Qwen3-1.7B L10, v2 runs): p rises, everything else stays flat',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
