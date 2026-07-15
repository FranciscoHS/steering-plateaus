#!/usr/bin/env python3
"""Clean, de-cluttered plot from run_probe_dict_vs_sae.py's results.pkl.

Keeps only four lines per panel: raw activations (ceiling), random OMP-k, the
best-of-our-dictionaries, and the best-of-the-SAE-variants. "Best" is the per-k
envelope (max over the family) computed separately for ID and OOD, since which
member wins differs by k/split. The argmax member at each point is printed so
the envelope doesn't hide it.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def envelope(results, keys, ks, split):
    """Per-k max over `keys` for the given split; returns (vals, winners)."""
    vals, winners = [], []
    for k in ks:
        best_v, best_key = -1.0, None
        for key in keys:
            if k in results.get(key, {}):
                v = results[key][k][split]
                if v > best_v:
                    best_v, best_key = v, key
        vals.append(best_v); winners.append(best_key)
    return vals, winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--k_max', type=int, default=40, help='drop k beyond this')
    args = ap.parse_args()

    results = pickle.load(open(args.results, 'rb'))
    dict_keys = [f'dict:{l}' for l in results['dict_labels']]
    sae_keys = ['sae_native', 'sae_omp']
    ks = [k for k in results['k_values'] if k <= args.k_max]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split, title in [(axes[0], 'id_acc', 'In-distribution (SST-2)'),
                             (axes[1], 'ood_acc', 'Out-of-distribution (Amazon)')]:
        best_dict, win_dict = envelope(results, dict_keys, ks, split)
        best_sae, win_sae = envelope(results, sae_keys, ks, split)
        rand = [results['random'][k][split] for k in ks]

        ax.plot(ks, best_dict, 's-', color='tab:red', lw=2, label='best found dict')
        ax.plot(ks, best_sae, 'o-', color='tab:blue', lw=2, label='best Qwen SAE')
        ax.plot(ks, rand, '^-', color='tab:gray', label='random')
        ax.axhline(results['raw'][split], ls='--', color='black', alpha=0.6,
                   label='raw activations')

        ax.set_xlabel('k (atoms / latents)'); ax.set_ylabel('accuracy')
        ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        print(f'\n{title}')
        for i, k in enumerate(ks):
            print(f'  k={k:>3}: dict={best_dict[i]:.3f} ({win_dict[i].replace("dict:","")})'
                  f'  sae={best_sae[i]:.3f} ({win_sae[i].replace("sae_","")})'
                  f'  rand={rand[i]:.3f}  raw={results["raw"][split]:.3f}')

    plt.tight_layout()
    out = args.out or os.path.join(os.path.dirname(args.results), 'probe_dict_vs_sae_clean.png')
    plt.savefig(out, dpi=150)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
