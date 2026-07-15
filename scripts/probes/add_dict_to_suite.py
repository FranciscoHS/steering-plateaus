#!/usr/bin/env python3
"""Add one more dictionary's OMP-k line to an existing probe-suite results.pkl.

Loads prior complete results, computes ONLY the new dict's OMP-k probe per
concept (mean-as-free-bias) from the cached activations, merges under
'dict:<label>', re-plots. Reuses cached acts — no model, fast.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.probes.run_probe_dict_vs_sae import omp_gpu, load_dict, run_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_results', required=True)
    ap.add_argument('--acts', required=True)
    ap.add_argument('--dict', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--omp_batch', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_k', type=int, default=None,
                    help='only compute the new dict up to this k (default: all)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = pickle.load(open(args.base_results, 'rb'))
    k_values = results['k_values']
    acts = torch.load(args.acts, weights_only=False)

    # report the dict's args so we know exactly what we added
    blob = torch.load(args.dict, map_location='cpu', weights_only=False)
    a = blob.get('args', {})
    print(f'dict {args.label}: n_done={blob.get("n_done")} lam_align={a.get("lam_align")} '
          f'lam_prev={a.get("lam_prev")} align_topk={a.get("align_topk")}', flush=True)
    D = load_dict(args.dict)
    key = f'dict:{args.label}'

    # which k's to compute for THIS dict (others keep their full grid)
    new_ks = [k for k in k_values if args.max_k is None or k <= args.max_k]

    for name, res in results['concepts'].items():
        d = acts[name]
        tr_a, tr_y, te_a, te_y = d['train'], d['train_y'], d['test'], d['test_y']
        abar = tr_a.mean(dim=0, keepdim=True)
        res[key] = {}
        for k in new_ks:
            trr = abar + omp_gpu(D, tr_a - abar, k, device, args.omp_batch)
            ter = abar + omp_gpu(D, te_a - abar, k, device, args.omp_batch)
            res[key][k] = run_probe(trr.numpy(), tr_y, ter.numpy(), te_y, args.seed)
        print(f'[{name}] {key}@{new_ks[-1]}={res[key][new_ks[-1]]:.3f}', flush=True)

    if args.label not in results['dict_labels']:
        results['dict_labels'].append(args.label)
    with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- re-plot --------------------------------------------------------
    names = list(results['concepts'].keys()); nC = len(names)

    def agg(method, k=None):
        vals = np.array([results['concepts'][n][method] if k is None
                         else results['concepts'][n][method][k] for n in names], float)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(nC)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    raw_m, _ = agg('raw')
    ax.axhline(raw_m, ls='--', color='black', alpha=0.6, label=f'raw ({raw_m:.3f})')
    specs = [('sae_native', 'Qwen SAE (native top-k)', 'o-', 'tab:blue'),
             ('sae_omp', 'Qwen SAE decoder (OMP-k)', 'D-', 'tab:cyan')]
    dcolors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown']
    for i, l in enumerate(results['dict_labels']):
        specs.append((f'dict:{l}', f'dict {l}', 's-', dcolors[i % len(dcolors)]))
    specs.append(('random', 'random', '^-', 'tab:gray'))
    for k_, lbl, style, color in specs:
        if k_ not in results['concepts'][names[0]]:
            continue
        ks = [k for k in k_values if k in results['concepts'][names[0]][k_]]
        ms = [agg(k_, k) for k in ks]
        ax.errorbar(ks, [m for m, _ in ms], yerr=[s for _, s in ms],
                    fmt=style, color=color, capsize=3, label=lbl)
    ax.set_xlabel('k (atoms / latents)'); ax.set_ylabel('accuracy')
    ax.set_xscale('log'); ax.set_xticks(k_values); ax.set_xticklabels(k_values)
    ax.set_title(f'Sparse probing over {nC} concepts (mean ± sem)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out, 'probe_suite.png')
    plt.savefig(out_png, dpi=150)
    print(f'\nPlot:    {out_png}\nResults: {os.path.join(args.out, "results.pkl")}')


if __name__ == '__main__':
    main()
