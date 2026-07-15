#!/usr/bin/env python3
"""Incrementally add the sae_omp line to an existing probe-suite results.pkl.

run_sparse_probe_suite.py recomputes every method from scratch; this avoids that
by loading the prior complete results, computing ONLY sae_omp (SAE decoder via
OMP-k, mean-as-free-bias) per concept from the cached activations, merging, and
re-plotting. Reuses the cached suite_acts.pt — no model, fast.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.probes.run_probe_dict_vs_sae import omp_gpu, load_qwen_sae, run_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_results', required=True, help='prior results.pkl')
    ap.add_argument('--acts', required=True, help='cached suite_acts.pt')
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--omp_batch', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = pickle.load(open(args.base_results, 'rb'))
    k_values = results['k_values']
    acts = torch.load(args.acts, weights_only=False)

    sae = load_qwen_sae(args.sae_ckpt, device)
    sae_dec = sae.W_dec.detach().cpu().float()
    sae_dec = sae_dec / (sae_dec.norm(dim=-1, keepdim=True) + 1e-12)
    print(f'SAE decoder dictionary: {sae_dec.shape[0]} atoms', flush=True)

    for name, res in results['concepts'].items():
        a = acts[name]
        tr_a, tr_y, te_a, te_y = a['train'], a['train_y'], a['test'], a['test_y']
        abar = tr_a.mean(dim=0, keepdim=True)
        res['sae_omp'] = {}
        for k in k_values:
            trr = abar + omp_gpu(sae_dec, tr_a - abar, k, device, args.omp_batch)
            ter = abar + omp_gpu(sae_dec, te_a - abar, k, device, args.omp_batch)
            res['sae_omp'][k] = run_probe(trr.numpy(), tr_y, ter.numpy(), te_y, args.seed)
        print(f'[{name}] sae_omp@{k_values[-1]}={res["sae_omp"][k_values[-1]]:.3f}', flush=True)

    with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- re-plot: mean +/- sem across concepts ---------------------------
    names = list(results['concepts'].keys()); nC = len(names)
    dicts = results['dict_labels']

    def agg(method, k=None):
        vals = np.array([results['concepts'][n][method] if k is None
                         else results['concepts'][n][method][k] for n in names], float)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(nC)

    fig, ax = plt.subplots(figsize=(7, 5))
    raw_m, raw_s = agg('raw')
    ax.axhline(raw_m, ls='--', color='black', alpha=0.6, label=f'raw ({raw_m:.3f})')
    specs = [('sae_native', 'Qwen SAE (native top-k)', 'o-', 'tab:blue'),
             ('sae_omp', 'Qwen SAE decoder (OMP-k)', 'D-', 'tab:cyan')]
    dcolors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
    for i, l in enumerate(dicts):
        specs.append((f'dict:{l}', f'dict {l}', 's-', dcolors[i % len(dcolors)]))
    specs.append(('random', 'random', '^-', 'tab:gray'))
    for key, lbl, style, color in specs:
        ms = [agg(key, k) for k in k_values]
        ax.errorbar(k_values, [m for m, _ in ms], yerr=[s for _, s in ms],
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
