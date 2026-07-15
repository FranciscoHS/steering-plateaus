#!/usr/bin/env python3
"""Alignment of contrastive diff-of-means directions with SAE decoder dictionaries.

For each contrastive direction, find its best-matching decoder atom (max cosine)
in each given SAE, against a random-unit-direction null over the same decoder.
With firing counts available (from sae_vs_sae_alignment.py's alignment.pkl or
recomputed here via --cache_path), also reports whether the matched atom is alive
and the best ALIVE match.

Usage:
    python scripts/dictionary_learning/contrastive_vs_sae_alignment.py \\
        --contrastive_pkls dirsA.pkl dirsB.pkl \\
        --saes popt=.../l10_lam1/sae_weights.pt baseline=.../l10_doubled_k32_2M/sae_weights.pt \\
        [--cache_path .../cached_activations.pt --n_cache_rows 200000] \\
        --out results/contrastive_vs_sae_l10
"""

import argparse
import importlib.util
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location(
    'svsa', os.path.join(os.path.dirname(__file__), 'sae_vs_sae_alignment.py'))
_svsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svsa)
load_decoder, firing_counts = _svsa.load_decoder, _svsa.firing_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrastive_pkls', nargs='+', required=True)
    ap.add_argument('--saes', nargs='+', required=True,
                    help='label=path pairs of SAE checkpoints')
    ap.add_argument('--cache_path', default=None)
    ap.add_argument('--n_cache_rows', type=int, default=200000)
    ap.add_argument('--n_random', type=int, default=2000)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # contrastive directions
    names, vecs = [], []
    for p in args.contrastive_pkls:
        d = pickle.load(open(p, 'rb'))
        tag = d.get('prompt_set', os.path.basename(p))
        for name, v in d['features'].items():
            names.append(f'{tag}:{name}')
            vecs.append(v.float())
        print(f'{p}: {len(d["features"])} dirs at layer {d.get("layer")}')
    C = torch.stack(vecs).to(device)
    C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
    M = C.shape[0]

    results = {'names': names, 'saes': {}}
    for spec in args.saes:
        label, path = spec.split('=', 1)
        W, sd, cfg = load_decoder(path)
        W = W.to(device)
        K = W.shape[0]

        fires = None
        if args.cache_path:
            meta_path = args.cache_path + '.meta.json'
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            mm = np.memmap(args.cache_path, dtype=np.dtype(meta['dtype']),
                           mode='r', shape=tuple(meta['shape']))
            X = torch.from_numpy(np.array(mm[:args.n_cache_rows]))
            fires = firing_counts(sd, X, cfg.get('k', 32), device)

        cos = C @ W.T                                      # [M, K]
        best = cos.max(dim=1)
        best_cos = best.values.cpu().numpy()
        best_idx = best.indices.cpu().numpy()

        alive_cos = alive_idx = None
        if fires is not None:
            mask = torch.from_numpy(fires > 0).to(device)
            cos_alive = cos.clone()
            cos_alive[:, ~mask] = -1.0
            ba = cos_alive.max(dim=1)
            alive_cos = ba.values.cpu().numpy()
            alive_idx = ba.indices.cpu().numpy()

        g = torch.Generator(device=device).manual_seed(args.rand_seed)
        Rnd = torch.randn(args.n_random, W.shape[1], generator=g, device=device)
        Rnd = Rnd / Rnd.norm(dim=-1, keepdim=True)
        rand_max = (Rnd @ W.T).max(dim=1).values.cpu().numpy()
        r95 = np.quantile(rand_max, 0.95)

        print(f'\n=== contrastive dirs vs {label} ({K} atoms'
              + (f', {int((fires>0).sum())} alive' if fires is not None else '')
              + f') ===   random null 95pct={r95:.3f}')
        for m in range(M):
            line = f'  {names[m]:>18s}: max_cos={best_cos[m]:.3f} (atom {best_idx[m]}'
            if fires is not None:
                line += f', fires={fires[best_idx[m]]}'
                line += (f'; best alive: {alive_cos[m]:.3f} '
                         f'(atom {alive_idx[m]}, fires={fires[alive_idx[m]]})')
            print(line + ')')

        results['saes'][label] = {
            'best_cos': best_cos, 'best_idx': best_idx,
            'alive_cos': alive_cos, 'alive_idx': alive_idx,
            'rand_max': rand_max, 'fires_of_best': (fires[best_idx]
                                                    if fires is not None else None)}

    with open(os.path.join(args.out, 'contrastive_alignment.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), **results}, f)
    print(f"\nSaved {os.path.join(args.out, 'contrastive_alignment.pkl')}")

    # dot plot: one row per contrastive dir, one marker per SAE (alive-best if known)
    fig, ax = plt.subplots(figsize=(8, 0.45 * M + 2))
    y = np.arange(M)
    for ci, (label, r) in enumerate(results['saes'].items()):
        vals = r['alive_cos'] if r['alive_cos'] is not None else r['best_cos']
        ax.scatter(vals, y, s=40, color=f'C{ci}', label=f'{label} (best alive atom)')
        r95 = np.quantile(r['rand_max'], 0.95)
        ax.axvline(r95, color=f'C{ci}', ls='--', lw=1, alpha=0.7,
                   label=f'{label} random 95pct={r95:.3f}')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('max cosine with any (alive) decoder atom')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('Contrastive directions vs SAE dictionaries')
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'contrastive_alignment.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
