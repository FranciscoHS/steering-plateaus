#!/usr/bin/env python3
"""Cosine alignment between two SAE decoder dictionaries (e.g. popt vs baseline).

For each decoder atom of SAE A, find its best-matching atom in SAE B (max cosine
over normalized decoder rows), and vice versa. Compared against a random-direction
null (a random unit vector's max cosine over the same reference set — the relevant
baseline since with 32k references even random directions get inflated best-matches).

Also reports the many-to-one structure of the argmax mapping in both directions:
how many distinct B-atoms are hit, and the collision-count distribution (how many
A-atoms map to the same B-atom).

Optionally, given an activation cache, computes latent firing counts for both SAEs
(top-k encode, matching src/autoencoders.py TopKSAE) so dead latents can be
separated out in the histogram — dead latents sit at their random init and can
contaminate the alignment picture.

Pure linear algebra + optional encoder passes; no model forward needed.

Usage:
    python scripts/dictionary_learning/sae_vs_sae_alignment.py \\
        --sae_a .../popt/sae_weights.pt --label_a popt \\
        --sae_b .../baseline/sae_weights.pt --label_b baseline \\
        [--cache_path .../cached_activations.pt --n_cache_rows 200000] \\
        --out results/sae_alignment_popt_vs_baseline
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_decoder(path):
    """Return (W_dec [d_sae, d_model] unit rows, state dict, config).

    Expects our TopKSAE checkpoints: {'state_dict': {W_dec, b_dec,
    encoder.weight, encoder.bias}, 'config': {d_in, d_sae, k}, ...}.
    """
    blob = torch.load(path, map_location='cpu', weights_only=False)
    sd = blob['state_dict'] if 'state_dict' in blob else blob
    cfg = blob.get('config', {})
    W = sd['W_dec'].to(torch.float32)                 # [d_sae, d_in]
    W = W / (W.norm(dim=-1, keepdim=True) + 1e-12)
    return W, sd, cfg


def max_cos_chunked(A, B, chunk=2048):
    """Max cosine and argmax of each row of A over rows of B (both unit-norm)."""
    vals, idxs = [], []
    for i in range(0, A.shape[0], chunk):
        cos = A[i:i + chunk] @ B.T
        m = cos.max(dim=1)
        vals.append(m.values)
        idxs.append(m.indices)
    return torch.cat(vals).cpu().numpy(), torch.cat(idxs).cpu().numpy()


def firing_counts(sd, X, topk, device, chunk=8192):
    """Top-k encode cache rows X (TopKSAE semantics: ReLU((x-b_dec)@W_enc.T+b)
    then top-k), return per-latent counts of nonzero top-k membership."""
    W_enc = sd['encoder.weight'].to(device, torch.float32)   # [d_sae, d_in]
    b_enc = sd['encoder.bias'].to(device, torch.float32)
    b_dec = sd['b_dec'].to(device, torch.float32)
    counts = torch.zeros(W_enc.shape[0], dtype=torch.long, device=device)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].to(device, torch.float32) - b_dec
        pre = torch.relu(x @ W_enc.T + b_enc)
        vals, top = pre.topk(topk, dim=-1)
        top = top[vals > 0]           # ReLU can zero out some of the top-k slots
        counts.scatter_add_(0, top.flatten(),
                            torch.ones_like(top.flatten(), dtype=torch.long))
    return counts.cpu().numpy()


def collision_stats(argmax_idx, n_ref):
    """Distribution of how many source atoms share the same best-match target."""
    binc = np.bincount(argmax_idx, minlength=n_ref)
    hit = binc[binc > 0]
    return {
        'n_source': len(argmax_idx),
        'n_ref': n_ref,
        'n_distinct_targets': int((binc > 0).sum()),
        'max_collision': int(binc.max()),
        'collision_hist': np.bincount(hit),   # collision_hist[c] = # targets hit by exactly c sources
        'target_counts': binc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae_a', required=True, help='the "new" dictionary (e.g. popt)')
    ap.add_argument('--sae_b', required=True, help='the reference SAE (e.g. baseline)')
    ap.add_argument('--label_a', default='A')
    ap.add_argument('--label_b', default='B')
    ap.add_argument('--n_random', type=int, default=2000)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--cache_path', default=None,
                    help='cached_activations.pt for firing counts')
    ap.add_argument('--n_cache_rows', type=int, default=200000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Wa, sd_a, cfg_a = load_decoder(args.sae_a)
    Wb, sd_b, cfg_b = load_decoder(args.sae_b)
    Wa, Wb = Wa.to(device), Wb.to(device)
    Ka, Kb = Wa.shape[0], Wb.shape[0]
    d_model = Wa.shape[1]
    print(f'{args.label_a}: {Ka} atoms, {args.label_b}: {Kb} atoms, d_model={d_model}',
          flush=True)

    # A -> B and B -> A best matches
    a2b_cos, a2b_idx = max_cos_chunked(Wa, Wb)
    b2a_cos, b2a_idx = max_cos_chunked(Wb, Wa)

    # random null: max cos of random unit dirs over B (and over A for the reverse)
    g = torch.Generator(device=device).manual_seed(args.rand_seed)
    Rnd = torch.randn(args.n_random, d_model, generator=g, device=device)
    Rnd = Rnd / Rnd.norm(dim=-1, keepdim=True)
    rand_vs_b, _ = max_cos_chunked(Rnd, Wb)
    rand_vs_a, _ = max_cos_chunked(Rnd, Wa)

    # optional firing counts to identify dead latents
    fires_a = fires_b = None
    if args.cache_path:
        meta_path = args.cache_path + '.meta.json'
        if os.path.exists(meta_path):        # raw memmap + sidecar (train_sae.py)
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            mm = np.memmap(args.cache_path, dtype=np.dtype(meta['dtype']),
                           mode='r', shape=tuple(meta['shape']))
            X = torch.from_numpy(np.array(mm[:args.n_cache_rows]))
        else:
            X = torch.load(args.cache_path, map_location='cpu')
            if isinstance(X, dict):
                X = X.get('acts', next(iter(X.values())))
            X = X[:args.n_cache_rows]
        ka = cfg_a.get('k', 32)
        kb = cfg_b.get('k', 32)
        print(f'firing counts over {X.shape[0]} cached rows '
              f'(top-{ka}/{kb})', flush=True)
        fires_a = firing_counts(sd_a, X, ka, device)
        fires_b = firing_counts(sd_b, X, kb, device)
        print(f'  dead latents: {args.label_a} {(fires_a == 0).sum()}/{Ka}, '
              f'{args.label_b} {(fires_b == 0).sum()}/{Kb}', flush=True)

    def report(tag, cos, idx, rand, n_ref, fires_src):
        r95 = np.quantile(rand, 0.95)
        cs = collision_stats(idx, n_ref)
        print(f'\n=== {tag} ===')
        print(f'  max-cos: mean={cos.mean():.3f} median={np.median(cos):.3f} '
              f'min={cos.min():.3f} max={cos.max():.3f}')
        print(f'  random null: median={np.median(rand):.3f} 95pct={r95:.3f}')
        print(f'  above random 95pct: {(cos > r95).sum()}/{len(cos)}')
        print(f'  argmax targets: {cs["n_distinct_targets"]} distinct of {n_ref} '
              f'(max {cs["max_collision"]} sources on one target)')
        ch = cs['collision_hist']
        desc = ', '.join(f'{c}x:{ch[c]}' for c in range(1, min(len(ch), 6)))
        tail = int(ch[6:].sum()) if len(ch) > 6 else 0
        print(f'  collision counts: {desc}' + (f', >=6x:{tail}' if tail else ''))
        if fires_src is not None:
            alive = fires_src > 0
            print(f'  alive sources only ({alive.sum()}): '
                  f'median max-cos={np.median(cos[alive]):.3f}; '
                  f'dead sources ({(~alive).sum()}): '
                  f'median={np.median(cos[~alive]):.3f}' if (~alive).any()
                  else f'  all sources alive')
        return cs

    cs_a2b = report(f'{args.label_a} -> {args.label_b}', a2b_cos, a2b_idx,
                    rand_vs_b, Kb, fires_a)
    cs_b2a = report(f'{args.label_b} -> {args.label_a}', b2a_cos, b2a_idx,
                    rand_vs_a, Ka, fires_b)

    with open(os.path.join(args.out, 'alignment.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args),
                     'a2b_cos': a2b_cos, 'a2b_idx': a2b_idx,
                     'b2a_cos': b2a_cos, 'b2a_idx': b2a_idx,
                     'rand_vs_b': rand_vs_b, 'rand_vs_a': rand_vs_a,
                     'fires_a': fires_a, 'fires_b': fires_b,
                     'collisions_a2b': cs_a2b, 'collisions_b2a': cs_b2a}, f)
    print(f"\nSaved {os.path.join(args.out, 'alignment.pkl')}", flush=True)

    # two-panel histogram: A->B and B->A, each with its random null
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [(a2b_cos, rand_vs_b, fires_a,
               f'{args.label_a} atoms -> best {args.label_b} match'),
              (b2a_cos, rand_vs_a, fires_b,
               f'{args.label_b} atoms -> best {args.label_a} match')]
    for ax, (cos, rand, fires, title) in zip(axes, panels):
        bins = np.linspace(0, 1, 80)
        ax.hist(rand, bins=bins, density=True, color='gray', alpha=0.5,
                label=f'random null (n={len(rand)})')
        if fires is not None and (fires == 0).any():
            alive = fires > 0
            ax.hist(cos[alive], bins=bins, density=True, color='C0', alpha=0.6,
                    label=f'alive atoms (n={alive.sum()})')
            ax.hist(cos[~alive], bins=bins, density=True, color='C3', alpha=0.5,
                    label=f'dead atoms (n={(~alive).sum()})')
        else:
            ax.hist(cos, bins=bins, density=True, color='C0', alpha=0.6,
                    label=f'atoms (n={len(cos)})')
        r95 = np.quantile(rand, 0.95)
        ax.axvline(r95, color='k', ls='--', lw=1, label=f'random 95pct={r95:.3f}')
        ax.set_xlabel('max cosine with any atom in the other dictionary')
        ax.set_ylabel('density')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'alignment_hist.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plot_path}', flush=True)


if __name__ == '__main__':
    main()
