#!/usr/bin/env python3
"""Max |cosine| alignment of each direction with a real activation sample.

For each direction v (unit) and a large sample of layer-L activations {a},
compute the single statistic

    s(v) = max_a |v . a| / ||a||      ( = max_a |cos(v, a)| )

i.e. how closely v ever lines up with an actual activation. RAW cosine (no
centering, no norm-weighting) per spec. We compute this for the optimized
dictionary directions and for the SAE decoder columns, and plot the two
distributions to compare.

Activation extraction mirrors rank_directions_featureness.py (FineWeb,
hidden_states[layer+1], pos-0/BOS skipped). Run on a GPU pod.

Usage:
    HF_HOME=/workspace/hf_cache python -u \\
      scripts/superellipse/max_cos_alignment.py \\
      --dict_path .../lam100_20k/dictionary.pt \\
      --sae_ckpt  .../qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt \\
      --model qwen3-1.7b --layer 14 --n_seqs 2000 --seq_len 64 \\
      --out .../max_cos_alignment
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

from src.model import load_model
from src.data import load_fineweb_fixed_length


def max_abs_cos(Ahat, V, dir_chunk, device):
    """Ahat: [M, d] unit activations (gpu). V: [K, d] unit directions (cpu/gpu).
    Returns [K] = max_a |Ahat @ v| over the M activations, chunked over dirs."""
    K = V.shape[0]
    out = np.zeros(K, dtype=np.float32)
    for c0 in range(0, K, dir_chunk):
        c1 = min(K, c0 + dir_chunk)
        Vc = V[c0:c1].to(device)
        S = Ahat @ Vc.T                       # [M, c]  raw cosine
        out[c0:c1] = S.abs().max(0).values.cpu().numpy()
        print(f'  dirs {c0}:{c1} done', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True,
                    help='optimized dictionary.pt')
    ap.add_argument('--sae_ckpt', required=True,
                    help='SAE checkpoint with W_dec [d, d_sae]')
    ap.add_argument('--contrastive_paths', nargs='*', default=[],
                    help='dictionary.pt files (with "names") for the contrastive '
                         'set; all concatenated into one "contrastive" series')
    ap.add_argument('--n_random', type=int, default=20000,
                    help='# random unit directions (0 to skip)')
    ap.add_argument('--random_seed', type=int, default=0)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--dir_chunk', type=int, default=2000,
                    help='# directions scored per matmul chunk (memory knob)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    # --- optimized dictionary directions (unit) ---
    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d_model = D.shape
    print(f'Dictionary: {K} atoms, d_model={d_model}', flush=True)

    # --- SAE decoder columns (unit) ---
    sae = torch.load(args.sae_ckpt, map_location='cpu', weights_only=False)
    W_dec = sae['W_dec'].to(torch.float32)                 # [d, d_sae]
    assert W_dec.shape[0] == d_model, f'{W_dec.shape} vs d_model={d_model}'
    Wd = (W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)).T  # [d_sae, d]
    n_sae = Wd.shape[0]
    print(f'SAE decoder: {n_sae} columns', flush=True)

    # --- contrastive set (concat of labeled contrastive dicts, unit) ---
    Vc = None
    if args.contrastive_paths:
        parts = []
        for p in args.contrastive_paths:
            cb = torch.load(p, map_location='cpu', weights_only=False)
            parts.append(cb['dictionary'].to(torch.float32))
            print(f'  contrastive {os.path.basename(os.path.dirname(p))}: '
                  f'{parts[-1].shape[0]} dirs', flush=True)
        Vc = torch.cat(parts, 0)
        Vc = Vc / (Vc.norm(dim=-1, keepdim=True) + 1e-12)
        print(f'Contrastive set: {Vc.shape[0]} dirs total', flush=True)

    # --- random unit directions ---
    Vr = None
    if args.n_random > 0:
        g = torch.Generator(device='cpu').manual_seed(args.random_seed)
        Vr = torch.randn(args.n_random, d_model, generator=g)
        Vr = Vr / Vr.norm(dim=-1, keepdim=True)
        print(f'Random: {args.n_random} unit dirs', flush=True)

    # --- activation sample (pos-0/BOS skipped) ---
    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape
    print('Capturing activations (pos-0 skipped)', flush=True)
    chunks = []
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        chunks.append(h[:, 1:, :].float().reshape(-1, d_model).cpu())
    A = torch.cat(chunks, 0)                                # [M, d] cpu
    M = A.shape[0]
    norms = A.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Ahat = (A / norms).to(device)                           # [M, d] unit, raw
    print(f'  M={M} token activations  mean||a||={norms.mean():.1f}', flush=True)

    # --- score every series ---
    series = [('optimized dict', D, 'C2'),
              ('SAE decoder', Wd, 'C1')]
    if Vc is not None:
        series.append(('contrastive', Vc, 'C3'))
    if Vr is not None:
        series.append(('random', Vr, 'C7'))

    scores = {}
    for label, V, _ in series:
        print(f'Scoring {label} ({V.shape[0]}) in chunks of {args.dir_chunk}',
              flush=True)
        scores[label] = max_abs_cos(Ahat, V, args.dir_chunk, device)

    out = {'args': vars(args), 'M': M, 'scores': scores}
    with open(os.path.join(args.out, 'max_cos_alignment.pkl'), 'wb') as f:
        pickle.dump(out, f)

    def q(x):
        return (f'median={np.median(x):.3f}  p10={np.quantile(x,.1):.3f}  '
                f'p90={np.quantile(x,.9):.3f}  min={x.min():.3f}  max={x.max():.3f}')
    print('', flush=True)
    for label, V, _ in series:
        print(f'{label:16s} max|cos|:  {q(scores[label])}', flush=True)

    # --- overlaid density histograms (counts differ across series) ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    lo = min(s.min() for s in scores.values())
    hi = max(s.max() for s in scores.values())
    bins = np.linspace(lo, hi, 80)
    for label, V, col in series:
        s = scores[label]
        ax.hist(s, bins=bins, density=True, color=col, alpha=0.50,
                label=f'{label} (n={s.size})')
        ax.axvline(np.median(s), color=col, ls='--', lw=1.5)
    ax.set_xlabel(r'$\max_a\,|\hat v^\top a|/\|a\|$  (max |cos| with a real activation)')
    ax.set_ylabel('density')
    ax.set_title(f'Max activation alignment by direction family\n'
                 f'{args.model} L{args.layer}  M={M} token activations')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = os.path.join(args.out, 'max_cos_alignment.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f'\nSaved {png}', flush=True)
    print(f'Saved {os.path.join(args.out, "max_cos_alignment.pkl")}', flush=True)


if __name__ == '__main__':
    main()
