#!/usr/bin/env python3
"""Rank dictionary directions by intrinsic feature-ness of their activation
distribution over a FineWeb corpus.

Per-token score is the CENTERED COSINE

    c_i = (a_i - abar) . d_k / || a_i - abar ||

where abar is the corpus-mean layer-L activation (pos-0/BOS skipped). Centering
removes the shared residual-stream mean component; cosine removes the norm
outliers (massive-activation tokens) that would otherwise dominate a raw dot.
So c isolates how *selectively* a direction aligns with token states, which is
what we want for "is this direction feature-like".

Per atom, over the corpus distribution of c:
  - excess kurtosis   (PRIMARY rank): heavy-tailed/peaked = sparse-firing =
    feature-like. The ICA / sparse-coding classic.
  - tail ratio p99.9(|c|) / p50(|c|)   (CROSS-CHECK): extreme vs typical
    magnitude; robust (percentile-based, not max-driven like kurtosis).
  - q999 / q001 of signed c, plus min/max, so the dominant firing side (+/-)
    is visible (a one-sided feature lives on one tail).

Writes ranking.pkl, a readable top/bottom report, top_atoms.txt (comma list for
the dashboard), and a kurtosis-distribution histogram. Run on a pod (needs the
model forward):

    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/rank_directions_featureness.py \
      --dict_path /tmp/dict_25k_snapshot.pt --model qwen3-1.7b --layer 14 \
      --n_seqs 2000 --seq_len 64 \
      --out results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_resample/featureness_25k
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


def kth(x, q, dim=0):
    """q-quantile per column via kthvalue (avoids torch.quantile's element cap)."""
    M = x.shape[dim]
    k = min(M, max(1, int(round(q * M))))
    return torch.kthvalue(x, k, dim=dim).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--atom_chunk', type=int, default=1000,
                    help='# atoms scored per matmul chunk (memory knob)')
    ap.add_argument('--topn_report', type=int, default=60)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d_model = D.shape
    print(f'Dictionary: {K} atoms, d_model={d_model}', flush=True)

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
    A = torch.cat(chunks, 0)                       # [M, d] cpu
    M = A.shape[0]
    print(f'  M={M} token activations  mean||a||={A.norm(dim=-1).mean():.1f}',
          flush=True)

    # Centered, unit-normalised activations -> centered cosine = Ahat @ d_k.
    abar = A.mean(0, keepdim=True)
    Ac = A - abar
    norms = Ac.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Ahat = (Ac / norms).to(device)                 # [M, d] gpu (~1GB f32)

    kurt = np.zeros(K); tail = np.zeros(K)
    q999 = np.zeros(K); q001 = np.zeros(K)
    smax = np.zeros(K); smin = np.zeros(K)
    smean = np.zeros(K); sstd = np.zeros(K)

    print(f'Scoring {K} atoms in chunks of {args.atom_chunk}', flush=True)
    for c0 in range(0, K, args.atom_chunk):
        c1 = min(K, c0 + args.atom_chunk)
        Dc = D[c0:c1].to(device)
        S = Ahat @ Dc.T                            # [M, c]  centered cosine
        mu = S.mean(0); sd = S.std(0).clamp_min(1e-9)
        z = (S - mu) / sd
        k_exc = (z ** 4).mean(0) - 3.0             # excess kurtosis
        absS = S.abs()
        p50 = kth(absS, 0.5).clamp_min(1e-9)
        p999 = kth(absS, 0.999)
        sl = slice(c0, c1)
        kurt[sl] = k_exc.cpu().numpy()
        tail[sl] = (p999 / p50).cpu().numpy()
        q999[sl] = kth(S, 0.999).cpu().numpy()
        q001[sl] = kth(S, 0.001).cpu().numpy()
        smax[sl] = S.max(0).values.cpu().numpy()
        smin[sl] = S.min(0).values.cpu().numpy()
        smean[sl] = mu.cpu().numpy()
        sstd[sl] = sd.cpu().numpy()
        print(f'  atoms {c0}:{c1} done', flush=True)

    order = np.argsort(-kurt)                      # most feature-like first

    out = {'args': vars(args), 'kurtosis': kurt, 'tail_ratio': tail,
           'q999': q999, 'q001': q001, 'smax': smax, 'smin': smin,
           'mean': smean, 'std': sstd, 'order': order,
           'abar': abar.squeeze(0).numpy()}
    with open(os.path.join(args.out, 'ranking.pkl'), 'wb') as f:
        pickle.dump(out, f)

    lines = [f'Feature-ness ranking (centered cosine) for {args.dict_path}',
             f'{K} atoms, M={M} tokens',
             f'kurtosis: mean={kurt.mean():.2f} median={np.median(kurt):.2f} '
             f'max={kurt.max():.2f} min={kurt.min():.2f}',
             f'tail_ratio: median={np.median(tail):.2f} '
             f'max={tail.max():.2f}', '',
             f'{"rank":>4} {"atom":>6} {"kurt":>8} {"tailR":>7} '
             f'{"q999":>7} {"q001":>7} side']
    for r, a in enumerate(order[:args.topn_report]):
        side = '+' if abs(smax[a]) >= abs(smin[a]) else '-'
        lines.append(f'{r:>4} {a:>6} {kurt[a]:>8.2f} {tail[a]:>7.2f} '
                     f'{q999[a]:>7.3f} {q001[a]:>7.3f} {side}')
    lines.append('')
    lines.append('--- bottom 15 (least feature-like) ---')
    for r, a in enumerate(order[-15:]):
        lines.append(f'{K-15+r:>4} {a:>6} {kurt[a]:>8.2f} {tail[a]:>7.2f}')
    report = os.path.join(args.out, 'ranking_report.txt')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:28]), flush=True)
    print(f'\nwrote {report}', flush=True)

    with open(os.path.join(args.out, 'top_atoms.txt'), 'w') as f:
        f.write(','.join(map(str, order[:args.topn_report].tolist())) + '\n')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(kurt, bins=80, color='C0', alpha=0.85)
    ax.set_xlabel('excess kurtosis of centered-cosine activations')
    ax.set_ylabel('# atoms')
    ax.set_title(f'Feature-ness (kurtosis) over {K} atoms — '
                 f'{args.model} L{args.layer}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'kurtosis_hist.png'), dpi=150)
    print('Saved kurtosis_hist.png\ndone', flush=True)


if __name__ == '__main__':
    main()
