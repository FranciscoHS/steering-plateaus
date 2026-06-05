#!/usr/bin/env python3
"""Max-activating corpus examples for found dictionary directions.

For each found atom d_k, project a corpus of layer-L activations (all token
positions, BOS/pos-0 skipped) onto d_k and report the highest-scoring tokens
with their left context. A quick interpretability read: do the optimized
directions align with anything human-legible in the corpus?

Score = a · d_k  (d_k unit). We also report the same with the corpus mean
activation removed, which isolates the variation aligned with d_k (the raw
dot can be dominated by the shared mean component).

Usage:
    python scripts/superellipse/dict_max_activating.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b --layer 14 \\
        --n_seqs 2000 --seq_len 64 --topk 25 --out .../max_activating
"""

import argparse
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model
from src.data import load_fineweb_fixed_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--topk', type=int, default=25)
    ap.add_argument('--ctx_left', type=int, default=12,
                    help='# tokens of left context to show per hit')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K, d_model = D.shape
    D_dev = D.to(device)
    print(f'Dictionary: {K} atoms, d_model={d_model}', flush=True)

    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)                  # [N, T] cpu

    # Collect all-position activations; skip pos 0 (BOS / attention sink).
    N, T = tok_ids.shape
    acts_chunks, meta = [], []      # meta: (seq_idx, pos)
    print('Capturing activations', flush=True)
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        h = h[:, 1:, :].float()                      # drop pos 0  -> [b, T-1, d]
        b = h.shape[0]
        acts_chunks.append(h.reshape(b * (T - 1), d_model).cpu())
        for s in range(b):
            for p in range(1, T):
                meta.append((i + s, p))
    acts = torch.cat(acts_chunks, 0)                 # [M, d] cpu
    M = acts.shape[0]
    print(f'  {M} token activations  mean||a||={acts.norm(dim=-1).mean():.1f}',
          flush=True)

    mean_act = acts.mean(0, keepdim=True)
    acts_dev = acts.to(device)
    mean_dev = mean_act.to(device)
    act_norms = acts_dev.norm(dim=-1).clamp_min(1e-9)  # [M]
    meta = np.array(meta)                             # [M, 2]

    def decode_hit(seq_idx, pos):
        lo = max(1, pos - args.ctx_left)
        ids = tok_ids[seq_idx]
        left = tokenizer.decode(ids[lo:pos]).replace('\n', '\\n')
        tgt = tokenizer.decode(ids[pos:pos + 1]).replace('\n', '\\n')
        return f'…{left}⟦{tgt}⟧'

    out_txt = os.path.join(args.out, 'max_activating.txt')
    lines = []
    for k in range(K):
        d = D_dev[k]
        raw = acts_dev @ d                           # [M]
        cen = (acts_dev - mean_dev) @ d              # mean-removed
        cos = raw / act_norms                        # cosine: norm-normalized
        for tag, scores in [('raw dot', raw), ('mean-removed', cen),
                            ('cosine', cos)]:
            top = torch.topk(scores, args.topk).indices.cpu().numpy()
            header = (f'\n=== atom {k}  |  {tag}  |  '
                      f'top {args.topk} (score: token ⟦…⟧ in context) ===')
            lines.append(header)
            print(header, flush=True)
            for rank, m in enumerate(top):
                s_idx, pos = int(meta[m, 0]), int(meta[m, 1])
                sc = float(scores[m].item())
                txt = decode_hit(s_idx, pos)
                line = f'  {rank+1:2d}. {sc:8.2f}  {txt}'
                lines.append(line)
                if rank < 10:        # console: only top 10 per (atom,tag)
                    print(line, flush=True)

    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved {out_txt}', flush=True)


if __name__ == '__main__':
    main()
