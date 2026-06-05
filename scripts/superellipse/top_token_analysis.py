#!/usr/bin/env python3
"""
Top-activating-token analysis for the learned (d1, d2) pair.

Runs a large FineWeb corpus through the model, records per-position layer-2
activations, computes their overlap with d1 and d2, and produces:
  - a histogram of per-position |overlap| per direction, with the strongly-
    aligned tail color-coded by the token strings (types) that dominate it;
  - a bar chart of the most frequent token types in the strong tail;
  - a text dump of the top-K aligned positions (token + context + value).

Overlap metric (--metric):
  cosine (default): cos = (a . d_i) / ||a||  -> alignment, norm-independent.
                    Removes the massive-activation/outlier-norm effect that
                    dominated the raw dot product.
  dot:              a . d_i (signed) -> raw projection.

Ranking/threshold/coloring use the ABSOLUTE value, so strong alignment of
either sign counts (a -200 overlap is as significant as +200). The sign is
still reported in the dump and as a diverging color in the histogram.

Position 0 (BOS) is excluded (outlier).

Usage:
    python scripts/superellipse/top_token_analysis.py \
        --dict_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl \
        --metric cosine --n_seq 4000 --seq_len 64
"""
import os, sys, pickle, argparse, time
from collections import Counter
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_pkl', type=str,
                    default='results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl')
    ap.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'dot'])
    ap.add_argument('--n_seq', type=int, default=4000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--top_k', type=int, default=40, help='top examples to dump')
    ap.add_argument('--tail_pct', type=float, default=99.0,
                    help='percentile of |overlap| defining the "strong" tail')
    ap.add_argument('--n_token_colors', type=int, default=8)
    ap.add_argument('--ctx_window', type=int, default=6)
    ap.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', type=str, default=None)
    args = ap.parse_args()

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions'].float()
    da = dd['args']
    model_name = da['model']; layer = da['layer']

    if args.results_dir is None:
        tag = 'top_tokens' if args.metric == 'dot' else f'top_tokens_{args.metric}'
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), tag)
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    print(f'Loading {model_name} (dtype={args.dtype})...')
    model, tokenizer, device = load_model(model_name, dtype=dtype_map[args.dtype])
    D = directions.to(device)  # [2, d]

    print(f'Loading {args.n_seq} FineWeb seqs (seq_len={args.seq_len}, seed={args.seed})...')
    toks = load_fineweb_fixed_length(args.n_seq, tokenizer, seq_len=args.seq_len, seed=args.seed)

    # Per-position overlap + token ids (exclude position 0 = BOS).
    all_val, all_tok = [], []
    t0 = time.time()
    for i in range(0, len(toks), args.batch_size):
        bt = torch.stack(toks[i:i + args.batch_size]).to(device)
        with torch.no_grad():
            hidden = model(bt, output_hidden_states=True).hidden_states[layer + 1].float()  # [B,L,d]
            dot = hidden @ D.T  # [B,L,2]
            if args.metric == 'cosine':
                val = dot / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                val = dot
        all_val.append(val[:, 1:, :].reshape(-1, 2).cpu())
        all_tok.append(bt[:, 1:].reshape(-1).cpu())
        if (i // args.batch_size) % 20 == 0:
            print(f'  {min(i+args.batch_size, len(toks))}/{len(toks)}', end='\r')
    val = torch.cat(all_val, 0).numpy()    # [n_pos, 2], signed
    tokids = torch.cat(all_tok, 0).numpy()
    print(f'\n  {val.shape[0]} positions, took {time.time()-t0:.1f}s')

    uniq = np.unique(tokids)
    tokstr = {int(t): tokenizer.decode([int(t)]) for t in uniq}
    L = args.seq_len
    seq_of = np.repeat(np.arange(len(toks)), L - 1)
    pos_of = np.tile(np.arange(1, L), len(toks))

    with open(os.path.join(args.results_dir, 'overlaps.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'model': model_name, 'layer': layer,
                     'metric': args.metric, 'val': val, 'tokids': tokids,
                     'directions': directions}, f)

    # ---- text dump: top-K by |overlap| (signed value shown) ----
    txt = os.path.join(args.results_dir, 'top_tokens.txt')
    with open(txt, 'w') as f:
        f.write(f'{model_name} layer {layer}: top-activating tokens, metric={args.metric}, ranked by |overlap|\n')
        f.write(f'cos(d1,d2)={(directions[0]@directions[1]).item():.4f}, {val.shape[0]} positions\n')
        for di, dname in enumerate(['d1', 'd2']):
            v = val[:, di]
            f.write(f'\n{"="*70}\n{dname}: range [{v.min():.3f}, {v.max():.3f}], '
                    f'mean {v.mean():.3f}, std {v.std():.3f}\n')
            f.write(f'--- {dname} top {args.top_k} by |overlap| ---\n')
            for idx in np.argsort(-np.abs(v))[:args.top_k]:
                s = seq_of[idx]; po = pos_of[idx]
                ctx = tokenizer.decode(toks[s][max(0, po - args.ctx_window):po + 1].tolist())
                f.write(f'  {v[idx]:+.3f}  tok={tokstr[int(tokids[idx])]!r:18s} ctx=...{ctx!r}\n')

    # ---- plots: |overlap| histogram (type-coded tail) + top-type bar, per dir ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for di, dname in enumerate(['d1', 'd2']):
        v = val[:, di]; av = np.abs(v)
        thr = np.percentile(av, args.tail_pct)
        strong = av >= thr
        strong_toks = np.array([tokstr[int(t)] for t in tokids[strong]])
        cnt = Counter(strong_toks.tolist())
        top_toks = [t for t, _ in cnt.most_common(args.n_token_colors)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_toks)))

        ax = axes[di, 0]
        bins = np.linspace(0, av.max(), 80)
        ax.hist(av, bins=bins, color='#cccccc', log=True, label='all positions')
        for tk, col in zip(top_toks, colors):
            idxs = np.where(strong)[0][strong_toks == tk]
            ax.hist(av[idxs], bins=bins, color=col, log=True, alpha=0.9,
                    label=f'{tk!r} (n={cnt[tk]})')
        ax.axvline(thr, color='k', ls=':', lw=1.2, label=f'{args.tail_pct:.0f}pct |overlap|')
        ax.set_xlabel(f'|overlap| with {dname}  (|{args.metric}|)')
        ax.set_ylabel('count (log)')
        ax.set_title(f'{dname}: per-position |overlap| histogram')
        ax.legend(fontsize=7, ncol=2)

        ax = axes[di, 1]
        labels = [t for t, _ in cnt.most_common(15)]
        counts = [cnt[t] for t in labels]
        ax.barh(range(len(labels))[::-1], counts, color='#377eb8')
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels([repr(t) for t in labels], fontsize=8)
        ax.set_xlabel('count in strong tail')
        ax.set_title(f'{dname}: top token types in top {100-args.tail_pct:.0f}% by |overlap|')

    fig.suptitle(f'{model_name} L{layer}: top-activating tokens (metric={args.metric}, |overlap|)', fontsize=14)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'top_tokens.png')
    plt.savefig(png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {os.path.join(args.results_dir, "overlaps.pkl")}')
    print(f'Saved: {txt}')
    print(f'Saved: {png}')


if __name__ == '__main__':
    main()
