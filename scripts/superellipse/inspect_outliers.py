#!/usr/bin/env python3
"""Characterize the extreme-projection outlier positions from top_tokens/projections.pkl.

Quantifies how many positions are extreme, the gap structure, and which token
types they are, for d1 and d2 (and their overlap). Loads only the tokenizer
(no model weights). Local CPU analysis.
"""
import sys, pickle, argparse
from collections import Counter
sys.path.insert(0, '.')
import numpy as np
from transformers import AutoTokenizer
from src.model import MODEL_ALIASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='results/superellipse_dict/qwen3-1.7b_L2_a30/top_tokens/projections.pkl')
    args = ap.parse_args()
    d = pickle.load(open(args.pkl, 'rb'))
    proj = d['proj']; tokids = d['tokids']; model = d['model']
    tok = AutoTokenizer.from_pretrained(MODEL_ALIASES.get(model, model))

    for di, name in enumerate(['d1', 'd2']):
        p = proj[:, di]
        print(f'\n=== {name}: n={len(p)}, range [{p.min():.1f}, {p.max():.1f}], '
              f'mean {p.mean():.2f}, std {p.std():.2f} ===')
        for thr in [100, 50, 20, 10, 5]:
            n = int((np.abs(p) > thr).sum())
            print(f'  |proj| > {thr:4d}: {n:6d} positions ({100*n/len(p):.3f}%)')
        # extreme negative positions (the heavy tail), sorted
        order = np.argsort(p)[:15]
        print(f'  most-extreme (negative) positions:')
        for idx in order:
            print(f'    {p[idx]:8.1f}  tok={tok.decode([int(tokids[idx])])!r}')

    # overlap of the extreme sets between d1 and d2
    for thr in [50, 20, 10]:
        e1 = set(np.where(np.abs(proj[:, 0]) > thr)[0])
        e2 = set(np.where(np.abs(proj[:, 1]) > thr)[0])
        inter = len(e1 & e2)
        print(f'\n|proj|>{thr}: d1={len(e1)}, d2={len(e2)}, shared positions={inter} '
              f'({100*inter/max(1,min(len(e1),len(e2))):.0f}% of smaller set)')

    # token-type composition of the |proj|>10 outlier set (both directions)
    for di, name in enumerate(['d1', 'd2']):
        idx = np.where(np.abs(proj[:, di]) > 10)[0]
        c = Counter(tok.decode([int(t)]) for t in tokids[idx])
        print(f'\n{name} |proj|>10 token types: {c.most_common(15)}')


if __name__ == '__main__':
    main()
