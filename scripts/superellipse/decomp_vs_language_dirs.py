#!/usr/bin/env python3
"""Cosine alignment of decomposition features d_k with contrastive
language and code direction vectors at the same layer.

Extracts any missing language/code directions via fsec-anonymous's
dirlib.extract_all (writes them into the scaffold pkl in place), then
reports the cosine matrix d_k vs each contrastive direction.

Must be run with fsec-anonymous as CWD (for its scripts.lib imports).
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import torch


LANG_KEYS = ['Portuguese', 'Chinese', 'Spanish', 'French', 'German', 'Dutch',
             'Italian', 'Japanese', 'Russian', 'Arabic']
CODE_KEYS = ['Python', 'JavaScript', 'Java', 'Rust', 'Haskell', 'Cpp', 'Go',
             'Lisp', 'TypeScript']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True)
    ap.add_argument('--scaffold_pkl',
                    default='results/directions/dirs_gemma_L15.pkl')
    ap.add_argument('--target', default='gemma')
    ap.add_argument('--layer', type=int, default=15)
    ap.add_argument('--names', nargs='+', default=None,
                    help='which contrastive directions to compare against; '
                         'defaults to all languages + codes')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    if args.names is None:
        args.names = LANG_KEYS + CODE_KEYS

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features']
    if not isinstance(feats, torch.Tensor):
        feats = torch.from_numpy(feats)
    feats = feats.float()
    K, hidden = feats.shape
    feats_unit = feats / feats.norm(dim=-1, keepdim=True)
    print(f'{K} features, hidden={hidden}', flush=True)

    with open(args.scaffold_pkl, 'rb') as f:
        scaf = pickle.load(f)
    scaf_dirs = scaf['directions']
    missing = [n for n in args.names if n not in scaf_dirs]
    if missing:
        print(f'extending scaffold with {len(missing)} missing: {missing}',
              flush=True)
        from src.model import load_model
        from scripts.lib import directions as dirlib
        dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                     'float32': torch.float32}
        model_name = 'google/gemma-2-9b' if args.target == 'gemma' else None
        if model_name is None:
            raise ValueError(f'No model mapping for target={args.target}')
        model, tokenizer, device = load_model(model_name,
                                              dtype=dtype_map[args.dtype])
        dirlib.extract_all(model, tokenizer, device, args.target,
                           args.layer, names=missing)
        with open(args.scaffold_pkl, 'rb') as f:
            scaf = pickle.load(f)
        scaf_dirs = scaf['directions']
        still_missing = [n for n in args.names if n not in scaf_dirs]
        if still_missing:
            raise KeyError(f'still missing: {still_missing}')

    # Build matrix (n_names, h) of unit-norm scaffold directions.
    rows = []
    for n in args.names:
        v = scaf_dirs[n].float()
        v = v / v.norm()
        rows.append(v)
    scaf_mat = torch.stack(rows)                                   # (N, h)
    cos = feats_unit @ scaf_mat.T                                  # (K, N)

    # Also include the scaffold/decomp self-overlap.
    print('\ncosine matrix (rows = d_k, cols = contrastive direction):',
          flush=True)
    hdr = '       ' + '  '.join(f'{n:>10s}' for n in args.names)
    print(hdr, flush=True)
    for k_idx in range(K):
        line = f'  d{k_idx+1}:  ' + '  '.join(f'{float(cos[k_idx, j]):+10.4f}'
                                              for j in range(len(args.names)))
        print(line, flush=True)

    out_txt = os.path.join(args.out_dir, 'decomp_vs_language_cos.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Cosine of decomposition features vs contrastive directions\n')
        f.write(f'  results_pkl: {args.results_pkl}\n')
        f.write(f'  scaffold_pkl: {args.scaffold_pkl}\n')
        f.write(f'  target={args.target}  layer={args.layer}\n')
        f.write(f'  names: {args.names}\n')
        f.write('=' * 90 + '\n\n')
        f.write(hdr + '\n')
        for k_idx in range(K):
            line = (f'  d{k_idx+1}:  ' +
                    '  '.join(f'{float(cos[k_idx, j]):+10.4f}'
                              for j in range(len(args.names))))
            f.write(line + '\n')
        f.write('\nMax-|cos| per d_k:\n')
        for k_idx in range(K):
            j = int(cos[k_idx].abs().argmax().item())
            f.write(f'  d{k_idx+1}: {args.names[j]:>12s}  '
                    f'cos={float(cos[k_idx, j]):+.4f}\n')
        f.write('\nTop 3 per d_k (by |cos|):\n')
        for k_idx in range(K):
            order = torch.argsort(cos[k_idx].abs(), descending=True)
            top3 = [(args.names[int(o)], float(cos[k_idx, int(o)]))
                    for o in order[:3]]
            f.write(f'  d{k_idx+1}: ' +
                    ', '.join(f'{n}({v:+.3f})' for n, v in top3) + '\n')

    out_pkl = os.path.join(args.out_dir, 'decomp_vs_language_cos.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'names': args.names,
            'cos': cos.numpy(),
            'feature_labels': [f'd{k+1}' for k in range(K)],
        }, f)
    print(f'\nwrote {out_txt}', flush=True)
    print(f'wrote {out_pkl}', flush=True)


if __name__ == '__main__':
    main()
