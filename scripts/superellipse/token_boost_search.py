#!/usr/bin/env python3
"""Token-anchored atom search: which dictionary atom most boosts a given
token under the logit lens?

For every atom d_k compute W_U @ ln_f(d_k) (same as the dashboard's logit
column) and, for each target token, list the atoms with the largest positive
boost AND the largest negative (an atom used with a negative OMP coefficient
boosts via its negative side). Tokens are encoded with the model tokenizer
and must be single tokens (multi-token strings are reported and skipped).

    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/token_boost_search.py \\
        --dict_path .../dictionary.pt --model qwen3-1.7b \\
        --tokens " woman" " women" " she" " France" " Paris" \\
        --out .../token_boost
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch

from src.model import load_model, _get_lm_head, _get_final_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--tokens', nargs='+', required=True)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model}', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    K = D.shape[0]
    print(f'Dictionary: {K} atoms', flush=True)

    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float()                  # [V, d]
    ln_dtype = next(ln_f.parameters()).dtype
    with torch.no_grad():
        ln_D = ln_f(D.to(ln_dtype)).float()                # [K, d]

    results = {}
    lines = ['# Token-anchored atom search (logit boost W_U·ln_f(d))', '']
    for ts in args.tokens:
        ids = tokenizer.encode(ts, add_special_tokens=False)
        if len(ids) != 1:
            print(f'SKIP {ts!r}: encodes to {len(ids)} tokens {ids}',
                  flush=True)
            lines.append(f'## {ts!r} — SKIPPED (multi-token: {ids})')
            continue
        tid = ids[0]
        with torch.no_grad():
            boost = ln_D @ W_U[tid].to(device)             # [K]
        top = boost.topk(args.topm)
        bot = (-boost).topk(args.topm)
        results[ts] = {'token_id': tid,
                       'top_atoms': top.indices.cpu().tolist(),
                       'top_boost': top.values.cpu().tolist(),
                       'bot_atoms': bot.indices.cpu().tolist(),
                       'bot_boost': (-bot.values).cpu().tolist(),
                       'boost_mean': float(boost.mean()),
                       'boost_std': float(boost.std())}
        r = results[ts]
        lines.append(f'## {ts!r} (id {tid})  '
                     f'[atom boost mean={r["boost_mean"]:.3f} '
                     f'std={r["boost_std"]:.3f}]')
        lines.append('- top + : ' + '  '.join(
            f'{a}:{b:+.2f}' for a, b in zip(r['top_atoms'], r['top_boost'])))
        lines.append('- top − : ' + '  '.join(
            f'{a}:{b:+.2f}' for a, b in zip(r['bot_atoms'], r['bot_boost'])))
        lines.append('')
        print('\n'.join(lines[-4:]), flush=True)

    with open(os.path.join(args.out, 'token_boost.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'results': results}, f)
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {os.path.join(args.out, 'report.md')}", flush=True)


if __name__ == '__main__':
    main()
