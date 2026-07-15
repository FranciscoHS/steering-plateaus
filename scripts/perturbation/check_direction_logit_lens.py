#!/usr/bin/env python3
"""Logit-lens a direction: W_U @ ln_f(d). Tells us which tokens +d promotes,
so we can check the SIGN of a contrastive direction (does +d point toward the
language / away from English?)."""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import torch
from src.model import load_model, _get_lm_head, _get_final_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir_pkls', nargs='+', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--topk', type=int, default=15)
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    W_U = _get_lm_head(model).weight.detach().float().to(device)   # [vocab, d]
    ln_f = _get_final_norm(model)
    ln_dtype = next(ln_f.parameters()).dtype

    for p in args.dir_pkls:
        d = pickle.load(open(p, 'rb'))['directions'][0].float()
        d = d / d.norm()
        with torch.no_grad():
            lg = (ln_f(d.to(device, ln_dtype)).float() @ W_U.T).cpu()
        pos = torch.topk(lg, args.topk).indices
        neg = torch.topk(-lg, args.topk).indices
        dec = lambda ids: ' '.join(repr(tokenizer.decode([int(i)])) for i in ids)
        print(f'\n=== {os.path.basename(p)} ===')
        print(f'  +d promotes: {dec(pos)}')
        print(f'  +d suppresses: {dec(neg)}')


if __name__ == '__main__':
    main()
