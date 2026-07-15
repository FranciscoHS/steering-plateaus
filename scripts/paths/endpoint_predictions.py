#!/usr/bin/env python3
"""
Check next-token predictions at the interpolation endpoints.

Hypothesis (Francisco): big->in has a plateau->boundary->plateau structure
because the next-token prediction changes a lot (an adjective ends the clause,
"in" demands an object), whereas big->{synonyms,antonyms} stay in the same
"plateau" because the prediction barely changes.

This script forwards each real prompt "<prefix> <word>" and looks at the
last-token next-token distribution: top-k tokens, and the divergence from the
anchor word's distribution. No interpolation needed — these ARE the endpoints
(prefixes are token-identical across words).

Usage:
    python scripts/paths/endpoint_predictions.py --model qwen3-0.6b \
        --prefix "The house was" --anchor big \
        --words large huge enormous small tiny in
"""

import argparse
import os

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def next_token_dist(model, prompt):
    with torch.no_grad():
        logits = model(prompt)          # [1, seq, vocab]
    return F.log_softmax(logits[0, -1, :].float(), dim=-1)   # log-probs [vocab]


def topk_str(model, log_probs, k=10):
    p = log_probs.exp()
    tp, ti = torch.topk(p, k)
    return [(model.to_string(i.item()), round(v.item(), 4)) for v, i in zip(tp, ti)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-0.6b')
    ap.add_argument('--prefix', default='The house was')
    ap.add_argument('--anchor', default='big')
    ap.add_argument('--words', nargs='+',
                    default=['large', 'huge', 'enormous', 'small', 'tiny', 'in'])
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--out', default='results/interpolation_flips/endpoint_predictions.txt')
    args = ap.parse_args()

    model = HookedTransformer.from_pretrained(args.model, device='cuda')
    model.eval()

    words = [args.anchor] + [w for w in args.words if w != args.anchor]
    dists = {w: next_token_dist(model, f"{args.prefix} {w}") for w in words}
    anchor_lp = dists[args.anchor]
    anchor_p = anchor_lp.exp()

    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f"model={args.model}  prefix={args.prefix!r}  anchor={args.anchor!r}\n")
    emit(f"{'word':10s} {'KL(anchor||w)':>13s} {'KL(w||anchor)':>13s} {'JS':>7s}  top-{args.k} next tokens")
    for w in words:
        lp = dists[w]
        p = lp.exp()
        kl_aw = (anchor_p * (anchor_lp - lp)).sum().item()        # KL(anchor || w)
        kl_wa = (p * (lp - anchor_lp)).sum().item()               # KL(w || anchor)
        m = 0.5 * (anchor_p + p)
        logm = m.clamp_min(1e-12).log()
        js = 0.5 * (anchor_p * (anchor_lp - logm)).sum().item() \
           + 0.5 * (p * (lp - logm)).sum().item()                 # Jensen-Shannon
        tk = topk_str(model, lp, args.k)
        tk_s = ' '.join(f"{t!r}:{v}" for t, v in tk)
        marker = '  (anchor)' if w == args.anchor else ''
        emit(f"{w:10s} {kl_aw:13.3f} {kl_wa:13.3f} {js:7.3f}  {tk_s}{marker}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f"\nsaved: {args.out}")


if __name__ == '__main__':
    main()
