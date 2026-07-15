#!/usr/bin/env python3
"""
Quick next-token prediction check for arbitrary full prompts.

Used to find pairs of prompts that share a continuation but reach it via
different computational pathways (e.g. "Paris is the capital of" -> France via
a capital-of relation; "Spain neighbors" -> France via geography). These are
candidate counterexamples for "boundary iff next-token prediction differs".

Forwards each prompt and prints the top-k next tokens + their probs. Also prints
the pairwise JS divergence between every prompt's next-token distribution so we
can see which pairs actually match.

Usage:
    python scripts/paths/check_prompt_predictions.py --model qwen3-0.6b \
        --prompts "Paris is the capital of" "Spain neighbors" "Madrid is the capital of"
"""

import argparse
import os

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-0.6b')
    ap.add_argument('--prompts', nargs='+', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', default='results/interpolation_flips/prompt_predictions.txt')
    args = ap.parse_args()

    model = HookedTransformer.from_pretrained(args.model, device='cuda')
    model.eval()

    lps = []
    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f"model={args.model}\n")
    for prompt in args.prompts:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(toks)
        lp = F.log_softmax(logits[0, -1, :].float(), dim=-1)
        lps.append(lp)
        p = lp.exp()
        tp, ti = torch.topk(p, args.k)
        tk = ' '.join(f"{model.to_string(i.item())!r}:{v.item():.3f}" for v, i in zip(tp, ti))
        emit(f"[{len(toks[0])} tok] {prompt!r}")
        emit(f"    -> {tk}")
        emit()

    # Pairwise JS divergence between next-token distributions
    emit("pairwise JS divergence (0 = identical prediction):")
    n = len(args.prompts)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = lps[i].exp(), lps[j].exp()
            m = 0.5 * (pi + pj)
            logm = m.clamp_min(1e-12).log()
            js = 0.5 * (pi * (lps[i] - logm)).sum().item() \
               + 0.5 * (pj * (lps[j] - logm)).sum().item()
            emit(f"    {js:.3f}   {args.prompts[i]!r}  vs  {args.prompts[j]!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f"\nsaved: {args.out}")


if __name__ == '__main__':
    main()
