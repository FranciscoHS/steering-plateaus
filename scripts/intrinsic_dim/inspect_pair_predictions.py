#!/usr/bin/env python3
"""Next-token predictions at the endpoints of one slerp path.

Given an anchor index i and a candidate index j from slerp_jacopt_to_reals.py,
print the top-k next-token distribution for:

  A_own    anchor i's real activation in anchor i's own context
  B_own    candidate j's real activation in candidate j's own context
  B_in_A   candidate j's activation transplanted into anchor i's context —
           this is the actual t=1 endpoint of the path, and the thing the
           relative-distance metric saw
  J_in_A   the jac-opt point of anchor i, in anchor i's context (the t=0 end
           of the jacopt condition), when --with_jacopt is passed

Output: printed, and written to <outdir>/pair_predictions_<i>_<j>.txt
"""

import argparse
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import torch
import torch.nn.functional as F

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length


def report(tok, name, seq_text, logits, topk):
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    H = -(p * logp).sum()
    t = p.topk(topk)
    lines = [f'{name}   context {seq_text!r}',
             f'    entropy {H.item():.3f} nats, perplexity exp(H) '
             f'{H.exp().item():.2f}']
    for prob, idx in zip(t.values.tolist(), t.indices.tolist()):
        lines.append(f'      {prob:6.3f}  {tok.decode([idx])!r}')
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--anchor', type=int, required=True)
    p.add_argument('--cand', type=int, required=True)
    p.add_argument('--n_real', type=int, default=200)
    p.add_argument('--real_seed', type=int, default=44)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--with_jacopt', action='store_true')
    p.add_argument('--jac_pt', default='results/intrinsic_dim/'
                                       'ctx_sweep_oldconfig_gpt2xl/jac_optimized.pt')
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/slerp_jacopt_to_reals_gpt2xl')
    args = p.parse_args()

    model, tok, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype

    blob = torch.load(args.jac_pt, map_location='cpu', weights_only=False)
    a_ids = blob['tokens'][args.anchor]
    cand_tokens = load_fineweb_fixed_length(args.n_real, tok,
                                            seq_len=args.seq_len,
                                            seed=args.real_seed,
                                            cache_dir='/workspace/fineweb_cache')
    b_ids = cand_tokens[args.cand]

    out = []
    hs = {}
    for tag, ids in [('A', a_ids), ('B', b_ids)]:
        with torch.no_grad():
            o = model(ids.unsqueeze(0).to(device), output_hidden_states=True)
        hs[tag] = {'ctx': o.hidden_states[args.layer + 1][:, :-1, :].float(),
                   'act': o.hidden_states[args.layer + 1][0, -1, :].float(),
                   'logits': o.logits[0, -1, :],
                   'text': tok.decode([int(x) for x in ids])}

    out += report(tok, 'A_own ', hs['A']['text'], hs['A']['logits'], args.topk)
    out.append('')
    out += report(tok, 'B_own ', hs['B']['text'], hs['B']['logits'], args.topk)
    out.append('')

    ctx_a = hs['A']['ctx'].to(device, model_dtype)
    with torch.no_grad():
        lg = forward_from_layer(
            model, ctx_a, hs['B']['act'][None, None].to(device, model_dtype),
            args.layer).float()
    out += report(tok, 'B_in_A', hs['A']['text'] + '   <- B act transplanted',
                  lg[0], args.topk)
    out.append('')

    if args.with_jacopt:
        xj = blob['X_opt'][args.anchor].float()
        with torch.no_grad():
            lg = forward_from_layer(
                model, ctx_a, xj[None, None].to(device, model_dtype),
                args.layer).float()
        out += report(tok, 'J_in_A', hs['A']['text'] + '   <- jac-opt point',
                      lg[0], args.topk)

    txt = '\n'.join(out)
    print(txt)
    path = os.path.join(args.outdir,
                        f'pair_predictions_{args.anchor}_{args.cand}.txt')
    with open(path, 'w') as f:
        f.write(txt + '\n')
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
