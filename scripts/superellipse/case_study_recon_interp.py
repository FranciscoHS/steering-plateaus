#!/usr/bin/env python3
"""Case study: find a held-out activation that our dictionary reconstructs
relatively well (OMP), then interpret the atoms that constitute it.

For each held FineWeb activation (last-token resid at layer L):
  - OMP up to K atoms on the dictionary; record rel. recon error + the selected
    atoms and their least-squares coefficients.
Pick the top-reconstructed activations. For the best one, decode its context and
interpret its top atoms (by |coeff|) two ways:
  - corpus max-activating contexts (cosine of a FineWeb token corpus onto the atom)
  - logit-lens: top promoted/suppressed tokens (W_U @ ln_f(atom)).
Then we can eyeball whether the constituent atoms match the activation's content.

Usage:
    python scripts/superellipse/case_study_recon_interp.py \\
        --dict_path /tmp/snap1100.pt --model qwen3-1.7b --layer 14 \\
        --held_seed 2024 --n_held 2000 --seq_len 10 \\
        --K 32 --top_atoms 10 --n_best 3 \\
        --n_corpus 2000 --corpus_seq_len 64 --out .../case_study
"""
import argparse
import os
import sys
import pickle
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import (load_model, _get_blocks, _get_lm_head,
                       _get_final_norm)
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)


def omp(a, D, K):
    """OMP of a [d] onto unit dictionary D [N,d]. Returns (sel_idx, coeffs,
    rel_err_sq) where rel_err_sq=||a-â||^2/||a||^2 at K atoms."""
    r = a.clone()
    sel = []
    for _ in range(K):
        proj = D @ r
        if sel:
            proj[sel] = 0.0
        j = int(proj.abs().argmax())
        sel.append(j)
        # least squares of a onto D[sel]
        Dsel = D[sel]                                  # [k, d]
        G = Dsel @ Dsel.T
        rhs = Dsel @ a
        coeffs = torch.linalg.solve(G, rhs)
        r = a - coeffs @ Dsel
    rel = (r @ r) / (a @ a)
    return sel, coeffs, float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--n_held', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--K', type=int, default=32, help='OMP atoms')
    ap.add_argument('--top_atoms', type=int, default=10,
                    help='# of top-|coeff| atoms to interpret for the best act')
    ap.add_argument('--n_best', type=int, default=3,
                    help='# of best-reconstructed activations to report')
    ap.add_argument('--n_corpus', type=int, default=2000)
    ap.add_argument('--corpus_seq_len', type=int, default=64)
    ap.add_argument('--corpus_seed', type=int, default=123)
    ap.add_argument('--ctx_left', type=int, default=12)
    ap.add_argument('--topk_corpus', type=int, default=12)
    ap.add_argument('--topk_tokens', type=int, default=15)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}

    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
    N, d_model = D.shape
    D_dev = D.to(device)
    print(f'Dictionary: {N} atoms (n_done={blob.get("n_done")})', flush=True)

    # Held set (regenerated to match held_acts.pt: same seed/seq_len) + tokens.
    tok_list = load_fineweb_fixed_length(args.n_held, tokenizer,
                                         seq_len=args.seq_len, seed=args.held_seed)
    held_tok = torch.stack(tok_list)                   # [n_held, T]
    _, acts = extract_context_and_activation(model, tok_list, args.layer, device)
    acts = acts.to(device=device, dtype=torch.float32)  # [n_held, d]
    print(f'Held: {acts.shape[0]} acts, OMP K={args.K}', flush=True)

    # OMP every held activation.
    rels, sels, coefs = [], [], []
    for i in range(acts.shape[0]):
        sel, coeffs, rel = omp(acts[i], D_dev, args.K)
        rels.append(rel); sels.append(sel); coefs.append(coeffs.cpu())
    rels = np.array(rels)
    order = np.argsort(rels)                            # best (lowest err) first
    print(f'rel_err^2 at K={args.K}: best={rels[order[0]]:.4f} '
          f'median={np.median(rels):.4f} worst={rels.max():.4f}', flush=True)

    # ---- Corpus for atom interpretation (cosine max-activating) ----
    print(f'Capturing corpus acts ({args.n_corpus} seqs '
          f'seq_len={args.corpus_seq_len})', flush=True)
    ctok = torch.stack(load_fineweb_fixed_length(
        args.n_corpus, tokenizer, seq_len=args.corpus_seq_len,
        seed=args.corpus_seed))
    Nc, Tc = ctok.shape
    cchunks, meta = [], []
    for i in range(0, Nc, 32):
        bt = ctok[i:i + 32].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        h = h[:, 1:, :].float()
        b = h.shape[0]
        cchunks.append(h.reshape(b * (Tc - 1), d_model).cpu())
        for s in range(b):
            for p in range(1, Tc):
                meta.append((i + s, p))
    cacts = torch.cat(cchunks, 0).to(device)
    cnorm = cacts.norm(dim=-1).clamp_min(1e-9)
    meta = np.array(meta)

    def decode_corpus_hit(s_idx, pos):
        lo = max(1, pos - args.ctx_left)
        ids = ctok[s_idx]
        left = tokenizer.decode(ids[lo:pos]).replace('\n', '\\n')
        tgt = tokenizer.decode(ids[pos:pos + 1]).replace('\n', '\\n')
        return f'…{left}⟦{tgt}⟧'

    lm_head = _get_lm_head(model); ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float().to(device)

    def interp_atom(k):
        d = D_dev[k]
        cos = (cacts @ d) / cnorm
        top = torch.topk(cos, args.topk_corpus).indices.cpu().numpy()
        corpus = [(float(cos[m]), decode_corpus_hit(int(meta[m, 0]), int(meta[m, 1])))
                  for m in top]
        with torch.no_grad():
            lnd = ln_f(d.unsqueeze(0).to(next(ln_f.parameters()).dtype)).float().squeeze(0)
        logits = (W_U @ lnd)
        promo = torch.topk(logits, args.topk_tokens).indices.cpu().numpy()
        supp = torch.topk(-logits, args.topk_tokens).indices.cpu().numpy()
        prom = [tokenizer.decode([int(t)]).replace('\n', '\\n') for t in promo]
        sup = [tokenizer.decode([int(t)]).replace('\n', '\\n') for t in supp]
        return corpus, prom, sup

    # ---- Report ----
    lines = []
    def w(s=''):
        lines.append(s); print(s, flush=True)

    w(f'Case study: {N}-atom dict, OMP K={args.K}, model {args.model} L{args.layer}')
    w('=' * 78)
    for rank in range(min(args.n_best, len(order))):
        idx = int(order[rank])
        seq = tokenizer.decode(held_tok[idx][1:]).replace('\n', '\\n')
        last = tokenizer.decode(held_tok[idx][-1:]).replace('\n', '\\n')
        sel = sels[idx]; cf = coefs[idx]
        w(f'\n##### Best #{rank+1}: held act {idx}  '
          f'rel_err^2={rels[idx]:.4f}  (energy explained {1-rels[idx]:.1%}) #####')
        w(f'  context (10 tok): "{seq}"')
        w(f'  activation = last-token resid after "{last}"')
        ordc = np.argsort(-cf.abs().numpy())
        w(f'  selected atoms (top {min(args.top_atoms,len(sel))} by |coeff|):')
        for j in ordc[:args.top_atoms]:
            w(f'    atom {sel[j]:5d}  coeff={float(cf[j]):+8.2f}')
        # interpret the top atoms only for the #1 best act
        if rank == 0:
            w(f'\n  --- interpretation of top {args.top_atoms} atoms ---')
            for j in ordc[:args.top_atoms]:
                k = sel[j]
                corpus, prom, sup = interp_atom(k)
                w(f'\n  atom {k} (coeff {float(cf[j]):+.2f}):')
                w(f'    promoted tokens: {prom}')
                w(f'    suppressed     : {sup}')
                w(f'    top corpus contexts (cosine):')
                for sc, txt in corpus:
                    w(f'      {sc:.3f}  {txt}')

    with open(os.path.join(args.out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines))
    with open(os.path.join(args.out, 'data.pkl'), 'wb') as f:
        pickle.dump({'rels': rels, 'sels': sels,
                     'coefs': [c.numpy() for c in coefs],
                     'order': order, 'args': vars(args)}, f)
    w(f'\nSaved {os.path.join(args.out, "report.txt")}')


if __name__ == '__main__':
    main()
