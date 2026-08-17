#!/usr/bin/env python3
"""Add the fully-random-sequence group to the per-context measurements.

The `randtok` group in ctx_jacopt_perplexity.py keeps a real 9-token carrier
and randomizes only the final token. The older sweeps
(scripts/superellipse/eval_real_vs_random_acts.py --random_tokens) instead used
a sequence in which EVERY token is random, so the carrier is random too. Those
runs found random-token anchors nearly as flat as real ones, which the randtok
group here does not reproduce.

This script measures the old definition with the current pipeline, as group
`randseq`: n_randseq sequences of seq_len uniformly random tokens, each anchor
swept inside its OWN random carrier. Same three measurements as the other
groups: ||J||_F, next-token perplexity exp(H), and the 0..max_angle sweep of
final-hidden L2 and output KL.

Note the structural difference: every other group shares one real carrier
context per panel, while randseq anchors each carry their own random context.
That is inherent to the old definition, not a choice here.

Token ids are drawn from tokenizer.vocab_size, as the old script did (the
randtok group used model.config.vocab_size; the two differ by ~0.2% of ids).

Results are written back into ctx{i}.pt under groups['randseq'], and
summary.txt / ppl.png / sweep.png are regenerated. The summary now reports
small-angle columns as well, since every group has saturated by 60 degrees.
"""

import argparse
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from ctx_jacopt_perplexity import jac_norm_fd, next_token_stats, plateau_sweep

ORDER = ['jacopt', 'gauss', 'randtok', 'randseq', 'real']
COLORS = {'jacopt': 'C0', 'gauss': 'C2', 'randtok': 'C1', 'randseq': 'C4',
          'real': 'k'}


def measure_randseq(model, tokenizer, device, layer, seq_len, n_seq,
                    angles_rad, n_dirs, seed):
    """Each anchor is the last-token activation of its own random sequence."""
    g = torch.Generator().manual_seed(seed)
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, vocab, (n_seq, seq_len), generator=g).to(device)
    jac, ppl, p_max, top_ids, top_p, norms = [], [], [], [], [], []
    L2s, KLs = [], []
    for i in range(n_seq):
        with torch.no_grad():
            o = model(ids[i:i + 1], output_hidden_states=True)
        h = o.hidden_states[layer + 1].float()
        ctx = h[:, :-1, :]
        a = h[:, -1, :]
        with torch.no_grad():
            jac.append(jac_norm_fd(model, ctx, a, layer))
            st = next_token_stats(model, ctx, a, layer)
            L2, KL = plateau_sweep(model, ctx, a, layer, angles_rad, n_dirs,
                                   seed + i)
        ppl.append(st['ppl'])
        p_max.append(st['p_max'])
        top_ids.append(st['top_ids'])
        top_p.append(st['top_p'])
        norms.append(a.norm(dim=-1).cpu())
        L2s.append(L2)
        KLs.append(KL)
    return {'jac': torch.cat(jac).numpy(), 'ppl': torch.cat(ppl).numpy(),
            'p_max': torch.cat(p_max).numpy(),
            'top_ids': torch.cat(top_ids).numpy(),
            'top_p': torch.cat(top_p).numpy(),
            'norm': torch.cat(norms).numpy(),
            'sweep_L2': np.concatenate(L2s), 'sweep_KL': np.concatenate(KLs),
            'token_ids': ids.cpu().numpy()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--n_randseq', type=int, default=20)
    p.add_argument('--n_dirs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='results/intrinsic_dim/ctx_jacopt_ppl')
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.outdir)
                   if f.startswith('ctx') and f.endswith('.pt'))
    print(f'{len(files)} context files in {args.outdir}', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.float32)

    blobs = []
    for f in files:
        path = os.path.join(args.outdir, f)
        b = torch.load(path, map_location='cpu', weights_only=False)
        ci = b['ctx_id']
        if 'randseq' in b['groups']:
            print(f'ctx {ci}: randseq already present, skipping', flush=True)
            blobs.append(b)
            continue
        print(f'\n=== ctx {ci}: adding randseq ===', flush=True)
        gr = measure_randseq(model, tokenizer, device, args.layer,
                             args.seq_len, args.n_randseq,
                             np.deg2rad(b['angles_deg']), args.n_dirs,
                             args.seed + 100 * ci)
        b['groups']['randseq'] = gr
        i10 = int(np.argmin(abs(b['angles_deg'] - 10.0)))
        print(f'  randseq  n={args.n_randseq:3d}  '
              f'||x||={np.median(gr["norm"]):.2f}  '
              f'||J||={np.median(gr["jac"]):8.1f}  '
              f'ppl={np.median(gr["ppl"]):8.2f}  '
              f'L2@10deg={np.median(gr["sweep_L2"][:, :, i10]):8.2f}  '
              f'L2@60deg={np.median(gr["sweep_L2"][:, :, -1]):8.2f}   '
              f'(medians)', flush=True)
        torch.save(b, path)
        blobs.append(b)

    n = len(blobs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    for i, b in enumerate(blobs):
        ax = axes[0, i]
        for name in ORDER:
            if name == 'real' or name not in b['groups']:
                continue
            ax.hist(b['groups'][name]['ppl'], bins=25, histtype='step',
                    color=COLORS[name], label=name, lw=1.5)
        ax.axvline(b['groups']['real']['ppl'][0], color='k', ls='--',
                   label='real token')
        ax.set_xscale('log')
        ax.set_xlabel('perplexity exp(H)')
        ax.set_title(f'ctx {b["ctx_id"]}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'ppl.png'), dpi=150)

    # Sweeps on log-log axes: the plateau lives at small angles, and a linear
    # 0..60 axis hides it.
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.4), squeeze=False)
    for i, b in enumerate(blobs):
        for r, key in enumerate(['sweep_L2', 'sweep_KL']):
            ax = axes[r, i]
            for name in ORDER:
                if name not in b['groups']:
                    continue
                v = b['groups'][name][key]
                med = np.median(v.reshape(-1, v.shape[-1]), axis=0)
                ax.plot(b['angles_deg'][1:], med[1:], color=COLORS[name],
                        label=name)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('angle (deg)')
            ax.set_ylabel(key)
            ax.set_title(f'ctx {b["ctx_id"]} — {key} (median)', fontsize=9)
            ax.grid(True, alpha=0.3, which='both')
            if i == 0 and r == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'sweep.png'), dpi=150)

    ang = blobs[0]['angles_deg']
    i5, i10, i20 = (int(np.argmin(abs(ang - t))) for t in (5.0, 10.0, 20.0))
    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'{args.model} L{args.layer}  n_ctx={n}\nmedians per group\n')
        f.write('L2/KL columns are at 5, 10, 20 and 60 deg; by 60 deg every '
                'group has saturated\n')
        for b in blobs:
            f.write(f'\nctx {b["ctx_id"]}: {b["text"]!r}\n')
            f.write(f'{"group":8s} {"n":>4s} {"||x||":>7s} {"||J||":>8s} '
                    f'{"ppl":>8s} {"L2@5":>8s} {"L2@10":>8s} {"L2@20":>8s} '
                    f'{"L2@60":>8s} {"KL@10":>7s} {"KL@60":>7s}\n')
            for name in ORDER:
                if name not in b['groups']:
                    continue
                gr = b['groups'][name]
                L2 = gr['sweep_L2'].reshape(-1, gr['sweep_L2'].shape[-1])
                KL = gr['sweep_KL'].reshape(-1, gr['sweep_KL'].shape[-1])
                f.write(f'{name:8s} {len(gr["ppl"]):4d} '
                        f'{np.median(gr["norm"]):7.2f} '
                        f'{np.median(gr["jac"]):8.1f} '
                        f'{np.median(gr["ppl"]):8.2f} '
                        f'{np.median(L2[:, i5]):8.2f} '
                        f'{np.median(L2[:, i10]):8.2f} '
                        f'{np.median(L2[:, i20]):8.2f} '
                        f'{np.median(L2[:, -1]):8.2f} '
                        f'{np.median(KL[:, i10]):7.3f} '
                        f'{np.median(KL[:, -1]):7.3f}\n')

        f.write('\npooled over contexts\n')
        f.write(f'{"group":8s} {"L2@5":>8s} {"L2@10":>8s} {"L2@20":>8s} '
                f'{"L2@60":>8s} {"KL@10":>7s} {"KL@60":>7s}\n')
        for name in ORDER:
            rows = [b['groups'][name]['sweep_L2'].reshape(
                -1, len(ang)) for b in blobs if name in b['groups']]
            rowsk = [b['groups'][name]['sweep_KL'].reshape(
                -1, len(ang)) for b in blobs if name in b['groups']]
            if not rows:
                continue
            L2 = np.concatenate(rows)
            KL = np.concatenate(rowsk)
            f.write(f'{name:8s} {np.median(L2[:, i5]):8.2f} '
                    f'{np.median(L2[:, i10]):8.2f} '
                    f'{np.median(L2[:, i20]):8.2f} '
                    f'{np.median(L2[:, -1]):8.2f} '
                    f'{np.median(KL[:, i10]):7.3f} '
                    f'{np.median(KL[:, -1]):7.3f}\n')
    print(f'\nSaved {args.outdir}/summary.txt, ppl.png, sweep.png')


if __name__ == '__main__':
    main()
