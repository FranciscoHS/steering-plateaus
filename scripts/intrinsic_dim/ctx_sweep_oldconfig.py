#!/usr/bin/env python3
"""The usual perturbation sweep, run at the OLD experiment's configuration.

Purpose: settle whether random-token and random-sequence anchors plateau. The
per-context run (ctx_jacopt_perplexity.py) used seq_len 10, the final block and
5 real anchors, and found them ~2.2-2.6x steeper than real. The earlier sweep
(results/real_vs_random_acts/qwen3-1.7b_L2_jacopt_randtok_gaussshift) used
seq_len 5, layer 26 and 30 anchors, and found ~1.3x.

Here every group is measured under the OLD settings, each anchor in its own
context, with median and inter-quartile shading:

  real      30 FineWeb seq_len-5 last-token activations
  randtok   the same 30 contexts with the FINAL token replaced by a random one
  randseq   30 sequences in which EVERY token is random
  gauss     cov-matched Gaussian vectors (fit on a pool of real activations)
            placed at the last position of the 30 real contexts

The old run's published medians for real and randseq are overlaid as dashed
reference lines, so any drift from the original is visible in the same figure.

jac-opt is not included: at one context per anchor it needs 30 separate
optimizations and takes about an hour. Pass --with_jacopt to add it.

Output: results/intrinsic_dim/ctx_sweep_oldconfig/{results.pkl,sweep.png}
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (load_model, _get_blocks, _is_rope_model,
                       _get_position_embeddings, _run_block)
from src.data import load_fineweb_fixed_length

# Published medians from the old run, for the overlay.
OLD = {'real': {10: 90.6, 20: 181.8, 40: 453.8, 60: 1182.0, 90: 1920.4},
       'randseq': {10: 117.9, 20: 240.7, 40: 579.7, 60: 1212.3, 90: 1787.0}}
COLORS = {'real': 'k', 'randtok': 'C1', 'randseq': 'C4', 'gauss': 'C2',
          'jacopt': 'C0'}
ORDER = ['jacopt', 'real', 'randtok', 'randseq', 'gauss']


def forward_hidden(model, hidden, layer, end_layer):
    """Run blocks (layer, end_layer] on a full [n, seq_len, d] hidden batch.

    Rows are independent under causal attention, so each row may carry its own
    context.
    """
    pos_emb = _get_position_embeddings(model, hidden) if _is_rope_model(model) else None
    blocks = _get_blocks(model)
    sel = blocks[layer + 1:] if end_layer is None else blocks[layer + 1:end_layer + 1]
    for block in sel:
        hidden = _run_block(block, hidden, model, pos_emb)
    return hidden[:, -1, :]


def forward_to_layer(model, ctx, points, layer, end_layer):
    """Run blocks (layer, end_layer] with `points` at the last position of the
    single context `ctx`."""
    n = points.shape[0]
    hidden = torch.cat([ctx.expand(n, -1, -1), points.unsqueeze(1)], dim=1)
    return forward_hidden(model, hidden, layer, end_layer)


def acts_from_token_lists(model, device, layer, token_lists):
    """Per sequence: its own carrier context and its last-token activation."""
    ctxs, acts = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(tids, output_hidden_states=True)
        h = o.hidden_states[layer + 1]
        ctxs.append(h[:, :-1, :].float())
        acts.append(h[:, -1, :].float())
    return ctxs, acts


def sweep(model, ctx, a, layer, end_layer, angles_rad, n_dirs, seed, dtype):
    """Norm-preserving rotation of one anchor in its own context."""
    a32 = a.float().reshape(-1)
    d = a32.shape[0]
    R = a32.norm()
    a_u = a32 / R
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=a32.device)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=a32.device)
    ref = forward_to_layer(model, ctx.to(dtype), a32[None].to(dtype), layer,
                           end_layer).float()
    out = np.zeros((n_dirs, len(angles_rad)), dtype=np.float32)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for j in range(n_dirs):
        u = torch.randn(d, generator=g).to(a32.device)
        u_o = F.normalize(u - (u @ a_u) * a_u, dim=0)
        pts = R * (c[:, None] * a_u[None] + s[:, None] * u_o[None])
        h = forward_to_layer(model, ctx.to(dtype), pts.to(dtype), layer,
                             end_layer).float()
        out[j] = (h - ref).norm(dim=-1).cpu().numpy()
    return out


def optimize_batch(model, x0, ctxs, layer, end_layer, n_steps, lr, lam,
                   n_probes, chunk):
    """min ||J||_F + lam*||x-x0||^2, one point per context, all at once.

    Same objective and hyperparameters as gpt2xl_shift_replication.py, but each
    row carries its own context, so the anchors are optimized in exactly the
    contexts they are later swept in. `end_layer` sets the map the Jacobian is
    taken of; the earlier runs used the final block.
    """
    N, d = x0.shape
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for step in range(n_steps):
        opt.zero_grad()
        tot_j, tot_d = 0.0, 0.0
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            xc = x[st:ed]
            hidden = torch.cat([ctxs[st:ed], xc.unsqueeze(1)], dim=1)
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.MATH):
                f = forward_hidden(model, hidden, layer, end_layer)
            frob_sq = torch.zeros(ed - st, device=x.device)
            for _ in range(n_probes):
                u = torch.randn(ed - st, d, device=x.device, dtype=f.dtype)
                gx = torch.autograd.grad((f * u).sum(), xc, create_graph=True,
                                         retain_graph=True)[0]
                frob_sq = frob_sq + (gx.float() ** 2).sum(-1)
            jac_loss = torch.sqrt(frob_sq / n_probes + 1e-8)
            dist_loss = lam * ((xc - x0[st:ed]).float() ** 2).sum(-1)
            ((jac_loss + dist_loss).sum() / N).backward()
            tot_j += jac_loss.sum().item()
            tot_d += dist_loss.sum().item()
        opt.step()
        if step % 20 == 0:
            move = (x.detach() - x0).float().norm(dim=-1).mean().item()
            print(f'    step {step}: jac={tot_j/N:.1f} dist={tot_d/N:.2f} '
                  f'move={move:.2f}', flush=True)
    return x.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-1.7b')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--measure_layer', type=int, default=26)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--n_act', type=int, default=30)
    p.add_argument('--n_pool', type=int, default=2000,
                   help='real activations used to fit the Gaussian mean/cov')
    p.add_argument('--n_dirs', type=int, default=10)
    p.add_argument('--max_angle', type=float, default=90.0)
    p.add_argument('--n_steps', type=int, default=46)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rand_tok_seed', type=int, default=1234)
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--with_jacopt', action='store_true')
    p.add_argument('--opt_steps', type=int, default=200)
    p.add_argument('--opt_lr', type=float, default=0.1)
    p.add_argument('--opt_lam', type=float, default=1e-3)
    p.add_argument('--opt_probes', type=int, default=5)
    p.add_argument('--opt_chunk', type=int, default=10,
                   help='points per double-backprop chunk (memory bound)')
    p.add_argument('--opt_end_layer', type=int, default=-1,
                   help='block the optimized Jacobian ends at; -1 means use '
                        '--measure_layer, so the objective matches the plot. '
                        'The earlier runs used the final block.')
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/ctx_sweep_oldconfig')
    p.add_argument('--zoom', action='store_true',
                   help='add a second panel zoomed to 0-20 deg')
    p.add_argument('--show_old', action='store_true',
                   help='overlay the earlier run\'s published medians')
    p.add_argument('--replot', action='store_true',
                   help='redraw sweep.png from the saved results.pkl only')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.replot:
        with open(os.path.join(args.outdir, 'results.pkl'), 'rb') as f:
            make_plot(pickle.load(f), args)
        return

    model, tokenizer, device = load_model(
        args.model, dtype={'bfloat16': torch.bfloat16,
                           'float16': torch.float16,
                           'float32': torch.float32}[args.dtype])
    # load_model ignores `dtype` for GPT-2 and always loads float32, so take
    # the dtype the weights actually have. eval_real_vs_random_acts.py does the
    # same, which is why its gpt2-xl runs were float32 despite --dtype bfloat16.
    dtype = next(model.parameters()).dtype
    print(f'model weights are {dtype}', flush=True)
    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    # ── anchors, each in its own context ────────────────────────────────
    real_tokens = load_fineweb_fixed_length(args.n_act, tokenizer,
                                            seq_len=args.seq_len,
                                            seed=args.seed)
    ctxs, real_acts = acts_from_token_lists(model, device, args.layer,
                                            real_tokens)

    g = torch.Generator().manual_seed(args.rand_tok_seed)
    vocab = tokenizer.vocab_size
    # randtok: the same contexts, only the final token replaced
    randtok_tokens = [torch.cat([t[:-1], torch.randint(0, vocab, (1,),
                                                       generator=g)])
                      for t in real_tokens]
    _, randtok_acts = acts_from_token_lists(model, device, args.layer,
                                            randtok_tokens)
    # randseq: every token random, so the carrier is random too
    randseq_tokens = [torch.randint(0, vocab, (args.seq_len,), generator=g)
                      for _ in range(args.n_act)]
    randseq_ctxs, randseq_acts = acts_from_token_lists(model, device,
                                                       args.layer,
                                                       randseq_tokens)

    # gauss: cov-matched to a pool of real activations, dropped into the real
    # contexts
    pool_tokens = load_fineweb_fixed_length(args.n_pool, tokenizer,
                                            seq_len=args.seq_len,
                                            seed=args.seed + 1)
    pool = []
    for st in range(0, len(pool_tokens), 64):
        batch = torch.stack(pool_tokens[st:st + 64]).to(device)
        with torch.no_grad():
            o = model(batch, output_hidden_states=True)
        pool.append(o.hidden_states[args.layer + 1][:, -1, :].float().cpu())
    pool = torch.cat(pool).numpy()
    mean, cov = pool.mean(0), np.cov(pool, rowvar=False)
    gauss = torch.from_numpy(np.random.RandomState(0).multivariate_normal(
        mean, cov, size=args.n_act)).float().to(device)
    gauss_acts = [gauss[i:i + 1] for i in range(args.n_act)]
    print(f'pool {pool.shape}  mean||a||={np.linalg.norm(pool, axis=1).mean():.2f}'
          f'  gauss mean||x||={gauss.norm(dim=-1).mean():.2f}', flush=True)

    groups = {'real': (ctxs, real_acts),
              'randtok': (ctxs, randtok_acts),
              'randseq': (randseq_ctxs, randseq_acts),
              'gauss': (ctxs, gauss_acts)}

    if args.with_jacopt:
        end = args.measure_layer if args.opt_end_layer < 0 else args.opt_end_layer
        print(f'optimizing {args.n_act} points, one per context, '
              f'Jacobian of layer {args.layer} -> block {end} ...', flush=True)
        ctx_stack = torch.cat([c for c in ctxs]).to(dtype)     # [n, seq-1, d]
        X_opt = optimize_batch(model, gauss.to(dtype), ctx_stack, args.layer,
                               end, args.opt_steps, args.opt_lr, args.opt_lam,
                               args.opt_probes, args.opt_chunk)
        opt_acts = [X_opt[i:i + 1].float() for i in range(args.n_act)]
        groups['jacopt'] = (ctxs, opt_acts)
        torch.save({'X_opt': X_opt.float().cpu(), 'X_start': gauss.cpu(),
                    'opt_end_layer': end, 'tokens': torch.stack(real_tokens)},
                   os.path.join(args.outdir, 'jac_optimized.pt'))

    # ── sweeps ──────────────────────────────────────────────────────────
    res = {'args': vars(args), 'angles_deg': angles_deg, 'curves': {},
           'norms': {}}
    for name in ORDER:
        if name not in groups:
            continue
        gctxs, gacts = groups[name]
        curves = np.stack([
            sweep(model, gctxs[i], gacts[i], args.layer, args.measure_layer,
                  angles_rad, args.n_dirs, args.seed + i, dtype)
            for i in range(len(gacts))])
        res['curves'][name] = curves
        res['norms'][name] = np.array([a.float().norm().item() for a in gacts])
        flat = curves.reshape(-1, len(angles_deg))
        med = np.median(flat, axis=0)
        line = f'  {name:8s} mean||a||={res["norms"][name].mean():6.2f}  '
        for t in [10, 20, 40, 60, 90]:
            if t > args.max_angle:
                continue
            i = int(np.argmin(abs(angles_deg - t)))
            # the published reference numbers are qwen3-1.7b only
            ref = (f' (old {OLD[name][t]:7.1f})'
                   if name in OLD and args.model == 'qwen3-1.7b' else '')
            line += f'{t}deg={med[i]:8.1f}{ref}  '
        print(line, flush=True)

    with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
        pickle.dump(res, f)

    make_plot(res, args)


def make_plot(res, args):
    """Median + IQR per group. --zoom adds a 0-20 deg panel, --show_old
    overlays the published medians of the earlier run."""
    angles_deg = res['angles_deg']
    xmaxes = [args.max_angle] + ([20.0] if args.zoom else [])
    fig, axes = plt.subplots(1, len(xmaxes), figsize=(6.0 * len(xmaxes), 4.6),
                             squeeze=False)
    for ax, xmax in zip(axes[0], xmaxes):
        keep = angles_deg <= xmax
        for name in ORDER:
            if name not in res['curves']:
                continue
            flat = res['curves'][name].reshape(-1, len(angles_deg))
            med = np.median(flat, axis=0)
            lo = np.percentile(flat, 25, axis=0)
            hi = np.percentile(flat, 75, axis=0)
            ax.plot(angles_deg[keep], med[keep], color=COLORS[name], lw=1.8,
                    label=name)
            ax.fill_between(angles_deg[keep], lo[keep], hi[keep],
                            color=COLORS[name], alpha=0.18, lw=0)
        if args.show_old:
            for name, old in OLD.items():
                xs = [a for a in old if a <= xmax]
                if xs:
                    ax.plot(xs, [old[a] for a in xs], color=COLORS[name],
                            ls='--', marker='o', ms=3, lw=1.0, alpha=0.7,
                            label=f'{name} (old run)')
        ax.set_xlabel('angle (deg)')
        ax.set_ylabel(f'L2 at layer {args.measure_layer}')
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title(f'{args.model} L{args.layer}, seq_len={args.seq_len}, '
                         f'n={args.n_act} anchors x {args.n_dirs} dirs, '
                         f'median + IQR', fontsize=9)
    if args.zoom:
        axes[0, 1].set_title('same, zoomed to 0-20 deg', fontsize=9)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'sweep.png'), dpi=150)
    print(f'\nSaved {args.outdir}/sweep.png')


if __name__ == '__main__':
    main()
