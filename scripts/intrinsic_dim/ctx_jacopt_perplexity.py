#!/usr/bin/env python3
"""Per-context jac-norm minimization: plateau check + next-token perplexity.

Francisco's design:
  1. take a given context
  2. optimize the last-token residual activation for low ||J||_F, starting from
     cov-matched Gaussian points, with THIS context as the carrier
  3. verify the optimized points sit on a plateau
  4. compare the perplexity of the next-token prediction against the same
     context with different (random) actual last tokens
  5. ask whether low values mean the same plateau

Unlike the April run, the carrier context of the optimization is saved, so
every later measurement uses the context the points were optimized for.

Per context (5 by default) three groups of anchors are compared:
  jacopt    n_opt points optimized on this context
  randtok   real activations of this context with a random token in the last
            position (run the model on ctx_ids[:-1] + t)
  real      the activation of the true last token

Measurements per anchor:
  ||J||_F           finite-difference Hutchinson, layer -> final block
  plateau sweep     spherical rotation by 0..max_angle along n_dirs random
                    directions; final-hidden L2 and output KL from the anchor
  perplexity        exp(H(p)) of the next-token distribution, plus p_max and
                    the top-10 tokens

Outputs in --outdir (default results/intrinsic_dim/ctx_jacopt_ppl/):
  ctx{i}.pt per context (resumable), summary.txt, ppl.png, sweep.png
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (load_model, _get_blocks, _get_final_norm, _get_lm_head,
                       _is_rope_model, _get_position_embeddings, _run_block)

JAC_DIR = 'results/intrinsic_dim'


# ── Forward helpers ─────────────────────────────────────────────────────

def forward_points(model, ctx, points, layer, want_logits=True):
    """Run blocks[layer+1:] on `points` placed at the last position of `ctx`.

    ctx:    [1, seq_len-1, d] residual stream at `layer`
    points: [n, d]
    Returns final-block hidden [n, d] and (optionally) logits [n, V].
    """
    n = points.shape[0]
    hidden = torch.cat([ctx.expand(n, -1, -1), points.unsqueeze(1)], dim=1)
    pos_emb = _get_position_embeddings(model, hidden) if _is_rope_model(model) else None
    for block in _get_blocks(model)[layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)
    h = hidden[:, -1, :]
    if not want_logits:
        return h, None
    return h, _get_lm_head(model)(_get_final_norm(model)(h))


def jac_norm_fd(model, ctx, points, layer, n_probes=10, eps=1e-3, batch=64):
    """Finite-difference Hutchinson estimate of ||J||_F, layer -> final block."""
    d = points.shape[1]
    out = torch.zeros(points.shape[0])
    g = torch.Generator(device='cpu').manual_seed(0)
    for st in range(0, points.shape[0], batch):
        x = points[st:st + batch]
        base, _ = forward_points(model, ctx, x, layer, want_logits=False)
        acc = torch.zeros(x.shape[0], device=base.device)
        for _ in range(n_probes):
            u = F.normalize(torch.randn(x.shape, generator=g), dim=-1).to(x.device)
            pert, _ = forward_points(model, ctx, x + eps * u, layer,
                                     want_logits=False)
            acc += ((pert - base).float() / eps).pow(2).sum(-1)
        out[st:st + batch] = (acc * d / n_probes).sqrt().cpu()
    return out


# ── Optimization ────────────────────────────────────────────────────────

def optimize_points(model, points, ctx, layer, n_steps, lr, lam, n_probes,
                    chunk, device):
    """min ||J||_F + lam*||x - x0||^2, same objective as the April run."""
    x0 = points.clone().to(device)
    x = points.clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    N, d = points.shape
    hist = {'jac': [], 'dist': []}
    for step in range(n_steps):
        opt.zero_grad()
        tot_j, tot_d = 0.0, 0.0
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            xc = x[st:ed]
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.MATH):
                f, _ = forward_points(model, ctx, xc, layer, want_logits=False)
            frob_sq = torch.zeros(ed - st, device=device)
            for _ in range(n_probes):
                u = torch.randn(ed - st, d, device=device)
                s = (f * u).sum()
                gc = torch.autograd.grad(s, xc, create_graph=True,
                                         retain_graph=True)[0]
                frob_sq = frob_sq + (gc ** 2).sum(-1)
            frob_sq = frob_sq / n_probes
            jac_loss = torch.sqrt(frob_sq + 1e-8)
            dist_loss = lam * ((xc - x0[st:ed]) ** 2).sum(-1)
            ((jac_loss + dist_loss).sum() / N).backward()
            tot_j += jac_loss.sum().item()
            tot_d += dist_loss.sum().item()
        opt.step()
        hist['jac'].append(tot_j / N)
        hist['dist'].append(tot_d / N)
        if step % 20 == 0:
            move = (x.detach() - x0).norm(dim=-1).mean().item()
            print(f'    step {step}: jac={tot_j/N:.1f} '
                  f'dist={tot_d/N:.2f} move={move:.2f}', flush=True)
    return x.detach(), hist


# ── Measurements ────────────────────────────────────────────────────────

def next_token_stats(model, ctx, points, layer, batch=32, topk=10):
    """exp(H(p)), p_max, and the top-k tokens of the next-token distribution."""
    ppl, pmax, top_ids, top_p = [], [], [], []
    for st in range(0, points.shape[0], batch):
        _, logits = forward_points(model, ctx, points[st:st + batch], layer)
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        H = -(p * logp).sum(-1)
        ppl.append(H.exp().cpu())
        t = p.topk(topk, dim=-1)
        pmax.append(t.values[:, 0].cpu())
        top_ids.append(t.indices.cpu())
        top_p.append(t.values.cpu())
    return {'ppl': torch.cat(ppl), 'p_max': torch.cat(pmax),
            'top_ids': torch.cat(top_ids), 'top_p': torch.cat(top_p)}


def plateau_sweep(model, ctx, anchors, layer, angles_rad, n_dirs, seed,
                  batch=64):
    """Spherical rotation sweep. Returns L2 and KL arrays [n_anchors, n_dirs,
    n_angles], both measured against the unperturbed anchor."""
    n_a, d = anchors.shape
    n_ang = len(angles_rad)
    device = anchors.device
    L2 = np.zeros((n_a, n_dirs, n_ang), dtype=np.float32)
    KL = np.zeros((n_a, n_dirs, n_ang), dtype=np.float32)
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for i in range(n_a):
        a = anchors[i].float()
        R = a.norm()
        a_u = a / R
        h_ref, lg_ref = forward_points(model, ctx, a[None].to(anchors.dtype), layer)
        logp_ref = F.log_softmax(lg_ref.float(), dim=-1)
        for j in range(n_dirs):
            u = torch.randn(d, generator=g).to(device)
            u_o = F.normalize(u - (u @ a_u) * a_u, dim=0)
            pts = R * (c[:, None] * a_u[None] + s[:, None] * u_o[None])
            for st in range(0, n_ang, batch):
                h, lg = forward_points(model, ctx,
                                       pts[st:st + batch].to(anchors.dtype), layer)
                L2[i, j, st:st + batch] = (
                    (h.float() - h_ref.float()).norm(dim=-1).cpu().numpy())
                logp = F.log_softmax(lg.float(), dim=-1)
                kl = (logp_ref.exp() * (logp_ref - logp)).sum(-1)
                KL[i, j, st:st + batch] = kl.cpu().numpy()
    return L2, KL


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=10)
    p.add_argument('--n_ctx', type=int, default=5)
    p.add_argument('--n_opt', type=int, default=20)
    p.add_argument('--n_randtok', type=int, default=50)
    p.add_argument('--n_sweep_randtok', type=int, default=20)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--lam', type=float, default=1e-3)
    p.add_argument('--probes', type=int, default=5)
    p.add_argument('--chunk', type=int, default=10)
    p.add_argument('--n_dirs', type=int, default=10)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--n_angles', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='results/intrinsic_dim/ctx_jacopt_ppl')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model, tokenizer, device = load_model(args.model, dtype=torch.float32)
    vocab = model.config.vocab_size
    angles_deg = np.linspace(0, args.max_angle, args.n_angles)
    angles_rad = np.deg2rad(angles_deg)

    # Contexts: FineWeb rows with a full seq_len of real tokens.
    print('Streaming FineWeb for contexts...', flush=True)
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True).shuffle(seed=args.seed)
    texts = []
    for ex in ds:
        if len(ex['text']) >= args.seq_len * 10:
            texts.append(ex['text'])
        if len(texts) >= args.n_ctx * 4:
            break
    tok = tokenizer(texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=args.seq_len)
    full = tok['attention_mask'].sum(-1) == args.seq_len
    ids = tok['input_ids'][full][:args.n_ctx]
    print(f'  {ids.shape[0]} contexts of {args.seq_len} tokens', flush=True)

    # Gaussian starts: the same cov-matched pool as the April run, rows far
    # from the 200 it optimized.
    filt = torch.load(os.path.join(JAC_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    pool = filt['X_gauss'].float()[1000:1000 + args.n_ctx * args.n_opt]

    summaries = []
    for ci in range(ids.shape[0]):
        out_path = os.path.join(args.outdir, f'ctx{ci}.pt')
        if os.path.exists(out_path):
            print(f'\n=== context {ci}: found {out_path}, skipping ===',
                  flush=True)
            summaries.append(torch.load(out_path, map_location='cpu',
                                        weights_only=False))
            continue
        ctx_ids = ids[ci:ci + 1].to(device)
        text = tokenizer.decode(ctx_ids[0])
        print(f'\n=== context {ci}: {text!r} ===', flush=True)

        with torch.no_grad():
            o = model(ctx_ids, output_hidden_states=True)
        h = o.hidden_states[args.layer + 1].float()
        ctx = h[:, :-1, :]                       # [1, seq_len-1, d]
        a_real = h[:, -1, :]                     # [1, d]

        # Random last tokens on the same context.
        g = torch.Generator().manual_seed(args.seed + ci)
        rand_ids = torch.randint(0, vocab, (args.n_randtok,), generator=g)
        seqs = ctx_ids[0, :-1].unsqueeze(0).repeat(args.n_randtok, 1)
        seqs = torch.cat([seqs, rand_ids[:, None].to(device)], dim=1)
        randtok_acts = []
        for st in range(0, args.n_randtok, 32):
            with torch.no_grad():
                oo = model(seqs[st:st + 32], output_hidden_states=True)
            randtok_acts.append(oo.hidden_states[args.layer + 1][:, -1, :].float())
        X_rand = torch.cat(randtok_acts)

        # Optimize.
        start = pool[ci * args.n_opt:(ci + 1) * args.n_opt].to(device)
        print(f'  optimizing {args.n_opt} points, {args.steps} steps ...',
              flush=True)
        X_opt, hist = optimize_points(model, start, ctx, args.layer,
                                      args.steps, args.lr, args.lam,
                                      args.probes, args.chunk, device)

        groups = {'jacopt': X_opt,
                  'randtok': X_rand,
                  'real': a_real}
        blob = {'ctx_id': ci, 'ctx_ids': ctx_ids.cpu(), 'text': text,
                'angles_deg': angles_deg, 'hist': hist,
                'X_opt': X_opt.cpu(), 'X_start': start.cpu(),
                'X_rand': X_rand.cpu(), 'a_real': a_real.cpu(),
                'rand_token_ids': rand_ids, 'groups': {}}

        with torch.no_grad():
            for name, X in groups.items():
                jn = jac_norm_fd(model, ctx, X, args.layer)
                st = next_token_stats(model, ctx, X, args.layer)
                sw = X if name != 'randtok' else X[:args.n_sweep_randtok]
                L2, KL = plateau_sweep(model, ctx, sw, args.layer, angles_rad,
                                       args.n_dirs, args.seed + ci)
                blob['groups'][name] = {
                    'jac': jn.numpy(), 'ppl': st['ppl'].numpy(),
                    'p_max': st['p_max'].numpy(),
                    'top_ids': st['top_ids'].numpy(),
                    'top_p': st['top_p'].numpy(),
                    'sweep_L2': L2, 'sweep_KL': KL,
                    'norm': X.norm(dim=-1).cpu().numpy()}
                print(f'  {name:8s} n={X.shape[0]:3d}  '
                      f'||J||={np.median(jn.numpy()):8.1f}  '
                      f'ppl={np.median(st["ppl"].numpy()):8.2f}  '
                      f'p_max={np.median(st["p_max"].numpy()):.3f}  '
                      f'KL@{args.max_angle:g}deg='
                      f'{np.median(KL[:, :, -1]):.3f}   (medians)', flush=True)

        torch.save(blob, out_path)
        summaries.append(blob)
        print(f'  saved {out_path}', flush=True)

    # ── Plots ───────────────────────────────────────────────────────────
    colors = {'jacopt': 'C0', 'randtok': 'C1', 'real': 'k'}
    n = len(summaries)

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    for i, b in enumerate(summaries):
        ax = axes[0, i]
        for name in ['jacopt', 'randtok']:
            ax.hist(b['groups'][name]['ppl'], bins=25, histtype='step',
                    color=colors[name], label=name, lw=1.5)
        ax.axvline(b['groups']['real']['ppl'][0], color='k', ls='--',
                   label='real token')
        ax.set_xscale('log')
        ax.set_xlabel('perplexity exp(H)')
        ax.set_title(f'ctx {i}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'ppl.png'), dpi=150)

    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.4), squeeze=False)
    for i, b in enumerate(summaries):
        for r, key in enumerate(['sweep_L2', 'sweep_KL']):
            ax = axes[r, i]
            for name in ['jacopt', 'randtok', 'real']:
                v = b['groups'][name][key]
                med = np.median(v.reshape(-1, v.shape[-1]), axis=0)
                ax.plot(b['angles_deg'], med, color=colors[name], label=name)
            ax.set_xlabel('angle (deg)')
            ax.set_ylabel(key)
            ax.set_title(f'ctx {i} — {key} (median)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0 and r == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'sweep.png'), dpi=150)
    print(f'\nSaved plots in {args.outdir}')

    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'{args.model} L{args.layer} seq_len={args.seq_len} '
                f'n_ctx={len(summaries)} n_opt={args.n_opt} '
                f'n_randtok={args.n_randtok}\n')
        f.write('medians per group\n')
        for b in summaries:
            f.write(f'\nctx {b["ctx_id"]}: {b["text"]!r}\n')
            f.write(f'{"group":8s} {"n":>4s} {"||x||":>8s} {"||J||":>9s} '
                    f'{"ppl":>9s} {"p_max":>7s} {"KLmax":>8s} {"L2max":>9s}\n')
            for name, gr in b['groups'].items():
                f.write(f'{name:8s} {len(gr["ppl"]):4d} '
                        f'{np.median(gr["norm"]):8.2f} '
                        f'{np.median(gr["jac"]):9.1f} '
                        f'{np.median(gr["ppl"]):9.2f} '
                        f'{np.median(gr["p_max"]):7.3f} '
                        f'{np.median(gr["sweep_KL"][:, :, -1]):8.3f} '
                        f'{np.median(gr["sweep_L2"][:, :, -1]):9.2f}\n')
    print(f'Saved {os.path.join(args.outdir, "summary.txt")}')


if __name__ == '__main__':
    main()
