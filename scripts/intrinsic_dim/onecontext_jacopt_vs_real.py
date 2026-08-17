#!/usr/bin/env python3
"""One carrier context: jac-opt points vs real activations, no transplanting.

Francisco's design, which fixes the flaw in slerp_jacopt_to_reals.py (there the
t=1 endpoint was a vector borrowed from another sentence, so it was not a state
the model ever produces in the carrier it was evaluated in):

  1. pick one context of 5 tokens; positions 1-4 are the carrier
  2. optimize ||J||_F at position 5, starting from cov-matched Gaussian points,
     in that carrier
  3. that gives n_jac jac-opt points sharing one context
     3a. check they are not degenerate, i.e. they landed at different points
  4. swap n_real random tokens into position 5
  5. that gives n_real real activations sharing the same context
  6. slerp between them and score path shape

Every endpoint is now a genuine model state in the context the path runs in.

Conditions (all slerps in the one carrier):
  jac_to_real    n_jac x n_real
  real_to_real   sampled pairs, the yardstick on the starting point
  jac_to_jac     all pairs of jac-opt points

Per path: rel(t) = d_A/(d_A+d_B) in logit space, 41 steps, plus max_dev,
rms_dev, max_step, width_25_75 AND endpoint_dist = ||f_B - f_A||. The endpoint
distance is recorded this time because a pair whose endpoints barely differ
produces a trivially linear curve; any ranking must be read with it in hand.

Output in --outdir: results.pkl, curves.png, degeneracy.txt
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

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length
from interp_jacopt_pairs import slerp_point
from ctx_sweep_oldconfig import forward_hidden, optimize_batch
from slerp_jacopt_to_reals import path_stats

STAT_NAMES = ['max_dev', 'rms_dev', 'max_step', 'width_25_75', 'endpoint_dist']


def plot_pair(args):
    """One slerp curve on its own, no reference line."""
    from transformers import AutoTokenizer
    cond, i, j = args.plot_pair.split(',')
    i, j = int(i), int(j)
    with open(os.path.join(args.outdir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)
    tok = AutoTokenizer.from_pretrained(args.model)
    rand_ids = res['rand_token_ids']
    k = res['pairs'][cond].index((i, j))

    carrier = tok.decode([int(x) for x in res['ctx_ids'][:-1]])

    def name(side, idx):
        """Full sequence, so the activation is identified by its context and
        its final token rather than by the token alone."""
        if cond.startswith('jac') and side == 0:
            return f'jac-opt {idx} in {carrier!r}'
        return repr(carrier + tok.decode([int(rand_ids[idx])]))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(res['t'], res['rel'][cond][k], color='C0', lw=2.0, marker='o',
            ms=3, label='slerp')
    ax.set_xlabel('interpolation t')
    ax.set_ylabel('relative logit distance')
    ax.set_title(f'{args.model} L{args.layer}\n'
                 f'{name(0, i)} → {name(1, j)}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(args.outdir, f'slerp_{cond}_{i}_{j}.png')
    plt.savefig(out, dpi=150)
    print(f'{cond} pair ({i},{j}) index {k}, stats '
          f'{dict(zip(STAT_NAMES, res["stats"][cond][k].round(4)))}')
    print(f'Saved {out}')


def inspect(args):
    """Most linear path per condition, with the tokens that produced it.

    endpoint_dist is printed next to every winner: a pair whose endpoints
    barely differ is linear for trivial reasons, so the ranking is only
    readable alongside it.
    """
    from transformers import AutoTokenizer
    with open(os.path.join(args.outdir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)
    tok = AutoTokenizer.from_pretrained(args.model)
    ts = res['t']
    rand_ids = res['rand_token_ids']
    ctx_ids = [int(i) for i in res['ctx_ids']]
    carrier = tok.decode(ctx_ids[:-1])

    def label(cond, side, k):
        if cond.startswith('jac') and side == 0:
            return f'jac-opt point {k}'
        return (f'real activation {k}, final token '
                f'{tok.decode([int(rand_ids[k])])!r}')

    lines = [f'carrier (positions 1-4): {carrier!r}',
             f'full source sequence: {tok.decode(ctx_ids)!r}', '']
    picks = {}
    for cond in ['jac_to_real', 'real_to_real']:
        s = res['stats'][cond]
        plist = res['pairs'][cond]
        order = np.argsort(s[:, 0])
        picks[cond] = int(order[0])
        med_ed = float(np.median(s[:, 4]))
        lines += [f'=== {cond}: most linear of {len(plist)} paths ===',
                  f'    median endpoint_dist over this condition: {med_ed:.1f}']
        for rank, k in enumerate(order[:args.top]):
            i, j = plist[int(k)]
            st = s[int(k)]
            lines += [
                f'  #{rank + 1}  max_dev={st[0]:.4f}  rms={st[1]:.4f}  '
                f'max_step={st[2]:.4f}  width={st[3]:.4f}  '
                f'endpoint_dist={st[4]:.1f}',
                f'        from {label(cond, 0, i)}',
                f'        to   {label(cond, 1, j)}']
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.outdir, 'best_paths.txt'), 'w') as f:
        f.write(txt + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (cond, color) in zip(axes, [('jac_to_real', 'C0'),
                                        ('real_to_real', 'k')]):
        k = picks[cond]
        i, j = res['pairs'][cond][k]
        st = res['stats'][cond][k]
        ax.plot(ts, ts, 'r--', lw=1.2, label='linear ramp')
        ax.plot(ts, res['rel'][cond][k], color=color, lw=2.0, marker='o', ms=3,
                label=cond)
        ax.set_title(f'{cond}: max_dev={st[0]:.3f}, endpoint_dist={st[4]:.0f}\n'
                     f'{label(cond, 0, i)}  ->  {label(cond, 1, j)}\n'
                     f'carrier {carrier!r}', fontsize=8)
        ax.set_xlabel('interpolation t (slerp)')
        ax.set_ylabel('relative logit distance')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(args.outdir, 'best_paths.png')
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')
    print(f'Saved {os.path.join(args.outdir, "best_paths.txt")}')


def predict(args):
    """Next-token distribution of chosen endpoints, in the shared carrier.

    Real activations here are genuine model states in this carrier, so their
    distribution is just the model run on carrier + that token. Jac-opt points
    are placed at the same position.
    """
    with open(os.path.join(args.outdir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)
    model, tok, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    ctx_ids = torch.tensor([int(i) for i in res['ctx_ids']])
    rand_ids = res['rand_token_ids']

    with torch.no_grad():
        o = model(ctx_ids.unsqueeze(0).to(device), output_hidden_states=True)
    ctx = o.hidden_states[args.layer + 1][:, :-1, :].float().to(model_dtype)
    carrier = tok.decode([int(i) for i in ctx_ids[:-1]])

    items = []
    for k in [int(x) for x in args.predict_real.split(',') if x.strip()]:
        items.append(('real', k, res['X_real'][k].float(),
                      tok.decode([int(rand_ids[k])])))
    for k in [int(x) for x in args.predict_jac.split(',') if x.strip()]:
        items.append(('jacopt', k, res['X_jac'][k].float(), None))

    out = [f'carrier: {carrier!r}', '']
    for kind, k, vec, tokstr in items:
        with torch.no_grad():
            lg = forward_from_layer(model, ctx,
                                    vec[None, None].to(device, model_dtype),
                                    args.layer).float()[0]
        logp = F.log_softmax(lg, dim=-1)
        p = logp.exp()
        H = -(p * logp).sum()
        t = p.topk(args.topk)
        head = (f'{kind} {k}: {carrier + tokstr!r}' if tokstr is not None
                else f'{kind} {k} (optimized vector at the last position)')
        out += [head,
                f'    ||x||={vec.norm():.2f}  entropy {H.item():.3f} nats  '
                f'perplexity exp(H) {H.exp().item():.2f}']
        out += [f'      {pr:6.3f}  {tok.decode([i])!r}'
                for pr, i in zip(t.values.tolist(), t.indices.tolist())]
        out.append('')
    txt = '\n'.join(out)
    print(txt)
    path = os.path.join(args.outdir, 'endpoint_predictions.txt')
    with open(path, 'w') as f:
        f.write(txt + '\n')
    print(f'Saved {path}')


def logits_at(model, ctx, points, layer, model_dtype, batch=64):
    out = []
    for st in range(0, points.shape[0], batch):
        with torch.no_grad():
            lg = forward_from_layer(
                model, ctx, points[st:st + batch].unsqueeze(1).to(model_dtype),
                layer)
        out.append(lg.float())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--ctx_index', type=int, default=0,
                   help='which FineWeb seq_len-5 sample supplies the carrier')
    p.add_argument('--random_ctx', action='store_true',
                   help='use a random-token carrier instead of a FineWeb one')
    p.add_argument('--n_jac', type=int, default=30)
    p.add_argument('--n_real', type=int, default=60)
    p.add_argument('--n_pairs_rr', type=int, default=600,
                   help='sampled real-real pairs for the control')
    p.add_argument('--n_pool', type=int, default=2000,
                   help='real activations used to fit the Gaussian mean/cov')
    p.add_argument('--n_t', type=int, default=41)
    p.add_argument('--opt_steps', type=int, default=200)
    p.add_argument('--opt_lr', type=float, default=0.1)
    p.add_argument('--opt_lam', type=float, default=1e-3)
    p.add_argument('--opt_probes', type=int, default=5)
    p.add_argument('--opt_chunk', type=int, default=10)
    p.add_argument('--opt_end_layer', type=int, default=-1,
                   help='block the optimized Jacobian ends at; -1 = final '
                        'block, matching the earlier runs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/onecontext_gpt2xl')
    p.add_argument('--inspect', action='store_true',
                   help='no measurement: load results.pkl, report and plot the '
                        'most linear path of each condition')
    p.add_argument('--top', type=int, default=5,
                   help='--inspect: how many of the most linear to list')
    p.add_argument('--predict_real', default='',
                   help='comma-separated real-activation indices; print their '
                        'next-token distributions in the carrier and exit')
    p.add_argument('--predict_jac', default='',
                   help='same for jac-opt point indices')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--plot_pair', default='',
                   help='"cond,i,j" — plot that single slerp curve on its own '
                        'and exit, e.g. real_to_real,40,50')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.plot_pair:
        plot_pair(args)
        return
    if args.inspect:
        inspect(args)
        return
    if args.predict_real or args.predict_jac:
        predict(args)
        return

    model, tok, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    vocab = tok.vocab_size
    print(f'model weights are {model_dtype}, vocab {vocab}', flush=True)

    # ── the one carrier ─────────────────────────────────────────────────
    g = torch.Generator().manual_seed(args.seed)
    if args.random_ctx:
        ctx_ids = torch.randint(0, vocab, (args.seq_len,), generator=g)
    else:
        toks = load_fineweb_fixed_length(args.ctx_index + 1, tok,
                                         seq_len=args.seq_len, seed=args.seed,
                                         cache_dir='/workspace/fineweb_cache')
        ctx_ids = toks[args.ctx_index]
    ctx_text = tok.decode([int(i) for i in ctx_ids])
    print(f'carrier: {ctx_text!r}', flush=True)

    with torch.no_grad():
        o = model(ctx_ids.unsqueeze(0).to(device), output_hidden_states=True)
    ctx = o.hidden_states[args.layer + 1][:, :-1, :].float()
    a_true = o.hidden_states[args.layer + 1][0, -1, :].float()

    # ── real activations: random final tokens in the same carrier ───────
    rand_ids = torch.randint(0, vocab, (args.n_real,), generator=g)
    seqs = torch.cat([ctx_ids[:-1].unsqueeze(0).repeat(args.n_real, 1),
                      rand_ids[:, None]], dim=1).to(device)
    reals = []
    for st in range(0, args.n_real, 32):
        with torch.no_grad():
            oo = model(seqs[st:st + 32], output_hidden_states=True)
        reals.append(oo.hidden_states[args.layer + 1][:, -1, :].float())
    X_real = torch.cat(reals)

    # ── jac-opt points in the same carrier ──────────────────────────────
    pool_tokens = load_fineweb_fixed_length(args.n_pool, tok,
                                            seq_len=args.seq_len,
                                            seed=args.seed + 1,
                                            cache_dir='/workspace/fineweb_cache')
    pool = []
    for st in range(0, len(pool_tokens), 64):
        batch = torch.stack(pool_tokens[st:st + 64]).to(device)
        with torch.no_grad():
            oo = model(batch, output_hidden_states=True)
        pool.append(oo.hidden_states[args.layer + 1][:, -1, :].float().cpu())
    pool = torch.cat(pool).numpy()
    gauss = torch.from_numpy(np.random.RandomState(0).multivariate_normal(
        pool.mean(0), np.cov(pool, rowvar=False),
        size=args.n_jac)).float().to(device)

    end = None if args.opt_end_layer < 0 else args.opt_end_layer
    print(f'optimizing {args.n_jac} points in this carrier, Jacobian of layer '
          f'{args.layer} -> {"final block" if end is None else f"block {end}"}',
          flush=True)
    ctx_stack = ctx.expand(args.n_jac, -1, -1).to(model_dtype).contiguous()
    X_jac = optimize_batch(model, gauss.to(model_dtype), ctx_stack, args.layer,
                           end, args.opt_steps, args.opt_lr, args.opt_lam,
                           args.opt_probes, args.opt_chunk).float()

    # ── 3a: are the jac-opt points distinct? ────────────────────────────
    def pdist_stats(X):
        d = torch.cdist(X[None], X[None])[0]
        iu = torch.triu_indices(len(X), len(X), offset=1)
        v = d[iu[0], iu[1]]
        return (v.median().item(), v.min().item(), v.max().item())

    lg_jac = logits_at(model, ctx.to(device, model_dtype), X_jac, args.layer,
                       model_dtype)
    lg_real = logits_at(model, ctx.to(device, model_dtype), X_real, args.layer,
                        model_dtype)

    def kl_stats(lg):
        logp = F.log_softmax(lg, dim=-1)
        p = logp.exp()
        kl = (p[:, None, :] * (logp[:, None, :] - logp[None, :, :])).sum(-1)
        iu = torch.triu_indices(len(lg), len(lg), offset=1)
        v = kl[iu[0], iu[1]]
        return (v.median().item(), v.min().item(), v.max().item())

    deg = {
        'ctx_text': ctx_text,
        'jac_pairwise_l2': pdist_stats(X_jac),
        'start_pairwise_l2': pdist_stats(gauss.float()),
        'real_pairwise_l2': pdist_stats(X_real),
        'jac_move_from_start': (X_jac - gauss.float()).norm(dim=-1),
        'jac_norms': X_jac.norm(dim=-1),
        'real_norms': X_real.norm(dim=-1),
        'jac_pairwise_kl': kl_stats(lg_jac),
        'real_pairwise_kl': kl_stats(lg_real),
    }
    dlines = [f'carrier: {ctx_text!r}',
              f'jac-opt n={args.n_jac}, real n={args.n_real}', '',
              'pairwise L2 (median, min, max)',
              f'  jac-opt points   {deg["jac_pairwise_l2"]}',
              f'  their starts     {deg["start_pairwise_l2"]}',
              f'  real activations {deg["real_pairwise_l2"]}', '',
              'pairwise output KL in this carrier, nats (median, min, max)',
              f'  jac-opt points   {deg["jac_pairwise_kl"]}',
              f'  real activations {deg["real_pairwise_kl"]}', '',
              f'||x||   jac-opt median {deg["jac_norms"].median():.2f}, '
              f'real median {deg["real_norms"].median():.2f}',
              f'move from start median '
              f'{deg["jac_move_from_start"].median():.2f}']
    print('\n'.join(dlines), flush=True)
    with open(os.path.join(args.outdir, 'degeneracy.txt'), 'w') as f:
        f.write('\n'.join(dlines) + '\n')

    # ── slerp paths ─────────────────────────────────────────────────────
    ts = np.linspace(0.0, 1.0, args.n_t)
    ts_f = [float(t) for t in ts]
    rng = np.random.RandomState(args.seed)
    pairs = {
        'jac_to_real': [(i, j) for i in range(args.n_jac)
                        for j in range(args.n_real)],
        'jac_to_jac': [(i, j) for i in range(args.n_jac)
                       for j in range(i + 1, args.n_jac)],
    }
    rr = [(i, j) for i in range(args.n_real) for j in range(i + 1, args.n_real)]
    idx = rng.choice(len(rr), size=min(args.n_pairs_rr, len(rr)),
                     replace=False)
    pairs['real_to_real'] = [rr[k] for k in idx]

    srcs = {'jac_to_real': (X_jac, X_real), 'jac_to_jac': (X_jac, X_jac),
            'real_to_real': (X_real, X_real)}
    ctx_d = ctx.to(device, model_dtype)
    res = {'args': vars(args), 't': ts, 'ctx_text': ctx_text,
           'ctx_ids': ctx_ids, 'rand_token_ids': rand_ids,
           'X_jac': X_jac.cpu(), 'X_real': X_real.cpu(),
           'X_start': gauss.float().cpu(), 'a_true': a_true.cpu(),
           'degeneracy': {k: (v.cpu() if torch.is_tensor(v) else v)
                          for k, v in deg.items()},
           'pairs': pairs, 'rel': {}, 'stats': {}}
    for cname, plist in pairs.items():
        A, B = srcs[cname]
        rel = np.zeros((len(plist), args.n_t), dtype=np.float32)
        stats = np.zeros((len(plist), 5), dtype=np.float32)
        for k, (i, j) in enumerate(plist):
            pts = torch.stack([slerp_point(A[i], B[j], t) for t in ts_f])
            with torch.no_grad():
                lg = forward_from_layer(
                    model, ctx_d, pts.unsqueeze(1).to(model_dtype),
                    args.layer).float()
            dA = (lg - lg[0]).norm(dim=-1)
            dB = (lg - lg[-1]).norm(dim=-1)
            rel[k] = (dA / (dA + dB + 1e-9)).cpu().numpy()
            stats[k, :4] = path_stats(rel[k], ts)
            stats[k, 4] = (lg[-1] - lg[0]).norm().item()
            if (k + 1) % 200 == 0:
                print(f'  {cname}: {k + 1}/{len(plist)}', flush=True)
        res['rel'][cname] = rel
        res['stats'][cname] = stats
        m = np.median(stats, axis=0)
        print(f'  {cname:12s} n={len(plist):5d}  ' + '  '.join(
            f'{n}={m[k]:.3f}' for k, n in enumerate(STAT_NAMES)), flush=True)
        with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
            pickle.dump(res, f)

    # ── plot ────────────────────────────────────────────────────────────
    colors = {'jac_to_real': 'C0', 'real_to_real': 'k', 'jac_to_jac': 'C1'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for cname, c in colors.items():
        rel = res['rel'][cname]
        axes[0].plot(ts, np.median(rel, axis=0), color=c, lw=2, label=cname)
        axes[0].fill_between(ts, np.percentile(rel, 25, axis=0),
                             np.percentile(rel, 75, axis=0), color=c,
                             alpha=0.15, lw=0)
        axes[1].hist(res['stats'][cname][:, 0], bins=50, histtype='step',
                     density=True, color=c, label=cname)
        axes[2].scatter(res['stats'][cname][:, 4], res['stats'][cname][:, 0],
                        s=5, alpha=0.35, color=c, label=cname)
    axes[0].plot(ts, ts, 'r--', lw=1.2, label='linear ramp')
    axes[0].set_xlabel('interpolation t (slerp)')
    axes[0].set_ylabel('relative logit distance')
    axes[0].set_title(f'median + IQR, carrier {ctx_text!r}', fontsize=8)
    axes[1].set_xlabel('max |rel(t) - t|')
    axes[1].set_ylabel('density')
    axes[1].set_title('path linearity', fontsize=9)
    axes[2].set_xlabel('endpoint logit distance ||f_B - f_A||')
    axes[2].set_ylabel('max |rel(t) - t|')
    axes[2].set_title('linearity vs how far apart the endpoints are',
                      fontsize=9)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'curves.png'), dpi=150)
    print(f'\nSaved {args.outdir}/results.pkl, curves.png, degeneracy.txt')


if __name__ == '__main__':
    main()
