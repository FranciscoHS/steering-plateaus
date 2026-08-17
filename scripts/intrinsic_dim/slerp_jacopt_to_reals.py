#!/usr/bin/env python3
"""Which real activation, if any, shares a plateau with a jac-opt point?

Francisco's design: rather than averaging matched-index interpolations (which
came out inconclusive), slerp from ONE jac-opt point to MANY real activations
and keep the paths that look least like they crossed a boundary. A path that
never leaves a plateau should trace the linear ramp in the relative-distance
plot; a boundary crossing shows up as a step.

Metric: relative logit distance rel(t) = d_A/(d_A + d_B), with d_A, d_B the
logit-space distances to the two endpoints, exactly as in interp_real_pairs.py
and interp_jacopt_pairs.py. Path: slerp_point (direction slerp, linear norm),
41 values of t.

Per path we record four shape statistics:
  max_dev   max_t |rel(t) - t|            0 for a perfectly linear ramp
  rms_dev   sqrt(mean((rel - t)^2))       same, less sensitive to one outlier
  max_step  max |rel(t_k+1) - rel(t_k)|   1/(n_t-1) for linear; large = a jump
  width     t(rel=0.75) - t(rel=0.25)     0.5 for linear; small = a sharp step

Conditions, both in the SAME carrier context (the context the jac-opt point was
optimized in), so the two differ only in where the path starts:
  jacopt -> real    A = the jac-opt point of context i
  real   -> real    A = the true real activation of context i   (the yardstick)
Both run against the same pool of real candidate activations B. The anchor's
own real activation is tracked separately, as candidate 'own'.

gpt2-xl by default, on the points from
results/intrinsic_dim/ctx_sweep_oldconfig_gpt2xl/jac_optimized.pt, which were
optimized in these very contexts.

Output in --outdir: results.pkl (all curves + stats), slerp_shape.png
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length
from interp_jacopt_pairs import slerp_point


def path_stats(rel, ts):
    """Shape statistics of one relative-distance curve."""
    dev = rel - ts
    max_dev = float(np.abs(dev).max())
    rms_dev = float(np.sqrt((dev ** 2).mean()))
    max_step = float(np.abs(np.diff(rel)).max())
    t25 = float(np.interp(0.25, rel, ts)) if rel[-1] > rel[0] else np.nan
    t75 = float(np.interp(0.75, rel, ts)) if rel[-1] > rel[0] else np.nan
    return max_dev, rms_dev, max_step, t75 - t25


def acts_and_contexts(model, device, layer, token_lists, model_dtype):
    ctxs, acts = [], []
    for tids in token_lists:
        with torch.no_grad():
            o = model(tids.unsqueeze(0).to(device), output_hidden_states=True)
        h = o.hidden_states[layer + 1]
        ctxs.append(h[:, :-1, :].float())
        acts.append(h[0, -1, :].float())
    return ctxs, torch.stack(acts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--jac_pt', default='results/intrinsic_dim/'
                                       'ctx_sweep_oldconfig_gpt2xl/jac_optimized.pt')
    p.add_argument('--n_anchor', type=int, default=30)
    p.add_argument('--n_real', type=int, default=200,
                   help='candidate real activations each anchor is slerped to')
    p.add_argument('--real_seed', type=int, default=44)
    p.add_argument('--n_t', type=int, default=41)
    p.add_argument('--outdir',
                   default='results/intrinsic_dim/slerp_jacopt_to_reals_gpt2xl')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model, tokenizer, device = load_model(args.model)
    model_dtype = next(model.parameters()).dtype
    print(f'model weights are {model_dtype}', flush=True)

    blob = torch.load(args.jac_pt, map_location='cpu', weights_only=False)
    X_opt = blob['X_opt'].float()[:args.n_anchor]
    anchor_tokens = [t for t in blob['tokens'][:args.n_anchor]]
    n_anchor = X_opt.shape[0]
    print(f'{n_anchor} jac-opt anchors from {args.jac_pt} '
          f'(opt_end_layer={blob.get("opt_end_layer")})', flush=True)

    ctxs, real_acts = acts_and_contexts(model, device, args.layer,
                                        anchor_tokens, model_dtype)
    cand_tokens = load_fineweb_fixed_length(args.n_real, tokenizer,
                                            seq_len=args.seq_len,
                                            seed=args.real_seed,
                                            cache_dir='/workspace/fineweb_cache')
    _, cand_acts = acts_and_contexts(model, device, args.layer, cand_tokens,
                                     model_dtype)
    print(f'{cand_acts.shape[0]} candidates, mean||b||='
          f'{cand_acts.norm(dim=-1).mean():.2f};  jac-opt mean||a||='
          f'{X_opt.norm(dim=-1).mean():.2f};  real mean||a||='
          f'{real_acts.norm(dim=-1).mean():.2f}', flush=True)

    ts = np.linspace(0.0, 1.0, args.n_t)
    ts_t = [float(t) for t in ts]
    conds = {'jacopt': X_opt.to(device), 'real': real_acts.to(device)}
    cand_dev = cand_acts.to(device)

    res = {'args': vars(args), 't': ts, 'stats': {}, 'rel': {}, 'own': {}}
    for cname, A_all in conds.items():
        rel_all = np.zeros((n_anchor, args.n_real, args.n_t), dtype=np.float32)
        stats = np.zeros((n_anchor, args.n_real, 4), dtype=np.float32)
        own = np.full((n_anchor, 4), np.nan, dtype=np.float32)
        own_rel = np.full((n_anchor, args.n_t), np.nan, dtype=np.float32)
        for i in range(n_anchor):
            ctx = ctxs[i].to(device, model_dtype)
            a = A_all[i]
            # the own-real target is degenerate for the real->real condition
            # (identical endpoints), so it is only meaningful for jacopt
            targets = list(cand_dev)
            if cname != 'real':
                targets = targets + [real_acts[i].to(device)]
            for j, b in enumerate(targets):
                pts = torch.stack([slerp_point(a, b, t) for t in ts_t])
                with torch.no_grad():
                    lg = forward_from_layer(
                        model, ctx, pts.unsqueeze(1).to(model_dtype),
                        args.layer).float()
                dA = (lg - lg[0]).norm(dim=-1)
                dB = (lg - lg[-1]).norm(dim=-1)
                rel = (dA / (dA + dB + 1e-9)).cpu().numpy()
                st = path_stats(rel, ts)
                if j < args.n_real:
                    rel_all[i, j] = rel
                    stats[i, j] = st
                else:
                    own_rel[i] = rel
                    own[i] = st
            print(f'  {cname}: anchor {i + 1}/{n_anchor} '
                  f'median max_dev={np.median(stats[i, :, 0]):.3f} '
                  f'best={stats[i, :, 0].min():.3f} '
                  f'own={own[i, 0]:.3f}', flush=True)
        res['stats'][cname] = stats
        res['rel'][cname] = rel_all
        res['own'][cname] = {'stats': own, 'rel': own_rel}
        with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
            pickle.dump(res, f)

    names = ['max_dev', 'rms_dev', 'max_step', 'width_25_75']
    print('\nmedians over all anchor x candidate paths')
    for cname in conds:
        s = res['stats'][cname].reshape(-1, 4)
        o = res['own'][cname]['stats']
        line = f'  {cname:7s} ' + '  '.join(
            f'{n}={np.median(s[:, k]):.3f}' for k, n in enumerate(names))
        if not np.all(np.isnan(o)):
            line += '   | own-real ' + '  '.join(
                f'{n}={np.nanmedian(o[:, k]):.3f}' for k, n in enumerate(names))
        print(line)
    print('\nbest-per-anchor (most linear candidate), median over anchors')
    for cname in conds:
        s = res['stats'][cname]
        best = s[:, :, 0].min(axis=1)
        print(f'  {cname:7s} max_dev best={np.median(best):.3f}  '
              f'[{best.min():.3f}, {best.max():.3f}]')

    # ── plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for cname, color in [('jacopt', 'C0'), ('real', 'k')]:
        s = res['stats'][cname].reshape(-1, 4)[:, 0]
        ax.hist(s, bins=60, histtype='step', color=color, density=True,
                label=f'{cname} -> real (all paths)')
        b = res['stats'][cname][:, :, 0].min(axis=1)
        ax.axvline(np.median(b), color=color, ls='--', lw=1.2,
                   label=f'{cname}: median best')
    ax.set_xlabel('max |rel(t) - t|')
    ax.set_ylabel('density')
    ax.set_title('path linearity, all candidates', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax, cname, color in [(axes[1], 'jacopt', 'C0'), (axes[2], 'real', 'k')]:
        s = res['stats'][cname][:, :, 0]
        for i in range(n_anchor):
            j = int(np.argmin(s[i]))
            ax.plot(ts, res['rel'][cname][i, j], color=color, alpha=0.35, lw=1)
        ax.plot(ts, ts, color='r', ls='--', lw=1.2, label='linear ramp')
        ax.set_xlabel('interpolation t (slerp)')
        ax.set_ylabel('relative logit distance')
        ax.set_title(f'{cname} -> best real candidate per anchor', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'slerp_shape.png'), dpi=150)
    print(f'\nSaved {args.outdir}/results.pkl, slerp_shape.png')


if __name__ == '__main__':
    main()
