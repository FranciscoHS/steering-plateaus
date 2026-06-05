#!/usr/bin/env python3
"""p-fit on the denoising toy model: feature x feature / feature x random.

Ground-truth-features control for the transformer p-fit pipeline. In the
denoising toy model the features are *exactly* the canonical basis e_0..e_{d-1}
of the input, and the trained tanh3 net cleanly recovers them (runs/phase1_tanh3,
encoder/decoder cos_mean ~ 0.91). We measure the superellipse-p of the
output-L2 response when perturbing along two true features (e_f, e_g) or along
one true feature and a random direction.

Method: this replicates the fsec-anonymous toy-model pipeline
(toy_model/scripts/random_sparse_lp_beeswarm.py and the denoising repo's
scripts/run_beeswarm_phase1.py) exactly, NOT the transformer canonical-contour
machinery. For each angle phi in [0, pi/2] we take the 2D direction
    dir(phi) = cos(phi)*u + sin(phi)*v
sweep the radius alpha in [0, alpha_max] (fine 1D grid), and record the
*first-crossing* radius r(phi) where ||f(x_clean + alpha*dir) - f(x_clean)||_2
first exceeds a FIXED ABSOLUTE threshold tau. We then fit n in
    (r cos phi / r(0))^n + (r sin phi / r(pi/2))^n = 1.

This first-crossing / fixed-tau "onset" boundary is where the plateau-breaking
geometry lives; the transformer pipeline's "0.5 x in-window-max" threshold with
an over-driven alpha_max collapses p<2 by measuring the saturated regime
instead (see scripts/superellipse/diagnose_toymodel_p.py).

Usage:
    python scripts/superellipse/fit_pairs_toymodel.py --leg2 feature \\
        --ckpt /workspace/denoising/runs/phase1_tanh3/model.pt \\
        --denoising_repo /workspace/denoising --device cuda
"""

import argparse
import math
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def load_toy_model(ckpt_path, denoising_repo, device):
    """Load the trained TwoLayerNet denoiser from a denoising-repo checkpoint."""
    repo = os.path.expanduser(denoising_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from mft_denoising.nn import TwoLayerNet  # noqa: E402

    ck = torch.load(os.path.expanduser(ckpt_path), map_location='cpu',
                    weights_only=False)
    cfg = ck['config']
    model = TwoLayerNet(
        input_size=cfg['d'], hidden_size=cfg['hidden_size'],
        activation=cfg['activation'],
        activation_kwargs=dict(theta_init=cfg['jumprelu_theta_init'],
                               learn_threshold=cfg['jumprelu_learn_threshold'],
                               eps=cfg['jumprelu_eps']),
    )
    model.load_state_dict(ck['state_dict'])
    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def boundary_radii(model, d, S, u, v, phis, taus, alpha_max, n_alpha):
    """First-crossing boundary radii r(phi) for each fixed absolute tau.

    Vectorized over phi: builds (n_phi * n_alpha, d) inputs and forwards in
    memory-bounded chunks. Mirrors fsec random_sparse_lp_beeswarm.boundary_radii.
    Returns {tau: rs[n_phi]} with nan where the threshold is never reached.
    """
    device = next(model.parameters()).device
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    alphas = torch.linspace(0.0, alpha_max, n_alpha, device=device)

    phis_t = torch.tensor(np.asarray(phis), dtype=torch.float32, device=device)
    n_phi = phis_t.numel()
    directions = (torch.cos(phis_t).unsqueeze(1) * u.unsqueeze(0)
                  + torch.sin(phis_t).unsqueeze(1) * v.unsqueeze(0))   # [n_phi, d]
    x = (x_clean.view(1, 1, d)
         + alphas.view(1, n_alpha, 1) * directions.view(n_phi, 1, d))
    x_flat = x.view(n_phi * n_alpha, d)

    chunk = max(1, 100_000_000 // max(d, 1))
    diff_norms = []
    for s_idx in range(0, x_flat.shape[0], chunk):
        out_chunk = model(x_flat[s_idx:s_idx + chunk])
        diff_norms.append((out_chunk - base.unsqueeze(0)).norm(p=2, dim=1))
    l2 = torch.cat(diff_norms, dim=0).view(n_phi, n_alpha)

    a_cpu = alphas.detach().cpu().numpy()
    out = {tau: np.full(n_phi, np.nan) for tau in taus}
    for tau in taus:
        above = (l2 > tau).int()
        any_above = above.any(dim=1).detach().cpu().numpy()
        first_idx = above.argmax(dim=1).detach().cpu().numpy()
        out[tau][any_above] = a_cpu[first_idx[any_above]]
    return out


def fit_n(xn, yn, bounds=(0.5, 60.0)):
    """Fit n in xn^n + yn^n = 1 (mirrors fsec fit_n)."""
    mask = (xn > 1e-3) & (yn > 1e-3) & np.isfinite(xn) & np.isfinite(yn)
    x = xn[mask]; y = yn[mask]
    if len(x) < 10:
        return np.nan
    def loss(n):
        return float(np.sum((x ** n + y ** n - 1.0) ** 2))
    res = minimize_scalar(loss, bounds=bounds, method='bounded',
                          options={'xatol': 1e-3})
    return float(res.x)


def fit_from_radii(rs, phis):
    """Normalize boundary radii by the two axis intercepts and fit n."""
    if not (np.isfinite(rs[0]) and np.isfinite(rs[-1])):
        return np.nan
    xs = rs * np.cos(phis) / rs[0]
    ys = rs * np.sin(phis) / rs[-1]
    return fit_from_xy(xs, ys)


def fit_from_xy(xs, ys):
    return fit_n(xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt',
                    help='denoising-repo TwoLayerNet checkpoint (model.pt)')
    ap.add_argument('--denoising_repo', default='~/Research/denoising',
                    help='path to the denoising repo (for mft_denoising import)')
    ap.add_argument('--leg2', default='random', choices=['random', 'feature'],
                    help="second leg: 'feature' (another true feature e_g, "
                         "g not in S, g != f) or 'random' (unit Gaussian, S "
                         "coords zeroed, orthogonalized to e_f)")
    ap.add_argument('--n_pairs', type=int, default=100)
    ap.add_argument('--taus', type=float, nargs='+', default=[0.1, 0.2, 0.5],
                    help='fixed absolute L2 thresholds (replicates phase1 beeswarm)')
    ap.add_argument('--n_phis', type=int, default=80)
    ap.add_argument('--alpha_max', type=float, default=15.0)
    ap.add_argument('--n_alpha', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None,
                    help='default: results/dictionary/toymodel_phase1_tanh3/'
                         'fit_pairs_feature_vs_<leg2>')
    args = ap.parse_args()

    if args.out is None:
        args.out = ('results/dictionary/toymodel_phase1_tanh3/'
                    f'fit_pairs_feature_vs_{args.leg2}')
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print(f'Loading toy model: {args.ckpt}', flush=True)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']
    sparsity = cfg['sparsity']
    S = tuple(range(sparsity))   # fixed two-hot anchor, matching fsec/phase1
    inactive = [i for i in range(d) if i not in S]
    print(f'  device={device} d={d} H={cfg["hidden_size"]} act={cfg["activation"]} '
          f'S={S}  leg2={args.leg2}', flush=True)
    print(f'  taus={args.taus}  alpha_max={args.alpha_max}  n_alpha={args.n_alpha}  '
          f'n_phis={args.n_phis}', flush=True)

    phis = np.linspace(0.0, math.pi / 2, args.n_phis)
    taus = list(args.taus)

    u_unit = torch.zeros(d, device=device)
    per_pair = []
    p_by_tau = {t: [] for t in taus}

    for k in range(args.n_pairs):
        f = int(rng.choice(inactive))
        u = torch.zeros(d, device=device); u[f] = 1.0
        if args.leg2 == 'feature':
            g = int(rng.choice([i for i in inactive if i != f]))
            v = torch.zeros(d, device=device); v[g] = 1.0
        else:
            g = None
            v_np = rng.standard_normal(d)
            v_np[list(S)] = 0.0
            v_np /= np.linalg.norm(v_np)
            # orthogonalize to u (=e_f) and renormalize (fsec random convention)
            v_np[f] = 0.0
            v_np /= np.linalg.norm(v_np)
            v = torch.from_numpy(v_np.astype(np.float32)).to(device)

        rs_dict = boundary_radii(model, d, S, u, v, phis, taus,
                                 args.alpha_max, args.n_alpha)
        ps = {t: fit_from_radii(rs_dict[t], phis) for t in taus}
        for t in taus:
            p_by_tau[t].append(ps[t])
        cos_fv = float((u * v).sum().item())
        per_pair.append({
            'f': f, 'g': g, 'cos': cos_fv,
            'p_by_tau': ps,
            'rs_by_tau': {t: rs_dict[t] for t in taus},
        })

        if (k + 1) % 20 == 0 or k == 0:
            leg2_lab = f'g={g:4d}' if g is not None else 'rand'
            ptxt = '  '.join(f'tau{t}:p={ps[t]:.2f}' for t in taus)
            print(f'  [{k + 1:>3}/{args.n_pairs}] f={f:4d} {leg2_lab}  {ptxt}',
                  flush=True)

    # Aggregate per tau
    summary = {}
    pair_lab = f'feature x {args.leg2}'
    print(f'\n=== Pair-level summary ({pair_lab}, d={d}, n={args.n_pairs}) ===')
    for t in taus:
        v = np.array(p_by_tau[t], float); v = v[np.isfinite(v)]
        if len(v):
            med = float(np.median(v)); mean = float(np.mean(v))
            iqr = (float(np.quantile(v, .25)), float(np.quantile(v, .75)))
            summary[t] = {'median': med, 'mean': mean, 'iqr': iqr, 'n': int(len(v))}
            print(f'  tau={t}: median={med:.3f}  mean={mean:.3f}  '
                  f'IQR=({iqr[0]:.3f}, {iqr[1]:.3f})  '
                  f'min={v.min():.3f}  max={v.max():.3f}  n={len(v)}')
        else:
            summary[t] = {'median': float('nan'), 'mean': float('nan'),
                          'iqr': (float('nan'), float('nan')), 'n': 0}
            print(f'  tau={t}: no finite fits')

    out_pkl = os.path.join(args.out, 'fit_pairs.pkl')
    with open(out_pkl, 'wb') as fout:
        pickle.dump({
            'args': vars(args), 'config': cfg, 'S': S,
            'phis': phis, 'taus': taus,
            'per_pair': per_pair,
            'p_by_tau': {t: np.array(p_by_tau[t], float) for t in taus},
            'summary': summary,
        }, fout)

    # Beeswarm across taus (matches fsec lp_exponent_beeswarm).
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    xpos = np.arange(len(taus))
    labels = []
    for ix, t in enumerate(taus):
        v = np.array(p_by_tau[t], float); v = v[np.isfinite(v)]
        if len(v) == 0:
            labels.append(f'tau={t}\n(no fits)'); continue
        jx = xpos[ix] + (rng.random(len(v)) - 0.5) * 0.35
        ax.scatter(jx, v, s=20, alpha=0.6, edgecolor='black', linewidth=0.3,
                   color='tab:blue')
        ax.hlines(np.median(v), xpos[ix] - 0.3, xpos[ix] + 0.3,
                  color='black', linewidth=2.0)
        labels.append(f'tau={t}\nmed={np.median(v):.2f}\nmean={np.mean(v):.2f}\n'
                      f'N={len(v)}')
    ax.axhline(2, color='tab:purple', ls='--', alpha=0.6, label='p=2 (L2)')
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('fitted Lp exponent n')
    ax.set_xlabel('output L2 threshold tau')
    ax.set_title(f'toy denoiser ({cfg["activation"]}, d={d}, H={cfg["hidden_size"]})  '
                 f'{pair_lab}  N={args.n_pairs} pairs/col')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.out, 'p_distribution.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
