#!/usr/bin/env python3
"""45 pair iso-plateau slices from 10 contrastive directions.

Qwen3-1.7b, layer 14 (where contrastive language features live).

Phase A: single-direction sweeps for the 10 contrastive directions + 10 random
orthogonalized directions, median over N_ACT activations → sensitivity baseline
showing contrastive directions break earlier.

Phase B: for each of the C(10,2)=45 pairs, run a 2D on-sphere angle grid and
collect cos/L2/KL response per activation. Save all grids for later contour
extraction at arbitrary thresholds.

Phase C (in plot): fit a superellipse exponent per pair per metric at the
hard-coded thresholds, produce overlay plot of normalized contours against
reference shapes.
"""
import os, sys, pickle, argparse, itertools, time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from src.model import (
    load_model, forward_from_layer_to_layer, forward_from_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import extract_context_and_activation


# --- Config ---
MODEL = 'qwen3-1.7b'
INJECT_LAYER = 14
MEASURE_LAYER = -2
SEQ_LEN = 5
DATA_SEED = 42
N_ACT = 30

N_CONTRASTIVE = 10
N_RAND = 10
RAND_SEED = 1

MAX_ANGLE_DEG_1D = 40.0
N_ANGLES_1D = 80
MAX_ANGLE_DEG_2D = 30.0
N_STEPS_2D = 31

# Hard-coded thresholds from the L7 run (may need recalibration for L14 — we
# save full grids so contours can be re-extracted without resweeping).
THRESHOLDS = {'cos': 0.005, 'l2': 352.20, 'kl': 0.05}

CONTRASTIVE_PKL = 'results/allpairs_contrastive/qwen3-1.7b/directions.pkl'
OUTPUT_DIR_BASE = 'results/pair_slices_10contrastive/qwen3-1.7b_L14'
MUTUAL_ORTH = False  # set by --mutual_orth CLI flag in main()


def forward_hidden(model, ctx, points, inject, measure):
    """points: [B, d_model] (on-sphere perturbed activations). Returns [B, d] hidden."""
    model_dtype = next(model.parameters()).dtype
    return forward_from_layer_to_layer(
        model, ctx.to(model_dtype), points.to(model_dtype).unsqueeze(1),
        inject, measure).float()


def forward_logp(model, ctx, points, inject):
    model_dtype = next(model.parameters()).dtype
    logits = forward_from_layer(
        model, ctx.to(model_dtype), points.to(model_dtype).unsqueeze(1), inject).float()
    return F.log_softmax(logits, dim=-1)


def sweep_1d(model, ctx, act, a_unit, a_norm, u, ref_L, ref_logp,
             inject, measure, device, angles_rad, batch_size=64):
    """Unit u orthogonal to a_unit. Sweep angles. Return dict of metrics, each [n]."""
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * u[None])
    n = len(angles_rad)
    out = {k: np.zeros(n) for k in ('cos', 'l2', 'kl')}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            p = perturbed[start:end]
            ctx_b = ctx.expand(end - start, -1, -1)
            hid = forward_hidden(model, ctx_b, p, inject, measure)
            out['cos'][start:end] = (1 - F.cosine_similarity(ref_L, hid, dim=-1)).cpu().numpy()
            out['l2'][start:end] = (hid - ref_L).norm(dim=-1).cpu().numpy()
            logp = forward_logp(model, ctx_b, p, inject)
            refp = ref_logp.exp()
            out['kl'][start:end] = (refp * (ref_logp - logp)).sum(dim=-1).cpu().numpy()
    return out


def sweep_2d_pair(model, ctx, act, a_unit, a_norm, d1_orth, d2_orth,
                  ref_L, ref_logp, inject, measure, device, angles_rad,
                  batch_size=64):
    """2D angle grid on the sphere using the pair (d1_orth, d2_orth) (already
    orthogonal to a_unit and mutually). Return [n, n] arrays per metric."""
    n = len(angles_rad)
    s1 = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    s2 = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))
    all_points = (a_norm * (C.reshape(-1, 1) * a_unit[None]
                            + S1.reshape(-1, 1) * d1_orth[None]
                            + S2.reshape(-1, 1) * d2_orth[None]))  # [n*n, d]

    total = n * n
    cos_flat = np.zeros(total)
    l2_flat = np.zeros(total)
    kl_flat = np.zeros(total)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            p = all_points[start:end]
            ctx_b = ctx.expand(end - start, -1, -1)
            hid = forward_hidden(model, ctx_b, p, inject, measure)
            cos_flat[start:end] = (1 - F.cosine_similarity(ref_L, hid, dim=-1)).cpu().numpy()
            l2_flat[start:end] = (hid - ref_L).norm(dim=-1).cpu().numpy()
            logp = forward_logp(model, ctx_b, p, inject)
            refp = ref_logp.exp()
            kl_flat[start:end] = (refp * (ref_logp - logp)).sum(dim=-1).cpu().numpy()
    return {
        'cos': cos_flat.reshape(n, n),
        'l2': l2_flat.reshape(n, n),
        'kl': kl_flat.reshape(n, n),
    }


# ---- superellipse contour fitting (adapted from plot_isoplateau.py) ----
def extract_contour(angles, grid, threshold):
    """Return contour points [M, 2] at given threshold from a 2D grid."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(angles, angles, grid.T, levels=[threshold])
    pts = []
    if hasattr(cs, 'allsegs'):
        for seg_list in cs.allsegs:
            for seg in seg_list:
                pts.append(seg)
    else:
        for coll in cs.collections:
            for path in coll.get_paths():
                pts.append(path.vertices)
    plt.close(fig_tmp)
    return np.concatenate(pts, axis=0) if pts else np.array([]).reshape(0, 2)


def find_axis_intercept(curve, angles, threshold):
    idx = np.searchsorted(curve, threshold)
    if idx <= 0 or idx >= len(angles):
        return float('nan')
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x0


def fit_superellipse_p(contour_norm):
    x = contour_norm[:, 0]; y = contour_norm[:, 1]
    mask = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    x, y = x[mask], y[mask]
    if len(x) < 3: return None
    def res(p): return np.sum((np.abs(x)**p + np.abs(y)**p - 1.0)**2)
    r = minimize_scalar(res, bounds=(0.5, 10.0), method='bounded')
    return float(r.x)


def superellipse_curve(p, n=200):
    t = np.linspace(0, np.pi/2, n)
    x = np.abs(np.cos(t))**(2/p) * np.sign(np.cos(t))
    y = np.abs(np.sin(t))**(2/p) * np.sign(np.sin(t))
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--mutual_orth', action='store_true',
                        help='Pre-Gram-Schmidt the 10 contrastive directions mutually '
                             '(order = languages list order) before the pair sweeps.')
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR_BASE + ('_mutual_orth' if args.mutual_orth else '')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pkl = os.path.join(OUTPUT_DIR, 'pair_slices.pkl')

    if args.plot_only:
        with open(out_pkl, 'rb') as f:
            results = pickle.load(f)
        make_plots(results)
        return

    # ---- Load model ----
    print(f'Loading {MODEL}...')
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    inject = INJECT_LAYER
    measure = MEASURE_LAYER if MEASURE_LAYER >= 0 else n_layers + MEASURE_LAYER
    print(f'  n_layers={n_layers}, inject@L{inject}, measure@L{measure}')
    d_model = model.config.hidden_size

    # ---- Load contrastive directions ----
    with open(CONTRASTIVE_PKL, 'rb') as f:
        c_data = pickle.load(f)
    langs = c_data['languages']
    assert len(langs) == N_CONTRASTIVE, f'Expected {N_CONTRASTIVE} langs, got {len(langs)}'
    contrastive = torch.stack([c_data['features'][k].to(device).float()
                               for k in langs])  # [10, d]
    contrastive = F.normalize(contrastive, dim=1)
    print(f'Loaded {len(langs)} contrastive directions: {langs}')
    pre_cos = (contrastive @ contrastive.T - torch.eye(len(langs), device=device)).abs().max().item()
    print(f'  pre-orth pairwise |cos| max: {pre_cos:.3f}')
    if args.mutual_orth:
        # Mutual Gram-Schmidt in the order of `langs`.
        orthed = torch.zeros_like(contrastive)
        for i in range(len(langs)):
            v = contrastive[i].clone()
            for j in range(i):
                v = v - (v @ orthed[j]) * orthed[j]
            orthed[i] = F.normalize(v, dim=0)
        contrastive = orthed
        post_cos = (contrastive @ contrastive.T - torch.eye(len(langs), device=device)).abs().max().item()
        print(f'  APPLIED mutual Gram-Schmidt, post-orth pairwise |cos| max: {post_cos:.2e}')

    # ---- Random directions for baseline ----
    g = torch.Generator(device='cpu').manual_seed(RAND_SEED)
    rand_dirs = torch.randn(N_RAND, d_model, generator=g).to(device).float()
    rand_dirs = F.normalize(rand_dirs, dim=1)

    # ---- Load N_ACT FineWeb activations ----
    print(f'Loading FineWeb (seed={DATA_SEED}, N_ACT={N_ACT})')
    toks = load_fineweb_fixed_length(max(100, N_ACT), tokenizer, SEQ_LEN, seed=DATA_SEED)
    ctx_all, act_all = extract_context_and_activation(model, tokenizer, toks, inject, device)
    ctx_all = ctx_all[:N_ACT].to(device)
    act_all = act_all[:N_ACT].to(device)
    a_norms = act_all.norm(dim=1).float()
    a_units = act_all / a_norms[:, None]
    print(f'  a_norms: min {a_norms.min().item():.2f}, '
          f'max {a_norms.max().item():.2f}, mean {a_norms.mean().item():.2f}')

    angles_1d = np.linspace(0, MAX_ANGLE_DEG_1D, N_ANGLES_1D + 1)[1:]
    angles_1d_rad = np.deg2rad(angles_1d)
    angles_2d = np.linspace(0, MAX_ANGLE_DEG_2D, N_STEPS_2D + 1)[1:]
    angles_2d_rad = np.deg2rad(angles_2d)

    model_dtype = next(model.parameters()).dtype

    # ---- Phase A: 1D sweeps (contrastive + random) ----
    print(f'\nPhase A: {N_CONTRASTIVE} contrastive + {N_RAND} random, '
          f'{N_ACT} acts, {N_ANGLES_1D} angles')
    phase_a = {m: np.zeros((N_ACT, N_CONTRASTIVE + N_RAND, N_ANGLES_1D))
               for m in ('cos', 'l2', 'kl')}
    t0 = time.time()
    for a in range(N_ACT):
        a_unit = a_units[a]; a_norm = a_norms[a]; ctx = ctx_all[a:a+1]
        with torch.no_grad():
            ref_L = forward_from_layer_to_layer(
                model, ctx.to(model_dtype), act_all[a][None, None].to(model_dtype),
                inject, measure).float()
            ref_logits = forward_from_layer(
                model, ctx.to(model_dtype), act_all[a][None, None].to(model_dtype),
                inject).float()
            ref_logp = F.log_softmax(ref_logits, dim=-1)
        for i, d in enumerate(torch.cat([contrastive, rand_dirs], dim=0)):
            d_orth = d - (d @ a_unit) * a_unit
            d_orth = F.normalize(d_orth, dim=0)
            out = sweep_1d(model, ctx, act_all[a], a_unit, a_norm, d_orth,
                           ref_L, ref_logp, inject, measure, device, angles_1d_rad)
            for m in ('cos', 'l2', 'kl'):
                phase_a[m][a, i] = out[m]
        if (a + 1) % 5 == 0:
            dt = time.time() - t0
            print(f'  {a+1}/{N_ACT} acts ({dt:.0f}s elapsed, '
                  f'ETA {dt/(a+1)*(N_ACT-a-1):.0f}s)')

    # ---- Phase B: 2D pair slices ----
    pairs = list(itertools.combinations(range(N_CONTRASTIVE), 2))
    print(f'\nPhase B: {len(pairs)} pair slices, {N_ACT} acts, '
          f'{N_STEPS_2D}x{N_STEPS_2D} grid')
    grids = {m: np.zeros((len(pairs), N_ACT, N_STEPS_2D, N_STEPS_2D))
             for m in ('cos', 'l2', 'kl')}
    t0 = time.time()
    for a in range(N_ACT):
        a_unit = a_units[a]; a_norm = a_norms[a]; ctx = ctx_all[a:a+1]
        with torch.no_grad():
            ref_L = forward_from_layer_to_layer(
                model, ctx.to(model_dtype), act_all[a][None, None].to(model_dtype),
                inject, measure).float()
            ref_logits = forward_from_layer(
                model, ctx.to(model_dtype), act_all[a][None, None].to(model_dtype),
                inject).float()
            ref_logp = F.log_softmax(ref_logits, dim=-1)

        # Orthogonalize all contrastive dirs against a_unit (each dir once per act)
        d_orth_all = contrastive - (contrastive @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        d_orth_all = F.normalize(d_orth_all, dim=1)  # [10, d]

        for p_idx, (i, j) in enumerate(pairs):
            di = d_orth_all[i]
            # Orthogonalize dj against di within the tangent space
            dj = d_orth_all[j]
            dj = dj - (dj @ di) * di
            dj = F.normalize(dj, dim=0)
            out = sweep_2d_pair(model, ctx, act_all[a], a_unit, a_norm, di, dj,
                                ref_L, ref_logp, inject, measure, device, angles_2d_rad)
            for m in ('cos', 'l2', 'kl'):
                grids[m][p_idx, a] = out[m]
        dt = time.time() - t0
        print(f'  act {a+1}/{N_ACT} ({dt:.0f}s elapsed, '
              f'ETA {dt/(a+1)*(N_ACT-a-1):.0f}s)')

    # ---- Save ----
    results = {
        'model': MODEL, 'inject': inject, 'measure': measure,
        'langs': list(langs),
        'pairs': pairs,
        'angles_1d_deg': angles_1d,
        'angles_2d_deg': angles_2d,
        'thresholds': dict(THRESHOLDS),
        'phase_a': phase_a,   # [N_ACT, 2K, N_ANGLES]
        'grids': grids,       # [N_PAIRS, N_ACT, N_STEPS, N_STEPS]
        'a_norms': a_norms.cpu().numpy(),
        'n_contrastive': N_CONTRASTIVE, 'n_rand': N_RAND,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {out_pkl}')

    make_plots(results)


def make_plots(results):
    langs = results['langs']
    pairs = results['pairs']
    angles_1d = results['angles_1d_deg']
    angles_2d = results['angles_2d_deg']
    thresholds = results['thresholds']
    K = results['n_contrastive']
    N_RAND_ = results['n_rand']

    # --- Phase A: sensitivity plot ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, m, label in zip(axes, ('cos', 'l2', 'kl'),
                             ('Cosine distance', 'L2 distance', 'KL divergence')):
        a = np.median(results['phase_a'][m], axis=0)  # [2K, N_ANGLES]
        for i in range(K):
            ax.plot(angles_1d, a[i], color='tab:blue', lw=1.3, alpha=0.9,
                    label=(f'contrastive ({K})' if i == 0 else None))
        for i in range(N_RAND_):
            ax.plot(angles_1d, a[K + i], color='tab:gray', lw=1.0, alpha=0.6,
                    ls='--', label=(f'random ({N_RAND_})' if i == 0 else None))
        ax.axhline(thresholds[m], color='k', ls=':', lw=1.2,
                   label=f'threshold {thresholds[m]:g}')
        ax.set_xlabel('Perturbation angle (deg)')
        ax.set_ylabel(label)
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle(f'qwen3-1.7b L{results["inject"]}: contrastive vs random sensitivity '
                 f'(median over {results["phase_a"][m].shape[0]} acts)', fontsize=12)
    plt.tight_layout()
    p0 = os.path.join(OUTPUT_DIR, 'phase_a_sensitivity.png')
    plt.savefig(p0, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p0}')

    # --- Phase B: per-pair superellipse fits ---
    fitted = {m: [] for m in ('cos', 'l2', 'kl')}
    contours_norm = {m: [] for m in ('cos', 'l2', 'kl')}
    for p_idx, (i, j) in enumerate(pairs):
        for m in ('cos', 'l2', 'kl'):
            median_grid = np.median(results['grids'][m][p_idx], axis=0)  # [N_STEPS, N_STEPS]
            thr = thresholds[m]
            # axis intercepts
            row0 = median_grid[:, 0]; col0 = median_grid[0, :]
            th1 = find_axis_intercept(row0, angles_2d, thr)
            th2 = find_axis_intercept(col0, angles_2d, thr)
            if not (np.isfinite(th1) and np.isfinite(th2) and th1 > 0 and th2 > 0):
                fitted[m].append(None)
                contours_norm[m].append(None)
                continue
            cp = extract_contour(angles_2d, median_grid, thr)
            if len(cp) == 0:
                fitted[m].append(None)
                contours_norm[m].append(None)
                continue
            cn = cp.copy().astype(float)
            cn[:, 0] /= th1; cn[:, 1] /= th2
            p_fit = fit_superellipse_p(cn)
            fitted[m].append(p_fit)
            contours_norm[m].append(cn)

    # --- Overlay plot ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, m, label in zip(axes, ('cos', 'l2', 'kl'),
                             ('Cosine distance', 'L2 distance', 'KL divergence')):
        # reference shapes
        xc, yc = superellipse_curve(2); ax.plot(xc, yc, '--', color='#888', lw=1.5, label='p=2 circle')
        x1 = 1 - np.linspace(0, 1, 200); y1 = np.linspace(0, 1, 200)
        ax.plot(x1, y1, ':', color='#ccc', lw=1.5, label='p=1 diamond')
        xs, ys = superellipse_curve(20); ax.plot(xs, ys, ':', color='#bbb', lw=1.5, label='p=∞ square')
        # data contours
        valid_ps = [p for p in fitted[m] if p is not None]
        for cn in contours_norm[m]:
            if cn is None: continue
            ang = np.arctan2(cn[:, 1], cn[:, 0])
            order = np.argsort(ang)
            ax.plot(cn[order, 0], cn[order, 1], '-', color='#377eb8',
                    lw=0.9, alpha=0.35)
        ax.plot(1, 0, 'kv', markersize=7); ax.plot(0, 1, 'k>', markersize=7)
        ax.set_xlim(-0.05, 1.5); ax.set_ylim(-0.05, 1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        med = np.median(valid_ps) if valid_ps else float('nan')
        if valid_ps:
            lo, hi = np.percentile(valid_ps, [5, 95])
            ax.set_title(f'{label}\n{len(valid_ps)}/{len(pairs)} pairs, '
                         f'p median={med:.2f} [5–95%: {lo:.2f}, {hi:.2f}]')
        else:
            ax.set_title(f'{label}\n(no valid fits at threshold={thresholds[m]})')
        ax.set_xlabel(r'$\alpha_i / \theta_i^*$')
        ax.set_ylabel(r'$\alpha_j / \theta_j^*$')
        ax.legend(fontsize=9, loc='upper right')
    plt.suptitle(f'qwen3-1.7b L{results["inject"]}: 45 pair iso-plateau slices '
                 f'(10 language contrastive directions)', fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, 'pair_slices_overlay.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p1}')

    # --- Histogram of fitted p ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m, label in zip(axes, ('cos', 'l2', 'kl'),
                             ('Cosine distance', 'L2 distance', 'KL divergence')):
        vals = [p for p in fitted[m] if p is not None]
        if vals:
            ax.hist(vals, bins=np.linspace(0.5, 6.0, 31), color='#377eb8', alpha=0.8)
            med = np.median(vals)
            ax.axvline(2, color='k', ls='--', lw=1, label='p=2')
            ax.axvline(med, color='red', lw=1.5, label=f'median={med:.2f}')
            ax.set_title(f'{label}\n{len(vals)}/{len(pairs)} pairs fitted')
        else:
            ax.set_title(f'{label}\n(no valid fits)')
        ax.set_xlabel('fitted p')
        ax.set_ylabel('count')
        ax.legend()
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'fitted_p_histogram.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p2}')

    # Save the fit summaries
    with open(os.path.join(OUTPUT_DIR, 'fitted_p_summary.pkl'), 'wb') as f:
        pickle.dump({'fitted': fitted, 'contours_norm': contours_norm,
                     'pairs': pairs, 'langs': langs}, f)


if __name__ == '__main__':
    main()
