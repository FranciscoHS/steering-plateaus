#!/usr/bin/env python3
"""10-direction plateau-form fit.

Takes 10 raw MELBO directions found on a single FineWeb activation
(qwen3-1.7b, inject L7, measure L-2), and:

  Phase 0: calibrate thresholds by sweeping one direction and inspecting curves.
  Phase A: sweep each of the 10 directions individually → per-direction
           breaking angle alpha_i* at the calibrated threshold.
  Phase B: sweep N=20 random dense unit combinations u^(j) = sum_i c_i^(j) d_i^perp
           → per-combination breaking angle alpha^(j).
  Phase C: fit a single scalar exponent n per metric to the iso-plateau condition
               sum_i |c_i^(j) * sin(alpha^(j)) / sin(alpha_i*)|^n = 1
           across all combinations (once c_i^(j) are the combination weights in
           the orthogonalized MELBO basis d_i^perp).

Output: results/ten_direction_fit/qwen3-1.7b_L7/ with data + plot.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from src.model import (
    load_model, forward_from_layer_to_layer, forward_from_layer,
    _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import extract_context_and_activation


# --- Config ---
OUTPUT_DIR = 'results/ten_direction_fit/qwen3-1.7b_L7_random_n30'
MODEL = 'qwen3-1.7b'
INJECT_LAYER = 7
MEASURE_LAYER = -2           # penultimate (n_layers-2)
SEQ_LEN = 5
N_ACT = 30                    # number of activations to sweep (medianed)
DATA_SEED = 42
K = 10                        # number of random unit directions
N_COMBO = 100                 # number of random combination directions
MAX_ANGLE_DEG = 40.0
N_ANGLES = 80
DIRECTIONS_SEED = 1           # seed for the K random directions
COMBO_SEED = 0                # seed for combinations


def sweep_unit_direction(model, ctx, a_unit, a_norm, u_unit, ref_L, ref_logp,
                         inject_layer, measure_layer, device, angles_rad,
                         batch_size=64):
    """Sweep along a unit direction (already orthogonal to a_unit, unit norm).

    Returns dict with cos_dist, l2, kl arrays of shape [n_angles].
    """
    model_dtype = next(model.parameters()).dtype
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    n = len(angles_rad)

    # Build perturbed points: [n, d_model]
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * u_unit[None])

    cos_arr = np.zeros(n)
    l2_arr = np.zeros(n)
    kl_arr = np.zeros(n)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = perturbed[start:end].to(model_dtype).unsqueeze(1)  # [B,1,d]
            ctx_b = ctx.expand(end - start, -1, -1)

            # Hidden state at measure layer (for cos + L2)
            pert_L = forward_from_layer_to_layer(
                model, ctx_b.to(model_dtype), batch, inject_layer, measure_layer
            ).float()
            cos_arr[start:end] = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
            l2_arr[start:end]  = (pert_L - ref_L).norm(dim=-1).cpu().numpy()

            # Logits for KL
            pert_logits = forward_from_layer(
                model, ctx_b.to(model_dtype), batch, inject_layer
            ).float()
            pert_logp = F.log_softmax(pert_logits, dim=-1)
            # KL(ref || pert) = sum ref * (log ref - log pert); ref_logp has shape [1, V]
            ref_p = ref_logp.exp()
            kl = (ref_p * (ref_logp - pert_logp)).sum(dim=-1)
            kl_arr[start:end] = kl.cpu().numpy()

    return {'cos': cos_arr, 'l2': l2_arr, 'kl': kl_arr}


def find_breaking_angle(angles_deg, response, threshold):
    """First angle at which `response` crosses `threshold` (linear interp).

    Returns NaN if never crossed.
    """
    if response[0] >= threshold:
        return 0.0
    above = np.where(response >= threshold)[0]
    if len(above) == 0:
        return float('nan')
    i = above[0]
    # Linear interp between angles[i-1] and angles[i]
    y0, y1 = response[i-1], response[i]
    x0, x1 = angles_deg[i-1], angles_deg[i]
    t = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
    return x0 + t * (x1 - x0)


def fit_general_B_n2(c_matrix, alpha_combos_deg, alpha_indiv_deg):
    """Fit general symmetric PSD-ish M = B/S_c assuming n=2 via LSQ.

    Pin the diagonal from individual sweeps: M_jj = 1/sin(alpha_j*)^2 (in radians).
    Fit off-diagonals M_jk (j<k) from combination data:
        c^T M c = 1/sin(alpha)^2
        => sum_j c_j^2 M_jj  +  2 sum_{j<k} c_j c_k M_jk  =  1/sin(alpha)^2.
    Subtract the pinned diagonal contribution from the RHS and LSQ the rest.

    Returns dict with M (KxK), n_combos_used, residual norms, and predicted
    breaking angles per combo.
    """
    K = c_matrix.shape[1]
    valid = np.isfinite(alpha_combos_deg) & np.all(np.isfinite(c_matrix), axis=1) \
            & np.all(np.isfinite(alpha_indiv_deg))
    if valid.sum() < K * (K - 1) // 2:
        return {'M': None, 'n_used': int(valid.sum()),
                'reason': 'insufficient combos for full off-diagonal fit',
                'alpha_pred_deg': np.full(len(alpha_combos_deg), np.nan)}

    a_indiv_rad = np.deg2rad(alpha_indiv_deg)
    M_diag = 1.0 / np.sin(a_indiv_rad) ** 2            # shape [K]
    c_v = c_matrix[valid]                               # [N_v, K]
    a_v = np.deg2rad(alpha_combos_deg[valid])           # [N_v]
    rhs_full = 1.0 / np.sin(a_v) ** 2                   # [N_v]
    diag_contrib = (c_v ** 2) @ M_diag                  # [N_v]
    rhs = rhs_full - diag_contrib                       # y for off-diag fit

    # Build design matrix of 2*c_j*c_k features for j<k
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
    X = np.stack([2 * c_v[:, j] * c_v[:, k] for j, k in pairs], axis=1)  # [N_v, K*(K-1)/2]
    sol, *_ = np.linalg.lstsq(X, rhs, rcond=None)

    # Build M
    M = np.diag(M_diag)
    for (j, k), v in zip(pairs, sol):
        M[j, k] = v
        M[k, j] = v

    # Predicted breaking angles (all combos, including invalid ones get nan)
    alpha_pred_deg = np.full(len(alpha_combos_deg), np.nan)
    quad = np.einsum('ni,ij,nj->n', c_matrix, M, c_matrix)  # c^T M c
    good = valid & (quad > 0)
    sin_pred = np.clip(1.0 / np.sqrt(quad[good]), -1, 1)
    alpha_pred_deg[good] = np.rad2deg(np.arcsin(sin_pred))

    # Residuals
    res_rhs = X @ sol - rhs
    return {
        'M': M,
        'n_used': int(valid.sum()),
        'sol': sol,
        'pairs': pairs,
        'rhs_residual_rms': float(np.sqrt(np.mean(res_rhs ** 2))),
        'rhs_std': float(np.std(rhs)),
        'rhs_frac_unexplained': float(np.sqrt(np.mean(res_rhs ** 2)) / (np.std(rhs) + 1e-20)),
        'alpha_pred_deg': alpha_pred_deg,
    }


def fit_n(c_matrix, alpha_combos_deg, alpha_indiv_deg):
    """Fit single exponent n to: sum_i |c_i^(j) sin(alpha^(j)) / sin(alpha_i*)|^n = 1

    c_matrix: [N_combo, K] (rows are combination weight vectors, normalized)
    alpha_combos_deg: [N_combo] breaking angles for combos
    alpha_indiv_deg: [K] breaking angles for individual dirs

    Returns (n_hat, residuals_at_n_hat).
    """
    valid = np.isfinite(alpha_combos_deg) & np.all(np.isfinite(c_matrix), axis=1)
    if valid.sum() < 2:
        return float('nan'), None
    c_m = c_matrix[valid]
    a_c = np.deg2rad(alpha_combos_deg[valid])
    a_i = np.deg2rad(alpha_indiv_deg)

    # ratio_ij = c_m[j,i] * sin(a_c[j]) / sin(a_i[i])
    ratio = c_m * (np.sin(a_c)[:, None] / np.sin(a_i)[None, :])
    # For each j: sum_i |ratio_ji|^n = 1

    def sse(n):
        s = np.sum(np.abs(ratio) ** n, axis=1)
        return np.sum((s - 1.0) ** 2)

    res = minimize_scalar(sse, bounds=(0.1, 10.0), method='bounded',
                          options={'xatol': 1e-4})
    n_hat = res.x
    s_fit = np.sum(np.abs(ratio) ** n_hat, axis=1)
    return n_hat, s_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Comma-separated cos,l2,kl thresholds to skip Phase 0. '
                             'If unset, auto-pick at 50%% of max response from Phase 0.')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pkl = os.path.join(OUTPUT_DIR, 'ten_direction_fit.pkl')

    if args.plot_only:
        with open(out_pkl, 'rb') as f:
            results = pickle.load(f)
        plot_results(results)
        return

    # ---- Load model ----
    print(f'Loading {MODEL}...')
    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    inject_layer = INJECT_LAYER
    measure_layer = MEASURE_LAYER
    if measure_layer < 0:
        measure_layer = n_layers + measure_layer
    print(f'  n_layers={n_layers}, inject@L{inject_layer}, measure@L{measure_layer}')
    d_model = model.config.hidden_size

    # ---- Load N_ACT activations ----
    print(f'Loading FineWeb (seed={DATA_SEED}, seq_len={SEQ_LEN}, N_ACT={N_ACT})')
    token_ids_list = load_fineweb_fixed_length(
        max(100, N_ACT), tokenizer, SEQ_LEN, seed=DATA_SEED)
    ctx_all, act_all = extract_context_and_activation(
        model, tokenizer, token_ids_list, inject_layer, device)
    ctx_all = ctx_all[:N_ACT].to(device)   # [N_ACT, seq_len-1, d]
    act_all = act_all[:N_ACT].to(device)   # [N_ACT, d]
    a_norms = act_all.norm(dim=1).float()  # [N_ACT]
    a_units = (act_all / a_norms[:, None]).float()  # [N_ACT, d]
    print(f'  a_norms: min {a_norms.min().item():.2f}, max {a_norms.max().item():.2f}, '
          f'mean {a_norms.mean().item():.2f}')

    # ---- Generate K random directions (mutually orthonormal via Gram-Schmidt) ----
    # We will also re-orthogonalize against each a_unit inside the per-activation loop.
    g = torch.Generator(device='cpu').manual_seed(DIRECTIONS_SEED)
    directions = torch.randn(K, d_model, generator=g).to(device).float()
    d_base = torch.zeros_like(directions)
    for i in range(K):
        v = directions[i].clone()
        for j in range(i):
            v = v - (v @ d_base[j]) * d_base[j]
        d_base[i] = F.normalize(v, dim=0)
    pw_base = (d_base @ d_base.T - torch.eye(K, device=device)).abs().max().item()
    print(f'  base K={K} random directions, mutual orth max = {pw_base:.2e}')

    melbo_args = {'note': 'random Gram-Schmidt directions, not MELBO',
                  'N_ACT': N_ACT, 'N_COMBO': N_COMBO, 'K': K}

    model_dtype = next(model.parameters()).dtype
    angles_deg = np.linspace(0.0, MAX_ANGLE_DEG, N_ANGLES + 1)[1:]  # exclude 0
    angles_rad = np.deg2rad(angles_deg)

    # ---- Generate combinations once (shared across activations) ----
    rng = np.random.RandomState(COMBO_SEED)
    c_matrix = rng.randn(N_COMBO, K)
    c_matrix = c_matrix / np.linalg.norm(c_matrix, axis=1, keepdims=True)  # [N_COMBO, K]
    c_t = torch.tensor(c_matrix, dtype=torch.float32, device=device)

    # Storage: [N_ACT, K, N_ANGLES] and [N_ACT, N_COMBO, N_ANGLES] per metric
    indiv_per_act = {m: np.zeros((N_ACT, K, N_ANGLES)) for m in ('cos', 'l2', 'kl')}
    combo_per_act = {m: np.zeros((N_ACT, N_COMBO, N_ANGLES)) for m in ('cos', 'l2', 'kl')}

    # Per-activation pairwise diagnostics
    all_pairwise = np.zeros(N_ACT)

    print(f'\nSweeping {N_ACT} activations x (K={K} + N={N_COMBO}) directions'
          f' x {N_ANGLES} angles')
    for a_idx in range(N_ACT):
        ctx_a = ctx_all[a_idx:a_idx+1]
        act_a = act_all[a_idx]
        a_norm = a_norms[a_idx]
        a_unit = a_units[a_idx]

        # Per-activation Gram-Schmidt: re-orthogonalize d_base against a_unit and each other.
        d_orth = d_base.clone()
        d_orth = d_orth - (d_orth @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        # Re-orthogonalize mutually after projection
        for i in range(K):
            for j in range(i):
                d_orth[i] = d_orth[i] - (d_orth[i] @ d_orth[j]) * d_orth[j]
            d_orth[i] = F.normalize(d_orth[i], dim=0)
        all_pairwise[a_idx] = (d_orth @ d_orth.T - torch.eye(K, device=device)).abs().max().item()

        # Reference forward for this activation
        with torch.no_grad():
            ref_L = forward_from_layer_to_layer(
                model, ctx_a.to(model_dtype), act_a[None, None].to(model_dtype),
                inject_layer, measure_layer).float()
            ref_logits = forward_from_layer(
                model, ctx_a.to(model_dtype), act_a[None, None].to(model_dtype),
                inject_layer).float()
            ref_logp = F.log_softmax(ref_logits, dim=-1)

        # Individual sweeps
        for i in range(K):
            out = sweep_unit_direction(
                model, ctx_a, a_unit, a_norm, d_orth[i], ref_L, ref_logp,
                inject_layer, measure_layer, device, angles_rad)
            for m in ('cos', 'l2', 'kl'):
                indiv_per_act[m][a_idx, i] = out[m]

        # Combination unit vectors in this activation's basis
        u_combos = c_t @ d_orth  # [N_COMBO, d_model]
        u_norms_here = u_combos.norm(dim=1, keepdim=True)
        u_combos = u_combos / u_norms_here

        for j in range(N_COMBO):
            out = sweep_unit_direction(
                model, ctx_a, a_unit, a_norm, u_combos[j], ref_L, ref_logp,
                inject_layer, measure_layer, device, angles_rad)
            for m in ('cos', 'l2', 'kl'):
                combo_per_act[m][a_idx, j] = out[m]

        if (a_idx + 1) % 5 == 0:
            print(f'  {a_idx+1}/{N_ACT} activations done'
                  f' (max pairwise so far: {all_pairwise[:a_idx+1].max():.2e})')

    print(f'  done. max pairwise across activations: {all_pairwise.max():.2e}')

    # ---- Median across activations ----
    indiv_sweeps = []
    for i in range(K):
        indiv_sweeps.append(
            {m: np.median(indiv_per_act[m][:, i, :], axis=0) for m in ('cos', 'l2', 'kl')}
        )
    for i in range(K):
        print(f'  dir {i} (median): cos max {indiv_sweeps[i]["cos"].max():.4f}, '
              f'l2 max {indiv_sweeps[i]["l2"].max():.2f}, '
              f'kl max {indiv_sweeps[i]["kl"].max():.3f}')

    # ---- Phase 0: threshold calibration ----
    # Use the median of max responses across individual directions, halved.
    # Or user-provided.
    if args.thresholds:
        c_thr, l_thr, k_thr = [float(x) for x in args.thresholds.split(',')]
    else:
        cos_maxes = np.array([s['cos'].max() for s in indiv_sweeps])
        l2_maxes = np.array([s['l2'].max() for s in indiv_sweeps])
        kl_maxes = np.array([s['kl'].max() for s in indiv_sweeps])
        c_thr = 0.5 * np.median(cos_maxes)
        l_thr = 0.5 * np.median(l2_maxes)
        k_thr = 0.5 * np.median(kl_maxes)
    print(f'\nThresholds: cos={c_thr:.4f}, l2={l_thr:.2f}, kl={k_thr:.3f}')

    alpha_indiv = {
        'cos': np.array([find_breaking_angle(angles_deg, s['cos'], c_thr) for s in indiv_sweeps]),
        'l2':  np.array([find_breaking_angle(angles_deg, s['l2'],  l_thr) for s in indiv_sweeps]),
        'kl':  np.array([find_breaking_angle(angles_deg, s['kl'],  k_thr) for s in indiv_sweeps]),
    }
    for m, arr in alpha_indiv.items():
        print(f'  alpha_i*[{m}]: {np.nanmin(arr):.2f} to {np.nanmax(arr):.2f} deg, '
              f'{np.sum(np.isnan(arr))} NaN')

    # ---- Combination median sweeps ----
    combo_sweeps = []
    alpha_combo = {'cos': [], 'l2': [], 'kl': []}
    for j in range(N_COMBO):
        cs = {m: np.median(combo_per_act[m][:, j, :], axis=0) for m in ('cos', 'l2', 'kl')}
        combo_sweeps.append(cs)
        for m, thr in (('cos', c_thr), ('l2', l_thr), ('kl', k_thr)):
            alpha_combo[m].append(find_breaking_angle(angles_deg, cs[m], thr))
    alpha_combo = {m: np.array(v) for m, v in alpha_combo.items()}
    print('  combo alpha* ranges:')
    for m in ('cos', 'l2', 'kl'):
        arr = alpha_combo[m]
        print(f'    {m}: {np.nanmin(arr):.2f} to {np.nanmax(arr):.2f} deg, '
              f'{int(np.sum(np.isnan(arr)))} NaN')
    c_matrix_eff = c_matrix  # d_base and d_orth are orthonormal; keep identity

    # ---- Phase C: fit n (axis-aligned) and general-B (n=2) ----
    print('\nPhase C: fits')
    fit = {}
    fit_B = {}
    for m in ('cos', 'l2', 'kl'):
        n_hat, s_fit = fit_n(c_matrix_eff, alpha_combo[m], alpha_indiv[m])
        fit[m] = {'n_hat': n_hat, 's_fit': s_fit}
        fit_B[m] = fit_general_B_n2(c_matrix_eff, alpha_combo[m], alpha_indiv[m])
        print(f'  {m}: axis-aligned n_hat = {n_hat:.3f}; '
              f'general-B n_used={fit_B[m]["n_used"]}, '
              f'rhs residual RMS / std = '
              f'{fit_B[m].get("rhs_frac_unexplained", float("nan")):.3f}')

    # ---- Save ----
    results = {
        'K': K, 'N_COMBO': N_COMBO, 'N_ACT': N_ACT,
        'angles_deg': angles_deg,
        'a_norms': a_norms.cpu().numpy(),
        'indiv_per_act': indiv_per_act,
        'combo_per_act': combo_per_act,
        'pairwise_per_act': all_pairwise,
        'melbo_args': melbo_args,
        'thresholds': {'cos': c_thr, 'l2': l_thr, 'kl': k_thr},
        'indiv_sweeps': indiv_sweeps,
        'combo_sweeps': combo_sweeps,
        'c_matrix': c_matrix,
        'c_matrix_eff': c_matrix_eff,
        'alpha_indiv': alpha_indiv,
        'alpha_combo': alpha_combo,
        'fit': fit,
        'fit_B': fit_B,
        'pairwise_orth_inner_abs_max': float(all_pairwise.max()),
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved {out_pkl}')

    plot_results(results)


def plot_results(results):
    K = results['K']
    fit = results['fit']
    fit_B = results.get('fit_B', {})
    alpha_indiv = results['alpha_indiv']
    alpha_combo = results['alpha_combo']
    c_matrix_eff = results['c_matrix_eff']

    # Two rows: (1) axis-aligned fit, (2) general-B (n=2) fit
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, (m, label) in enumerate(zip(('cos', 'l2', 'kl'),
                                          ('Cosine distance', 'L2 distance',
                                           'KL divergence'))):
        obs_deg = alpha_combo[m]
        a_i = np.deg2rad(alpha_indiv[m])
        a_c_deg = alpha_combo[m]

        # Row 0: axis-aligned fit
        ax = axes[0, col]
        n_hat = fit[m]['n_hat']
        # finite mask EXCLUDING NaN in alpha_indiv (otherwise prediction is NaN everywhere)
        indiv_finite = np.isfinite(alpha_indiv[m])
        if indiv_finite.sum() >= 1 and np.isfinite(n_hat):
            a_i_safe = np.deg2rad(alpha_indiv[m][indiv_finite])
            c_safe = c_matrix_eff[:, indiv_finite]
            coeff = np.sum(np.abs(c_safe / np.sin(a_i_safe)[None, :]) ** n_hat, axis=1)
            coeff = np.where(coeff > 0, coeff, np.nan)
            sin_pred = np.clip(coeff ** (-1.0 / n_hat), -1, 1)
            a_pred_deg = np.rad2deg(np.arcsin(sin_pred))
        else:
            a_pred_deg = np.full_like(obs_deg, np.nan, dtype=float)
        valid = np.isfinite(a_c_deg) & np.isfinite(a_pred_deg)
        ax.scatter(a_pred_deg[valid], obs_deg[valid], s=30, alpha=0.7)
        if valid.sum() >= 2:
            pd_v = a_pred_deg[valid]
            od_v = obs_deg[valid]
            lo = min(pd_v.min(), od_v.min())
            hi = max(pd_v.max(), od_v.max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
            r = np.corrcoef(pd_v, od_v)[0, 1] if valid.sum() >= 3 else np.nan
            resid = od_v - pd_v
            ax.text(0.03, 0.97,
                    f'n valid = {valid.sum()}\nr = {r:.2f}\nRMSE = {np.sqrt(np.mean(resid**2)):.2f} deg',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.set_xlabel('Predicted breaking angle (deg)')
        ax.set_ylabel('Measured breaking angle (deg)')
        ax.set_title(f'{label}\naxis-aligned fit: n = {n_hat:.3f}')

        # Row 1: general-B (n=2) fit
        ax2 = axes[1, col]
        fb = fit_B.get(m, {})
        a_pred_B = fb.get('alpha_pred_deg', np.full_like(obs_deg, np.nan, dtype=float))
        valid_B = np.isfinite(a_c_deg) & np.isfinite(a_pred_B)
        ax2.scatter(a_pred_B[valid_B], obs_deg[valid_B], s=30, alpha=0.7, color='#d62728')
        if valid_B.sum() >= 2:
            pd_v = a_pred_B[valid_B]
            od_v = obs_deg[valid_B]
            lo = min(pd_v.min(), od_v.min())
            hi = max(pd_v.max(), od_v.max())
            ax2.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
            r = np.corrcoef(pd_v, od_v)[0, 1] if valid_B.sum() >= 3 else np.nan
            resid = od_v - pd_v
            frac = fb.get('rhs_frac_unexplained', float('nan'))
            ax2.text(0.03, 0.97,
                     f'n valid = {valid_B.sum()}\nr = {r:.2f}\n'
                     f'RMSE = {np.sqrt(np.mean(resid**2)):.2f} deg\n'
                     f'rhs residual/std = {frac:.2f}',
                     transform=ax2.transAxes, va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax2.set_xlabel('Predicted breaking angle (deg)')
        ax2.set_ylabel('Measured breaking angle (deg)')
        ax2.set_title(f'general-B fit (assumes n=2)')

    plt.suptitle(f'qwen3-1.7b L7: K={K} random orthonormal directions, '
                 f'N={results["N_COMBO"]} random unit combinations',
                 fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'ten_direction_fit.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
