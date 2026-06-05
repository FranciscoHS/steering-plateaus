#!/usr/bin/env python3
"""
Experiment 1: fit the superellipse exponent n for a learned (d1, d2) pair.

Given the pair learned by learn_superellipse_dict.py, run the full plateau-form
fit in the (d1, d2) plane:
  - per activation, orthogonalize d1, d2 against a (Gram-Schmidt -> orthonormal
    tangent basis d1_perp, d2_perp);
  - sweep the mixing angle phi in [0, 90] deg, forming the unit direction
        u(phi) = cos(phi) d1_perp + sin(phi) d2_perp;
  - for each phi, sweep the perturbation angle theta and find the breaking angle
    alpha(phi) where the downstream response first crosses a threshold;
  - the axis sweeps (phi=0, 90) give alpha_1*, alpha_2*;
  - fit a single exponent n to  sum_i |c_i sin(alpha)/sin(alpha_i*)|^n = 1
    over the intermediate-phi combinations (c = (cos phi, sin phi)).

n >> 2 confirms the super-additive (superellipse) interference geometry the
optimization targeted. n ~ 2 means the loss found something that is NOT
superellipse-like (a sign the procedure exploited F's nonlinearity rather than
the |.|^n feature structure).

Runs on BOTH the activations the pair was optimized on AND a fresh random set.

Usage:
    python scripts/superellipse/fit_superellipse_pair.py \
        --dict_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl
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
    load_model, forward_from_layer_to_layer, forward_from_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length


# ---- helpers copied from sweep_ten_direction_fit.py (kept self-contained) ----

def extract_context_and_activation(model, token_ids_list, layer, device, batch_size=64):
    all_contexts, all_activations = [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def sweep_unit_direction(model, ctx, a_unit, a_norm, u_unit, ref_L, ref_logp,
                         inject_layer, measure_layer, device, angles_rad, batch_size=64):
    """Sweep along a unit direction (orthogonal to a_unit). Returns cos/l2/kl arrays."""
    model_dtype = next(model.parameters()).dtype
    cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    n = len(angles_rad)
    perturbed = a_norm * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * u_unit[None])
    cos_arr = np.zeros(n); l2_arr = np.zeros(n); kl_arr = np.zeros(n)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = perturbed[start:end].to(model_dtype).unsqueeze(1)
            ctx_b = ctx.expand(end - start, -1, -1)
            pert_L = forward_from_layer_to_layer(
                model, ctx_b.to(model_dtype), batch, inject_layer, measure_layer).float()
            cos_arr[start:end] = (1 - F.cosine_similarity(ref_L, pert_L, dim=-1)).cpu().numpy()
            l2_arr[start:end] = (pert_L - ref_L).norm(dim=-1).cpu().numpy()
            pert_logits = forward_from_layer(model, ctx_b.to(model_dtype), batch, inject_layer).float()
            pert_logp = F.log_softmax(pert_logits, dim=-1)
            ref_p = ref_logp.exp()
            kl = (ref_p * (ref_logp - pert_logp)).sum(dim=-1)
            kl_arr[start:end] = kl.cpu().numpy()
    return {'cos': cos_arr, 'l2': l2_arr, 'kl': kl_arr}


def find_breaking_angle(angles_deg, response, threshold):
    """First angle where response crosses threshold (linear interp). NaN if never."""
    if response[0] >= threshold:
        return 0.0
    above = np.where(response >= threshold)[0]
    if len(above) == 0:
        return float('nan')
    i = above[0]
    y0, y1 = response[i - 1], response[i]
    x0, x1 = angles_deg[i - 1], angles_deg[i]
    t = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
    return x0 + t * (x1 - x0)


def fit_n(c_matrix, alpha_combos_deg, alpha_indiv_deg):
    """Fit single exponent n: sum_i |c_i sin(alpha)/sin(alpha_i*)|^n = 1."""
    valid = np.isfinite(alpha_combos_deg) & np.all(np.isfinite(c_matrix), axis=1)
    if valid.sum() < 2:
        return float('nan'), None, valid
    c_m = c_matrix[valid]
    a_c = np.deg2rad(alpha_combos_deg[valid])
    a_i = np.deg2rad(alpha_indiv_deg)
    ratio = c_m * (np.sin(a_c)[:, None] / np.sin(a_i)[None, :])

    def sse(n):
        s = np.sum(np.abs(ratio) ** n, axis=1)
        return np.sum((s - 1.0) ** 2)

    res = minimize_scalar(sse, bounds=(0.1, 30.0), method='bounded', options={'xatol': 1e-4})
    n_hat = res.x
    s_fit = np.sum(np.abs(ratio) ** n_hat, axis=1)
    return n_hat, s_fit, valid


# ---- core: sweep the (d1,d2) plane for one activation set ----

def sweep_set(model, ctx_all, act_all, d1, d2, phis_deg, angles_rad, angles_deg,
              inject_layer, measure_layer, device):
    """Return per-activation breaking-angle sweeps over phi, medianed.

    For each activation: orthonormalize (d1,d2) in the tangent plane at a, sweep
    u(phi) = cos(phi) d1p + sin(phi) d2p over theta. Returns dict of medianed
    response surfaces [n_phi, n_angles] per metric.
    """
    n_act = act_all.shape[0]
    n_phi = len(phis_deg)
    n_ang = len(angles_rad)
    model_dtype = next(model.parameters()).dtype
    phis_rad = np.deg2rad(phis_deg)

    surf = {m: np.zeros((n_act, n_phi, n_ang)) for m in ('cos', 'l2', 'kl')}
    orth_resid = np.zeros(n_act)  # |d1p . d2p| before re-orthogonalization

    for ai in range(n_act):
        a = act_all[ai].to(device).float()
        a_norm = a.norm()
        a_unit = a / a_norm
        ctx_a = ctx_all[ai:ai + 1].to(device)

        # tangent-plane orthonormal basis from d1, d2
        d1p = d1 - (d1 @ a_unit) * a_unit
        d1p = F.normalize(d1p, dim=0)
        d2p = d2 - (d2 @ a_unit) * a_unit
        orth_resid[ai] = abs((d1p @ F.normalize(d2p, dim=0)).item())
        d2p = d2p - (d2p @ d1p) * d1p       # Gram-Schmidt against d1p
        d2p = F.normalize(d2p, dim=0)

        with torch.no_grad():
            ref_L = forward_from_layer_to_layer(
                model, ctx_a.to(model_dtype), a[None, None].to(model_dtype),
                inject_layer, measure_layer).float()
            ref_logits = forward_from_layer(
                model, ctx_a.to(model_dtype), a[None, None].to(model_dtype), inject_layer).float()
            ref_logp = F.log_softmax(ref_logits, dim=-1)

        for pi, phi in enumerate(phis_rad):
            u = np.cos(phi) * d1p + np.sin(phi) * d2p
            u = F.normalize(u, dim=0)
            out = sweep_unit_direction(model, ctx_a, a_unit, a_norm, u, ref_L, ref_logp,
                                       inject_layer, measure_layer, device, angles_rad)
            for m in ('cos', 'l2', 'kl'):
                surf[m][ai, pi] = out[m]

    median_surf = {m: np.median(surf[m], axis=0) for m in ('cos', 'l2', 'kl')}
    return median_surf, surf, orth_resid


def fit_from_surface(median_surf, phis_deg, angles_deg, thresholds):
    """Find breaking angles and fit n per metric. phi=0 -> d1, phi=90 -> d2."""
    out = {}
    i0 = int(np.argmin(np.abs(phis_deg - 0.0)))
    i90 = int(np.argmin(np.abs(phis_deg - 90.0)))
    combo_idx = [i for i in range(len(phis_deg)) if i not in (i0, i90)]
    for m in ('cos', 'l2', 'kl'):
        thr = thresholds[m]
        alpha = np.array([find_breaking_angle(angles_deg, median_surf[m][pi], thr)
                          for pi in range(len(phis_deg))])
        alpha1, alpha2 = alpha[i0], alpha[i90]
        alpha_indiv = np.array([alpha1, alpha2])
        phis_combo = phis_deg[combo_idx]
        c_matrix = np.stack([np.cos(np.deg2rad(phis_combo)),
                             np.sin(np.deg2rad(phis_combo))], axis=1)
        alpha_combo = alpha[combo_idx]
        n_hat, s_fit, valid = fit_n(c_matrix, alpha_combo, alpha_indiv)
        out[m] = {
            'threshold': thr, 'alpha_all': alpha, 'alpha1': alpha1, 'alpha2': alpha2,
            'phis_combo': phis_combo, 'alpha_combo': alpha_combo,
            'c_matrix': c_matrix, 'n_hat': n_hat, 's_fit': s_fit, 'n_valid': int(valid.sum()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_pkl', type=str,
                        default='results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl')
    parser.add_argument('--n_phi', type=int, default=19,
                        help='phi grid points from 0 to 90 deg inclusive')
    parser.add_argument('--max_angle', type=float, default=80.0,
                        help='max theta sweep angle (deg)')
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--n_act_fresh', type=int, default=30)
    parser.add_argument('--fresh_seed', type=int, default=999)
    parser.add_argument('--thresholds', type=str, default=None,
                        help='cos,l2,kl thresholds. Default: 0.5*median(axis max).')
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions']           # [2, d] (cpu)
    da = dd['args']
    model_name = da['model']; inject_layer = da['layer']; seq_len = da['seq_len']
    train_seed = da['seed']; n_act_train = da['n_act']
    measure_layer = dd.get('measure_layer')

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), 'fit')
    os.makedirs(args.results_dir, exist_ok=True)

    print(f'Loading {model_name} (inject@L{inject_layer}, measure@L{measure_layer})...')
    model, tokenizer, device = load_model(model_name)
    n_layers = len(_get_blocks(model))
    if measure_layer is None:
        measure_layer = n_layers - 2
    d1 = directions[0].to(device).float()
    d2 = directions[1].to(device).float()
    print(f'  d1.d2 (global) = {(F.normalize(d1,dim=0) @ F.normalize(d2,dim=0)).item():.4f}')

    phis_deg = np.linspace(0.0, 90.0, args.n_phi)
    angles_deg = np.linspace(0.0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # --- two activation sets ---
    sets = {}
    # optimized: exact training activations (same seed/seq_len, first n_act_train)
    tok_opt = load_fineweb_fixed_length(max(100, n_act_train), tokenizer, seq_len, seed=train_seed)
    ctx_o, act_o = extract_context_and_activation(model, tok_opt, inject_layer, device)
    sets['optimized'] = (ctx_o[:n_act_train], act_o[:n_act_train])
    # fresh: different seed, more activations
    tok_fresh = load_fineweb_fixed_length(args.n_act_fresh, tokenizer, seq_len, seed=args.fresh_seed)
    ctx_f, act_f = extract_context_and_activation(model, tok_fresh, inject_layer, device)
    sets['fresh'] = (ctx_f[:args.n_act_fresh], act_f[:args.n_act_fresh])

    results = {'args': vars(args), 'dict_args': da, 'measure_layer': measure_layer,
               'phis_deg': phis_deg, 'angles_deg': angles_deg, 'sets': {}}

    for set_name, (ctx_all, act_all) in sets.items():
        print(f'\n=== Set: {set_name} ({act_all.shape[0]} activations) ===')
        median_surf, surf, orth_resid = sweep_set(
            model, ctx_all, act_all, d1, d2, phis_deg, angles_rad, angles_deg,
            inject_layer, measure_layer, device)
        print(f'  tangent orthogonality |d1p.d2p|: max={orth_resid.max():.3f}, '
              f'mean={orth_resid.mean():.3f} (Gram-Schmidt removes it)')

        # thresholds: per set, 0.5 * median over phi of the per-phi max response,
        # restricted to the two axis sweeps so axes always break.
        if args.thresholds:
            cth, lth, kth = [float(x) for x in args.thresholds.split(',')]
        else:
            i0 = int(np.argmin(np.abs(phis_deg - 0.0)))
            i90 = int(np.argmin(np.abs(phis_deg - 90.0)))
            cth = 0.5 * np.median([median_surf['cos'][i0].max(), median_surf['cos'][i90].max()])
            lth = 0.5 * np.median([median_surf['l2'][i0].max(), median_surf['l2'][i90].max()])
            kth = 0.5 * np.median([median_surf['kl'][i0].max(), median_surf['kl'][i90].max()])
        thresholds = {'cos': cth, 'l2': lth, 'kl': kth}
        print(f'  thresholds: cos={cth:.4f}, l2={lth:.2f}, kl={kth:.3f}')

        fit = fit_from_surface(median_surf, phis_deg, angles_deg, thresholds)
        for m in ('l2', 'cos', 'kl'):
            f = fit[m]
            print(f'  [{m}] n_hat = {f["n_hat"]:.3f}  (alpha1={f["alpha1"]:.1f}, '
                  f'alpha2={f["alpha2"]:.1f} deg, n_valid_combo={f["n_valid"]}, '
                  f'censored={int(np.sum(~np.isfinite(f["alpha_combo"])))})')

        results['sets'][set_name] = {
            'median_surf': median_surf, 'orth_resid': orth_resid,
            'thresholds': thresholds, 'fit': fit, 'n_act': int(act_all.shape[0]),
        }

    with open(os.path.join(args.results_dir, 'fit.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- plot: breaking angle vs phi, with fitted superellipse, per set (l2) ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for row, set_name in enumerate(['optimized', 'fresh']):
        sd = results['sets'][set_name]
        for col, m in enumerate(['l2', 'cos', 'kl']):
            ax = axes[row, col]
            f = sd['fit'][m]
            ax.plot(phis_deg, f['alpha_all'], 'o-', ms=4, label='measured alpha(phi)')
            # superellipse prediction at fitted n
            n_hat = f['n_hat']; a1 = np.deg2rad(f['alpha1']); a2 = np.deg2rad(f['alpha2'])
            if np.isfinite(n_hat) and np.isfinite(a1) and np.isfinite(a2):
                phr = np.deg2rad(phis_deg)
                coeff = (np.abs(np.cos(phr) / np.sin(a1)) ** n_hat
                         + np.abs(np.sin(phr) / np.sin(a2)) ** n_hat)
                coeff = np.where(coeff > 0, coeff, np.nan)
                sin_pred = np.clip(coeff ** (-1.0 / n_hat), -1, 1)
                ax.plot(phis_deg, np.rad2deg(np.arcsin(sin_pred)), 'r--',
                        label=f'superellipse n={n_hat:.2f}')
            ax.set_xlabel('phi (deg)  [0=d1, 90=d2]')
            ax.set_ylabel('breaking angle alpha (deg)')
            ax.set_title(f'{set_name} / {m}: n={f["n_hat"]:.2f}')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{model_name} L{inject_layer}->{measure_layer}: superellipse fit for learned pair',
                 fontsize=13)
    plt.tight_layout()
    png = os.path.join(args.results_dir, 'superellipse_fit.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {os.path.join(args.results_dir, "fit.pkl")}')
    print(f'Saved: {png}')


if __name__ == '__main__':
    main()
