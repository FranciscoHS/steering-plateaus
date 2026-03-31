#!/usr/bin/env python3
"""3D iso-plateau sweep for French × Rust × Gender on Gemma-2-9B.

Perturbs along three directions simultaneously on the sphere, scanning a 3D grid
of (α₁, α₂, α₃) angles. Extracts the iso-plateau surface, normalizes, and fits
a superellipsoid exponent p: (α₁/θ₁*)^p + (α₂/θ₂*)^p + (α₃/θ₃*)^p = 1.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    compute_contrastive_direction,
    extract_activations_holdout,
)

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 20
MAX_ANGLE = 12.0
N_STEPS = 21

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
DIRECTIONS_FR_RUST = os.path.join(RESULTS_DIR, 'directions_french_rust_L20.pkl')
DIRECTIONS_GENDER = os.path.join(RESULTS_DIR, 'directions_gender_L20.pkl')
OUTPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau_3d.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep()
    plot_results()
    fit_and_report()


def sweep_3d(model, ctx, act, d1, d2, d3, layer, measure_layer, device,
             angles_rad, batch_size=64):
    """Sweep a 3D grid of combined perturbations on the sphere.

    The combined perturbation is:
        p = R * (c * a_unit + sin(α₁)*d1_orth + sin(α₂)*d2_orth + sin(α₃)*d3_orth)
    where c = sqrt(1 - sin²α₁ - sin²α₂ - sin²α₃).
    d1, d2, d3 are Gram-Schmidt orthogonalized against a_unit and each other.

    Returns:
        l2_grid: [n, n, n] array of L2 distances
        cos_grid: [n, n, n] array of cosine distances
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    # Gram-Schmidt: orthogonalize d1, d2, d3 against a_unit and each other
    d1_d = d1.to(device=device, dtype=torch.float32)
    d1_orth = d1_d - (d1_d @ a_unit) * a_unit
    d1_orth = F.normalize(d1_orth, dim=0)

    d2_d = d2.to(device=device, dtype=torch.float32)
    d2_orth = d2_d - (d2_d @ a_unit) * a_unit - (d2_d @ d1_orth) * d1_orth
    d2_orth = F.normalize(d2_orth, dim=0)

    d3_d = d3.to(device=device, dtype=torch.float32)
    d3_orth = d3_d - (d3_d @ a_unit) * a_unit - (d3_d @ d1_orth) * d1_orth - (d3_d @ d2_orth) * d2_orth
    d3_orth = F.normalize(d3_orth, dim=0)

    n = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)

    # Build 3D meshgrid
    S1, S2, S3 = torch.meshgrid(s, s, s, indexing='ij')
    sum_sq = S1**2 + S2**2 + S3**2

    # Mask out invalid points where sin²α₁ + sin²α₂ + sin²α₃ > 1
    valid = sum_sq <= 1.0
    C = torch.sqrt(torch.clamp(1.0 - sum_sq, min=0.0))

    # Flatten valid points only
    valid_flat = valid.reshape(-1)
    C_flat = C.reshape(-1)[valid_flat]
    S1_flat = S1.reshape(-1)[valid_flat]
    S2_flat = S2.reshape(-1)[valid_flat]
    S3_flat = S3.reshape(-1)[valid_flat]

    all_points = (R * (C_flat.unsqueeze(1) * a_unit.unsqueeze(0)
                       + S1_flat.unsqueeze(1) * d1_orth.unsqueeze(0)
                       + S2_flat.unsqueeze(1) * d2_orth.unsqueeze(0)
                       + S3_flat.unsqueeze(1) * d3_orth.unsqueeze(0))).to(model_dtype)

    n_points = all_points.shape[0]

    # Reference (unperturbed)
    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype), layer, measure_layer).float()

    l2_valid = np.zeros(n_points)
    cos_valid = np.zeros(n_points)

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            l2_valid[start:end] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos_valid[start:end] = (1.0 - F.cosine_similarity(
                ref_act, pf, dim=-1)).cpu().numpy()

    # Reconstruct full grid (invalid points get NaN)
    l2_grid = np.full(n**3, np.nan)
    cos_grid = np.full(n**3, np.nan)
    l2_grid[valid_flat.cpu().numpy()] = l2_valid
    cos_grid[valid_flat.cpu().numpy()] = cos_valid

    return l2_grid.reshape(n, n, n), cos_grid.reshape(n, n, n)


def run_sweep():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load directions
    print("Loading directions...")
    with open(DIRECTIONS_FR_RUST, 'rb') as f:
        fr_data = pickle.load(f)
    with open(DIRECTIONS_GENDER, 'rb') as f:
        gen_data = pickle.load(f)

    d_french = fr_data['d_french']
    d_rust = fr_data['d_rust']
    d_gender = gen_data['d_to_female']

    # Report pairwise cosines
    cos_fr_ru = (d_french @ d_rust).item()
    cos_fr_ge = (d_french @ d_gender).item()
    cos_ru_ge = (d_rust @ d_gender).item()
    print(f"  cos(French, Rust) = {cos_fr_ru:.4f}")
    print(f"  cos(French, Gender) = {cos_fr_ge:.4f}")
    print(f"  cos(Rust, Gender) = {cos_ru_ge:.4f}")

    # Load model
    print("Loading Gemma-2-9B...")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Holdout activations
    from data.coding_prompts_agnostic import CODING_PROMPTS
    holdout = CODING_PROMPTS[:N_ACT]
    print(f"Extracting activations from {len(holdout)} coding-agnostic holdout prompts...")
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Angle grid
    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)
    n_valid = np.sum(np.sin(angles_rad[:, None, None])**2
                     + np.sin(angles_rad[None, :, None])**2
                     + np.sin(angles_rad[None, None, :])**2 <= 1.0)
    print(f"Grid: {N_STEPS}³ = {N_STEPS**3} total, {n_valid} valid points per activation")
    print(f"  0-{MAX_ANGLE}° in {angles_deg[1]-angles_deg[0]:.2f}° steps")

    all_l2, all_cos = [], []
    for i in range(N_ACT):
        l2_grid, cos_grid = sweep_3d(
            model, contexts[i], activations[i],
            d_french, d_rust, d_gender,
            LAYER, measure_layer, device, angles_rad)
        all_l2.append(l2_grid)
        all_cos.append(cos_grid)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_ACT}")

    results = {
        'angles_deg': angles_deg,
        'l2': np.stack(all_l2),      # [N_ACT, N_STEPS, N_STEPS, N_STEPS]
        'cosine': np.stack(all_cos),
        'layer': LAYER,
        'measure_layer': measure_layer,
        'direction_labels': ('French', 'Rust', 'Gender'),
        'pairwise_cosines': {
            'french_rust': cos_fr_ru,
            'french_gender': cos_fr_ge,
            'rust_gender': cos_ru_ge,
        },
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUTPUT_PKL}")


def find_threshold_1d(curve, angles, thresh):
    """Interpolate the angle where curve crosses thresh."""
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx-1], angles[idx]
    y0, y1 = curve[idx-1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def extract_isosurface(angles, median_grid, threshold):
    """Extract iso-surface points using marching cubes."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from skimage import measure
        marching_cubes = measure.marching_cubes

    # Replace NaN with a large value (outside surface)
    grid = median_grid.copy()
    grid[np.isnan(grid)] = threshold * 100

    # marching_cubes expects the level to cross inside the grid
    if grid.min() > threshold or np.nanmax(median_grid) < threshold:
        return np.array([]).reshape(0, 3)

    try:
        verts, faces, _, _ = marching_cubes(grid, level=threshold)
    except (ValueError, RuntimeError):
        return np.array([]).reshape(0, 3)

    # Convert vertex indices to angle values
    step = angles[1] - angles[0]
    pts = verts * step + angles[0]
    return pts


def fit_superellipsoid(pts_norm):
    """Fit superellipsoid exponent p to normalized 3D iso-surface points.

    Model: |x|^p + |y|^p + |z|^p = 1
    """
    from scipy.optimize import minimize_scalar

    x, y, z = pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]

    # Filter away from axes and to reasonable range
    mask = (x > 0.05) & (y > 0.05) & (z > 0.05) & (x < 1.5) & (y < 1.5) & (z < 1.5)
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 5:
        return None, None, None

    def residual(p):
        pred = np.abs(x)**p + np.abs(y)**p + np.abs(z)**p
        return np.sum((pred - 1.0)**2)

    result = minimize_scalar(residual, bounds=(0.5, 10.0), method='bounded')
    p_fit = result.x

    # Bootstrap CI
    n = len(x)
    p_boots = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        xb, yb, zb = x[idx], y[idx], z[idx]
        def res_boot(p):
            return np.sum((np.abs(xb)**p + np.abs(yb)**p + np.abs(zb)**p - 1.0)**2)
        rb = minimize_scalar(res_boot, bounds=(0.5, 10.0), method='bounded')
        p_boots.append(rb.x)
    ci = np.percentile(p_boots, [2.5, 97.5])

    # Radial error
    lp_norm = (np.abs(x)**p_fit + np.abs(y)**p_fit + np.abs(z)**p_fit)**(1/p_fit)
    frac_err = np.abs(lp_norm - 1.0)
    avg_theta = np.mean([x.mean(), y.mean(), z.mean()])  # rough scale
    fit_info = {
        'mean_radial_frac': frac_err.mean(),
        'max_radial_frac': frac_err.max(),
        'n_pts': n,
        'n_pts_total': len(pts_norm),
    }

    return p_fit, ci, fit_info


def plot_results():
    """Plot 2D slices of the 3D sweep."""
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    labels = results['direction_labels']

    for metric, threshold, units in [
        ('l2', 50.0, 'L2=50'),
        ('cosine', 0.01, 'cos=0.01'),
    ]:
        data = results[metric]
        median_grid = np.nanmedian(data, axis=0)

        # Plot 3 slices: α₃=0, α₂=0, α₁=0
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        slice_configs = [
            (0, median_grid[:, :, 0], labels[0], labels[1], f'{labels[2]}=0°'),
            (1, median_grid[:, 0, :], labels[0], labels[2], f'{labels[1]}=0°'),
            (2, median_grid[0, :, :], labels[1], labels[2], f'{labels[0]}=0°'),
        ]

        for ax, (idx, slc, xl, yl, title_extra) in zip(axes, slice_configs):
            cf = ax.contourf(angles, angles, slc.T, levels=50, cmap='viridis')
            plt.colorbar(cf, ax=ax, label=metric.upper())
            try:
                cs = ax.contour(angles, angles, slc.T, levels=[threshold],
                                colors='red', linewidths=2)
                ax.clabel(cs, fmt=f'{threshold}', fontsize=8, colors='red')
            except ValueError:
                pass
            ax.set_xlabel(f'{xl} α (degrees)', fontsize=10)
            ax.set_ylabel(f'{yl} α (degrees)', fontsize=10)
            ax.set_title(title_extra, fontsize=11)
            ax.set_aspect('equal')

        fig.suptitle(f'3D Iso-Plateau Slices ({units})\n'
                     f'Gemma-2-9B · layer 20 · median over {data.shape[0]} prompts',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        png_path = os.path.join(RESULTS_DIR, f'isoplateau_3d_slices_{metric}.png')
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        plt.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight',
                    facecolor='white')
        plt.close()


def fit_and_report():
    """Extract iso-surface, fit superellipsoid, report results."""
    with open(OUTPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']
    labels = results['direction_labels']

    thresholds = {
        'l2': np.array([20, 30, 40, 50, 60, 70, 80]),
        'cosine': np.array([0.002, 0.005, 0.008, 0.01, 0.015, 0.02]),
    }

    for metric in ['l2', 'cosine']:
        data = results[metric]
        median_grid = np.nanmedian(data, axis=0)

        # Axis intercepts
        curve_1 = median_grid[:, 0, 0]
        curve_2 = median_grid[0, :, 0]
        curve_3 = median_grid[0, 0, :]

        print(f"\n{metric.upper()} metric:")
        print(f"  {'threshold':>10s}  {'θ₁':>6s}  {'θ₂':>6s}  {'θ₃':>6s}  "
              f"{'p_fit':>6s}  {'CI':>14s}  {'frac_err':>8s}  {'n_pts':>5s}")

        for thresh in thresholds[metric]:
            t1 = find_threshold_1d(curve_1, angles, thresh)
            t2 = find_threshold_1d(curve_2, angles, thresh)
            t3 = find_threshold_1d(curve_3, angles, thresh)

            pts = extract_isosurface(angles, median_grid, thresh)
            if len(pts) == 0:
                print(f"  {thresh:>10g}  no surface found")
                continue

            pts_norm = pts.copy()
            pts_norm[:, 0] /= t1
            pts_norm[:, 1] /= t2
            pts_norm[:, 2] /= t3

            p_fit, ci, fit_info = fit_superellipsoid(pts_norm)
            if p_fit is None:
                print(f"  {thresh:>10g}  {t1:>5.1f}°  {t2:>5.1f}°  {t3:>5.1f}°  "
                      f"fit failed (too few interior pts)")
                continue

            print(f"  {thresh:>10g}  {t1:>5.1f}°  {t2:>5.1f}°  {t3:>5.1f}°  "
                  f"{p_fit:>6.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]  "
                  f"{fit_info['mean_radial_frac']:>7.4f}  "
                  f"{fit_info['n_pts']:>5d}")


if __name__ == '__main__':
    main()
