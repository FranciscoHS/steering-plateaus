#!/usr/bin/env python3
"""Plot iso-plateau contours with superellipse fit.

Extracts the iso-plateau contour from the 2D sweep data, normalizes by
individual axis thresholds, fits a superellipse exponent p, and plots
the result against reference shapes (diamond p=1, circle p=2, square p→∞).
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = 'results/mandarin_haskell/gemma-2-9b'
INPUT_PKL = os.path.join(RESULTS_DIR, 'sweep_isoplateau.pkl')


def extract_contour_points(angles, median_grid, threshold):
    """Extract contour points at the given threshold using matplotlib contour."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(angles, angles, median_grid.T, levels=[threshold])
    points = []
    # Handle both old and new matplotlib API
    if hasattr(cs, 'allsegs'):
        for seg_list in cs.allsegs:
            for seg in seg_list:
                points.append(seg)
    else:
        for collection in cs.collections:
            for path in collection.get_paths():
                points.append(path.vertices)
    plt.close(fig_tmp)
    if points:
        return np.concatenate(points, axis=0)
    return np.array([]).reshape(0, 2)


def fit_superellipse(contour_pts_norm, avg_theta):
    """Fit superellipse exponent p to normalized contour points.

    Model: (x)^p + (y)^p = 1, where x = α₁/θ₁* and y = α₂/θ₂*.
    Minimizes sum of squared residuals: (x^p + y^p - 1)^2.

    Returns:
        p_fit: fitted exponent
        ci: 95% bootstrap confidence interval [lo, hi]
        fit_info: dict with radial error stats
    """
    x = contour_pts_norm[:, 0]
    y = contour_pts_norm[:, 1]

    # Filter to first quadrant, away from axes (avoid numerical issues)
    mask = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return None, None, None

    def residual(p):
        pred = np.abs(x)**p + np.abs(y)**p
        return np.sum((pred - 1.0)**2)

    result = minimize_scalar(residual, bounds=(0.5, 10.0), method='bounded')
    p_fit = result.x

    # Bootstrap confidence interval
    n = len(x)
    p_boots = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        xb, yb = x[idx], y[idx]
        def res_boot(p):
            return np.sum((np.abs(xb)**p + np.abs(yb)**p - 1.0)**2)
        rb = minimize_scalar(res_boot, bounds=(0.5, 10.0), method='bounded')
        p_boots.append(rb.x)
    ci = np.percentile(p_boots, [2.5, 97.5])

    # Fit quality: radial error
    lp_norm = (np.abs(x)**p_fit + np.abs(y)**p_fit)**(1/p_fit)
    frac_err = np.abs(lp_norm - 1.0)
    fit_info = {
        'mean_radial_frac': frac_err.mean(),
        'max_radial_frac': frac_err.max(),
        'mean_radial_deg': frac_err.mean() * avg_theta,
        'max_radial_deg': frac_err.max() * avg_theta,
        'n_pts': n,
    }

    return p_fit, ci, fit_info


def superellipse_curve(p, n_pts=200):
    """Generate a quarter superellipse in the first quadrant."""
    t = np.linspace(0, np.pi / 2, n_pts)
    x = np.abs(np.cos(t))**(2/p) * np.sign(np.cos(t))
    y = np.abs(np.sin(t))**(2/p) * np.sign(np.sin(t))
    return x, y


def plot_one(angles, median_grid, threshold, metric_label, units_label,
             save_path):
    """Plot normalized iso-plateau contour with reference shapes and fit."""
    # Find axis intercepts
    pt_curve = median_grid[:, 0]   # α₂=0
    gen_curve = median_grid[0, :]  # α₁=0

    # Interpolate threshold crossings
    def find_threshold(curve, angles, thresh):
        idx = np.searchsorted(curve, thresh)
        if idx == 0 or idx >= len(angles):
            return angles[idx] if idx < len(angles) else angles[-1]
        # Linear interpolation
        x0, x1 = angles[idx-1], angles[idx]
        y0, y1 = curve[idx-1], curve[idx]
        return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)

    theta_pt = find_threshold(pt_curve, angles, threshold)
    theta_gen = find_threshold(gen_curve, angles, threshold)

    # Extract and normalize contour
    contour_pts = extract_contour_points(angles, median_grid, threshold)
    if len(contour_pts) == 0:
        print(f"  No contour found for {metric_label} at {threshold}")
        return

    contour_norm = contour_pts.copy()
    contour_norm[:, 0] /= theta_pt
    contour_norm[:, 1] /= theta_gen

    # Fit superellipse
    avg_theta = (theta_pt + theta_gen) / 2
    p_fit, ci, fit_info = fit_superellipse(contour_norm, avg_theta)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference shapes
    t = np.linspace(0, np.pi/2, 200)

    # Diamond (p=1)
    x1 = 1 - np.linspace(0, 1, 200)
    y1 = np.linspace(0, 1, 200)
    ax.plot(x1, y1, '--', color='#aaaaaa', lw=1.5, label='p=1 (diamond)')

    # Circle (p=2)
    xc, yc = superellipse_curve(2)
    ax.plot(xc, yc, '--', color='#888888', lw=1.5, label='p=2 (circle)')

    # Square (p=20, approximation)
    xs, ys = superellipse_curve(20)
    ax.plot(xs, ys, '--', color='#bbbbbb', lw=1.5, label='p→∞ (square)')

    # Fitted superellipse
    if p_fit is not None:
        xf, yf = superellipse_curve(p_fit)
        ax.plot(xf, yf, '-', color='#e41a1c', lw=2.5,
                label=f'fit: p={p_fit:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')
        # Fit quality annotation
        ax.text(0.03, 0.97,
                f'radial error: {fit_info["mean_radial_deg"]:.2f}° mean, '
                f'{fit_info["max_radial_deg"]:.2f}° max\n'
                f'({fit_info["n_pts"]} contour pts)',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # Actual contour (scatter for raw data)
    # Sort by angle for clean line
    contour_angle = np.arctan2(contour_norm[:, 1], contour_norm[:, 0])
    order = np.argsort(contour_angle)
    ax.plot(contour_norm[order, 0], contour_norm[order, 1], 'o',
            color='#377eb8', markersize=3, alpha=0.7, label='data')

    ax.set_xlabel('α₁ / θ₁*  (Portuguese, normalized)', fontsize=12)
    ax.set_ylabel('α₂ / θ₂*  (Gender, normalized)', fontsize=12)
    ax.set_title(f'Iso-Plateau Shape ({units_label})\n'
                 f'θ_PT*={theta_pt:.1f}°, θ_gen*={theta_gen:.1f}°',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.05, 1.5)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Mark (1,0) and (0,1)
    ax.plot(1, 0, 'kv', markersize=8, zorder=5)
    ax.plot(0, 1, 'k>', markersize=8, zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    with open(INPUT_PKL, 'rb') as f:
        results = pickle.load(f)

    angles = results['angles_deg']

    for metric, threshold, label, units in [
        ('l2', 50.0, 'L2 distance', 'L2=50'),
        ('cosine', 0.01, 'Cosine distance', 'cos=0.01'),
    ]:
        median_grid = np.median(results[metric], axis=0)
        save_path = os.path.join(RESULTS_DIR, f'isoplateau_shape_{metric}.png')
        print(f"\n{metric} (threshold={threshold}):")
        plot_one(angles, median_grid, threshold, label, units, save_path)


if __name__ == '__main__':
    main()
