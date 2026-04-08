#!/usr/bin/env python3
"""Fit power laws to MELBO spectra and Jacobian SVs."""

import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def power_law(x, a, b):
    return a * x**b


def fit_and_report(x, y, label):
    """Fit power law, return (a, b, R^2) or None on failure."""
    try:
        popt, _ = curve_fit(power_law, x, y, p0=[y[0], -0.5], maxfev=5000)
        residuals = y - power_law(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        print(f'{label}: {popt[0]:.2f} * k^{popt[1]:.3f}  (R^2={r2:.4f}, n={len(y)})')
        return popt[0], popt[1], r2
    except Exception as e:
        print(f'{label}: fit failed: {e}')
        return None


results_path = 'results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1/results.pkl'
with open(results_path, 'rb') as f:
    r = pickle.load(f)

out_dir = 'results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1'

# === Jacobian SVs ===
print('=== Jacobian SVs ===')
svs = r['jacobian_svs']
x_jac = np.arange(1, len(svs) + 1, dtype=float)
jac_fit = fit_and_report(x_jac, svs, 'Jacobian SVs')

# === MELBO spectra at each angle ===
print('\n=== MELBO spectra (L2 objective) ===')
fits = {}
for angle in r['opt_angles_deg']:
    objs = np.array(r['melbo_results'][angle]['objs'])
    x = np.arange(1, len(objs) + 1, dtype=float)
    result = fit_and_report(x, objs, f'{angle:5.1f}°')
    if result:
        fits[angle] = result

# === Summary: exponent vs angle ===
print('\n=== Exponent vs angle ===')
for angle in sorted(fits.keys()):
    a, b, r2 = fits[angle]
    print(f'  {angle:5.1f}°: exponent = {b:.3f}, R^2 = {r2:.4f}')

# === Plot: data + fits ===
n_panels = len(r['opt_angles_deg']) + 1  # +1 for Jacobian
ncols = 3
nrows = (n_panels + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
axes = axes.flatten()

# Jacobian SVs panel
ax = axes[0]
ax.plot(x_jac, svs, 'o', markersize=3, color='C0', label='Jacobian SVs')
if jac_fit:
    a, b, r2 = jac_fit
    x_fit = np.linspace(1, len(svs), 200)
    ax.plot(x_fit, power_law(x_fit, a, b), 'r-', linewidth=1.5,
            label=f'{a:.0f}·k^{b:.2f} (R²={r2:.3f})')
ax.set_xlabel('Singular value index')
ax.set_ylabel('Singular value')
ax.set_title('Jacobian SVs')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.set_xscale('log')

# MELBO panels
for idx, angle in enumerate(r['opt_angles_deg']):
    ax = axes[idx + 1]
    objs = np.array(r['melbo_results'][angle]['objs'])
    x = np.arange(1, len(objs) + 1, dtype=float)
    ax.plot(x, objs, 'o', markersize=3, label='MELBO')

    if angle in fits:
        a, b, r2 = fits[angle]
        x_fit = np.linspace(1, len(objs), 200)
        ax.plot(x_fit, power_law(x_fit, a, b), 'r-', linewidth=1.5,
                label=f'{a:.0f}·k^{b:.2f} (R²={r2:.3f})')

    ax.set_xlabel('Direction index')
    ax.set_ylabel('L2')
    ax.set_title(f'{angle}°')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')

# Hide unused axes
for idx in range(n_panels, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Power law fits to MELBO spectrum (L2, act 1)\nqwen3-1.7b, layer 2 → L26',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{out_dir}/power_law_fits.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_dir}/power_law_fits.png')

# === Plot: exponent vs angle ===
angles_sorted = sorted(fits.keys())
exponents = [fits[a][1] for a in angles_sorted]
r2s = [fits[a][2] for a in angles_sorted]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(angles_sorted, exponents, 'ko-', markersize=6)
ax1.set_xlabel('Optimization angle (degrees)')
ax1.set_ylabel('Power law exponent')
ax1.set_title('Exponent vs angle')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', ls=':', alpha=0.5)

ax2.plot(angles_sorted, r2s, 'ko-', markersize=6)
ax2.set_xlabel('Optimization angle (degrees)')
ax2.set_ylabel('R²')
ax2.set_title('Goodness of fit vs angle')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.suptitle('Power law fit quality across angles\nqwen3-1.7b, layer 2 → L26, act 1', fontsize=13)
plt.tight_layout()
plt.savefig(f'{out_dir}/exponent_vs_angle.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_dir}/exponent_vs_angle.png')
