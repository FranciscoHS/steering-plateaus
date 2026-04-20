#!/usr/bin/env python3
"""Fit power law to the 645-direction uncapped MELBO spectrum at 10°."""

import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def power_law(x, a, b):
    return a * x**b


results_path = 'results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1_10deg_uncapped/results.pkl'
out_dir = 'results/melbo_spectrum/qwen3-1.7b_layer2_l2_tgt26_act1_10deg_uncapped'

with open(results_path, 'rb') as f:
    r = pickle.load(f)

objs = np.array(r['melbo_results'][10.0]['objs'])
x = np.arange(1, len(objs) + 1, dtype=float)
random_objs = r['random_objs']
sweep_angles_deg = r['sweep_angles_deg']

# Random median at 10°
ang_idx = np.argmin(np.abs(sweep_angles_deg - 10.0))
rand_med = np.median(random_objs[ang_idx])

# Fit power law
popt, _ = curve_fit(power_law, x, objs, p0=[objs[0], -0.5], maxfev=10000)
residuals = objs - power_law(x, *popt)
r2 = 1 - np.sum(residuals**2) / np.sum((objs - np.mean(objs))**2)

# Jacobian SVs
svs = r['jacobian_svs']
x_jac = np.arange(1, len(svs) + 1, dtype=float)
popt_j, _ = curve_fit(power_law, x_jac, svs, p0=[svs[0], -0.5], maxfev=5000)
r2_j = 1 - np.sum((svs - power_law(x_jac, *popt_j))**2) / np.sum((svs - np.mean(svs))**2)

print(f'MELBO spectrum: {popt[0]:.1f} * k^{popt[1]:.4f}  (R²={r2:.4f}, n={len(objs)})')
print(f'Jacobian SVs:   {popt_j[0]:.1f} * k^{popt_j[1]:.4f}  (R²={r2_j:.4f}, n={len(svs)})')

# === Plot ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: MELBO spectrum + fit (log-log)
ax1.scatter(x, objs, s=8, alpha=0.4, color='#E8820C', label='MELBO directions', zorder=2)
x_fit = np.linspace(1, len(objs), 500)
ax1.plot(x_fit, power_law(x_fit, *popt), 'k-', linewidth=2,
         label=f'{popt[0]:.0f} · k$^{{{popt[1]:.2f}}}$ (R²={r2:.3f})', zorder=3)
ax1.axhline(rand_med, color='#2B4E72', ls='--', linewidth=1.5,
            label=f'Random median ({rand_med:.0f})')
ax1.axhline(5 * rand_med, color='#2B4E72', ls=':', linewidth=1.5,
            label=f'5× random ({5*rand_med:.0f})')
ax1.set_xlabel('Direction index k')
ax1.set_ylabel('L2 distance at 10°')
ax1.set_title(f'MELBO spectrum at 10° ({len(objs)} directions)')
ax1.legend(fontsize=9)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Right: Jacobian SVs + fit (log-log)
ax2.scatter(x_jac, svs, s=15, color='#2B4E72', label='Jacobian SVs', zorder=2)
x_fit_j = np.linspace(1, len(svs), 500)
ax2.plot(x_fit_j, power_law(x_fit_j, *popt_j), 'k-', linewidth=2,
         label=f'{popt_j[0]:.0f} · k$^{{{popt_j[1]:.2f}}}$ (R²={r2_j:.3f})', zorder=3)
ax2.set_xlabel('Singular value index k')
ax2.set_ylabel('Singular value')
ax2.set_title(f'Jacobian SV spectrum (top {len(svs)})')
ax2.legend(fontsize=9)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

fig.suptitle('Qwen3-1.7B, layer 2 → L26, activation 1', fontsize=13)
plt.tight_layout()
out_path = f'{out_dir}/power_law_fit.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
