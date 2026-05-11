"""Spot-check: does interpolating the non-uniform L2 grid onto a uniform
grid before contour extraction change p_fit?

Audit flagged that matplotlib's contour() does linear interpolation between
adjacent grid points, and our angle grid jumps from 0.1deg spacing to 2deg
spacing at alpha=3deg. Contours that straddle this seam (roughly T = 50..300)
could be distorted.

This script re-fits using a uniformly-spaced (0.1deg) grid built by 2D linear
interpolation of the median L2 surface, then compares per-threshold (theta1,
theta2, p_fit) against the non-uniform original.
"""
from __future__ import annotations
import sys, pickle
sys.path.insert(0, ".")
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse,
)

PKL = ("results/exp_map/data/sweeps_2d/"
       "sweep2d_gemma_L2_Gender__Tense_fineweb_90deg_sbr_d3-0.1-2_fp32.pkl")
THRESHS = [0.5, 1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 550]
UNIFORM_STEP = 0.1


def fit_at(angles, median_grid, thresh):
    theta1 = axis_intercept(angles, median_grid[:, 0], thresh)
    theta2 = axis_intercept(angles, median_grid[0, :], thresh)
    if not (np.isfinite(theta1) and np.isfinite(theta2)
            and theta1 > 0 and theta2 > 0):
        return None
    contour = extract_contour(angles, median_grid, thresh)
    if contour.size == 0:
        return None
    cn = contour.astype(float).copy()
    cn[:, 0] /= theta1; cn[:, 1] /= theta2
    keep = (cn[:, 0] > 0.05) & (cn[:, 1] > 0.05) \
        & (cn[:, 0] < 1.5) & (cn[:, 1] < 1.5)
    cn = cn[keep]
    if cn.shape[0] < 4:
        return {"theta1": theta1, "theta2": theta2,
                "p": float("nan"), "n_pts": int(cn.shape[0])}
    fit = fit_superellipse(cn[:, 0], cn[:, 1])
    return {"theta1": theta1, "theta2": theta2,
            "p": fit["p"], "n_pts": fit["n_pts"],
            "mrf": fit["mean_radial_frac"]}


def main():
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    angles = d["angles_deg"]
    median_grid = np.median(d["l2"], axis=0)

    # Uniform grid 0..89 (the original max) at UNIFORM_STEP
    uniform_angles = np.arange(0.0, angles[-1] + 1e-9, UNIFORM_STEP)
    interp = RegularGridInterpolator(
        (angles, angles), median_grid,
        method="linear", bounds_error=False, fill_value=np.nan)
    Aa, Bb = np.meshgrid(uniform_angles, uniform_angles, indexing="ij")
    uniform_grid = interp(np.stack([Aa.ravel(), Bb.ravel()], axis=-1)
                          ).reshape(len(uniform_angles), len(uniform_angles))

    print(f"original: n_angles={len(angles)}, non-uniform")
    print(f"uniform:  n_angles={len(uniform_angles)} at {UNIFORM_STEP}deg")
    print()
    print(f"{'T':>6}  {'theta1_o':>9} {'theta2_o':>9} {'p_o':>6} {'n_o':>4}  "
          f"{'theta1_u':>9} {'theta2_u':>9} {'p_u':>6} {'n_u':>4}  "
          f"{'dp':>7}")
    for T in THRESHS:
        ro = fit_at(angles, median_grid, T)
        ru = fit_at(uniform_angles, uniform_grid, T)
        if ro is None or ru is None:
            print(f"{T:6.1f}  skip (axis intercepts out of range)")
            continue
        dp = ru["p"] - ro["p"]
        print(f"{T:6.1f}  "
              f"{ro['theta1']:9.3f} {ro['theta2']:9.3f} "
              f"{ro['p']:6.3f} {ro['n_pts']:4d}  "
              f"{ru['theta1']:9.3f} {ru['theta2']:9.3f} "
              f"{ru['p']:6.3f} {ru['n_pts']:4d}  "
              f"{dp:+7.3f}")


if __name__ == "__main__":
    main()
