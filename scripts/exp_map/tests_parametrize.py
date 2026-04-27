"""Unit tests for scripts/exp_map/lib/parametrize.py.

Run before any sweep. All tests are CPU/float64; no model needed.

  python scripts/exp_map/tests_parametrize.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
import torch

from parametrize import (
    orthonormalize,
    exp_map_2d,
    exp_map_1d,
    chart_distance,
    project_to_chart,
    DegenerateDirection,
)


D = 64
TOL_NORM = 1e-10
TOL_ANGLE = 1e-10
F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def random_setup(seed: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(D, generator=g, dtype=torch.float64)
    d1 = torch.randn(D, generator=g, dtype=torch.float64)
    d2 = torch.randn(D, generator=g, dtype=torch.float64)
    return a, d1, d2


def test_norm_matched():
    """||new_a|| == R for 100 random configurations."""
    fails = 0
    for s in range(100):
        a, d1, d2 = random_setup(s)
        a_hat = a / a.norm()
        d1p, d2p = orthonormalize(a_hat, d1, d2)
        # angles up to 80 degrees on each axis (chart has no per-axis cap;
        # the only restriction we self-impose is r < pi/2 to stay on the
        # upper hemisphere, but tests should cover up through that).
        alpha1 = _t(0.6 + 0.5 * (s % 5) / 5)  # rad
        alpha2 = _t(0.5 + 0.4 * ((s + 1) % 7) / 7)
        if (alpha1**2 + alpha2**2).sqrt() >= 1.55:  # r < pi/2 - eps
            alpha1 = alpha1 * 0.5
            alpha2 = alpha2 * 0.5
        new_a = exp_map_2d(a, d1p, d2p, alpha1, alpha2)
        if abs(new_a.norm().item() - a.norm().item()) > TOL_NORM:
            fails += 1
            print(f"  norm mismatch seed={s}: {new_a.norm().item()} vs {a.norm().item()}")
    assert fails == 0, f"norm-matched: {fails}/100 failed"
    print(f"  norm-matched: 100/100 passed (tol {TOL_NORM})")


def test_chart_distance():
    """arccos(a_hat . new_a / R) == ||(alpha1, alpha2)||."""
    fails = 0
    for s in range(100):
        a, d1, d2 = random_setup(s)
        a_hat = a / a.norm()
        d1p, d2p = orthonormalize(a_hat, d1, d2)
        alpha1 = _t(0.3 + 0.4 * (s % 5) / 5)
        alpha2 = _t(0.2 + 0.5 * ((s + 1) % 7) / 7)
        r_expect = torch.sqrt(alpha1**2 + alpha2**2).item()
        new_a = exp_map_2d(a, d1p, d2p, alpha1, alpha2)
        r_got = chart_distance(a_hat, new_a).item()
        if abs(r_got - r_expect) > TOL_ANGLE:
            fails += 1
            print(f"  chart-dist seed={s}: {r_got} vs {r_expect}")
    assert fails == 0, f"chart-distance: {fails}/100 failed"
    print(f"  chart-distance: 100/100 passed (tol {TOL_ANGLE})")


def test_single_direction_reduction():
    """alpha2 = 0 -> new_a = R(cos a1 a_hat + sin a1 d1_perp)."""
    for s in range(20):
        a, d1, _ = random_setup(s)
        a_hat = a / a.norm()
        d1p, _ = orthonormalize(a_hat, d1, None)
        alpha = _t(0.7 + 0.05 * s)
        from2d = exp_map_2d(a, d1p, torch.zeros_like(d1p), alpha, _t(0.0))
        from1d = exp_map_1d(a, d1p, alpha)
        diff = (from2d - from1d).norm().item()
        assert diff < 1e-12, f"reduction failed seed={s}: diff={diff}"
    print(f"  single-direction reduction: 20/20 passed")


def test_sign_symmetry():
    """Simultaneous flip alpha_i->-alpha_i, d_i->-d_i gives the same new_a."""
    for s in range(20):
        a, d1, d2 = random_setup(s)
        a_hat = a / a.norm()
        d1p, d2p = orthonormalize(a_hat, d1, d2)
        a1 = _t(0.4 + 0.02 * s)
        a2 = _t(-0.3 - 0.01 * s)
        ref = exp_map_2d(a, d1p, d2p, a1, a2)
        flipped = exp_map_2d(a, -d1p, -d2p, -a1, -a2)
        diff = (ref - flipped).norm().item()
        assert diff < 1e-12, f"sign symmetry failed seed={s}: diff={diff}"
    print(f"  sign symmetry: 20/20 passed")


def test_degeneracy():
    """Collinear directions raise DegenerateDirection."""
    a, d1, _ = random_setup(0)
    a_hat = a / a.norm()
    # d1 || a_hat -> tangent component vanishes
    try:
        orthonormalize(a_hat, a_hat * 7.0, None)
        assert False, "expected raise (d1 || a_hat)"
    except DegenerateDirection:
        pass
    # d2 || d1 -> second tangent component vanishes
    try:
        orthonormalize(a_hat, d1, d1 * 3.0)
        assert False, "expected raise (d2 || d1)"
    except DegenerateDirection:
        pass
    print(f"  degeneracy: 2/2 passed")


def test_numerical_edge():
    """alpha1 = alpha2 = 1e-15 -> new_a == R * a_hat exactly."""
    a, d1, d2 = random_setup(0)
    a_hat = a / a.norm()
    d1p, d2p = orthonormalize(a_hat, d1, d2)
    new_a = exp_map_2d(a, d1p, d2p, _t(1e-15), _t(1e-15))
    diff = (new_a - a).norm().item()
    assert diff < 1e-14, f"numerical edge: diff={diff} (expected exact)"
    print(f"  numerical edge: passed (diff={diff:.2e})")


def test_project_to_chart_roundtrip():
    """Recovered (alpha1, alpha2) == input."""
    fails = 0
    for s in range(50):
        a, d1, d2 = random_setup(s)
        a_hat = a / a.norm()
        d1p, d2p = orthonormalize(a_hat, d1, d2)
        a1 = _t(0.3 + 0.01 * s)
        a2 = _t(-0.2 + 0.005 * s)
        new_a = exp_map_2d(a, d1p, d2p, a1, a2)
        r1, r2 = project_to_chart(a, d1p, d2p, new_a)
        if abs(r1.item() - a1.item()) > 1e-10 or abs(r2.item() - a2.item()) > 1e-10:
            fails += 1
            print(f"  roundtrip seed={s}: ({r1.item():.3e}, {r2.item():.3e}) "
                  f"vs ({a1.item():.3e}, {a2.item():.3e})")
    assert fails == 0, f"roundtrip: {fails}/50 failed"
    print(f"  project-to-chart roundtrip: 50/50 passed")


if __name__ == "__main__":
    print("running tests_parametrize.py")
    test_norm_matched()
    test_chart_distance()
    test_single_direction_reduction()
    test_sign_symmetry()
    test_degeneracy()
    test_numerical_edge()
    test_project_to_chart_roundtrip()
    print("\nall tests passed")
