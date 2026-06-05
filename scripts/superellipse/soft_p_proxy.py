#!/usr/bin/env python3
"""Differentiable estimators of the superellipse exponent p, in the eval frame.

The dictionary build's original objective minimizes R_mix / (R_f + R_c) at a
single input angle -- a surrogate for p that lives in the "fixed input angle ->
output response" frame and tracks the eval-time p only weakly. This module
implements estimators that live in the SAME frame as the eval fit (mirror
`plot_boundary_pair.boundary_for_set` / `canonical/superellipse.py`):

  - sample the output L2 response along n_phi rays in the (theta_d, theta_c)
    input-angle plane (phi=0 is the d-axis, phi=90 the c-axis), each at n_r
    geodesic radii;
  - threshold T = 0.5 * min(max R on d-axis, max R on c-axis)  (eval's rule);
  - intercept r_phi where R(r_phi) = T per ray;
  - exact-geodesic normalization (sin(r)/sin(theta_axis) per axis), identical to
    the eval.

Two ways to turn that into a scalar:
  1. `soft_p_estimate`  -- fit |x|^p+|y|^p=1 via Gauss-Newton (calibrates well
     as a *measurement*: Spearman~0.86 vs eval-p; but a poor optimization
     objective -- its gradient doesn't drive p up).
  2. `diag_rho_estimate` -- maximize the geodesic-normalized diagonal intercept
     rho (monotone in p via p = 1/(1/2 - log2 rho)). No nonlinear fit, no cap;
     combined with a smooth (all-radii) intercept this is meant to give a clean
     gradient to push the contour toward the corner.

Geometry matches `canonical/sweep_core` / `parametrize.exp_map_2d` exactly:
along ray phi at radius r the perturbed activation is
    a' = R (cos r * a_hat + sin r * (cos phi * d_perp + sin phi * c_perp))
with d_perp, c_perp the Gram-Schmidt orthonormal tangent dirs at a_hat.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import forward_from_layer_to_layer


# ---------- intercept helpers ----------

def soft_intercept(radii, responses, level):
    """First-crossing intercept of `responses` through `level` (piecewise diff).

    Mirrors `canonical.superellipse.axis_intercept`: first segment where
    (R[i-1]-level)(R[i]-level) <= 0, linearly interpolated. The bracket index is
    chosen by a non-differentiable argmax; only the bracketing pair of R's
    carries gradient.

    radii: [n_r] constants. responses: [B, n_r]. level: [B].
    Returns (intercept [B], crossed [B] bool). Non-crossing rows cap at radii[-1].
    """
    B, n_r = responses.shape
    device = responses.device
    L = level.unsqueeze(-1)
    diff = responses - L
    sign_change = (diff[:, :-1] * diff[:, 1:]) <= 0
    crossed = sign_change.any(dim=-1)
    idx = torch.argmax(sign_change.float(), dim=-1)
    ar = torch.arange(B, device=device)
    R0 = responses[ar, idx]
    R1 = responses[ar, idx + 1]
    r0 = radii[idx]
    r1 = radii[idx + 1]
    denom = R1 - R0
    denom_safe = torch.where(denom.abs() > 1e-9, denom, torch.ones_like(denom))
    t = (level - R0) / denom_safe
    inter = r0 + t * (r1 - r0)
    inter = torch.where(crossed, inter, radii[-1].expand_as(inter))
    return inter, crossed


def smooth_intercept(radii, responses, level, tau_frac=0.15):
    """Fully-differentiable level-crossing intercept using ALL radii.

    For a (roughly) monotone-increasing R(r), the crossing of `level` is located
    by soft bracket weights: segment i gets weight
        w_i = sigmoid((L - R_i)/tau) * sigmoid((R_{i+1} - L)/tau),
    which is ~1 only where R_i < L < R_{i+1}. The within-segment crossing is the
    clamped linear-interp radius; the intercept is the w-weighted mean of those.
    Every R sample feeds the gradient (no hard argmax, no jumps).

    tau = tau_frac * level sets the smoothing scale (per row, scale-invariant).
    radii: [n_r]. responses: [B, n_r]. level: [B].
    Returns (intercept [B], crossed [B] bool). No-crossing rows cap at radii[-1].
    """
    B, n_r = responses.shape
    L = level.unsqueeze(-1)                                   # [B, 1]
    tau = (tau_frac * level).clamp_min(1e-6).unsqueeze(-1)    # [B, 1]
    R0 = responses[:, :-1]
    R1 = responses[:, 1:]
    r0 = radii[:-1].unsqueeze(0)                              # [1, n_r-1]
    r1 = radii[1:].unsqueeze(0)
    w = torch.sigmoid((L - R0) / tau) * torch.sigmoid((R1 - L) / tau)
    denom = R1 - R0
    denom_safe = torch.where(denom.abs() > 1e-9, denom, torch.ones_like(denom))
    frac = ((L - R0) / denom_safe).clamp(0.0, 1.0)
    cross = r0 + frac * (r1 - r0)                             # [B, n_r-1]
    wsum = w.sum(-1)
    inter = (w * cross).sum(-1) / (wsum + 1e-9)
    crossed = wsum > 1e-3
    inter = torch.where(crossed, inter, radii[-1].expand_as(inter))
    return inter, crossed


def gauss_newton_p(x, y, n_iter=20, p0=2.0, p_min=0.5, p_max=10.0, eps=1e-9):
    """Fit p in |x|^p + |y|^p = 1 by Gauss-Newton, differentiable in (x, y).

    x, y: [..., M]. p clamped to [p_min, p_max] each step. Returns p [...].
    """
    lx = torch.log(x.clamp_min(1e-6))
    ly = torch.log(y.clamp_min(1e-6))
    p = torch.full(x.shape[:-1], float(p0), device=x.device, dtype=x.dtype)
    for _ in range(n_iter):
        pe = p.unsqueeze(-1)
        xp = torch.exp(pe * lx)
        yp = torch.exp(pe * ly)
        s = xp + yp - 1.0
        sp = xp * lx + yp * ly
        num = (s * sp).sum(-1)
        den = (sp * sp).sum(-1) + eps
        p = (p - num / den).clamp(p_min, p_max)
    return p


# ---------- shared ray-response builder ----------

def _ray_responses(model, points, anchor_idx, ctx_stack, f_refs_stack,
                   layer, measure_layer, chunk):
    """Forward `points` [P, d] and return L2 vs each point's anchor ref [P]."""
    model_dtype = next(model.parameters()).dtype
    P = points.shape[0]
    chunks = []
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        pts = points[s:e].unsqueeze(1).to(model_dtype)
        ai = anchor_idx[s:e]
        ctx_b = ctx_stack[ai]
        f = forward_from_layer_to_layer(
            model, ctx_b, pts, layer, measure_layer).float()
        chunks.append((f - f_refs_stack[ai]).norm(dim=-1))
    return torch.cat(chunks)


def ray_response_grids(model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
                       layer, measure_layer, phis_rad, radii_rad,
                       anchor_reduce='median', chunk=512):
    """Anchor-reduced L2 responses along the d-axis, c-axis and interior rays.

    Returns:
        radii  [n_r] tensor,  phis_pos [n_pp] tensor (phi>0 rays),
        R_dax  [n_r]          (phi=0 d-axis, shared across contrasts),
        R_cax  [K, n_r]       (phi=90 c-axis, per contrast),
        R_int  [K, n_int, n_r](interior rays, per contrast).
    All differentiable w.r.t. d (and contrast).
    """
    device = d.device
    n_act, d_model = acts_stack.shape
    K = contrast.shape[0]
    n_phi = len(phis_rad)
    n_r = len(radii_rad)
    n_pp = n_phi - 1
    n_int = n_pp - 1

    a_norm = acts_stack.norm(dim=-1, keepdim=True)
    a_hat = acts_stack / (a_norm + 1e-10)

    d_e = d.unsqueeze(0).expand(n_act, -1)
    d_p = d_e - (d_e * a_hat).sum(-1, keepdim=True) * a_hat
    d_p = d_p / (d_p.norm(dim=-1, keepdim=True) + 1e-10)

    c_e = contrast.unsqueeze(0).expand(n_act, K, d_model)
    a_hat_e = a_hat.unsqueeze(1)
    c_p = c_e - (c_e * a_hat_e).sum(-1, keepdim=True) * a_hat_e
    d_p_e = d_p.unsqueeze(1)
    c_p = c_p - (c_p * d_p_e).sum(-1, keepdim=True) * d_p_e
    c_p = c_p / (c_p.norm(dim=-1, keepdim=True) + 1e-10)

    radii = torch.as_tensor(np.asarray(radii_rad), dtype=torch.float32,
                            device=device)
    phis = torch.as_tensor(np.asarray(phis_rad), dtype=torch.float32,
                           device=device)
    cos_r = torch.cos(radii)
    sin_r = torch.sin(radii)
    phis_pos = phis[1:]
    cos_p = torch.cos(phis_pos)
    sin_p = torch.sin(phis_pos)

    # d-axis ray (phi=0), shared across contrasts.
    pts_d = (a_norm.unsqueeze(1) *
             (cos_r.view(1, n_r, 1) * a_hat.unsqueeze(1)
              + sin_r.view(1, n_r, 1) * d_p.unsqueeze(1)))    # [n_act, n_r, d]
    aidx_d = torch.arange(n_act, device=device).repeat_interleave(n_r)
    R_d = _ray_responses(model, pts_d.reshape(-1, d_model), aidx_d,
                         ctx_stack, f_refs_stack, layer, measure_layer,
                         chunk).reshape(n_act, n_r)

    # phi>0 rays (interior + c-axis), per contrast.
    v = (cos_p.view(1, 1, n_pp, 1) * d_p.view(n_act, 1, 1, d_model)
         + sin_p.view(1, 1, n_pp, 1) * c_p.view(n_act, K, 1, d_model))
    pts_ray = (a_norm.view(n_act, 1, 1, 1, 1) *
               (cos_r.view(1, 1, 1, n_r, 1) * a_hat.view(n_act, 1, 1, 1, d_model)
                + sin_r.view(1, 1, 1, n_r, 1) * v.unsqueeze(3)))
    aidx_ray = torch.arange(n_act, device=device).repeat_interleave(
        K * n_pp * n_r)
    R_ray = _ray_responses(model, pts_ray.reshape(-1, d_model), aidx_ray,
                           ctx_stack, f_refs_stack, layer, measure_layer,
                           chunk).reshape(n_act, K, n_pp, n_r)

    if anchor_reduce == 'median':
        R_d_m = torch.quantile(R_d, 0.5, dim=0)
        R_ray_m = torch.quantile(R_ray, 0.5, dim=0)
    elif anchor_reduce == 'mean':
        R_d_m = R_d.mean(0)
        R_ray_m = R_ray.mean(0)
    else:
        raise ValueError(anchor_reduce)

    R_dax = R_d_m                          # [n_r]
    R_cax = R_ray_m[:, n_pp - 1, :]        # [K, n_r]
    R_int = R_ray_m[:, :n_int, :]          # [K, n_int, n_r]
    return radii, phis_pos, R_dax, R_cax, R_int


def _threshold_and_axes(radii, R_dax, R_cax, intercept_fn):
    """T = 0.5*min(d-axis max, c-axis max); axis intercepts theta1, theta2."""
    K = R_cax.shape[0]
    n_r = radii.shape[0]
    d_max = R_dax.max()
    c_max = R_cax.max(dim=-1).values
    T = 0.5 * torch.minimum(d_max.expand_as(c_max), c_max)        # [K]
    theta1, c1 = intercept_fn(radii, R_dax.unsqueeze(0).expand(K, n_r), T)
    theta2, c2 = intercept_fn(radii, R_cax, T)
    return T, theta1, theta2, c1, c2


# ---------- estimator 1: Gauss-Newton p fit ----------

def soft_p_estimate(model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
                    layer, measure_layer, phis_rad, radii_rad,
                    p_cap=10.0, n_newton=20, anchor_reduce='median',
                    chunk=512, return_diag=False):
    """Differentiable p_hat per contrast (Gauss-Newton fit), mean over contrasts.

    Good as a measurement of p; weak as an optimization objective.
    """
    radii, phis_pos, R_dax, R_cax, R_int = ray_response_grids(
        model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
        layer, measure_layer, phis_rad, radii_rad, anchor_reduce, chunk)
    K = R_cax.shape[0]
    n_r = radii.shape[0]
    n_int = R_int.shape[1]

    T, theta1, theta2, c1, c2 = _threshold_and_axes(
        radii, R_dax, R_cax, soft_intercept)

    r_int, cint = soft_intercept(
        radii, R_int.reshape(K * n_int, n_r),
        T.view(K, 1).expand(K, n_int).reshape(K * n_int))
    r_int = r_int.reshape(K, n_int)

    phi_int = phis_pos[:n_int]
    sin_t1 = torch.sin(theta1).clamp_min(1e-6).unsqueeze(-1)
    sin_t2 = torch.sin(theta2).clamp_min(1e-6).unsqueeze(-1)
    xn = (torch.sin(r_int) / sin_t1) * torch.cos(phi_int).unsqueeze(0)
    yn = (torch.sin(r_int) / sin_t2) * torch.sin(phi_int).unsqueeze(0)

    p_k = gauss_newton_p(xn, yn, n_iter=n_newton, p_max=p_cap)
    p_hat = p_k.mean()
    if return_diag:
        return p_hat, {
            'p_k': p_k.detach(), 'T': T.detach(),
            'theta1_deg': torch.rad2deg(theta1).detach(),
            'theta2_deg': torch.rad2deg(theta2).detach(),
            'r_int_deg': torch.rad2deg(r_int).detach(),
            'xn': xn.detach(), 'yn': yn.detach(),
            'crossed_int': cint.reshape(K, n_int).detach(),
        }
    return p_hat


# ---------- estimator 2: diagonal-rho (simplified objective) ----------

def _p_from_rho(rho):
    """Closed form p = 1/(1/2 - log2 rho); rho=1->2, sqrt2->inf, 1/sqrt2->1."""
    log2rho = torch.log(rho.clamp_min(1e-6)) / np.log(2.0)
    return 1.0 / (0.5 - log2rho).clamp_min(1e-3)


def diag_rho_estimate(model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
                      layer, measure_layer, phis_rad, radii_rad,
                      anchor_reduce='median', tau_frac=0.15, chunk=512,
                      return_diag=False):
    """Geodesic-normalized diagonal intercept rho (monotone in p), mean over K.

    rho_k = mean over interior rays of the normalized point radius
            sqrt(xn^2 + yn^2). Uses the smooth (all-radii) intercept so every
            forward feeds the gradient; no Gauss-Newton, no cap. Maximizing rho
            maximizes p. Returns (rho_hat, [diag]); rho_hat is the mean rho.
    """
    radii, phis_pos, R_dax, R_cax, R_int = ray_response_grids(
        model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
        layer, measure_layer, phis_rad, radii_rad, anchor_reduce, chunk)
    K = R_cax.shape[0]
    n_r = radii.shape[0]
    n_int = R_int.shape[1]

    intercept_fn = lambda rr, resp, lvl: smooth_intercept(rr, resp, lvl, tau_frac)
    T, theta1, theta2, c1, c2 = _threshold_and_axes(
        radii, R_dax, R_cax, intercept_fn)

    r_int, cint = smooth_intercept(
        radii, R_int.reshape(K * n_int, n_r),
        T.view(K, 1).expand(K, n_int).reshape(K * n_int), tau_frac)
    r_int = r_int.reshape(K, n_int)

    phi_int = phis_pos[:n_int]
    sin_t1 = torch.sin(theta1).clamp_min(1e-6).unsqueeze(-1)
    sin_t2 = torch.sin(theta2).clamp_min(1e-6).unsqueeze(-1)
    xn = (torch.sin(r_int) / sin_t1) * torch.cos(phi_int).unsqueeze(0)
    yn = (torch.sin(r_int) / sin_t2) * torch.sin(phi_int).unsqueeze(0)
    rho_rays = torch.sqrt(xn * xn + yn * yn + 1e-12)              # [K, n_int]
    rho_k = rho_rays.mean(dim=-1)                                 # [K]
    rho_hat = rho_k.mean()
    if return_diag:
        return rho_hat, {
            'rho_k': rho_k.detach(),
            'p_monitor': _p_from_rho(rho_k).detach(),
            'T': T.detach(),
            'theta1_deg': torch.rad2deg(theta1).detach(),
            'theta2_deg': torch.rad2deg(theta2).detach(),
            'r_int_deg': torch.rad2deg(r_int).detach(),
            'crossed_int': cint.reshape(K, n_int).detach(),
        }
    return rho_hat


def default_phis_radii(n_rays, n_radii, max_angle_deg):
    """Standard ray/radius grids: phi in [0, 90] deg, r in [0, max_angle] deg."""
    phis_rad = np.deg2rad(np.linspace(0.0, 90.0, n_rays))
    radii_rad = np.deg2rad(np.linspace(0.0, max_angle_deg, n_radii))
    return phis_rad, radii_rad
