"""Shared sweep primitives in the exp-map chart.

Both sweep_1d and sweep_2d build perturbed activations with parametrize.exp_map_*
and then push them through the model from perturb_layer to measure_layer using
src.model.forward_from_layer_to_layer. Returns L^2 between perturbed and
unperturbed measurement-layer activations.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
from src.model import forward_from_layer_to_layer

from scripts.exp_map.lib.parametrize import (
    orthonormalize, exp_map_2d, exp_map_1d,
)


@torch.no_grad()
def sweep_1d_one_anchor(model, ctx, anchor, d, perturb_layer: int,
                        measure_layer: int, device, alphas_rad: np.ndarray,
                        batch_size: int = 64) -> np.ndarray:
    """L^2 distance at measure_layer for each alpha along direction d.

    ctx: (1, T-1, D) on device, model_dtype.
    anchor: (D,) float32 on device.
    d: (D,) float32 on device, not necessarily normalized or tangent.
    Returns (n_steps,) float32.
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d_perp, _ = orthonormalize(a_hat, d.to(torch.float32), None)
    alphas_t = torch.from_numpy(alphas_rad.astype(np.float32)).to(device)
    pts = exp_map_1d(a32, d_perp, alphas_t).to(model_dtype)
    n = pts.shape[0]
    ref = forward_from_layer_to_layer(
        model, ctx, a32[None, None].to(model_dtype),
        perturb_layer, measure_layer).float()
    out = np.zeros(n, dtype=np.float32)
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        batch = pts[st:ed].unsqueeze(1)  # (B, 1, D)
        pf = forward_from_layer_to_layer(
            model, ctx, batch, perturb_layer, measure_layer).float()
        out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def sweep_2d_one_anchor(model, ctx, anchor, d1, d2, perturb_layer: int,
                        measure_layer: int, device, alphas_rad: np.ndarray,
                        batch_size: int = 64) -> dict:
    """2D grid sweep in (alpha_1, alpha_2). Returns {'l2': (n, n)}.

    grid[i, j] = L2 at alpha_1 = alphas_rad[i], alpha_2 = alphas_rad[j].
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(torch.float32), d2.to(torch.float32))
    alphas_t = torch.from_numpy(alphas_rad.astype(np.float32)).to(device)
    A1, A2 = torch.meshgrid(alphas_t, alphas_t, indexing="ij")
    pts = exp_map_2d(a32, d1p, d2p, A1.flatten(), A2.flatten()).to(model_dtype)
    n_total = pts.shape[0]
    ref = forward_from_layer_to_layer(
        model, ctx, a32[None, None].to(model_dtype),
        perturb_layer, measure_layer).float()
    l2 = np.zeros(n_total, dtype=np.float32)
    for st in range(0, n_total, batch_size):
        ed = min(st + batch_size, n_total)
        batch = pts[st:ed].unsqueeze(1)
        pf = forward_from_layer_to_layer(
            model, ctx, batch, perturb_layer, measure_layer).float()
        l2[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    n = len(alphas_rad)
    return {"l2": l2.reshape(n, n)}


def run_1d_per_anchors(model, contexts, activations, d, perturb_layer,
                        measure_layer, device, alphas_rad) -> np.ndarray:
    """L2 grid (n_anchors, n_steps)."""
    model_dtype = next(model.parameters()).dtype
    out = np.zeros((len(activations), len(alphas_rad)), dtype=np.float32)
    for i in range(len(activations)):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a = activations[i].to(device=device)
        d_dev = d.to(device=device, dtype=torch.float32)
        out[i] = sweep_1d_one_anchor(model, ctx, a, d_dev, perturb_layer,
                                     measure_layer, device, alphas_rad)
    return out


def run_2d_per_anchors(model, contexts, activations, d1, d2,
                        perturb_layer, measure_layer, device,
                        alphas_rad) -> dict:
    """L2 grid (n_anchors, n, n)."""
    model_dtype = next(model.parameters()).dtype
    n = len(alphas_rad)
    grids = np.zeros((len(activations), n, n), dtype=np.float32)
    for i in range(len(activations)):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a = activations[i].to(device=device)
        d1_dev = d1.to(device=device, dtype=torch.float32)
        d2_dev = d2.to(device=device, dtype=torch.float32)
        r = sweep_2d_one_anchor(model, ctx, a, d1_dev, d2_dev,
                                perturb_layer, measure_layer, device,
                                alphas_rad)
        grids[i] = r["l2"]
    return {"l2": grids}
