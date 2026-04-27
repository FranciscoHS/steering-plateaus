"""E4 validation suite. Run before any production sweep, on one
(model, pair). Default: gemma at L=2, Gender x Tense.

Tests:
  1. Identity: 2D sweep slice alpha_2=0 == 1D sweep along Gender.
  2. Permutation: swap (d1, d2) -> L2_grid.T.
  3. Round-trip: project_to_chart recovers (alpha_1, alpha_2) from new_a.
  4. Zero: alpha_1=alpha_2=0 -> exactly anchor.

Outputs results/exp_map/data/validation_<target>_L<layer>_<d1>__<d2>.pkl
plus prints PASS/FAIL with maxabs.
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib, activations as actlib
from scripts.exp_map.lib.sweep_core import (
    sweep_1d_one_anchor, sweep_2d_one_anchor,
)
from scripts.exp_map.lib.parametrize import (
    orthonormalize, exp_map_2d, project_to_chart, eps_thresholds, CHART,
)

OUT_DIR = "results/exp_map/data"


def ts(): return time.strftime("%H:%M:%S")


def test_identity(model, ctx, anchor, d1, d2, perturb_layer, measure_layer,
                  device, angles_rad):
    r2 = sweep_2d_one_anchor(model, ctx, anchor, d1, d2,
                             perturb_layer, measure_layer, device, angles_rad)
    l2_2d_slice = r2["l2"][:, 0]  # alpha_2 = 0
    l2_1d = sweep_1d_one_anchor(model, ctx, anchor, d1, perturb_layer,
                                measure_layer, device, angles_rad)
    diff = float(np.max(np.abs(l2_2d_slice - l2_1d)))
    return diff, l2_2d_slice, l2_1d


def test_permutation(model, ctx, anchor, d1, d2, perturb_layer, measure_layer,
                     device, angles_rad):
    r12 = sweep_2d_one_anchor(model, ctx, anchor, d1, d2,
                              perturb_layer, measure_layer, device, angles_rad)
    r21 = sweep_2d_one_anchor(model, ctx, anchor, d2, d1,
                              perturb_layer, measure_layer, device, angles_rad)
    diff = float(np.max(np.abs(r12["l2"] - r21["l2"].T)))
    return diff


def test_roundtrip(anchor, d1, d2, angles_rad):
    """Pure CPU/algebra. Synthesize new_a from (alpha1, alpha2),
    project_to_chart back, compare."""
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(torch.float32),
                              d2.to(torch.float32))
    A1, A2 = np.meshgrid(angles_rad, angles_rad, indexing="ij")
    a1 = torch.from_numpy(A1.astype(np.float32)).flatten()
    a2 = torch.from_numpy(A2.astype(np.float32)).flatten()
    new_a = exp_map_2d(a32, d1p, d2p, a1, a2)
    r1, r2 = project_to_chart(a32, d1p, d2p, new_a)
    diff = float(max((r1 - a1).abs().max().item(),
                     (r2 - a2).abs().max().item()))
    return diff


def test_zero(anchor, d1, d2):
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(torch.float32),
                              d2.to(torch.float32))
    new_a = exp_map_2d(a32, d1p, d2p, torch.tensor(0.0), torch.tensor(0.0))
    diff = float((new_a - a32).abs().max())
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--d1", default="Gender")
    ap.add_argument("--d2", default="Tense")
    ap.add_argument("--n_steps", type=int, default=21)
    ap.add_argument("--max_angle", type=float, default=60.0)
    args = ap.parse_args()

    cfg = registry.TARGETS[args.target]
    print(f"[{ts()}] loading {cfg['model']}", flush=True)
    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    perturb_layer = args.layer

    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, perturb_layer)
    d1 = dirs_blob["directions"][args.d1].float()
    d2 = dirs_blob["directions"][args.d2].float()

    # FineWeb anchor 0
    fw = actlib.fineweb_acts(model, tokenizer, device, args.target,
                              perturb_layer, n=1)
    ctx = fw["contexts"][0].to(device=device,
                               dtype=next(model.parameters()).dtype)
    anchor = fw["activations"][0].to(device)
    d1 = d1.to(device); d2 = d2.to(device)

    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    print(f"[{ts()}] running identity test", flush=True)
    id_diff, l2_2d, l2_1d = test_identity(
        model, ctx, anchor, d1, d2, perturb_layer, measure_layer, device,
        angles_rad)
    print(f"  identity max|diff|: {id_diff:.3e}", flush=True)

    print(f"[{ts()}] running permutation test", flush=True)
    pm_diff = test_permutation(
        model, ctx, anchor, d1, d2, perturb_layer, measure_layer, device,
        angles_rad)
    print(f"  permutation max|diff|: {pm_diff:.3e}", flush=True)

    print(f"[{ts()}] running roundtrip test", flush=True)
    rt_diff = test_roundtrip(anchor.cpu(), d1.cpu(), d2.cpu(), angles_rad)
    print(f"  roundtrip max|diff| (rad): {rt_diff:.3e}", flush=True)

    print(f"[{ts()}] running zero test", flush=True)
    z_diff = test_zero(anchor.cpu(), d1.cpu(), d2.cpu())
    print(f"  zero max|diff|: {z_diff:.3e}", flush=True)

    # Threshold sanity: forward path uses bf16, expect identity/permutation
    # to be at the bf16 fp noise floor (around 1e-2 for an L^2 magnitude
    # ~10). roundtrip and zero are pure-algebra and should be 1e-5 / 0.
    passed = (id_diff < 5e-2 and pm_diff < 5e-2
              and rt_diff < 1e-5 and z_diff < 1e-5)
    print(f"\n{'PASS' if passed else 'FAIL'}", flush=True)

    out = {
        "chart": CHART,
        "eps_thresholds": eps_thresholds(),
        "target": args.target, "perturb_layer": perturb_layer,
        "measure_layer": measure_layer,
        "directions": (args.d1, args.d2),
        "angles_deg": angles_deg,
        "identity_max_diff": id_diff, "identity_l2_2d": l2_2d,
        "identity_l2_1d": l2_1d,
        "permutation_max_diff": pm_diff,
        "roundtrip_max_diff_rad": rt_diff,
        "zero_max_diff": z_diff,
        "passed": passed,
    }
    out_pkl = os.path.join(OUT_DIR,
                           f"validation_{args.target}_L{perturb_layer}_"
                           f"{args.d1}__{args.d2}.pkl")
    with open(out_pkl, "wb") as f: pickle.dump(out, f)
    print(f"  saved {out_pkl}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
