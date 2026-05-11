"""Diagnose the L^2 baseline floor at alpha=(0,0) in exp_map sweep_2d.

At alpha=(0,0) the perturbed point equals the unperturbed anchor exactly
(verified from exp_map_2d), yet sweep_2d reports L^2 ~= 8 there. This
script disentangles:

  (A) bf16 forward pass non-determinism: ref vs pf with batch=1
  (B) batched-vs-unbatched matmul: ref vs pf at index 0 of a batch of 64
  (C) within-process determinism: same call twice
  (D) anchor scaling: does the floor scale with |anchor|?

Usage:
  python scripts/exp_map/debug_floor.py --target gemma --layer 2 \\
      --n_anchors 8 --batch_size 64
"""
from __future__ import annotations
import os, sys, argparse, json
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, forward_from_layer_to_layer
from scripts.exp_map.lib import registry
from scripts.exp_map.lib.activations import fineweb_acts
from scripts.exp_map.lib.parametrize import orthonormalize, exp_map_2d
from scripts.exp_map.lib.sweep_core import build_perturbed_pts_2d


def _l2(a, b):
    return (a - b).norm(dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--n_anchors", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--measure_offset", type=int, default=-2)
    ap.add_argument("--big_grid", type=int, default=31,
                    help="grid size for the (D) single-big-batch test")
    ap.add_argument("--big_max_rad", type=float, default=np.pi/2,
                    help="alpha_max in radians for the (D) test "
                         "(default pi/2 = 90 deg)")
    ap.add_argument("--out", default="results/exp_map/debug/floor_diag.json")
    args = ap.parse_args()

    cfg = registry.TARGETS[args.target]
    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)
    n_layers = sum(1 for _ in model.modules() if False)  # placeholder
    from src.model import _get_blocks
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers + args.measure_offset
    perturb_layer = args.layer
    print(f"perturb_layer={perturb_layer}  measure_layer={measure_layer}  "
          f"n_layers={n_layers}  dtype=bf16", flush=True)

    acts = fineweb_acts(model, tokenizer, device, args.target, perturb_layer)
    activations = acts["activations"][:args.n_anchors]
    contexts = acts["contexts"][:args.n_anchors]
    print(f"loaded {len(activations)} anchors", flush=True)

    model_dtype = next(model.parameters()).dtype
    rng = np.random.default_rng(0)
    rows = []
    for i in range(len(activations)):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a = activations[i].to(device=device).to(torch.float32)
        a_norm = float(a.norm().cpu())

        # Two random orthogonal-ish directions for filling out the batch.
        d1 = torch.from_numpy(rng.standard_normal(a.shape[-1]).astype(np.float32)
                              ).to(device)
        d2 = torch.from_numpy(rng.standard_normal(a.shape[-1]).astype(np.float32)
                              ).to(device)
        a_hat = a / a.norm()
        d1p, d2p = orthonormalize(a_hat, d1, d2)

        # ref_act: forward the bare anchor (batch=1)
        ref_a = forward_from_layer_to_layer(
            model, ctx, a[None, None].to(model_dtype),
            perturb_layer, measure_layer).float()
        ref_b = forward_from_layer_to_layer(
            model, ctx, a[None, None].to(model_dtype),
            perturb_layer, measure_layer).float()
        within_proc_det = float(_l2(ref_a, ref_b)[0])  # (C)

        # (A) pf at alpha=(0,0), batch=1
        pt_solo = a[None, None].to(model_dtype)
        pf_solo = forward_from_layer_to_layer(
            model, ctx, pt_solo, perturb_layer, measure_layer).float()
        l2_A = float(_l2(pf_solo, ref_a)[0])

        # (B) pf at alpha=(0,0) in a batch of args.batch_size, at index 0
        # Build the batch like sweep_core does, with non-trivial alphas
        # for the rest so the matmul shapes/positions match the real sweep.
        n_grid = int(np.ceil(np.sqrt(args.batch_size)))
        # use a coarse 2D grid of small angles
        alphas = torch.linspace(0.0, 0.5, n_grid, device=device)  # rad
        pts_batch = build_perturbed_pts_2d(
            a, d1p, d2p, alphas, mode="geodesic")  # (n_grid*n_grid, D)
        # ensure index 0 is the (0,0) point — exp_map_2d puts (alpha1,alpha2)
        # in row-major, so [0] corresponds to (alphas[0], alphas[0]) = (0,0).
        # Truncate to batch_size.
        pts_batch = pts_batch[:args.batch_size].to(model_dtype)
        # Sanity: confirm pts_batch[0] equals a in fp32.
        delta_in = float((pts_batch[0].float() - a).norm().cpu())
        pf_batch = forward_from_layer_to_layer(
            model, ctx, pts_batch.unsqueeze(1),
            perturb_layer, measure_layer).float()
        l2_B = float(_l2(pf_batch[0:1], ref_a)[0])

        # (B') same batch run twice
        pf_batch_2 = forward_from_layer_to_layer(
            model, ctx, pts_batch.unsqueeze(1),
            perturb_layer, measure_layer).float()
        within_batch_det = float(_l2(pf_batch[0:1], pf_batch_2[0:1])[0])

        # (D) Single big batch covering a full grid; use pf[0] as the
        # within-batch reference and check that L2(pf[0] - pf[0]) = 0
        # exactly, plus report the next-cell L2 (smallest non-zero alpha)
        # to compare against the cross-batch floor (~8 from B).
        n_grid_big = args.big_grid
        alphas_big = torch.linspace(
            0.0, args.big_max_rad, n_grid_big, device=device)
        pts_big = build_perturbed_pts_2d(
            a, d1p, d2p, alphas_big, mode="geodesic").to(model_dtype)
        # row-major index: [i*n + j] corresponds to (alphas[i], alphas[j])
        # so [0] is (0,0) and [1] is (0, alphas[1]).
        n_pts = pts_big.shape[0]
        # Replace a few non-anchor rows with an exact copy of the anchor,
        # to directly test whether row-position-within-batch affects output.
        # Use indices 7, 100, 500, 800 (away from index 0).
        copy_idxs = [7, 100, 500, 800]
        copy_idxs = [k for k in copy_idxs if k < n_pts]
        for k in copy_idxs:
            pts_big[k] = pts_big[0]
        try:
            pf_big = forward_from_layer_to_layer(
                model, ctx, pts_big.unsqueeze(1),
                perturb_layer, measure_layer).float()
            # NON-tautological self-test: same input at different positions
            # in the same batch.
            row_pos_diffs = [
                float(_l2(pf_big[k:k+1], pf_big[0:1])[0]) for k in copy_idxs
            ]
            l2_D_self_max = max(row_pos_diffs) if row_pos_diffs else float("nan")
            # next cell along axis 2 (smallest non-zero alpha) — but now
            # idx 1 is still alpha=(0, alphas[1]) since we only overwrote
            # 7, 100, 500, 800.
            l2_D_smallest = float(_l2(pf_big[1:2], pf_big[0:1])[0])
            # also: L2 of pf_big[0] against the solo ref_a — measures the
            # cross-batch-shape rounding bias for an even bigger batch.
            l2_D_pf0_vs_solo = float(_l2(pf_big[0:1], ref_a)[0])
            l2_D_self = l2_D_self_max  # keep key name; now meaningful
            big_n = int(n_pts)
            big_ok = True
        except torch.cuda.OutOfMemoryError as e:
            l2_D_self = float("nan")
            l2_D_smallest = float("nan")
            l2_D_pf0_vs_solo = float("nan")
            big_n = int(n_pts)
            big_ok = False
            print(f"  (D) OOM at batch={n_pts}: {e}", flush=True)

        rows.append({
            "anchor_idx": i,
            "anchor_norm": a_norm,
            "delta_input_at_0": delta_in,
            "ref_within_proc_det": within_proc_det,
            "L2_solo_vs_ref": l2_A,
            "L2_inbatch_vs_ref": l2_B,
            "L2_inbatch_within_proc_det": within_batch_det,
            "batch_size": int(pts_batch.shape[0]),
            "big_batch_size": big_n,
            "big_batch_ok": big_ok,
            "L2_bigbatch_self": l2_D_self,
            "L2_bigbatch_smallest_step": l2_D_smallest,
            "L2_bigbatch_pf0_vs_solo": l2_D_pf0_vs_solo,
        })
        print(json.dumps(rows[-1], indent=None), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "perturb_layer": perturb_layer,
            "measure_layer": measure_layer,
            "model": cfg["model"],
            "rows": rows,
        }, f, indent=2)
    print(f"saved {args.out}", flush=True)

    # Quick summary
    arr = lambda k: np.array([r[k] for r in rows])
    print("\nSummary (over anchors):")
    for k in ("L2_solo_vs_ref", "L2_inbatch_vs_ref",
              "ref_within_proc_det", "L2_inbatch_within_proc_det",
              "L2_bigbatch_self", "L2_bigbatch_smallest_step",
              "L2_bigbatch_pf0_vs_solo",
              "delta_input_at_0", "anchor_norm"):
        v = arr(k)
        print(f"  {k}: median={np.median(v):.4f}  min={v.min():.4f}  "
              f"max={v.max():.4f}  std={v.std():.4f}")


if __name__ == "__main__":
    main()
