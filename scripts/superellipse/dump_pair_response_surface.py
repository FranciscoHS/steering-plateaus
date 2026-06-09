#!/usr/bin/env python3
"""2D response surface R(alpha1, alpha2) for specific dict atom pairs.

Tests the 'non-monotonic / cancellation' hypothesis for the worst-fit
candidate x candidate pairs: does adding direction 2 at fixed direction 1
*reduce* the response (R(a1,a2) < R(a1,0))? The superellipse/first-crossing
fit assumes R is monotonically increasing in radius, so cancellation breaks it
and pushes the iso-contour 'beyond the square'.

Saves a heatmap (with iso-threshold contour) per pair + prints diagnostics.

Usage:
    python scripts/superellipse/dump_pair_response_surface.py \\
        --dict_path .../dictionary_snap20k.pt --model qwen3-1.7b --layer 14 \\
        --pairs "10854,20425;12186,13086;8495,11971" --out .../response_surfaces
"""
import argparse, os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.superellipse.learn_superellipse_dict import extract_context_and_activation
from scripts.superellipse.canonical.sweep_core import run_2d_per_anchors


def diagnostics(grid, angles):
    """grid[i,j] = R(angle1=angles[i], angle2=angles[j])."""
    n = grid.shape[0]
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = 0.5 * base
    # cancellation: adding the OTHER direction lowers response below the single-dir value
    # row i: R(a1_i, a2_j) vs R(a1_i, 0)
    drop_rows = (grid[:, [0]] - grid)          # >0 where adding a2 reduced R
    drop_cols = (grid[[0], :] - grid)          # >0 where adding a1 reduced R
    max_cancel = float(max(drop_rows.max(), drop_cols.max()))
    max_cancel_frac = max_cancel / base if base > 0 else np.nan
    # also vs the LARGER single-direction response (stronger cancellation test)
    single_max = np.maximum(grid[:, [0]], grid[[0], :])
    cancel_vs_max = float((single_max - grid).max()) / base if base > 0 else np.nan
    # monotonicity: fraction of rows/cols that ever decrease as the angle grows
    nonmono_rows = int(np.sum(np.any(np.diff(grid, axis=1) < -1e-6 * base, axis=1)))
    nonmono_cols = int(np.sum(np.any(np.diff(grid, axis=0) < -1e-6 * base, axis=0)))
    corner = float(grid[-1, -1]); axis_end = float(min(grid[-1, 0], grid[0, -1]))
    return dict(thresh=thresh, base=base, max_cancel_frac=max_cancel_frac,
                cancel_vs_max_frac=cancel_vs_max, nonmono_rows=nonmono_rows,
                nonmono_cols=nonmono_cols, n=n, corner=corner, axis_end=axis_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_anchors', type=int, default=20)
    ap.add_argument('--held_seed', type=int, default=2024)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps_grid', type=int, default=31)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = [tuple(int(x) for x in p.split(',')) for p in args.pairs.split(';') if p.strip()]
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2

    blob = torch.load(args.dict_path, map_location='cpu')
    D = blob['dictionary'].to(device=device, dtype=torch.float32)
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)

    tok = load_fineweb_fixed_length(args.n_anchors, tokenizer, seq_len=args.seq_len, seed=args.held_seed)
    ctx, act = extract_context_and_activation(model, tok, args.layer, device)
    contexts = [ctx[i:i+1].float().cpu() for i in range(ctx.shape[0])]
    activations = act.float().cpu()
    angles = np.linspace(0.0, args.max_angle, args.n_steps_grid)
    angles_rad = np.deg2rad(angles)

    saved = {}
    for (i, j) in pairs:
        r = run_2d_per_anchors(model, contexts, activations, D[i].cpu(), D[j].cpu(),
                               args.layer, measure_layer, device, angles_rad,
                               mode='geodesic', single_batch_ref=True)
        grid = np.median(r['l2'], axis=0)              # [n,n], grid[i_a1, j_a2]
        dg = diagnostics(grid, angles)
        saved[(i, j)] = {'grid': grid, 'angles': angles, 'diag': dg}
        print(f"pair ({i},{j}): cancel(vs single)={dg['max_cancel_frac']*100:.1f}%  "
              f"cancel(vs max-single)={dg['cancel_vs_max_frac']*100:.1f}%  "
              f"nonmono rows={dg['nonmono_rows']}/{dg['n']} cols={dg['nonmono_cols']}/{dg['n']}  "
              f"corner={dg['corner']:.1f} axis_end={dg['axis_end']:.1f} base={dg['base']:.1f}", flush=True)

        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        im = ax.pcolormesh(angles, angles, grid.T, shading='auto', cmap='viridis')
        cs = ax.contour(angles, angles, grid.T, levels=[dg['thresh']], colors='white', linewidths=2)
        ax.clabel(cs, fmt={dg['thresh']: 'iso-thresh'}, fontsize=8)
        plt.colorbar(im, ax=ax, label=r'response $\|\Delta\|_2$')
        ax.set_xlabel(rf'$\alpha_1$ (atom {i}, deg)'); ax.set_ylabel(rf'$\alpha_2$ (atom {j}, deg)')
        ax.set_title(f'response surface atoms ({i},{j})  '
                     f"cancel={dg['cancel_vs_max_frac']*100:.0f}%  "
                     f"nonmono r/c={dg['nonmono_rows']}/{dg['nonmono_cols']}", fontsize=10)
        plt.tight_layout()
        png = os.path.join(args.out, f'response_surface_{i}_{j}.png')
        fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  saved {png}', flush=True)

    with open(os.path.join(args.out, 'surfaces.pkl'), 'wb') as f:
        pickle.dump(saved, f)
    print(f"saved {os.path.join(args.out,'surfaces.pkl')}", flush=True)


if __name__ == '__main__':
    main()
