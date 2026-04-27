#!/usr/bin/env python3
"""Extended-angle 2D iso-plateau sweep for Portuguese x Gender at Gemma-2-9B L2.

Differences from sweep_isoplateau_L2_wide.py:
  - Non-uniform angle grid spanning [0°, 90°]:
      * Dense 0.05° steps from 0 to 1°  -> resolves linear regime, fits at
        very low L2 thresholds (~1 to ~10).
      * Medium 1° steps from 1° to 30°  -> matches the previous wide grid in
        the main p~2 regime.
      * Coarse 3° steps from 30° to 90° -> captures high-threshold contours
        that needed angles past 60°.
    Total ~71 points per axis (vs 61 in the wide sweep).
  - Same-batch reference fix retained (so noise floor is 0).

Output: results/paper_L2_gemma2_9b/sweep_isoplateau_L2_xwide.pkl
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_lesswrong_figures import extract_activations_holdout
from scripts.perturbation.sweep_isoplateau_L2_wide import sweep_2d_samebatch

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2

OUT_DIR = 'results/paper_L2_gemma2_9b'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_PKL = os.path.join(OUT_DIR, 'sweep_isoplateau_L2_xwide.pkl')


def build_angle_grid():
    """Non-uniform angle grid in degrees."""
    seg1 = np.arange(0.0, 1.0, 0.05)        # 0.00, 0.05, ..., 0.95  (20 pts)
    seg2 = np.arange(1.0, 30.0, 1.0)        # 1.0, 2.0, ..., 29.0    (29 pts)
    seg3 = np.arange(30.0, 90.0 + 1e-6, 3.0)  # 30, 33, ..., 90      (21 pts)
    grid = np.concatenate([seg1, seg2, seg3])
    return grid


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("Loading directions...", flush=True)
    with open(DIR_PKL, 'rb') as f:
        dirs = pickle.load(f)
    d_pt = dirs['d_portuguese']
    d_gender = dirs['d_gender']

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}",
          flush=True)

    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting {len(holdout)} holdout activations at L{LAYER}...",
          flush=True)
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    angles_deg = build_angle_grid()
    angles_rad = np.deg2rad(angles_deg)
    n_steps = len(angles_deg)
    print(f"Non-uniform grid: {n_steps} angles, range "
          f"[{angles_deg[0]:.2f}, {angles_deg[-1]:.2f}]°", flush=True)
    print(f"  Steps: {np.unique(np.round(np.diff(angles_deg), 3))}", flush=True)
    print(f"  Total: {n_steps**2} points x {N_ACT} acts = "
          f"{n_steps**2 * N_ACT} forward passes", flush=True)

    all_l2, all_cos = [], []
    for i in range(N_ACT):
        l2, cos = sweep_2d_samebatch(
            model, contexts[i], activations[i],
            d_pt, d_gender, LAYER, measure_layer, device,
            angles_rad, angles_rad)
        all_l2.append(l2)
        all_cos.append(cos)
        if i == 0:
            print(f"  noise floor (0,0) on first activation: "
                  f"L2={l2[0,0]:.4g}  cos={cos[0,0]:.4g}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_ACT}", flush=True)

    results = {
        'angles_deg': angles_deg,
        'l2': np.stack(all_l2),
        'cosine': np.stack(all_cos),
        'layer': LAYER,
        'measure_layer': measure_layer,
        'direction_labels': ('Portuguese', 'Gender'),
        'config': {'grid': 'non-uniform', 'max_angle': 90.0,
                   'n_steps': n_steps, 'n_act': N_ACT,
                   'same_batch_reference': True, 'dtype': 'bfloat16',
                   'segments': [(0.0, 1.0, 0.05), (1.0, 30.0, 1.0),
                                (30.0, 90.0, 3.0)]},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUT_PKL}", flush=True)


if __name__ == '__main__':
    main()
