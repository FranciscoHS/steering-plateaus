#!/usr/bin/env python3
"""Iso-plateau sweeps + superellipse p-fits for J-Lens direction pairs.

Qwen3-8B, perturb at L20, measure L2 response at L34 (n_layers-2). For each
direction pair (EN×EN and ZH×ZH categories from build_jlens_directions.py),
sweeps the 2D spherical angle grid around 30 FineWeb base activations,
checkpointing one pkl per pair so spot interruptions lose at most one pair.

After sweeping, fits the superellipse exponent p per pair at several L2
thresholds and writes a summary pkl + text file.

Run on a pod:  python -u scripts/jlens/sweep_isoplateau_jlens.py
Fit-only:      python -u scripts/jlens/sweep_isoplateau_jlens.py --fit-only
"""
import os, sys, pickle, argparse, json
sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_isoplateau import sweep_2d
from scripts.perturbation.plot_isoplateau import (
    extract_contour_points, fit_superellipse)

# --- Config ---
MODEL_NAME = 'qwen3-8b'
LAYER = 20
N_ACT = 30
SEED = 123
MAX_ANGLE = 20.0    # degrees per axis
N_STEPS = 41
SEQ_LEN = 5         # matches gemma-2-9b baseline sweeps (SEQ_LEN=5 fineweb)
SWEEP_BATCH = 64
# Response scale at qwen3-8b L20→L34 is much larger than gemma-2-9b: noise
# floor ~10, axis max ~200. Thresholds must sit well above the floor.
L2_THRESHOLDS = [15.0, 20.0, 30.0, 50.0, 80.0, 120.0]

RESULTS_DIR = 'results/jlens/qwen3-8b'
DIRECTIONS_PKL = os.path.join(RESULTS_DIR, f'jlens_direction_pairs_L{LAYER}.pkl')
SWEEP_DIR = os.path.join(RESULTS_DIR, 'sweeps')
PROMPTS_JSON = os.path.join(RESULTS_DIR, 'base_prompts.json')
SUMMARY_PKL = os.path.join(RESULTS_DIR, f'jlens_p_summary_L{LAYER}.pkl')
SUMMARY_TXT = os.path.join(RESULTS_DIR, f'jlens_p_summary_L{LAYER}.txt')


def find_threshold(curve, angles, thresh):
    idx = np.searchsorted(curve, thresh)
    if idx == 0 or idx >= len(angles):
        return angles[idx] if idx < len(angles) else angles[-1]
    x0, x1 = angles[idx - 1], angles[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    return x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)


def iter_pairs(dirs):
    for cat, key in [('EN', 'en_pairs'), ('ZH', 'zh_pairs')]:
        for i, pair in enumerate(dirs[key]):
            yield cat, i, pair


def run_sweeps():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SWEEP_DIR, exist_ok=True)

    with open(DIRECTIONS_PKL, 'rb') as f:
        dirs = pickle.load(f)

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}")

    # Base prompts: FineWeb, exactly SEQ_LEN tokens (matches gemma baselines).
    # Cache is keyed by tokenizer inside load_fineweb_fixed_length.
    token_ids_list = load_fineweb_fixed_length(
        N_ACT, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    with open(PROMPTS_JSON, 'w') as f:
        json.dump([tokenizer.decode(t) for t in token_ids_list], f)
    print(f"Saved base prompts: {PROMPTS_JSON}")

    print("Extracting base activations...")
    contexts, activations = [], []
    for ids in token_ids_list:
        with torch.no_grad():
            out = model(ids.unsqueeze(0).to(device), output_hidden_states=True)
        hidden = out.hidden_states[LAYER + 1]
        contexts.append(hidden[:, :-1, :].float().cpu())
        activations.append(hidden[:, -1, :].squeeze(0).float().cpu())
    activations = torch.stack(activations)

    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)

    for cat, i, pair in iter_pairs(dirs):
        out_pkl = os.path.join(SWEEP_DIR, f'sweep_{cat}_{i:02d}.pkl')
        if os.path.exists(out_pkl):
            print(f"[{cat} {i}] exists, skipping: {out_pkl}")
            continue
        t1, t2 = pair['tokens']
        print(f"[{cat} {i}] {t1!r} x {t2!r} (cos={pair['cosine']:+.3f})")
        d1 = torch.tensor(pair['vecs'][0], dtype=torch.float32)
        d2 = torch.tensor(pair['vecs'][1], dtype=torch.float32)

        all_l2, all_cos = [], []
        for k in range(len(token_ids_list)):
            l2_grid, cos_grid = sweep_2d(
                model, contexts[k], activations[k], d1, d2,
                LAYER, measure_layer, device, angles_rad, angles_rad,
                batch_size=SWEEP_BATCH)
            all_l2.append(l2_grid)
            all_cos.append(cos_grid)
            if (k + 1) % 10 == 0:
                print(f"    act {k + 1}/{len(token_ids_list)}")

        with open(out_pkl, 'wb') as f:
            pickle.dump({
                'category': cat, 'pair_idx': i, 'tokens': (t1, t2),
                'cosine': pair['cosine'], 'angles_deg': angles_deg,
                'l2': np.stack(all_l2), 'cosine_dist': np.stack(all_cos),
                'layer': LAYER, 'measure_layer': measure_layer,
                'model': MODEL_NAME,
            }, f)
        print(f"    saved: {out_pkl}")


def fit_all():
    with open(DIRECTIONS_PKL, 'rb') as f:
        dirs = pickle.load(f)

    rows = []
    lines = [f"Superellipse p fits — {MODEL_NAME} L{LAYER}, "
             f"J-Lens direction pairs",
             f"{'cat':>4s} {'idx':>3s}  {'tokens':<30s} {'thresh':>6s} "
             f"{'theta1':>7s} {'theta2':>7s} {'p':>6s} {'CI':>14s} "
             f"{'err':>7s} {'npts':>4s}"]

    for cat, i, pair in iter_pairs(dirs):
        pkl = os.path.join(SWEEP_DIR, f'sweep_{cat}_{i:02d}.pkl')
        if not os.path.exists(pkl):
            lines.append(f"{cat:>4s} {i:>3d}  (no sweep file)")
            continue
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
        angles = res['angles_deg']
        median_grid = np.median(res['l2'], axis=0)
        tok_label = f"{res['tokens'][0]!r}x{res['tokens'][1]!r}"

        for thresh in L2_THRESHOLDS:
            theta_1 = find_threshold(median_grid[:, 0], angles, thresh)
            theta_2 = find_threshold(median_grid[0, :], angles, thresh)
            if (theta_1 <= angles[1] or theta_2 <= angles[1]
                    or theta_1 >= angles[-1] or theta_2 >= angles[-1]):
                continue
            contour = extract_contour_points(angles, median_grid, thresh)
            if len(contour) == 0:
                continue
            contour_norm = contour.copy()
            contour_norm[:, 0] /= theta_1
            contour_norm[:, 1] /= theta_2
            p_fit, ci, info = fit_superellipse(
                contour_norm, (theta_1 + theta_2) / 2)
            if p_fit is None:
                continue
            rows.append({'category': cat, 'pair_idx': i,
                         'tokens': res['tokens'], 'cosine': res['cosine'],
                         'threshold': thresh, 'theta_1': theta_1,
                         'theta_2': theta_2, 'p': p_fit, 'ci': ci,
                         'mean_radial_deg': info['mean_radial_deg'],
                         'n_pts': info['n_pts']})
            lines.append(
                f"{cat:>4s} {i:>3d}  {tok_label:<30s} {thresh:>6g} "
                f"{theta_1:>6.2f}° {theta_2:>6.2f}° {p_fit:>6.2f} "
                f"[{ci[0]:5.2f},{ci[1]:5.2f}] "
                f"{info['mean_radial_deg']:>6.3f}° {info['n_pts']:>4d}")

    with open(SUMMARY_PKL, 'wb') as f:
        pickle.dump(rows, f)
    txt = '\n'.join(lines)
    with open(SUMMARY_TXT, 'w') as f:
        f.write(txt)
    print(txt)
    print(f"\nSaved: {SUMMARY_PKL}\n       {SUMMARY_TXT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    if not args.fit_only:
        run_sweeps()
    fit_all()


if __name__ == '__main__':
    main()
