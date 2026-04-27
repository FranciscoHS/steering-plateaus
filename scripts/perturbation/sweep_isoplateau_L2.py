#!/usr/bin/env python3
"""2D iso-plateau sweep for Portuguese x Gender at Gemma-2-9B layer 2.

Reuses sweep_2d from sweep_isoplateau.py; only the layer and direction source
change (directions extracted at L2, saved in paper_L2_gemma2_9b/directions_L2.pkl).
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.perturbation.generate_lesswrong_figures import extract_activations_holdout
from scripts.perturbation.sweep_isoplateau import sweep_2d

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2
MAX_ANGLE = 25.0
N_STEPS = 41

OUT_DIR = 'results/paper_L2_gemma2_9b'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_PKL = os.path.join(OUT_DIR, 'sweep_isoplateau_L2.pkl')


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

    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)
    print(f"Grid: {N_STEPS}x{N_STEPS} = {N_STEPS**2} points x {N_ACT} acts "
          f"= {N_STEPS**2 * N_ACT} forward passes", flush=True)

    all_l2, all_cos = [], []
    for i in range(N_ACT):
        l2, cos = sweep_2d(
            model, contexts[i], activations[i],
            d_pt, d_gender, LAYER, measure_layer, device,
            angles_rad, angles_rad)
        all_l2.append(l2)
        all_cos.append(cos)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_ACT}", flush=True)

    results = {
        'angles_deg': angles_deg,
        'l2': np.stack(all_l2),
        'cosine': np.stack(all_cos),
        'layer': LAYER,
        'measure_layer': measure_layer,
        'direction_labels': ('Portuguese', 'Gender'),
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUT_PKL}", flush=True)


if __name__ == '__main__':
    main()
