#!/usr/bin/env python3
"""Wide-angle 2D iso-plateau sweep for Portuguese x Gender at Gemma-2-9B L2.

Differences from sweep_isoplateau_L2.py:
  - MAX_ANGLE bumped 25° -> 60° to capture high-threshold contours that
    previously hit the grid ceiling.
  - N_STEPS bumped 41 -> 61 to keep ~1° angular resolution.
  - Same-batch reference: the (0, 0) (unperturbed) point is included as the
    first row of every batched forward pass and used as the reference, instead
    of running a separate forward pass on a_d alone. This eliminates batch
    nondeterminism and makes the bf16 (0,0) -> measure-layer L2 distance
    bit-zero by construction, dropping the noise floor.

Output: results/paper_L2_gemma2_9b/sweep_isoplateau_L2_wide.pkl
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import extract_activations_holdout

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2
MAX_ANGLE = 60.0
N_STEPS = 61

OUT_DIR = 'results/paper_L2_gemma2_9b'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_PKL = os.path.join(OUT_DIR, 'sweep_isoplateau_L2_wide.pkl')


def sweep_2d_samebatch(model, ctx, act, d1, d2, layer, measure_layer, device,
                       angles1_rad, angles2_rad, batch_size=64):
    """Same as sweep_isoplateau.sweep_2d but with same-batch reference.

    The unperturbed (0,0) point is included as the first row of every batched
    forward pass, and the L2 distance / cosine distance is taken against that
    in-batch reference rather than a separately forwarded a_d. Eliminates
    bf16 batch-reduction nondeterminism between the reference path and the
    perturbed path.
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R

    d1_d = d1.to(device=device, dtype=torch.float32)
    d1_orth = d1_d - (d1_d @ a_unit) * a_unit
    d1_orth = F.normalize(d1_orth, dim=0)

    d2_d = d2.to(device=device, dtype=torch.float32)
    d2_orth = d2_d - (d2_d @ a_unit) * a_unit - (d2_d @ d1_orth) * d1_orth
    d2_orth = F.normalize(d2_orth, dim=0)

    n1, n2 = len(angles1_rad), len(angles2_rad)
    s1 = torch.tensor(np.sin(angles1_rad), dtype=torch.float32, device=device)
    s2 = torch.tensor(np.sin(angles2_rad), dtype=torch.float32, device=device)
    S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))

    all_points = (R * (C.reshape(-1, 1) * a_unit.unsqueeze(0)
                       + S1.reshape(-1, 1) * d1_orth.unsqueeze(0)
                       + S2.reshape(-1, 1) * d2_orth.unsqueeze(0))).to(model_dtype)

    n_points = n1 * n2
    l2_flat = np.zeros(n_points)
    cos_flat = np.zeros(n_points)

    # Build the (0,0) reference identically to how all_points is built, so it
    # is in the same numerical space.
    ref_point = (R * a_unit).to(model_dtype).view(1, 1, -1)  # [1, 1, d_model]

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch = all_points[start:end].unsqueeze(1)            # [B, 1, d]
            batch_with_ref = torch.cat([ref_point, batch], dim=0)  # [B+1, 1, d]
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch_with_ref, layer, measure_layer).float()
            ref_out = pf[0:1]   # [1, d]
            pf_pert = pf[1:]    # [B, d]
            l2_flat[start:end] = (pf_pert - ref_out).norm(dim=-1).cpu().numpy()
            cos_flat[start:end] = (1.0 - F.cosine_similarity(
                pf_pert, ref_out, dim=-1)).cpu().numpy()

    return l2_flat.reshape(n1, n2), cos_flat.reshape(n1, n2)


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
    print(f"  MAX_ANGLE={MAX_ANGLE}°, step={angles_deg[1]-angles_deg[0]:.3f}°",
          flush=True)

    all_l2, all_cos = [], []
    for i in range(N_ACT):
        l2, cos = sweep_2d_samebatch(
            model, contexts[i], activations[i],
            d_pt, d_gender, LAYER, measure_layer, device,
            angles_rad, angles_rad)
        all_l2.append(l2)
        all_cos.append(cos)
        # Sanity: noise floor is the (0,0) entry; should be exactly 0 now
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
        'config': {'max_angle': MAX_ANGLE, 'n_steps': N_STEPS, 'n_act': N_ACT,
                   'same_batch_reference': True, 'dtype': 'bfloat16'},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {OUT_PKL}", flush=True)


if __name__ == '__main__':
    main()
