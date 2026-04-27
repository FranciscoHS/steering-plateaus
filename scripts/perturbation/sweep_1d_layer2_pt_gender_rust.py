#!/usr/bin/env python3
"""1D spherical perturbation sweeps at layer 2 for PT, Gender, Rust + random.

Gemma-2-9B, perturb@L2, measure@penultimate. Each direction swept 0..60 deg.
Analogue of Figure 2 but with an early-layer perturbation to check whether
the plateau-then-break shape is more dramatic when there's more propagation
distance before the measurement.
"""
import os, sys, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    compute_contrastive_direction,
    extract_activations_holdout,
)

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2
MAX_ANGLE = 60.0
N_STEPS = 61   # 1-deg resolution
N_RANDOM = 6   # number of random baseline directions

OUT_DIR = 'results/paper_L2_gemma2_9b'
os.makedirs(OUT_DIR, exist_ok=True)
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_PKL = os.path.join(OUT_DIR, 'sweep_1d_L2.pkl')


def sweep_1d_spherical(model, ctx, act, direction, layer, measure_layer,
                       device, angles_rad, batch_size=64):
    """Spherical rotate of `act` toward `direction` on the sphere, measure L2 at
    measure_layer of the downstream residual stream.
    """
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R
    d_d = direction.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos().unsqueeze(1)
    sin_t = thetas.sin().unsqueeze(1)
    all_points = (R * (cos_t * a_unit.unsqueeze(0)
                       + sin_t * d_orth.unsqueeze(0))).to(model_dtype)

    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype),
            layer, measure_layer).float()

    n = len(angles_rad)
    l2 = np.zeros(n)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch, layer, measure_layer).float()
            l2[start:end] = (pf - ref_act).norm(dim=-1).cpu().numpy()
    return l2


def extract_directions_L2(model, tokenizer, device):
    print("[dir] Portuguese (EN→PT)...", flush=True)
    from data.multilingual_prompts import PROMPT_SETS as ML_SETS
    en = [ps['en'] for ps in ML_SETS]
    pt = [ps['pt'] for ps in ML_SETS]
    d_pt = compute_contrastive_direction(model, tokenizer, device, LAYER, en, pt)

    print("[dir] Gender (Male→Female)...", flush=True)
    from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
    d_gender = compute_contrastive_direction(
        model, tokenizer, device, LAYER, MALE_PROMPTS, FEMALE_PROMPTS)

    print("[dir] Rust (Python→Rust)...", flush=True)
    from data.programming_prompts import PROMPT_SETS as PROG_SETS
    py = [ps['python'] for ps in PROG_SETS]
    ru = [ps['rust'] for ps in PROG_SETS]
    d_rust = compute_contrastive_direction(
        model, tokenizer, device, LAYER, py, ru)

    # Python direction via EN prose -> Python code DoM.
    # Prompt sets are different lengths (EN=36, PY=30); trim to the smaller.
    print("[dir] Python (EN prose → Python code)...", flush=True)
    from data.programming_prompts import PYTHON_PROMPTS
    en_for_py = en[:len(PYTHON_PROMPTS)]
    d_python = compute_contrastive_direction(
        model, tokenizer, device, LAYER, en_for_py, PYTHON_PROMPTS)

    print("[dir] Sentiment (Positive → Negative)...", flush=True)
    from data.sentiment_prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS
    d_sentiment = compute_contrastive_direction(
        model, tokenizer, device, LAYER, POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

    out = {'d_portuguese': d_pt, 'd_gender': d_gender, 'd_rust': d_rust,
           'd_python': d_python, 'd_sentiment': d_sentiment, 'layer': LAYER}
    with open(DIR_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f"[dir] Saved {DIR_PKL}", flush=True)
    return out


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers}, perturb@L{LAYER}, measure@L{measure_layer}",
          flush=True)

    required_keys = {'d_portuguese', 'd_gender', 'd_rust', 'd_python', 'd_sentiment'}
    if os.path.exists(DIR_PKL):
        with open(DIR_PKL, 'rb') as f:
            dirs = pickle.load(f)
        if required_keys.issubset(dirs.keys()):
            print(f"[dir] Reusing {DIR_PKL}", flush=True)
        else:
            print(f"[dir] {DIR_PKL} missing keys, re-extracting", flush=True)
            dirs = extract_directions_L2(model, tokenizer, device)
    else:
        dirs = extract_directions_L2(model, tokenizer, device)

    # Holdout activations: male holdout (same as the flagship PT × Gender run
    # so the random baseline applies equivalently)
    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
    print(f"Extracting {len(holdout)} holdout activations at L{LAYER}...",
          flush=True)
    contexts, activations = extract_activations_holdout(
        model, tokenizer, device, LAYER, holdout)

    # Build random directions (unit vectors in the residual dim)
    d_model = activations.shape[1]
    rng = np.random.RandomState(SEED)
    rand_dirs = torch.tensor(
        rng.randn(N_RANDOM, d_model), dtype=torch.float32)
    rand_dirs = F.normalize(rand_dirs, dim=-1)

    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)

    directions = [
        ('Portuguese', dirs['d_portuguese']),
        ('Gender',     dirs['d_gender']),
        ('Rust',       dirs['d_rust']),
        ('Python',     dirs['d_python']),
        ('Sentiment',  dirs['d_sentiment']),
    ]
    for k in range(N_RANDOM):
        directions.append((f'Random_{k}', rand_dirs[k]))

    curves = {}   # name -> [N_ACT, N_STEPS]
    for name, d in directions:
        print(f"[sweep] {name}", flush=True)
        mat = np.zeros((N_ACT, N_STEPS))
        for i in range(N_ACT):
            mat[i] = sweep_1d_spherical(
                model, contexts[i], activations[i], d,
                LAYER, measure_layer, device, angles_rad)
        curves[name] = mat
        print(f"         median at 60°: L2 = {np.median(mat[:, -1]):.1f}",
              flush=True)

    out = {
        'angles_deg': angles_deg,
        'layer': LAYER,
        'measure_layer': measure_layer,
        'curves': curves,
        'n_activations': N_ACT,
        'n_random': N_RANDOM,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved: {OUT_PKL}", flush=True)


if __name__ == '__main__':
    main()
