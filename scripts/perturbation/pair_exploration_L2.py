#!/usr/bin/env python3
"""Autonomous pair-exploration pipeline at Gemma-2-9B L2.

Goal: find contrastive-direction pairs that (i) each individually break the
plateau early, (ii) produce good composite steering (low mutual cos sim),
(iii) have the lowest threshold-sweep variation in the superellipse fit p.

Pipeline phases (checkpointed — skips phases whose output pkl exists):
  A. Extract contrastive directions for a family of language / semantic /
     programming features. Save directions pkl.
  B. 1D spherical sweeps 0..60 deg for every direction plus random baseline.
     Save curves pkl.
  C. Filter to 'working' directions (breaking angle at L2=50 below
     0.6 * random pooled breaking). Form candidate pairs with
     |cos| < 0.5 prioritized by feature-type diversity. Save pair list.
  D. 2D iso-plateau sweep per selected pair, MAX_ANGLE chosen generously.
     Save one pkl per pair.

Threshold-sensitivity analysis + ranking is done locally after this script
completes; the pkls produced here are the raw inputs.
"""
import os, sys, pickle, json, itertools, argparse, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    compute_contrastive_direction, extract_activations_holdout)
from scripts.perturbation.sweep_isoplateau import sweep_2d

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2
MAX_ANGLE_1D = 60.0
N_STEPS_1D = 61
N_RANDOM_1D = 6

MAX_ANGLE_2D = 30.0
N_STEPS_2D = 41

THRESH_L2 = 50.0

OUT_DIR = 'results/pair_exploration_L2'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
CURVES_PKL = os.path.join(OUT_DIR, 'sweep_1d.pkl')
PAIR_JSON = os.path.join(OUT_DIR, 'pair_list.json')

# Feature-type labels used to prioritise pair diversity
FEATURE_TYPE = {
    'Portuguese': 'lang', 'Chinese': 'lang', 'Spanish': 'lang',
    'French': 'lang', 'German': 'lang', 'Dutch': 'lang',
    'Italian': 'lang', 'Japanese': 'lang', 'Russian': 'lang',
    'Arabic': 'lang',
    'Gender': 'semantic',
    'Python': 'code',
}

LANG_SUBSETS = {
    'Portuguese': 'pt', 'Chinese': 'zh', 'Spanish': 'es', 'French': 'fr',
    'German': 'de', 'Dutch': 'nl', 'Italian': 'it', 'Japanese': 'ja',
    'Russian': 'ru', 'Arabic': 'ar',
}


def ts():
    return time.strftime('%H:%M:%S')


def safe_pkl_dump(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Phase A: directions
# -----------------------------------------------------------------------------

def extract_directions(model, tokenizer, device, reuse):
    dirs = dict(reuse) if reuse else {}
    from data.multilingual_prompts import PROMPT_SETS as ML_SETS
    en_prompts = [ps['en'] for ps in ML_SETS]

    for lang_name, lang_key in LANG_SUBSETS.items():
        if lang_name in dirs:
            print(f"  [dir] skip {lang_name} (already present)", flush=True)
            continue
        other = [ps[lang_key] for ps in ML_SETS]
        print(f"  [dir] {lang_name} (EN -> {lang_name}) ...", flush=True)
        d = compute_contrastive_direction(
            model, tokenizer, device, LAYER, en_prompts, other)
        dirs[lang_name] = d

    if 'Gender' not in dirs:
        from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
        print("  [dir] Gender (Male -> Female) ...", flush=True)
        dirs['Gender'] = compute_contrastive_direction(
            model, tokenizer, device, LAYER, MALE_PROMPTS, FEMALE_PROMPTS)

    if 'Python' not in dirs:
        from data.programming_prompts import PYTHON_PROMPTS
        en_for_py = en_prompts[:len(PYTHON_PROMPTS)]
        print("  [dir] Python (EN prose -> Python code) ...", flush=True)
        dirs['Python'] = compute_contrastive_direction(
            model, tokenizer, device, LAYER, en_for_py, PYTHON_PROMPTS)

    out = {'directions': dirs, 'layer': LAYER}
    safe_pkl_dump(DIR_PKL, out)
    print(f"  [dir] saved {DIR_PKL}", flush=True)
    return dirs


def load_directions_maybe_reuse_existing():
    # Reuse directions from paper_L2_gemma2_9b if available (PT, Gender, Python).
    reuse = {}
    old_path = 'results/paper_L2_gemma2_9b/directions_L2.pkl'
    if os.path.exists(old_path):
        with open(old_path, 'rb') as f:
            d = pickle.load(f)
        mapping = {'d_portuguese': 'Portuguese', 'd_gender': 'Gender',
                   'd_python': 'Python'}
        for k, name in mapping.items():
            if k in d:
                reuse[name] = d[k]
        print(f"  [dir] reusing {len(reuse)} from {old_path}: "
              f"{sorted(reuse.keys())}", flush=True)
    return reuse


# -----------------------------------------------------------------------------
# Phase B: 1D sweeps
# -----------------------------------------------------------------------------

def sweep_1d_spherical(model, ctx, act, direction, layer, measure_layer,
                       device, angles_rad, batch_size=64):
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


def run_1d_sweeps(model, tokenizer, device, dirs):
    if os.path.exists(CURVES_PKL):
        with open(CURVES_PKL, 'rb') as f:
            stored = pickle.load(f)
        curves = stored.get('curves', {})
        ctx = stored.get('contexts')
        acts = stored.get('activations')
        angles = stored['angles_deg']
        measure_layer = stored['measure_layer']
        print(f"  [1d] reuse cached with {len(curves)} curves", flush=True)
    else:
        curves = {}
        ctx, acts = None, None
        angles = np.linspace(0, MAX_ANGLE_1D, N_STEPS_1D)
        n_layers = len(_get_blocks(model))
        measure_layer = n_layers - 2

    angles_rad = np.deg2rad(angles)

    if ctx is None or acts is None:
        from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
        holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
        print(f"  [1d] extracting {N_ACT} holdout activations", flush=True)
        ctx, acts = extract_activations_holdout(
            model, tokenizer, device, LAYER, holdout)

    d_model = acts.shape[1]

    # random directions baseline
    rng = np.random.RandomState(SEED)
    rand_dirs = F.normalize(
        torch.tensor(rng.randn(N_RANDOM_1D, d_model), dtype=torch.float32), dim=-1)
    to_run = [(f'Random_{k}', rand_dirs[k]) for k in range(N_RANDOM_1D)
              if f'Random_{k}' not in curves]
    for name, d in dirs.items():
        if name not in curves:
            to_run.append((name, d))

    if not to_run:
        print("  [1d] nothing new to sweep", flush=True)
    for name, d in to_run:
        print(f"  [1d] {name} [{ts()}]", flush=True)
        mat = np.zeros((N_ACT, len(angles)))
        for i in range(N_ACT):
            mat[i] = sweep_1d_spherical(
                model, ctx[i], acts[i], d, LAYER, measure_layer, device,
                angles_rad)
        curves[name] = mat
        med60 = float(np.median(mat[:, -1]))
        print(f"       median@60° L2={med60:.1f}", flush=True)
        # checkpoint after every direction
        safe_pkl_dump(CURVES_PKL, {
            'angles_deg': angles,
            'layer': LAYER,
            'measure_layer': measure_layer,
            'curves': curves,
            'contexts': ctx,
            'activations': acts,
            'n_activations': N_ACT,
        })

    return curves, angles, ctx, acts, measure_layer


# -----------------------------------------------------------------------------
# Phase C: pair selection
# -----------------------------------------------------------------------------

def breaking_angle(curves_mat, angles_deg, thr):
    med = np.median(curves_mat, axis=0)
    above = np.where(med >= thr)[0]
    if len(above) == 0:
        return None
    k = above[0]
    if k == 0: return float(angles_deg[0])
    v0, v1 = med[k-1], med[k]
    return float(angles_deg[k-1] + (thr - v0) / (v1 - v0) * (angles_deg[k] - angles_deg[k-1]))


def select_pairs(curves, angles, dirs, out_path):
    # Random baseline break
    rnd_stack = np.concatenate(
        [curves[k] for k in curves if k.startswith('Random_')], axis=0)
    rnd_break = breaking_angle(rnd_stack, angles, THRESH_L2)
    print(f"  [select] random break @ L2={THRESH_L2}: {rnd_break:.2f}°",
          flush=True)

    breaks = {}
    for name in dirs:
        b = breaking_angle(curves[name], angles, THRESH_L2)
        breaks[name] = b
        ratio = b / rnd_break if b else None
        print(f"  [select] {name:12s} break={b:.2f}°  "
              f"ratio-to-random={ratio:.2f}", flush=True)

    valid = [n for n, b in breaks.items()
             if b is not None and b < 0.6 * rnd_break]
    print(f"  [select] valid directions (break < 0.6 * random): {valid}",
          flush=True)

    # Pair enumeration with |cos| < 0.5, rank by feature-type diversity
    def diversity_rank(a, b):
        ta, tb = FEATURE_TYPE.get(a, '?'), FEATURE_TYPE.get(b, '?')
        if ta != tb: return 0          # cross-type pairs rank highest
        if ta == 'lang': return 1       # same-type language pairs second
        return 2

    pair_infos = []
    for a, b in itertools.combinations(sorted(valid), 2):
        da = F.normalize(dirs[a].float(), dim=0)
        db = F.normalize(dirs[b].float(), dim=0)
        c = float((da @ db).item())
        if abs(c) >= 0.5:
            continue
        pair_infos.append({
            'pair': [a, b],
            'cos': c,
            'break_a': breaks[a],
            'break_b': breaks[b],
            'rank_key': (diversity_rank(a, b), abs(c), max(breaks[a], breaks[b])),
        })

    pair_infos.sort(key=lambda r: r['rank_key'])
    for p in pair_infos:
        p.pop('rank_key')
    print(f"  [select] {len(pair_infos)} pairs pass filter:", flush=True)
    for p in pair_infos:
        print(f"     {p['pair'][0]:12s} × {p['pair'][1]:12s}  "
              f"cos={p['cos']:+.3f}  "
              f"breaks=({p['break_a']:.2f}°, {p['break_b']:.2f}°)",
              flush=True)

    with open(out_path, 'w') as f:
        json.dump({'random_break': rnd_break, 'breaks': breaks,
                   'pairs': pair_infos}, f, indent=2)
    return pair_infos


# -----------------------------------------------------------------------------
# Phase D: 2D sweeps
# -----------------------------------------------------------------------------

def run_2d_sweeps(model, ctx, acts, measure_layer, device, dirs, pair_infos,
                  max_pairs):
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    print(f"  [2d] grid {N_STEPS_2D}x{N_STEPS_2D}, max_angle={MAX_ANGLE_2D}°, "
          f"{N_ACT} acts, {min(max_pairs, len(pair_infos))} pairs", flush=True)

    for p_info in pair_infos[:max_pairs]:
        a, b = p_info['pair']
        pair_key = f'{a}_{b}'.replace(' ', '_')
        pkl_path = os.path.join(OUT_DIR, f'sweep_2d_{pair_key}.pkl')
        if os.path.exists(pkl_path):
            print(f"  [2d] skip {pair_key} (already exists)", flush=True)
            continue
        print(f"  [2d] {pair_key}  [{ts()}]", flush=True)

        all_l2, all_cos = [], []
        for i in range(N_ACT):
            l2, cos = sweep_2d(
                model, ctx[i], acts[i], dirs[a], dirs[b],
                LAYER, measure_layer, device, angles_rad, angles_rad)
            all_l2.append(l2)
            all_cos.append(cos)
            if (i + 1) % 5 == 0:
                print(f"       {i+1}/{N_ACT}", flush=True)

        out = {
            'angles_deg': angles,
            'l2': np.stack(all_l2),
            'cosine': np.stack(all_cos),
            'layer': LAYER,
            'measure_layer': measure_layer,
            'direction_labels': (a, b),
            'pair_info': p_info,
        }
        safe_pkl_dump(pkl_path, out)
        print(f"       saved {pkl_path}", flush=True)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_pairs', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"[{ts()}] Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    # --- Phase A ---
    if os.path.exists(DIR_PKL):
        with open(DIR_PKL, 'rb') as f:
            dirs = pickle.load(f)['directions']
        print(f"[{ts()}] loaded {len(dirs)} directions from cache: "
              f"{sorted(dirs.keys())}", flush=True)
    else:
        print(f"[{ts()}] extracting directions", flush=True)
        reuse = load_directions_maybe_reuse_existing()
        dirs = extract_directions(model, tokenizer, device, reuse)

    # --- Phase B ---
    print(f"[{ts()}] 1D sweeps", flush=True)
    curves, angles, ctx, acts, measure_layer = run_1d_sweeps(
        model, tokenizer, device, dirs)

    # --- Phase C ---
    print(f"[{ts()}] pair selection", flush=True)
    pair_infos = select_pairs(curves, angles, dirs, PAIR_JSON)

    # --- Phase D ---
    print(f"[{ts()}] 2D sweeps", flush=True)
    run_2d_sweeps(model, ctx, acts, measure_layer, device, dirs, pair_infos,
                  args.max_pairs)

    print(f"[{ts()}] DONE", flush=True)


if __name__ == '__main__':
    main()
