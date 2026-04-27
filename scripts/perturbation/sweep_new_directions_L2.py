#!/usr/bin/env python3
"""1D plateau-shape check at Gemma-2-9B L2 for new candidate binary directions.

Mirrors phases A+B of pair_exploration_L2_v2 but only for directions we haven't
tried yet (Honesty, Refusal, Tense), plus reuses Gender as the reference and a
few code directions for visual comparison. Saves curves to a fresh pkl and
produces a comparison plot.

Output:
  results/new_directions_L2/directions_L2.pkl
  results/new_directions_L2/sweep_1d.pkl
  results/new_directions_L2/curves_comparison.png
"""
import os, sys, pickle, time, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    compute_contrastive_direction, extract_activations_holdout)

# --- Config ---
N_ACT = 30
SEED = 123
LAYER = 2
MAX_ANGLE = 60.0
N_STEPS = 61
N_RANDOM = 6

OUT_DIR = 'results/new_directions_L2'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
CURVES_PKL = os.path.join(OUT_DIR, 'sweep_1d.pkl')
PLOT_PNG = os.path.join(OUT_DIR, 'curves_comparison.png')


def ts():
    return time.strftime('%H:%M:%S')


def safe_pkl_dump(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Direction extraction
# -----------------------------------------------------------------------------

def load_reuse_existing():
    """Pull whatever's already cached from pair_exploration_L2_v2."""
    reuse = {}
    path = 'results/pair_exploration_L2_v2/directions_L2.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for k, v in d.get('directions', {}).items():
            reuse[k] = v
    return reuse


def extract_directions(model, tokenizer, device):
    if os.path.exists(DIR_PKL):
        with open(DIR_PKL, 'rb') as f:
            cached = pickle.load(f)['directions']
        print(f"[{ts()}] [dir] cached: {sorted(cached.keys())}", flush=True)
    else:
        cached = {}

    # Reuse existing Gender + a couple code directions for the comparison plot
    reuse = load_reuse_existing()
    for name in ('Gender', 'Python', 'Haskell', 'Go'):
        if name in reuse and name not in cached:
            cached[name] = reuse[name]
            print(f"[{ts()}] [dir] reused {name}", flush=True)

    def do(name, a_prompts, b_prompts, tag):
        if name in cached:
            return
        print(f"[{ts()}]  [dir] {name} ({tag})", flush=True)
        n = min(len(a_prompts), len(b_prompts))
        cached[name] = compute_contrastive_direction(
            model, tokenizer, device, LAYER, a_prompts[:n], b_prompts[:n])

    # Honesty (long version, round 1)
    from data.honesty_prompts import HONEST_PROMPTS, DISHONEST_PROMPTS
    do('Honesty', DISHONEST_PROMPTS, HONEST_PROMPTS, 'Dishonest→Honest')

    # Refusal (round 1)
    from data.refusal_prompts import COMPLY_PROMPTS, REFUSE_PROMPTS
    do('Refusal', COMPLY_PROMPTS, REFUSE_PROMPTS, 'Comply→Refuse')

    # Tense (past↔future, round 1) — kept under name 'Tense' for back-compat
    from data.tense_prompts import PAST_PROMPTS, FUTURE_PROMPTS
    do('Tense', PAST_PROMPTS, FUTURE_PROMPTS, 'Past→Future')

    # Round 2: gender-mechanism candidates
    # Person (1st↔3rd) — pronoun routing
    from data.person_prompts import FIRST_PROMPTS, THIRD_PROMPTS
    do('Person', FIRST_PROMPTS, THIRD_PROMPTS, '1st→3rd')

    # Past↔Present (morphological, no "will")
    from data.tense_present_prompts import PRESENT_PROMPTS, PAST_PROMPTS as PAST2
    do('TensePP', PRESENT_PROMPTS, PAST2, 'Present→Past')

    # Era (medieval-fantasy↔modern) — vocabulary domain swap
    from data.era_prompts import MEDIEVAL_PROMPTS, MODERN_PROMPTS
    do('Era', MODERN_PROMPTS, MEDIEVAL_PROMPTS, 'Modern→Medieval')

    # Honesty short-form — methodological check
    from data.honesty_short_prompts import (
        HONEST_SHORT_PROMPTS, DISHONEST_SHORT_PROMPTS)
    do('HonestyShort', DISHONEST_SHORT_PROMPTS, HONEST_SHORT_PROMPTS,
       'Dishonest→Honest (short)')

    # Round 3: person-attribute axes that should compose with Gender
    from data.age_prompts import YOUNG_PROMPTS, OLD_PROMPTS
    do('Age', YOUNG_PROMPTS, OLD_PROMPTS, 'Young→Old')

    from data.number_prompts import SINGULAR_PROMPTS, PLURAL_PROMPTS
    do('Number', SINGULAR_PROMPTS, PLURAL_PROMPTS, 'Singular→Plural')

    from data.status_prompts import ROYAL_PROMPTS, COMMONER_PROMPTS
    do('Status', COMMONER_PROMPTS, ROYAL_PROMPTS, 'Commoner→Royal')

    safe_pkl_dump(DIR_PKL, {'directions': cached, 'layer': LAYER})
    print(f"[{ts()}] [dir] saved {DIR_PKL}  ({len(cached)} total)", flush=True)
    return cached


# -----------------------------------------------------------------------------
# 1D sweep (copied from pair_exploration_L2_v2.sweep_1d_spherical)
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
        curves = stored['curves']
        ctx = stored['contexts']
        acts = stored['activations']
        angles = stored['angles_deg']
        measure_layer = stored['measure_layer']
    else:
        curves = {}
        ctx, acts = None, None
        angles = np.linspace(0, MAX_ANGLE, N_STEPS)
        n_layers = len(_get_blocks(model))
        measure_layer = n_layers - 2

    angles_rad = np.deg2rad(angles)

    if ctx is None or acts is None:
        from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
        holdout = MALE_PROMPTS_HOLDOUT[:N_ACT]
        print(f"[{ts()}] [1d] extracting {N_ACT} holdout activations", flush=True)
        ctx, acts = extract_activations_holdout(
            model, tokenizer, device, LAYER, holdout)

    d_model = acts.shape[1]
    rng = np.random.RandomState(SEED)
    rand_dirs = F.normalize(
        torch.tensor(rng.randn(N_RANDOM, d_model), dtype=torch.float32), dim=-1)

    to_run = []
    for k in range(N_RANDOM):
        if f'Random_{k}' not in curves:
            to_run.append((f'Random_{k}', rand_dirs[k]))
    for name, d in dirs.items():
        if name not in curves:
            to_run.append((name, d))

    for name, d in to_run:
        print(f"[{ts()}] [1d] {name}", flush=True)
        mat = np.zeros((N_ACT, len(angles)))
        for i in range(N_ACT):
            mat[i] = sweep_1d_spherical(
                model, ctx[i], acts[i], d, LAYER, measure_layer, device,
                angles_rad)
        curves[name] = mat
        med60 = float(np.median(mat[:, -1]))
        print(f"       median@60° L2={med60:.1f}", flush=True)
        safe_pkl_dump(CURVES_PKL, {
            'angles_deg': angles, 'layer': LAYER,
            'measure_layer': measure_layer, 'curves': curves,
            'contexts': ctx, 'activations': acts, 'n_activations': N_ACT,
        })
    return curves, angles


# -----------------------------------------------------------------------------
# Comparison plot
# -----------------------------------------------------------------------------

def plot_comparison(curves, angles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    randoms = []
    for k in curves:
        if k.startswith('Random_'):
            randoms.append(np.median(curves[k], axis=0))
    rand_band = np.percentile(np.stack(randoms), [25, 50, 75], axis=0)

    round1_dirs = ['Honesty', 'Refusal', 'Tense']
    round2_dirs = ['Person', 'TensePP', 'Era', 'HonestyShort']
    round3_dirs = ['Age', 'Number', 'Status']
    ref_dirs = ['Gender', 'Python', 'Haskell', 'Go']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (title, names) in zip(axes,
                                  [('round 3 (person-attribute, composes with Gender)', round3_dirs),
                                   ('reference', ref_dirs),
                                   ('round 2 (gender-mechanism candidates)', round2_dirs),
                                   ('round 1 (semantic axes)', round1_dirs)]):
        ax.fill_between(angles, rand_band[0], rand_band[2],
                        color='gray', alpha=0.25, label='random IQR')
        ax.plot(angles, rand_band[1], color='k', ls=':', lw=1, label='random med')
        for n in names:
            if n in curves:
                med = np.median(curves[n], axis=0)
                ax.plot(angles, med, lw=2, label=n)
        ax.set_title(title)
        ax.set_xlim(0, 50); ax.set_ylim(0, 400)
        ax.set_xlabel('angle (deg)')
        ax.set_ylabel('L2 response')
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'[{ts()}] saved {PLOT_PNG}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"[{ts()}] Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    dirs = extract_directions(model, tokenizer, device)
    curves, angles = run_1d_sweeps(model, tokenizer, device, dirs)

    if not args.skip_plot:
        plot_comparison(curves, angles)

    # Quick numerical summary
    print(f"\n[{ts()}] === plateau-shape diagnostic ===")
    randoms = []
    for k in curves:
        if k.startswith('Random_'):
            randoms.append(np.median(curves[k], axis=0))
    rand_med = np.median(np.stack(randoms), axis=0)
    def cross(curve, val):
        for i in range(1, len(curve)):
            if curve[i] >= val:
                x0, x1 = angles[i-1], angles[i]
                y0, y1 = curve[i-1], curve[i]
                return float(x0 + (val - y0)/(y1-y0)*(x1-x0))
        return float('nan')
    print(f'{"direction":12s}  {"@L2=50":>7s}  {"@L2=200":>7s}  {"gap":>5s}')
    for k in sorted(curves.keys()):
        if k.startswith('Random_'): continue
        med = np.median(curves[k], axis=0)
        a50 = cross(med, 50); a200 = cross(med, 200)
        gap = a200 - a50 if not (np.isnan(a50) or np.isnan(a200)) else float('nan')
        print(f'{k:12s}  {a50:7.1f}  {a200:7.1f}  {gap:5.1f}')
    a50r = cross(rand_med, 50); a200r = cross(rand_med, 200)
    gapr = a200r - a50r if not (np.isnan(a50r) or np.isnan(a200r)) else float('nan')
    print(f'{"random":12s}  {a50r:7.1f}  {a200r:7.1f}  {gapr:5.1f}')


if __name__ == '__main__':
    main()
