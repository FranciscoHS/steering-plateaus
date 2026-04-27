#!/usr/bin/env python3
"""Expanded pair-exploration pipeline at Gemma-2-9B L2 (v2).

Adds many direction families beyond languages: gender, sentiment, register
(formal/literary), genre (children/news), and programming languages (python,
js, java, rust, haskell, c++, go, lisp, typescript). Runs 2D iso-plateau
sweeps on every valid pair (|cos| < 0.7, both breaking-ratios < 0.8 vs
random), then composition-generation verification on the top threshold-stable
candidates.

Designed to run unattended end-to-end: single launch, checkpoints between
phases, minimal per-file I/O.
"""
import os, sys, pickle, json, itertools, argparse, time, glob
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from scripts.perturbation.generate_lesswrong_figures import (
    compute_contrastive_direction, extract_activations_holdout)
from scripts.perturbation.sweep_isoplateau import sweep_2d
from scripts.perturbation.generate_composition_L2 import (
    make_steering_hook, generate)

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
BREAK_RATIO_CUTOFF = 0.8          # more permissive than v1 (was 0.6)
COS_CUTOFF = 0.7                  # more permissive than v1 (was 0.5)

N_COMPOSITION_TOP = 8             # run composition gens on this many
N_COMP_SAMPLES = 3                # per condition per pair
COMP_PROMPT = "They elected a new chairman"
COMP_ANGLE_DEG = 20.0

OUT_DIR = 'results/pair_exploration_L2_v2'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
CURVES_PKL = os.path.join(OUT_DIR, 'sweep_1d.pkl')
PAIR_JSON = os.path.join(OUT_DIR, 'pair_list.json')
SUMMARY_JSON = os.path.join(OUT_DIR, 'summary.json')

FEATURE_TYPE = {}
LANG_SUBSETS = {
    'Portuguese': 'pt', 'Chinese': 'zh', 'Spanish': 'es', 'French': 'fr',
    'German': 'de', 'Dutch': 'nl', 'Italian': 'it', 'Japanese': 'ja',
    'Russian': 'ru', 'Arabic': 'ar',
}
for n in LANG_SUBSETS: FEATURE_TYPE[n] = 'lang'

CODE_SUBSETS = {
    'Python': 'python', 'JavaScript': 'javascript', 'Java': 'java',
    'Rust': 'rust', 'Haskell': 'haskell', 'Cpp': 'cpp',
    'Go': 'go', 'Lisp': 'lisp', 'TypeScript': 'typescript',
}
for n in CODE_SUBSETS: FEATURE_TYPE[n] = 'code'

SEMANTIC_DIRS = ['Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News']
for n in SEMANTIC_DIRS: FEATURE_TYPE[n] = 'semantic'


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

def load_reuse_existing():
    """Pull whatever's already cached from older runs."""
    reuse = {}
    paths = [
        ('results/pair_exploration_L2/directions_L2.pkl', 'directions'),
        ('results/paper_L2_gemma2_9b/directions_L2.pkl', None),
    ]
    for path, key in paths:
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if key and key in d:
            for k, v in d[key].items():
                if k not in reuse: reuse[k] = v
        else:
            # older format: d_portuguese style
            mapping = {'d_portuguese': 'Portuguese', 'd_gender': 'Gender',
                       'd_python': 'Python'}
            for k, name in mapping.items():
                if k in d and name not in reuse:
                    reuse[name] = d[k]
    return reuse


def extract_directions(model, tokenizer, device):
    if os.path.exists(DIR_PKL):
        with open(DIR_PKL, 'rb') as f:
            cached = pickle.load(f)['directions']
        print(f"[{ts()}] [dir] cached: {sorted(cached.keys())}", flush=True)
    else:
        cached = load_reuse_existing()
        print(f"[{ts()}] [dir] reuse {len(cached)}: {sorted(cached.keys())}",
              flush=True)

    from data.multilingual_prompts import PROMPT_SETS as ML_SETS
    en_prompts = [ps['en'] for ps in ML_SETS]

    def do(name, a_prompts, b_prompts, tag):
        if name in cached: return
        print(f"[{ts()}]  [dir] {name} ({tag})", flush=True)
        n = min(len(a_prompts), len(b_prompts))
        cached[name] = compute_contrastive_direction(
            model, tokenizer, device, LAYER, a_prompts[:n], b_prompts[:n])

    # Languages
    for lang_name, lang_key in LANG_SUBSETS.items():
        other = [ps[lang_key] for ps in ML_SETS]
        do(lang_name, en_prompts, other, f'EN→{lang_name}')

    # Gender
    from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
    do('Gender', MALE_PROMPTS, FEMALE_PROMPTS, 'Male→Female')

    # Code
    from data.programming_prompts import PROMPT_SETS as PROG_SETS
    for code_name, code_key in CODE_SUBSETS.items():
        code_prompts = [ps[code_key] for ps in PROG_SETS]
        do(code_name, en_prompts, code_prompts, f'EN→{code_name}')

    # Sentiment
    from data.sentiment_prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS
    do('Sentiment', POSITIVE_PROMPTS, NEGATIVE_PROMPTS, 'Pos→Neg')

    # Register (formal vs casual; literary vs slang)
    from data.register_prompts import PROMPT_SETS as REG_SETS
    fml = [ps['formal'] for ps in REG_SETS]
    csl = [ps['casual'] for ps in REG_SETS]
    do('Formal', csl, fml, 'Casual→Formal')
    lit = [ps['literary'] for ps in REG_SETS]
    slg = [ps['slang'] for ps in REG_SETS]
    do('Literary', slg, lit, 'Slang→Literary')

    # Genre (children vs scientific; news vs dialogue)
    from data.genre_prompts import PROMPT_SETS as GEN_SETS
    sci = [ps['scientific'] for ps in GEN_SETS]
    chi = [ps['children'] for ps in GEN_SETS]
    do('Children', sci, chi, 'Scientific→Children')
    news = [ps['news'] for ps in GEN_SETS]
    dlg = [ps['dialogue'] for ps in GEN_SETS]
    do('News', dlg, news, 'Dialogue→News')

    safe_pkl_dump(DIR_PKL, {'directions': cached, 'layer': LAYER})
    print(f"[{ts()}] [dir] saved {DIR_PKL}  ({len(cached)} total)", flush=True)
    return cached


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
        curves = stored['curves']
        ctx = stored['contexts']
        acts = stored['activations']
        angles = stored['angles_deg']
        measure_layer = stored['measure_layer']
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
        print(f"[{ts()}] [1d] extracting {N_ACT} holdout activations", flush=True)
        ctx, acts = extract_activations_holdout(
            model, tokenizer, device, LAYER, holdout)

    d_model = acts.shape[1]
    rng = np.random.RandomState(SEED)
    rand_dirs = F.normalize(
        torch.tensor(rng.randn(N_RANDOM_1D, d_model), dtype=torch.float32), dim=-1)

    to_run = []
    for k in range(N_RANDOM_1D):
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
    return float(angles_deg[k-1] + (thr - v0) / (v1 - v0)
                 * (angles_deg[k] - angles_deg[k-1]))


def select_pairs(curves, angles, dirs):
    rnd_stack = np.concatenate(
        [curves[k] for k in curves if k.startswith('Random_')], axis=0)
    rnd_break = breaking_angle(rnd_stack, angles, THRESH_L2)
    print(f"[{ts()}] random break @ L2={THRESH_L2}: {rnd_break:.2f}°", flush=True)

    breaks = {}
    for name in dirs:
        b = breaking_angle(curves[name], angles, THRESH_L2)
        breaks[name] = b
        r = b / rnd_break if b else None
        flag = '✓' if r is not None and r < BREAK_RATIO_CUTOFF else '×'
        print(f"  {flag} {name:14s} break={b:5.2f}°  ratio={r:.2f}",
              flush=True)

    valid = [n for n, b in breaks.items()
             if b is not None and b / rnd_break < BREAK_RATIO_CUTOFF]
    print(f"[{ts()}] valid directions ({len(valid)}): {valid}", flush=True)

    def same_family(a, b):
        return FEATURE_TYPE.get(a) == FEATURE_TYPE.get(b)

    pair_infos = []
    for a, b in itertools.combinations(sorted(valid), 2):
        da = F.normalize(dirs[a].float(), dim=0)
        db = F.normalize(dirs[b].float(), dim=0)
        c = float((da @ db).item())
        if abs(c) >= COS_CUTOFF:
            continue
        pair_infos.append({
            'pair': [a, b],
            'cos': c,
            'same_family': same_family(a, b),
            'types': [FEATURE_TYPE.get(a), FEATURE_TYPE.get(b)],
            'break_a': breaks[a], 'break_b': breaks[b],
        })
    # rank: cross-family first, then by |cos| ascending
    pair_infos.sort(key=lambda r: (r['same_family'], abs(r['cos'])))
    print(f"[{ts()}] {len(pair_infos)} pairs pass filter "
          f"(|cos|<{COS_CUTOFF})", flush=True)

    with open(PAIR_JSON, 'w') as f:
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
    print(f"[{ts()}] [2d] grid {N_STEPS_2D}² × {N_ACT} acts, "
          f"{min(max_pairs, len(pair_infos))} pairs, "
          f"max_angle={MAX_ANGLE_2D}°", flush=True)

    for p_info in pair_infos[:max_pairs]:
        a, b = p_info['pair']
        pair_key = f'{a}_{b}'.replace(' ', '_')
        pkl_path = os.path.join(OUT_DIR, f'sweep_2d_{pair_key}.pkl')
        if os.path.exists(pkl_path):
            print(f"  [2d] skip {pair_key} (exists)", flush=True); continue
        t0 = time.time()
        all_l2, all_cos = [], []
        for i in range(N_ACT):
            l2, cos = sweep_2d(
                model, ctx[i], acts[i], dirs[a], dirs[b],
                LAYER, measure_layer, device, angles_rad, angles_rad)
            all_l2.append(l2)
            all_cos.append(cos)
        dt = time.time() - t0
        safe_pkl_dump(pkl_path, {
            'angles_deg': angles,
            'l2': np.stack(all_l2), 'cosine': np.stack(all_cos),
            'layer': LAYER, 'measure_layer': measure_layer,
            'direction_labels': (a, b), 'pair_info': p_info,
        })
        print(f"  [2d] {pair_key}  {dt:.1f}s  saved", flush=True)


# -----------------------------------------------------------------------------
# Phase E: local analysis  (runs as part of this script — no separate step)
# -----------------------------------------------------------------------------

def analyse_2d_pkls():
    """Threshold-sensitivity std per pair. Returns list sorted by std ascending."""
    from scripts.perturbation.plot_isoplateau import (
        extract_contour_points, fit_superellipse)
    THRESHOLDS = np.array(
        [10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400])

    def find_theta(curve, ang, thr):
        idx = np.searchsorted(curve, thr)
        if idx == 0 or idx >= len(ang):
            return ang[min(idx, len(ang) - 1)]
        x0, x1 = ang[idx-1], ang[idx]; y0, y1 = curve[idx-1], curve[idx]
        if y1 == y0: return x0
        return x0 + (thr - y0) / (y1 - y0) * (x1 - x0)

    results = []
    for pkl in sorted(glob.glob(os.path.join(OUT_DIR, 'sweep_2d_*.pkl'))):
        with open(pkl, 'rb') as f: res = pickle.load(f)
        angles = res['angles_deg']
        grid = np.median(res['l2'], axis=0)
        c1, c2 = grid[:, 0], grid[0, :]
        noise = grid[0, 0]
        rows = []
        for thr in THRESHOLDS:
            if thr < noise * 1.5: continue
            t1 = find_theta(c1, angles, thr)
            t2 = find_theta(c2, angles, thr)
            if t1 >= angles[-1] * 0.97 or t2 >= angles[-1] * 0.97: continue
            if t1 <= angles[1] or t2 <= angles[1]: continue
            pts = extract_contour_points(angles, grid, thr)
            if len(pts) == 0: continue
            cn = pts.astype(float).copy()
            cn[:, 0] /= t1; cn[:, 1] /= t2
            p, ci, info = fit_superellipse(cn, (t1+t2)/2)
            if p is None: continue
            rows.append({'thr': float(thr), 'p': float(p),
                         'ci': [float(ci[0]), float(ci[1])],
                         'err': float(info['mean_radial_frac']),
                         't1': float(t1), 't2': float(t2)})
        if not rows:
            continue
        ps = np.array([r['p'] for r in rows])
        results.append({
            'pair': list(res['direction_labels']),
            'pair_info': res.get('pair_info'),
            'rows': rows,
            'p_median': float(np.median(ps)),
            'p_std': float(ps.std()),
            'p_min': float(ps.min()),
            'p_max': float(ps.max()),
            'n_thr': int(len(ps)),
            'err_mean': float(np.mean([r['err'] for r in rows])),
        })
    results.sort(key=lambda r: r['p_std'])
    with open(SUMMARY_JSON, 'w') as f:
        json.dump({'ranked': results}, f, indent=2)
    print(f"[{ts()}] analysis saved {SUMMARY_JSON}", flush=True)
    return results


# -----------------------------------------------------------------------------
# Phase F: composition verification on top-ranked
# -----------------------------------------------------------------------------

def run_composition_verification(model, tokenizer, device, dirs, ranking,
                                 n_top=N_COMPOSITION_TOP):
    import scripts.perturbation.generate_composition_L2 as gcl
    gcl.LAYER = LAYER
    gcl.PROMPT = COMP_PROMPT

    dir_norm = {n: F.normalize(d.float(), dim=0) for n, d in dirs.items()}
    top = ranking[:n_top]
    alpha = float(np.deg2rad(COMP_ANGLE_DEG))

    for r in top:
        a, b = r['pair']
        out_path = os.path.join(OUT_DIR,
                                f'generations_{a}_{b}.json'.replace(' ', '_'))
        if os.path.exists(out_path):
            print(f"[{ts()}] [gen] skip {a}×{b} (exists)", flush=True)
            continue
        print(f"[{ts()}] [gen] {a} × {b}", flush=True)
        da, db = dir_norm[a], dir_norm[b]
        conds = [('baseline', None, None, 0.0, 0.0),
                 ('a_only',   da, None, alpha, 0.0),
                 ('b_only',   None, db, 0.0, alpha),
                 ('a_and_b',  da, db, alpha, alpha)]
        result = {'pair': [a, b], 'prompt': COMP_PROMPT, 'layer': LAYER,
                  'angle_deg': COMP_ANGLE_DEG, 'conditions': {}}
        for cname, dp, dg, ap, ag in conds:
            hook = (make_steering_hook(dp, dg, ap, ag)
                    if dp is not None or dg is not None else None)
            outs = generate(model, tokenizer, device, hook, N_COMP_SAMPLES,
                            hash((a, b, cname)) % 10000)
            result['conditions'][cname] = {
                'angle_a_deg': float(np.rad2deg(ap)),
                'angle_b_deg': float(np.rad2deg(ag)),
                'samples': outs,
            }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"       saved {out_path}", flush=True)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_pairs', type=int, default=200)
    ap.add_argument('--skip_composition', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"[{ts()}] Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    dirs = extract_directions(model, tokenizer, device)

    curves, angles, ctx, acts, measure_layer = run_1d_sweeps(
        model, tokenizer, device, dirs)

    pair_infos = select_pairs(curves, angles, dirs)

    run_2d_sweeps(model, ctx, acts, measure_layer, device, dirs, pair_infos,
                  args.max_pairs)

    ranking = analyse_2d_pkls()
    print(f"\n[{ts()}] TOP 15 by threshold-stability:")
    for r in ranking[:15]:
        pi = r.get('pair_info') or {}
        print(f"  {r['pair'][0]:12s} × {r['pair'][1]:12s}  "
              f"std={r['p_std']:.3f}  med={r['p_median']:.2f}  "
              f"range=[{r['p_min']:.2f},{r['p_max']:.2f}]  "
              f"cos={pi.get('cos', float('nan')):+.2f}", flush=True)

    if not args.skip_composition:
        run_composition_verification(model, tokenizer, device, dirs, ranking)

    print(f"[{ts()}] DONE", flush=True)


if __name__ == '__main__':
    main()
