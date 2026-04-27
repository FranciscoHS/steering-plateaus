#!/usr/bin/env python3
"""Fig 4 robustness orchestrator.

Sweeps fitted superellipse exponent p across many axes of variation:
  A. pair identity (~30 contrastive pairs, default config)
  B. random baselines (ctr×rand + rand×rand)
  C. threshold sweep (Formal × PT, L^2)
  D. metric (L^2, cosine, KL on 5 representative pairs)
  E. perturbation layer (L=2/5/10/20, 5 reps)
  F. measurement layer (penult, penult-5, penult-10, 5 reps)
  G. perturbation method (additive vs spherical, 5 reps)
  H. model (Gemma / Qwen3 / Llama at default config, 5 reps)

One pod per model (fig4-gemma, retrieve, fig4-llama). Same script, --target
selects the model and which phases to run. Output pkls under
results/fig4_robustness/.
"""
import os, sys, pickle, json, argparse, time, itertools
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer, forward_from_layer
from src.data import load_fineweb_fixed_length

OUT_DIR = 'results/fig4_robustness'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Common config ---
N_ACT = 30
SEQ_LEN = 5
SEED = 42
N_STEPS_2D = 41
MAX_ANGLE_2D = 30.0
DEFAULT_THRESH_L2 = 50.0
DEFAULT_THRESH_COS = 0.01
DEFAULT_THRESH_KL = 0.05

# --- Per-target config ---
TARGETS = {
    'gemma': dict(model='gemma-2-9b', perturb_layer=2),
    'qwen':  dict(model='qwen3-1.7b', perturb_layer=7),
    'llama': dict(model='llama-3.1-8b', perturb_layer=4),
}
TARGET_PHASES = {
    'gemma': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'qwen':  ['H'],
    'llama': ['H'],
}

# Representative pairs for phases C–H (always cross-family contrastive)
REP_PAIRS = [
    ('Formal',     'Portuguese'),
    ('Gender',     'Portuguese'),
    ('French',     'Gender'),
    ('Python',     'Portuguese'),
    ('Sentiment',  'Portuguese'),
]
THRESHOLDS_L2 = [20, 30, 40, 50, 60, 75, 100, 125, 150, 200]
PERTURB_LAYERS_GEMMA = [2, 5, 10, 20]
MEASURE_LAYERS_GEMMA_OFFSETS = [-2, -7, -12]   # relative to last layer

# --- Direction prompt sources ---
LANG_KEYS = {
    'Portuguese': 'pt', 'Chinese': 'zh', 'Spanish': 'es', 'French': 'fr',
    'German': 'de', 'Dutch': 'nl', 'Italian': 'it', 'Japanese': 'ja',
    'Russian': 'ru', 'Arabic': 'ar',
}
CODE_KEYS = {
    'Python': 'python', 'JavaScript': 'javascript', 'Java': 'java',
    'Rust': 'rust', 'Haskell': 'haskell', 'Cpp': 'cpp', 'Go': 'go',
    'Lisp': 'lisp', 'TypeScript': 'typescript',
}


def ts():
    return time.strftime('%H:%M:%S')


def safe_pkl(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: pickle.dump(obj, f)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Direction extraction
# -----------------------------------------------------------------------------

def compute_dom(model, tokenizer, device, layer, prompts_a, prompts_b):
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tok = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outs = model(**tok, output_hidden_states=True)
            acts.append(outs.hidden_states[layer + 1][:, -1, :]
                        .squeeze(0).float().cpu())
        return torch.stack(acts)
    a = get_acts(prompts_a); b = get_acts(prompts_b)
    return -F.normalize((a - b).mean(0), dim=0)   # points toward B


def extract_all_directions(model, tokenizer, device, layer, dir_pkl):
    if os.path.exists(dir_pkl):
        with open(dir_pkl, 'rb') as f: return pickle.load(f)['directions']
    print(f"[{ts()}] [dir] extracting at L{layer}", flush=True)
    from data.multilingual_prompts import PROMPT_SETS as ML_SETS
    en = [ps['en'] for ps in ML_SETS]
    dirs = {}
    for lang, key in LANG_KEYS.items():
        other = [ps[key] for ps in ML_SETS]
        n = min(len(en), len(other))
        dirs[lang] = compute_dom(model, tokenizer, device, layer, en[:n], other[:n])
        print(f"  {lang}", flush=True)
    from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
    dirs['Gender'] = compute_dom(model, tokenizer, device, layer,
                                  MALE_PROMPTS, FEMALE_PROMPTS)
    print("  Gender", flush=True)
    from data.programming_prompts import PROMPT_SETS as PROG_SETS
    for cname, ckey in CODE_KEYS.items():
        code = [ps[ckey] for ps in PROG_SETS]
        n = min(len(en), len(code))
        dirs[cname] = compute_dom(model, tokenizer, device, layer,
                                  en[:n], code[:n])
        print(f"  {cname}", flush=True)
    from data.sentiment_prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS
    dirs['Sentiment'] = compute_dom(model, tokenizer, device, layer,
                                    POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
    print("  Sentiment", flush=True)
    from data.register_prompts import PROMPT_SETS as REG_SETS
    fml = [ps['formal'] for ps in REG_SETS]
    csl = [ps['casual'] for ps in REG_SETS]
    dirs['Formal'] = compute_dom(model, tokenizer, device, layer, csl, fml)
    print("  Formal", flush=True)
    lit = [ps['literary'] for ps in REG_SETS]
    slg = [ps['slang'] for ps in REG_SETS]
    dirs['Literary'] = compute_dom(model, tokenizer, device, layer, slg, lit)
    print("  Literary", flush=True)
    from data.genre_prompts import PROMPT_SETS as GEN_SETS
    sci = [ps['scientific'] for ps in GEN_SETS]
    chi = [ps['children'] for ps in GEN_SETS]
    dirs['Children'] = compute_dom(model, tokenizer, device, layer, sci, chi)
    print("  Children", flush=True)
    news = [ps['news'] for ps in GEN_SETS]
    dlg = [ps['dialogue'] for ps in GEN_SETS]
    dirs['News'] = compute_dom(model, tokenizer, device, layer, dlg, news)
    print("  News", flush=True)
    safe_pkl(dir_pkl, {'directions': dirs, 'layer': layer})
    return dirs


# -----------------------------------------------------------------------------
# Activations from FineWeb
# -----------------------------------------------------------------------------

def fineweb_contexts_acts(model, tokenizer, device, layer, target,
                          n=N_ACT, seq_len=SEQ_LEN):
    """Returns lists of (context, activation) extracted from FineWeb chunks.

    Cache is per-target to avoid mixing tokenizers across models."""
    cache_dir = f'/workspace/fineweb_cache_{target}'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n, tokenizer, seq_len=seq_len, seed=SEED, cache_dir=cache_dir)
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        hidden = outs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].float().cpu())
        activations.append(hidden[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


# -----------------------------------------------------------------------------
# Sweep primitives
# -----------------------------------------------------------------------------

def sweep_2d_spherical(model, ctx, act, d1, d2, layer, measure_layer, device,
                        angles_rad, want_kl=False, batch_size=64):
    """2D on-sphere sweep. Returns dict with 'l2', 'cosine', and optionally
    'kl' grids."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    a_unit = a_d / R
    d1_d = d1.to(device=device, dtype=torch.float32)
    d2_d = d2.to(device=device, dtype=torch.float32)
    d1_o = F.normalize(d1_d - (d1_d @ a_unit) * a_unit, dim=0)
    d2_o = d2_d - (d2_d @ a_unit) * a_unit
    d2_o = F.normalize(d2_o - (d2_o @ d1_o) * d1_o, dim=0)
    n1, n2 = len(angles_rad), len(angles_rad)
    s1 = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    s2 = s1.clone()
    S1, S2 = torch.meshgrid(s1, s2, indexing='ij')
    C = torch.sqrt(torch.clamp(1.0 - S1**2 - S2**2, min=0.0))
    pts = (R * (C.reshape(-1, 1) * a_unit.unsqueeze(0)
                + S1.reshape(-1, 1) * d1_o.unsqueeze(0)
                + S2.reshape(-1, 1) * d2_o.unsqueeze(0))).to(model_dtype)
    n = n1 * n2
    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype),
            layer, measure_layer).float()
        if want_kl:
            ref_logits = forward_from_layer(
                model, ctx_d, a_d[None, None].to(model_dtype), layer).float()
            ref_logp = F.log_softmax(ref_logits, dim=-1)
    l2 = np.zeros(n); cos = np.zeros(n)
    kl = np.zeros(n) if want_kl else None
    with torch.no_grad():
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            batch = pts[st:ed].unsqueeze(1)
            pf = forward_from_layer_to_layer(model, ctx_d, batch, layer,
                                             measure_layer).float()
            l2[st:ed] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos[st:ed] = (1.0 - F.cosine_similarity(ref_act, pf, dim=-1)).cpu().numpy()
            if want_kl:
                logits = forward_from_layer(model, ctx_d, batch, layer).float()
                logp = F.log_softmax(logits, dim=-1)
                # forward KL: sum p_ref * (log_ref - log_perturbed) over vocab
                kld = (ref_logp.exp() * (ref_logp - logp)).sum(dim=-1).squeeze()
                kl[st:ed] = kld.cpu().numpy()
    out = {'l2': l2.reshape(n1, n2), 'cosine': cos.reshape(n1, n2)}
    if want_kl: out['kl'] = kl.reshape(n1, n2)
    return out


def sweep_2d_additive(model, ctx, act, d1, d2, layer, measure_layer, device,
                      mags, batch_size=64):
    """2D additive sweep: act + m1*d1 + m2*d2 (no normalization). mags are
    magnitudes in units of ||a|| (i.e., m=0.3 → add 0.3*||a||*d).
    Returns dict with 'l2' and 'cosine' grids."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=torch.float32)
    R = a_d.norm()
    d1_d = F.normalize(d1.to(device=device, dtype=torch.float32), dim=0)
    d2_d = F.normalize(d2.to(device=device, dtype=torch.float32), dim=0)
    n = len(mags)
    M1, M2 = torch.meshgrid(
        torch.tensor(mags, dtype=torch.float32, device=device),
        torch.tensor(mags, dtype=torch.float32, device=device), indexing='ij')
    pts = (a_d.unsqueeze(0) + R * (M1.reshape(-1, 1) * d1_d.unsqueeze(0)
                                   + M2.reshape(-1, 1) * d2_d.unsqueeze(0))
           ).to(model_dtype)
    nn = n * n
    with torch.no_grad():
        ref_act = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None].to(model_dtype),
            layer, measure_layer).float()
    l2 = np.zeros(nn); cos = np.zeros(nn)
    with torch.no_grad():
        for st in range(0, nn, batch_size):
            ed = min(st + batch_size, nn)
            batch = pts[st:ed].unsqueeze(1)
            pf = forward_from_layer_to_layer(model, ctx_d, batch, layer,
                                             measure_layer).float()
            l2[st:ed] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            cos[st:ed] = (1.0 - F.cosine_similarity(ref_act, pf, dim=-1)).cpu().numpy()
    return {'l2': l2.reshape(n, n), 'cosine': cos.reshape(n, n)}


def run_2d_per_acts(model, ctx, acts, d1, d2, layer, measure_layer, device,
                    angles_rad, want_kl=False, additive_mags=None):
    out = {'l2': [], 'cosine': []}
    if want_kl: out['kl'] = []
    for i in range(len(acts)):
        if additive_mags is not None:
            r = sweep_2d_additive(model, ctx[i], acts[i], d1, d2,
                                  layer, measure_layer, device, additive_mags)
        else:
            r = sweep_2d_spherical(model, ctx[i], acts[i], d1, d2,
                                   layer, measure_layer, device, angles_rad,
                                   want_kl=want_kl)
        for k in out: out[k].append(r[k])
    return {k: np.stack(v) for k, v in out.items()}


# -----------------------------------------------------------------------------
# Phase implementations
# -----------------------------------------------------------------------------

def pair_key(a, b): return f'{a}_{b}'.replace(' ', '_')


def phase_A(model, tokenizer, device, dirs, ctx, acts, perturb_layer,
            measure_layer, target):
    """30 cross-family contrastive pairs at default config."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    # Sample 30 pairs prioritizing cross-family
    LANG = list(LANG_KEYS); CODE = list(CODE_KEYS)
    SEM = ['Gender', 'Sentiment', 'Formal', 'Literary', 'Children', 'News']
    pairs = []
    # cross-family
    for s in SEM:
        for l in LANG: pairs.append((s, l))
    for s in SEM[:3]:
        for c in CODE[:5]: pairs.append((s, c))
    for l in LANG[:5]:
        for c in CODE[:5]: pairs.append((l, c))
    pairs = pairs[:30]
    print(f"[{ts()}] [A] {len(pairs)} pairs", flush=True)
    for a, b in pairs:
        out = os.path.join(OUT_DIR, f'A_{target}_{pair_key(a, b)}.pkl')
        if os.path.exists(out): print(f"  skip {a}×{b}"); continue
        t0 = time.time()
        r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                            perturb_layer, measure_layer, device, angles_rad)
        safe_pkl(out, {'angles_deg': angles, **r,
                       'layer': perturb_layer, 'measure_layer': measure_layer,
                       'direction_labels': (a, b), 'phase': 'A',
                       'model': target})
        print(f"  {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_B(model, tokenizer, device, dirs, ctx, acts, perturb_layer,
            measure_layer, target):
    """ctr×rand and rand×rand baselines."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    rng = np.random.RandomState(SEED)
    d_model = next(iter(dirs.values())).numel()
    rand_dirs = F.normalize(torch.tensor(rng.randn(20, d_model),
                                          dtype=torch.float32), dim=-1)
    base_dirs = ['Portuguese', 'Gender', 'Formal', 'Python', 'French']
    # 10 ctr × rand
    ctr_rand = [(base_dirs[i % len(base_dirs)], f'Random_{i}') for i in range(10)]
    rand_rand = [(f'Random_{i*2}', f'Random_{i*2+1}') for i in range(10)]
    all_pairs = [('ctr_rand', a, b) for a, b in ctr_rand] + \
                [('rand_rand', a, b) for a, b in rand_rand]
    print(f"[{ts()}] [B] {len(all_pairs)} pairs", flush=True)
    for kind, a, b in all_pairs:
        out = os.path.join(OUT_DIR, f'B_{kind}_{target}_{pair_key(a, b)}.pkl')
        if os.path.exists(out): print(f"  skip {a}×{b}"); continue
        d1 = dirs[a] if a in dirs else rand_dirs[int(a.split('_')[1])]
        d2 = dirs[b] if b in dirs else rand_dirs[int(b.split('_')[1])]
        t0 = time.time()
        r = run_2d_per_acts(model, ctx, acts, d1, d2,
                            perturb_layer, measure_layer, device, angles_rad)
        safe_pkl(out, {'angles_deg': angles, **r,
                       'layer': perturb_layer, 'measure_layer': measure_layer,
                       'direction_labels': (a, b), 'phase': 'B', 'kind': kind,
                       'model': target})
        print(f"  [{kind}] {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_C(*args, **kwargs):
    """Threshold sweep is post-hoc on phase A's flagship pkl. No new compute."""
    print("[C] no new compute (threshold sweep done in analysis)", flush=True)


def phase_D(model, tokenizer, device, dirs, ctx, acts, perturb_layer,
            measure_layer, target):
    """5 representative pairs at L^2 + cosine + KL (KL needs new sweep)."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    print(f"[{ts()}] [D] {len(REP_PAIRS)} pairs with KL", flush=True)
    for a, b in REP_PAIRS:
        out = os.path.join(OUT_DIR, f'D_{target}_{pair_key(a, b)}_withkl.pkl')
        if os.path.exists(out): print(f"  skip {a}×{b}"); continue
        if a not in dirs or b not in dirs:
            print(f"  miss dir for {a}×{b}"); continue
        t0 = time.time()
        r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                            perturb_layer, measure_layer, device, angles_rad,
                            want_kl=True)
        safe_pkl(out, {'angles_deg': angles, **r,
                       'layer': perturb_layer, 'measure_layer': measure_layer,
                       'direction_labels': (a, b), 'phase': 'D',
                       'model': target})
        print(f"  {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_E(model, tokenizer, device, ctx_acts_by_layer, target):
    """Vary perturb layer, fix measure at penult. ctx_acts_by_layer is dict
    layer→(ctx, acts, dirs)."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    for L in PERTURB_LAYERS_GEMMA:
        ctx, acts, dirs = ctx_acts_by_layer[L]
        for a, b in REP_PAIRS:
            out = os.path.join(OUT_DIR, f'E_{target}_L{L}_{pair_key(a, b)}.pkl')
            if os.path.exists(out): print(f"  skip L{L} {a}×{b}"); continue
            if a not in dirs or b not in dirs: continue
            t0 = time.time()
            r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                                L, measure_layer, device, angles_rad)
            safe_pkl(out, {'angles_deg': angles, **r,
                           'layer': L, 'measure_layer': measure_layer,
                           'direction_labels': (a, b), 'phase': 'E',
                           'model': target})
            print(f"  E L{L} {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_F(model, tokenizer, device, dirs, ctx, acts, perturb_layer, target):
    """Vary measure layer, fix perturb at default."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    n_layers = len(_get_blocks(model))
    for off in MEASURE_LAYERS_GEMMA_OFFSETS:
        meas_L = n_layers + off    # off is negative
        for a, b in REP_PAIRS:
            out = os.path.join(OUT_DIR,
                               f'F_{target}_meas{meas_L}_{pair_key(a, b)}.pkl')
            if os.path.exists(out): print(f"  skip meas{meas_L} {a}×{b}"); continue
            if a not in dirs or b not in dirs: continue
            t0 = time.time()
            r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                                perturb_layer, meas_L, device, angles_rad)
            safe_pkl(out, {'angles_deg': angles, **r,
                           'layer': perturb_layer, 'measure_layer': meas_L,
                           'direction_labels': (a, b), 'phase': 'F',
                           'model': target})
            print(f"  F meas{meas_L} {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_G(model, tokenizer, device, dirs, ctx, acts, perturb_layer,
            measure_layer, target):
    """Additive perturbation; compare to spherical (already in phase A pkls)."""
    n_steps = N_STEPS_2D
    # Magnitude grid: 0..0.5 in units of ||a||; 0.5 ≈ tan(30°)
    mags = np.linspace(0, 0.55, n_steps)
    print(f"[{ts()}] [G] additive on {len(REP_PAIRS)} pairs", flush=True)
    for a, b in REP_PAIRS:
        out = os.path.join(OUT_DIR, f'G_{target}_additive_{pair_key(a, b)}.pkl')
        if os.path.exists(out): print(f"  skip {a}×{b}"); continue
        if a not in dirs or b not in dirs: continue
        t0 = time.time()
        r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                            perturb_layer, measure_layer, device, None,
                            additive_mags=mags)
        safe_pkl(out, {'mags': mags, **r,
                       'layer': perturb_layer, 'measure_layer': measure_layer,
                       'direction_labels': (a, b), 'phase': 'G_additive',
                       'model': target})
        print(f"  G {a}×{b}  {time.time()-t0:.1f}s", flush=True)


def phase_H(model, tokenizer, device, dirs, ctx, acts, perturb_layer,
            measure_layer, target):
    """Phase H is just Phase A on a non-Gemma model: 5 representative pairs at
    that model's default config."""
    angles = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles)
    print(f"[{ts()}] [H] {len(REP_PAIRS)} pairs on {target}", flush=True)
    for a, b in REP_PAIRS:
        out = os.path.join(OUT_DIR, f'H_{target}_{pair_key(a, b)}.pkl')
        if os.path.exists(out): print(f"  skip {a}×{b}"); continue
        if a not in dirs or b not in dirs:
            print(f"  miss dir {a}×{b}"); continue
        t0 = time.time()
        r = run_2d_per_acts(model, ctx, acts, dirs[a], dirs[b],
                            perturb_layer, measure_layer, device, angles_rad)
        safe_pkl(out, {'angles_deg': angles, **r,
                       'layer': perturb_layer, 'measure_layer': measure_layer,
                       'direction_labels': (a, b), 'phase': 'H',
                       'model': target})
        print(f"  H {a}×{b}  {time.time()-t0:.1f}s", flush=True)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(TARGETS))
    ap.add_argument('--phases', default=None,
                    help='comma-separated phase letters; default is per-target')
    args = ap.parse_args()

    cfg = TARGETS[args.target]
    phases = (args.phases.split(',') if args.phases
              else TARGET_PHASES[args.target])
    print(f"[{ts()}] target={args.target}  model={cfg['model']}  "
          f"perturb_layer={cfg['perturb_layer']}  phases={phases}", flush=True)

    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    perturb_layer = cfg['perturb_layer']
    print(f"  n_layers={n_layers}  measure_layer={measure_layer}", flush=True)

    # Default-layer dirs + activations
    dir_pkl = os.path.join(OUT_DIR,
                           f'directions_{args.target}_L{perturb_layer}.pkl')
    dirs = extract_all_directions(
        model, tokenizer, device, perturb_layer, dir_pkl)
    print(f"[{ts()}] extracting FineWeb activations at L{perturb_layer}",
          flush=True)
    ctx, acts = fineweb_contexts_acts(
        model, tokenizer, device, perturb_layer, args.target)

    if 'A' in phases:
        phase_A(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, measure_layer, args.target)
    if 'B' in phases:
        phase_B(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, measure_layer, args.target)
    if 'C' in phases:
        phase_C()
    if 'D' in phases:
        phase_D(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, measure_layer, args.target)
    if 'E' in phases:
        # Phase E needs directions + activations at multiple perturb layers
        ctx_acts_by_layer = {perturb_layer: (ctx, acts, dirs)}
        for L in PERTURB_LAYERS_GEMMA:
            if L == perturb_layer: continue
            print(f"[{ts()}] [E] extracting dirs+acts for L={L}", flush=True)
            dir_pkl_L = os.path.join(OUT_DIR,
                                     f'directions_{args.target}_L{L}.pkl')
            dirs_L = extract_all_directions(
                model, tokenizer, device, L, dir_pkl_L)
            ctx_L, acts_L = fineweb_contexts_acts(
                model, tokenizer, device, L, args.target)
            ctx_acts_by_layer[L] = (ctx_L, acts_L, dirs_L)
        phase_E(model, tokenizer, device, ctx_acts_by_layer, args.target)
    if 'F' in phases:
        phase_F(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, args.target)
    if 'G' in phases:
        phase_G(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, measure_layer, args.target)
    if 'H' in phases:
        phase_H(model, tokenizer, device, dirs, ctx, acts,
                perturb_layer, measure_layer, args.target)

    print(f"[{ts()}] DONE", flush=True)


if __name__ == '__main__':
    main()
