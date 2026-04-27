#!/usr/bin/env python3
"""Coherence grading on the (alpha_pt, alpha_gen) plane (cheap version).

For each cell of an 11x11 grid in [0, 90]^2 deg (skipping cells with
sin^2 a1 + sin^2 a2 > 1), generate `N_PROMPTS` x 50 tokens from Gemma-2-9B
with the on-sphere perturbation hooked at L2. Score each generation with two
cheap signals:
  - repetition: 1 - unique 3-grams / total 3-grams (high = bad)
  - perplexity: avg next-token log-prob of the generation under unsteered
    Gemma-2-9B (high = bad)
A generation is `coherent` iff PPL < ppl_threshold AND repetition < 0.3,
where ppl_threshold = 5 * baseline_ppl_at_(0,0).

Saves per-cell coherence_frac (0..1) to a pkl. A separate plot script
overlays the 0.5 contour onto the three threshold panels.
"""
import os, sys, pickle, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks

# --- Config ---
LAYER = 2
N_PROMPTS = 5
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 42
GRID_DEG = np.linspace(0.0, 90.0, 11)   # 11x11 -> 121 cells
REP_THRESHOLD = 0.3
PPL_MULT = 5.0  # threshold = PPL_MULT * baseline_ppl_at_(0,0)

OUT_DIR = 'results/paper_L2_gemma2_9b'
DIR_PKL = os.path.join(OUT_DIR, 'directions_L2.pkl')
OUT_PKL = os.path.join(OUT_DIR, 'coherence_pt_gender.pkl')


def make_steering_hook(d_pt, d_gen, alpha_pt_r, alpha_gen_r):
    """Spherical perturbation hook applied to LAST position only."""
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        a = hs[:, -1, :].float()
        R = a.norm(dim=-1, keepdim=True)
        a_unit = a / (R + 1e-12)

        def orth(d, basis):
            x = d
            for b in basis:
                x = x - (x * b).sum(dim=-1, keepdim=True) * b
            n = x.norm(dim=-1, keepdim=True)
            return x / (n + 1e-12)

        basis = [a_unit]
        pt_perp = torch.zeros_like(a_unit)
        gen_perp = torch.zeros_like(a_unit)
        if d_pt is not None:
            dp = d_pt.to(a.device, a.dtype).unsqueeze(0).expand_as(a_unit)
            pt_perp = orth(dp, basis); basis = basis + [pt_perp]
        if d_gen is not None:
            dg = d_gen.to(a.device, a.dtype).unsqueeze(0).expand_as(a_unit)
            gen_perp = orth(dg, basis)

        s_pt = float(np.sin(alpha_pt_r))
        s_gen = float(np.sin(alpha_gen_r))
        c = float(np.sqrt(max(0.0, 1.0 - s_pt**2 - s_gen**2)))
        new_a = R * (c * a_unit + s_pt * pt_perp + s_gen * gen_perp)
        hs[:, -1, :] = new_a.to(hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs
    return hook


@torch.no_grad()
def generate_one(model, tokenizer, device, prompt, hook_fn, seed):
    blocks = _get_blocks(model)
    handle = blocks[LAYER].register_forward_hook(hook_fn) if hook_fn else None
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    try:
        torch.manual_seed(seed)
        out = model.generate(
            **inp, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0, inp.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()
    return text, gen_ids


@torch.no_grad()
def perplexity(model, tokenizer, device, text):
    """PPL of `text` (no prompt prefix) under unsteered Gemma."""
    if not text.strip():
        return float('nan')
    ids = tokenizer(text, return_tensors='pt', truncation=True,
                    max_length=128).input_ids.to(device)
    if ids.shape[1] < 2:
        return float('nan')
    out = model(ids)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.reshape(-1))
    return float(loss.exp().item())


def repetition_score(text, n=3):
    chars = list(text.replace(' ', ''))
    if len(chars) < n:
        return 0.0
    ngrams = [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("Loading directions...", flush=True)
    with open(DIR_PKL, 'rb') as f:
        dirs = pickle.load(f)
    d_pt = F.normalize(dirs['d_portuguese'].float(), dim=0)
    d_gen = F.normalize(dirs['d_gender'].float(), dim=0)

    print("Loading Gemma-2-9B...", flush=True)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
    prompts = MALE_PROMPTS_HOLDOUT[:N_PROMPTS]
    print(f"Using {len(prompts)} prompts:", flush=True)
    for p in prompts: print(f"  - {p}", flush=True)

    n = len(GRID_DEG)
    coherence_frac = np.full((n, n), np.nan)
    ppl_grid = np.full((n, n, N_PROMPTS), np.nan)
    rep_grid = np.full((n, n, N_PROMPTS), np.nan)
    gens_grid = [[None]*n for _ in range(n)]

    # Step 1: baseline ppl at (0,0)
    print("\nBaseline PPL at (0,0)...", flush=True)
    base_ppls = []
    for k, prompt in enumerate(prompts):
        hook = make_steering_hook(d_pt, d_gen, 0.0, 0.0)
        text, _ = generate_one(model, tokenizer, device, prompt, hook,
                               SEED + k)
        ppl = perplexity(model, tokenizer, device, text)
        base_ppls.append(ppl)
        print(f"  '{prompt[:30]}...' -> PPL={ppl:.1f}  text: {text[:60]!r}",
              flush=True)
    baseline_ppl = float(np.nanmedian(base_ppls))
    ppl_thr = PPL_MULT * baseline_ppl
    print(f"\nBaseline median PPL = {baseline_ppl:.1f}  -> "
          f"threshold = {ppl_thr:.1f}", flush=True)

    # Step 2: full grid
    t0 = time.time()
    total = 0
    for i, a1 in enumerate(GRID_DEG):
        for j, a2 in enumerate(GRID_DEG):
            s1, s2 = np.sin(np.deg2rad(a1)), np.sin(np.deg2rad(a2))
            if s1**2 + s2**2 > 1.0:
                continue
            cell_gens, cell_ppls, cell_reps = [], [], []
            for k, prompt in enumerate(prompts):
                hook = make_steering_hook(
                    d_pt, d_gen, np.deg2rad(a1), np.deg2rad(a2))
                text, _ = generate_one(model, tokenizer, device, prompt, hook,
                                       SEED + 1000 * i + 100 * j + k)
                ppl = perplexity(model, tokenizer, device, text)
                rep = repetition_score(text)
                cell_gens.append(text)
                cell_ppls.append(ppl)
                cell_reps.append(rep)
            ppl_arr = np.array(cell_ppls)
            rep_arr = np.array(cell_reps)
            coherent = (ppl_arr < ppl_thr) & (rep_arr < REP_THRESHOLD)
            coherence_frac[i, j] = coherent.mean()
            ppl_grid[i, j] = ppl_arr
            rep_grid[i, j] = rep_arr
            gens_grid[i][j] = cell_gens
            total += 1
            elapsed = time.time() - t0
            print(f"  ({a1:5.1f}°, {a2:5.1f}°) coh={coherence_frac[i,j]:.2f} "
                  f"PPL_med={np.nanmedian(ppl_arr):.1f} "
                  f"rep_med={np.nanmedian(rep_arr):.2f}  "
                  f"[{total} cells, {elapsed/60:.1f} min]",
                  flush=True)

    out = {
        'grid_deg': GRID_DEG,
        'coherence_frac': coherence_frac,
        'ppl_grid': ppl_grid,
        'rep_grid': rep_grid,
        'baseline_ppl': baseline_ppl,
        'ppl_threshold': ppl_thr,
        'rep_threshold': REP_THRESHOLD,
        'prompts': prompts,
        'gens_grid': gens_grid,
        'config': {'layer': LAYER, 'n_prompts': N_PROMPTS,
                   'max_new_tokens': MAX_NEW_TOKENS,
                   'temperature': TEMPERATURE, 'top_p': TOP_P, 'seed': SEED,
                   'ppl_mult': PPL_MULT, 'rep_threshold': REP_THRESHOLD},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {OUT_PKL}", flush=True)


if __name__ == '__main__':
    main()
