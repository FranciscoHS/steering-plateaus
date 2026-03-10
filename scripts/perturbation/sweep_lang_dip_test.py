#!/usr/bin/env python3
"""Test whether the rise-and-fall pattern depends on starting activations.

Hypothesis: steering along zh-en from English activations (where en is active)
produces a dip in cosine distance, but steering along the same direction from
Arabic activations (where neither zh nor en is active) does not.

Uses multi-contrastive language features from the allpairs estimator.
"""

import os, sys, pickle, argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model, forward_from_layer_to_layer
from data.multilingual_prompts import PROMPT_SETS

# Config
LAYER = 14
MEASURE_LAYER = 34
MODEL = 'qwen3-4b'
MAX_ANGLE = 60.0
N_ANGLES = 120
N_RAND = 20
SEED = 42
RESULTS_DIR = 'results/lang_dip_test'


def get_activations(model, tokenizer, device, prompts, layer):
    """Extract last-token activations at a given layer for a list of prompts."""
    acts = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            out = model(tokens, output_hidden_states=True)
            h = out.hidden_states[layer + 1]  # +1 because index 0 is embeddings
            acts.append(h[0, -1, :].float().cpu())
    return torch.stack(acts)


def extract_context_and_act(model, tokenizer, device, prompt, layer):
    """Extract context (all but last token) and last-token activation at layer."""
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
        h = out.hidden_states[layer + 1]
    ctx = h[:, :-1, :]   # [1, seq-1, d]
    act = h[0, -1, :]    # [d]
    return ctx, act


def sweep_dir(model, ctx, act, direction, layer, measure_layer, angles_rad, device):
    """Spherical sweep of one direction, return cosine distance array."""
    dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=dtype)
    act_d = act.to(device=device, dtype=dtype)
    d = direction.to(device=device, dtype=dtype)

    a_norm = act_d.norm()
    a_unit = act_d / a_norm

    # Project direction orthogonal to activation
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)

    # Reference at measure layer
    ref = forward_from_layer_to_layer(
        model, ctx_d, act_d[None, None], layer, measure_layer).float()

    cos_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        for j, theta in enumerate(angles_rad):
            p = a_norm * (np.cos(theta) * a_unit + np.sin(theta) * d)
            pf = forward_from_layer_to_layer(
                model, ctx_d, p[None, None], layer, measure_layer).float()
            cos_arr[j] = (1.0 - F.cosine_similarity(ref, pf, dim=-1)).item()

    return cos_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--layer', type=int, default=LAYER)
    parser.add_argument('--measure_layer', type=int, default=MEASURE_LAYER)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    d_model = model._plateau_info['d_model']

    # Compute diff-of-means zh - en direction from prompts
    print("Computing diff-of-means zh - en direction...")
    n_prompts_for_dir = len(PROMPT_SETS)
    zh_acts = get_activations(model, tokenizer, device,
                              [ps['zh'] for ps in PROMPT_SETS], args.layer)
    en_acts = get_activations(model, tokenizer, device,
                              [ps['en'] for ps in PROMPT_SETS], args.layer)
    zh_en_dom = F.normalize(zh_acts.mean(0) - en_acts.mean(0), dim=0)
    print(f"Direction: zh - en (diff-of-means), norm before normalize: "
          f"{(zh_acts.mean(0) - en_acts.mean(0)).norm():.4f}")

    # Angles
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Pick prompts - use a few different ones and average
    n_prompts = min(10, len(PROMPT_SETS))

    # --- Sweep from English activations ---
    print("\nSweeping from English activations...")
    en_cos_dom = []
    en_cos_rand = []
    for pi in range(n_prompts):
        prompt = PROMPT_SETS[pi]['en']
        ctx, act = extract_context_and_act(model, tokenizer, device, prompt, args.layer)
        en_cos_dom.append(sweep_dir(model, ctx, act, zh_en_dom, args.layer, args.measure_layer, angles_rad, device))
        rand_this = []
        for _ in range(N_RAND):
            rd = F.normalize(torch.randn(d_model), dim=0)
            rand_this.append(sweep_dir(model, ctx, act, rd, args.layer, args.measure_layer, angles_rad, device))
        en_cos_rand.append(np.mean(rand_this, axis=0))
        print(f"  prompt {pi+1}/{n_prompts} done")

    en_cos_dom = np.array(en_cos_dom)
    en_cos_rand = np.array(en_cos_rand)

    # --- Sweep from Arabic activations ---
    print("\nSweeping from Arabic activations...")
    ar_cos_dom = []
    ar_cos_rand = []
    for pi in range(n_prompts):
        prompt = PROMPT_SETS[pi]['ar']
        ctx, act = extract_context_and_act(model, tokenizer, device, prompt, args.layer)
        ar_cos_dom.append(sweep_dir(model, ctx, act, zh_en_dom, args.layer, args.measure_layer, angles_rad, device))
        rand_this = []
        for _ in range(N_RAND):
            rd = F.normalize(torch.randn(d_model), dim=0)
            rand_this.append(sweep_dir(model, ctx, act, rd, args.layer, args.measure_layer, angles_rad, device))
        ar_cos_rand.append(np.mean(rand_this, axis=0))
        print(f"  prompt {pi+1}/{n_prompts} done")

    ar_cos_dom = np.array(ar_cos_dom)
    ar_cos_rand = np.array(ar_cos_rand)

    # Save
    results = {
        'angles_deg': angles_deg,
        'en_cos_dom': en_cos_dom, 'en_cos_rand': en_cos_rand,
        'ar_cos_dom': ar_cos_dom, 'ar_cos_rand': ar_cos_rand,
        'config': {'model': args.model, 'layer': args.layer,
                   'measure_layer': args.measure_layer, 'n_prompts': n_prompts},
    }
    pkl_path = os.path.join(RESULTS_DIR, 'lang_dip_dom.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {pkl_path}")

    # Plot: two panels, English vs Arabic starting activations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, dom_arr, rand_arr, title in [
        (ax1, en_cos_dom, en_cos_rand, 'From English activations'),
        (ax2, ar_cos_dom, ar_cos_rand, 'From Arabic activations'),
    ]:
        rand_mean = rand_arr.mean(axis=0)
        rand_std = rand_arr.std(axis=0)
        ax.fill_between(angles_deg, rand_mean - rand_std, rand_mean + rand_std,
                         color='C0', alpha=0.15)
        ax.plot(angles_deg, rand_mean, color='C0', lw=2, label='Random')
        ax.plot(angles_deg, dom_arr.mean(axis=0), color='C3', lw=2, label='zh - en (diff-of-means)')
        ax.fill_between(angles_deg,
                         dom_arr.mean(0) - dom_arr.std(0),
                         dom_arr.mean(0) + dom_arr.std(0),
                         color='C3', alpha=0.15)
        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel('Cosine distance at L34')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{args.model}: zh-en diff-of-means dip test (layer {args.layer} → {args.measure_layer})',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'lang_dip_dom.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
