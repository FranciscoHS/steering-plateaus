#!/usr/bin/env python3
"""
Perturbation sweep for JavaScript direction orthogonalized against Python.

Compares: JS, JS_ortho_python, Python, random.
"""

import sys, os, pickle, argparse
import numpy as np

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model
from scripts.directions.sweep_programming_code_prompts import (
    extract_context_and_activation_from_prompts,
    forward_to_logits,
    sweep_direction,
    sweep_random,
)

PROMPTS = [
    "The following code",
    "The code below",
    "Here is some code",
    "Consider the following program",
    "The following function",
    "Below is an implementation",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = f'results/programming_contrastive/{args.model}'

    # Load directions
    with open(os.path.join(results_dir, 'contrastive_results.pkl'), 'rb') as f:
        contrastive = pickle.load(f)
    f_hat = contrastive['f_hat']

    js = f_hat['javascript']
    py = f_hat['python']

    # Orthogonalize JS against Python
    proj = (js @ py) / (py @ py) * py
    js_ortho = js - proj
    js_ortho = js_ortho / js_ortho.norm()  # renormalize to unit vector
    # Also normalize the originals for fair comparison
    js_unit = js / js.norm()
    py_unit = py / py.norm()

    cos_orig = (js_unit @ py_unit).item()
    cos_ortho = (js_ortho @ py_unit).item()
    print(f"cos(JS, Python) = {cos_orig:.4f}")
    print(f"cos(JS_ortho, Python) = {cos_ortho:.6f}")
    print(f"cos(JS, JS_ortho) = {(js_unit @ js_ortho).item():.4f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Extract activations
    print(f"Extracting activations for {len(PROMPTS)} prompts...")
    contexts, activations = extract_context_and_activation_from_prompts(
        model, tokenizer, PROMPTS, args.layer, device)
    contexts = contexts.to(device)
    activations = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    angles_deg = np.linspace(1, args.max_angle, args.n_angles)
    angles_rad = np.radians(angles_deg)

    # Sweep all three directions
    directions = {
        'f_hat_javascript': js_unit,
        'f_hat_javascript_ortho_python': js_ortho,
        'f_hat_python': py_unit,
    }

    sweep_results = {}
    for name, d in directions.items():
        print(f"  Sweeping {name}...")
        kl, l2 = sweep_direction(
            model, contexts, activations, ref_final, ref_log_probs,
            d, angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}

    print("  Sweeping random directions...")
    rand_kl, rand_l2 = sweep_random(
        model, contexts, activations, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=10)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}

    # Save
    save_data = {
        'angles_deg': angles_deg,
        'sweep_results': sweep_results,
        'prompts': PROMPTS,
        'cos_js_python': cos_orig,
        'cos_jsortho_python': cos_ortho,
    }
    out_path = os.path.join(results_dir, 'sweep_js_ortho_python.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
