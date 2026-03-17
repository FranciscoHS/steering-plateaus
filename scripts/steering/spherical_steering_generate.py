#!/usr/bin/env python3
"""
Spherical steering during generation: rotate activations toward Chinese direction.

Extracts PC1 from English-Chinese difference matrix at a given layer,
then generates text with spherical steering (norm-preserving rotation)
applied to all newly generated token positions at each step.

This is the generative equivalent of the spherical perturbation sweep,
so angles map directly onto the sweep_comparison plot x-axis.

Usage:
    python scripts/spherical_steering_generate.py
    python scripts/spherical_steering_generate.py --angles 0,5,10,15,20
"""

import argparse
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '.')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


def extract_pc1(model, tokenizer, device, layer, skip_svd=False):
    """Extract PC1 from English-Chinese activation difference matrix.

    D = en_acts - zh_acts at last token position.
    PC1 = first right singular vector of D (direction of max variance in diffs).

    Returns:
        pc1: [d_model] unit vector pointing English-ward
        mean_dir: [d_model] unit mean-difference direction
        mean_norm: float, mean activation norm at this layer
    """
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).cpu())
        return torch.stack(acts)

    print(f"Extracting bilingual activations at layer {layer}...")
    en_acts = get_acts(ENGLISH_PROMPTS)
    zh_acts = get_acts(CHINESE_PROMPTS)

    en_acts = en_acts.float()
    zh_acts = zh_acts.float()
    all_acts = torch.cat([en_acts, zh_acts], dim=0)
    mean_norm = all_acts.norm(dim=-1).mean().item()

    D = en_acts - zh_acts  # [n_pairs, d_model]
    mean_vec = D.mean(dim=0)
    mean_dir = mean_vec / mean_vec.norm()

    print(f"  D shape: {D.shape}, mean diff norm: {mean_vec.norm():.2f}")
    print(f"  Mean activation norm: {mean_norm:.2f}")

    if skip_svd:
        return mean_dir, mean_norm

    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    pc1 = Vh[0]  # [d_model]

    # Ensure PC1 points English-ward (positive cosine with mean diff)
    if (pc1 @ mean_dir) < 0:
        pc1 = -pc1

    var_explained = S[0]**2 / (S**2).sum()
    print(f"  PC1 variance explained: {var_explained:.3f}")
    print(f"  cos(PC1, mean): {(pc1 @ mean_dir).item():.3f}")

    return pc1, mean_dir, mean_norm


def generate_with_spherical_steering(model, tokenizer, prompt, steer_direction,
                                     angle_deg, steer_layer, device,
                                     n_tokens=20, steer_mode='rolling'):
    """Generate tokens with spherical steering at each step.

    No KV cache — full sequence reprocessed at each step for correctness.

    Args:
        steer_direction: [d_model] unit vector to rotate toward
        angle_deg: rotation angle in degrees (0 = no steering)
        steer_mode: 'rolling' steers all positions from last prompt token onward,
                    'last_token' steers only the final token (like You et al. 2026)
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    direction = steer_direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1  # last prompt token

    generated_ids = []

    for step in range(n_tokens):
        # Hook: spherical rotation at steered positions
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if steer_mode == 'last_token':
                steer_slice = hidden[0, -1:, :]  # [1, d] — only current token
            else:
                steer_slice = hidden[0, first_steer_pos:, :]  # [n_steer, d]

            if steer_slice.shape[0] == 0:
                return output

            R = steer_slice.norm(dim=-1, keepdim=True)        # [n_steer, 1]
            a_unit = steer_slice / (R + 1e-10)                # [n_steer, d]

            # Project direction orthogonal to each activation
            proj = (a_unit * direction).sum(dim=-1, keepdim=True)  # [n_steer, 1]
            v_perp = direction.unsqueeze(0) - proj * a_unit        # [n_steer, d]
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)

            # Spherical rotation: stay on the sphere of radius R
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            if steer_mode == 'last_token':
                hidden[0, -1:, :] = steered
            else:
                hidden[0, first_steer_pos:, :] = steered

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        handle.remove()

        # Greedy decoding
        logits = outputs.logits[0, -1, :]
        next_token_id = logits.argmax().item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def generate_with_additive_steering(model, tokenizer, prompt, steer_direction,
                                     magnitude, steer_layer, device,
                                     n_tokens=20, steer_mode='rolling'):
    """Generate tokens with additive steering at each step.

    Adds magnitude * steer_direction to the activation at layer L.
    Unlike spherical, this changes the norm.

    Args:
        steer_direction: [d_model] unit vector to steer toward
        magnitude: scalar multiplier for the steering vector
        steer_mode: 'rolling' steers from last prompt token onward,
                    'last_token' steers only the final token
    """
    direction = steer_direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if steer_mode == 'last_token':
                hidden[0, -1:, :] += magnitude * direction.unsqueeze(0)
            else:
                hidden[0, first_steer_pos:, :] += magnitude * direction.unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[steer_layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        handle.remove()

        logits = outputs.logits[0, -1, :]
        next_token_id = logits.argmax().item()
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def chinese_char_fraction(text):
    """Fraction of characters in text that are CJK unified ideographs."""
    if not text:
        return 0.0
    n_chinese = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    n_total = sum(1 for ch in text if not ch.isspace() and ch not in '.,;:!?()[]{}"\'-/')
    if n_total == 0:
        return 0.0
    return n_chinese / n_total


def plot_language_transition(all_results, x_values, save_dir, model_name, layer,
                             dir_label='PC1', x_key='angle_deg',
                             x_label='Steering angle (degrees)'):
    """Plot fraction of Chinese characters vs steering strength.

    Shows the English -> Mixed -> Mandarin transition.
    """
    # Group results by prompt
    prompts = []
    seen = set()
    for r in all_results:
        if r['prompt'] not in seen:
            prompts.append(r['prompt'])
            seen.add(r['prompt'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-prompt lines
    fractions_by_prompt = []
    for prompt in prompts:
        fracs = []
        for xv in x_values:
            match = [r for r in all_results
                     if r['prompt'] == prompt and r[x_key] == xv]
            fracs.append(chinese_char_fraction(match[0]['generated_text']))
        fractions_by_prompt.append(fracs)
        # Short label for legend
        label = prompt[:30] + '...' if len(prompt) > 30 else prompt
        ax.plot(x_values, fracs, 'o-', alpha=0.5, markersize=4, label=label)

    # Median across prompts
    fracs_array = np.array(fractions_by_prompt)  # [n_prompts, n_values]
    median = np.median(fracs_array, axis=0)
    ax.plot(x_values, median, 'k-', linewidth=3, label='Median', zorder=10)

    # Shade the three predicted regimes
    ax.axhspan(0, 0.1, color='blue', alpha=0.05)
    ax.axhspan(0.1, 0.9, color='yellow', alpha=0.05)
    ax.axhspan(0.9, 1.0, color='red', alpha=0.05)

    method_label = 'additive' if x_key == 'magnitude' else 'spherical'
    ax.set_xlabel(x_label)
    ax.set_ylabel('Fraction Chinese characters')
    ax.set_title(f'{model_name}, layer {layer}: {method_label} steering toward Chinese ({dir_label})')
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'language_transition.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--direction', type=str, default='pc1',
                        choices=['pc1', 'mean', 'melbo1', 'melbo2', 'melbo3'],
                        help='Steering direction: pc1, mean (diff-of-means), or melbo1/2/3')
    parser.add_argument('--melbo_results', type=str, default=None,
                        help='Path to MELBO results .pkl (auto-detected if None)')
    parser.add_argument('--method', type=str, default='spherical',
                        choices=['spherical', 'additive'],
                        help='Steering method: spherical (norm-preserving rotation) or additive')
    parser.add_argument('--angles', type=str,
                        default=','.join(str(i) for i in range(21)),
                        help='Comma-separated angles in degrees (for spherical, default: 0-20)')
    parser.add_argument('--magnitudes', type=str, default=None,
                        help='Comma-separated magnitudes (for additive, default: 0,5,10,15,20,25,30,40,50,65)')
    parser.add_argument('--n_tokens', type=int, default=20)
    parser.add_argument('--n_prompts', type=int, default=5,
                        help='Number of English prompts to test')
    parser.add_argument('--steer_mode', type=str, default='rolling',
                        choices=['rolling', 'last_token'],
                        help='rolling: steer all positions from last prompt token onward; '
                             'last_token: steer only the current token (like You et al. 2026)')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Model dtype (use bfloat16 for large models)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.method == 'additive':
        if args.magnitudes is None:
            args.magnitudes = '0,5,10,15,20,25,30,40,50,65'
        magnitudes = [float(m) for m in args.magnitudes.split(',')]
    else:
        magnitudes = None

    angles = [float(a) for a in args.angles.split(',')]

    if args.results_dir is None:
        mode_suffix = f'_{args.steer_mode}' if args.steer_mode != 'rolling' else ''
        method_prefix = 'additive_steering_gen' if args.method == 'additive' else 'spherical_steering_gen'
        args.results_dir = f'results/{method_prefix}/{args.model}/{args.direction}{mode_suffix}'
    os.makedirs(args.results_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Config: model={args.model}, layer={args.layer}, direction={args.direction}, method={args.method}, steer_mode={args.steer_mode}")
    if args.method == 'additive':
        print(f"Magnitudes: {magnitudes}")
    else:
        print(f"Angles: {angles}")
    print(f"Tokens per generation: {args.n_tokens}")

    # Load model
    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    load_dtype = dtype_map[args.dtype]
    print(f"\nLoading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)

    # Get steering direction
    if args.direction == 'mean':
        # Compute difference-of-means directly (no SVD needed)
        mean_dir, mean_norm = extract_pc1(model, tokenizer, device, args.layer, skip_svd=True)
        steer_unit = mean_dir
        dir_label = 'Diff-of-means'
    elif args.direction == 'pc1':
        pc1, mean_dir, mean_norm = extract_pc1(model, tokenizer, device, args.layer)
        steer_unit = pc1
        dir_label = 'PC1'
    else:
        # Load from MELBO results
        melbo_path = args.melbo_results
        if melbo_path is None:
            melbo_path = f'results/melbo_constrained/{args.model}/results.pkl'
        print(f"Loading MELBO results from {melbo_path}")
        with open(melbo_path, 'rb') as f:
            melbo_data = pickle.load(f)

        # Also extract mean_norm for the magnitude table
        _, mean_dir, mean_norm = extract_pc1(model, tokenizer, device, args.layer)

        melbo_idx = int(args.direction[-1]) - 1  # melbo1 -> 0, melbo2 -> 1, etc.
        steer_unit = melbo_data['melbo_directions'][melbo_idx]
        dir_label = f'MELBO_{melbo_idx + 1}'

        # Ensure points English-ward (positive cosine with mean diff)
        if (steer_unit @ mean_dir) < 0:
            steer_unit = -steer_unit
        print(f"  {dir_label} loaded, cos(dir, mean)={steer_unit @ mean_dir:.3f}")

    # Chinese-ward direction (negate, since directions point English-ward)
    zh_direction = -steer_unit

    print(f"\nSteering direction: {dir_label} (negated -> Chinese-ward)")
    print(f"Mean activation norm: {mean_norm:.2f}")

    # Select prompts
    prompts = ENGLISH_PROMPTS[:args.n_prompts]

    mode_desc = 'last-token only' if args.steer_mode == 'last_token' else 'rolling window'
    all_results = []

    if args.method == 'additive':
        # Show magnitude → equivalent angle mapping
        print(f"\nMagnitude -> equivalent angle (R={mean_norm:.1f}):")
        for mag in magnitudes:
            if mag > 0:
                equiv_angle = np.degrees(np.arctan(mag / mean_norm))
                print(f"  mag {mag:5.1f} -> angle ~ {equiv_angle:.1f} deg")

        print(f"\n{'='*70}")
        print(f"ADDITIVE STEERING TOWARD CHINESE ({dir_label}, {mode_desc})")
        print(f"Layer {args.layer}, {args.n_tokens} tokens, greedy decoding")
        print(f"{'='*70}")

        for prompt in prompts:
            print(f"\nPrompt: \"{prompt}\"")
            for mag in magnitudes:
                text, ids = generate_with_additive_steering(
                    model, tokenizer, prompt, zh_direction, mag,
                    args.layer, device, args.n_tokens,
                    steer_mode=args.steer_mode
                )
                label = f"mag {mag:5.1f}"
                print(f"  {label}: {text[:120]}")

                all_results.append({
                    'prompt': prompt,
                    'magnitude': mag,
                    'generated_text': text,
                    'generated_ids': ids,
                })

        # Plot with magnitude on x-axis
        plot_language_transition(all_results, magnitudes, args.results_dir,
                                 args.model, args.layer, dir_label=dir_label,
                                 x_key='magnitude', x_label='Steering magnitude')

        results_data = {
            'args': vars(args),
            'direction_name': args.direction,
            'direction_label': dir_label,
            'magnitudes': magnitudes,
            'mean_norm': mean_norm,
            'steer_unit': steer_unit.cpu(),
            'results': all_results,
        }
    else:
        # Spherical steering
        print(f"\nAngle -> equivalent additive magnitude (R={mean_norm:.1f}):")
        for ang in angles:
            if ang > 0:
                equiv_mag = mean_norm * np.tan(np.radians(ang))
                print(f"  {ang:5.1f} deg -> mag ~ {equiv_mag:.1f}")

        print(f"\n{'='*70}")
        print(f"SPHERICAL STEERING TOWARD CHINESE ({dir_label}, {mode_desc})")
        print(f"Layer {args.layer}, {args.n_tokens} tokens, greedy decoding")
        print(f"{'='*70}")

        for prompt in prompts:
            print(f"\nPrompt: \"{prompt}\"")
            for angle in angles:
                text, ids = generate_with_spherical_steering(
                    model, tokenizer, prompt, zh_direction, angle,
                    args.layer, device, args.n_tokens,
                    steer_mode=args.steer_mode
                )
                label = f"{angle:5.1f} deg"
                print(f"  {label}: {text[:120]}")

                all_results.append({
                    'prompt': prompt,
                    'angle_deg': angle,
                    'generated_text': text,
                    'generated_ids': ids,
                })

        plot_language_transition(all_results, angles, args.results_dir,
                                 args.model, args.layer, dir_label=dir_label)

        results_data = {
            'args': vars(args),
            'direction_name': args.direction,
            'direction_label': dir_label,
            'angles': angles,
            'mean_norm': mean_norm,
            'steer_unit': steer_unit.cpu(),
            'results': all_results,
        }

    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\nSaved results to {results_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
