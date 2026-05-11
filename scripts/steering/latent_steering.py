#!/usr/bin/env python3
"""Latent-space steering: steer in the AE's 30-dim bottleneck space.

Compares:
  1. Raw: spherical steering in activation space using raw contrastive direction
  2. Latent: encode activation → add latent contrastive vector → decode back

Usage:
    python -u scripts/steering/latent_steering.py \
        --ae_weights results/on_manifold_steering/qwen3-1.7b_layer2/ae_weights.pt
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.autoencoders import Autoencoder
PROMPT_SETS = {
    'language': 'data.bilingual_prompts_hq',
    'gender': 'data.gender_prompts',
}


# ── Activation extraction ────────────────────────────────────────────────

def get_paired_acts(model, tokenizer, device, layer, prompt_pairs):
    """Extract last-token activations for source and target prompts."""
    def get_acts(prompts):
        acts = []
        for p in prompts:
            toks = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**toks, output_hidden_states=True)
            acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
        return torch.stack(acts).float().cpu()

    src_prompts = [p[0] for p in prompt_pairs]
    tgt_prompts = [p[1] for p in prompt_pairs]
    print(f"Extracting paired activations at layer {layer} ({len(prompt_pairs)} pairs)...")

    src_acts = get_acts(src_prompts)
    tgt_acts = get_acts(tgt_prompts)
    print(f"  Source: {src_acts.shape}, Target: {tgt_acts.shape}")

    return src_acts, tgt_acts


# ── Direction computation ────────────────────────────────────────────────

def compute_raw_direction(src_acts, tgt_acts):
    """Standard contrastive direction in activation space."""
    diff = (tgt_acts - src_acts).mean(dim=0)
    direction = F.normalize(diff, dim=0)
    print(f"  Raw direction: ||mean diff|| = {diff.norm():.2f}")
    return direction


def compute_latent_direction(src_acts, tgt_acts, ae, device):
    """Contrastive direction in AE latent space (30-dim).

    Returns the normalized mean difference in latent space.
    """
    ae.eval()
    with torch.no_grad():
        z_src = ae.encode(src_acts.to(device))  # [n, k]
        z_tgt = ae.encode(tgt_acts.to(device))  # [n, k]

    z_diff = (z_tgt - z_src).mean(dim=0).cpu()  # [k]
    z_dir = F.normalize(z_diff, dim=0)
    print(f"  Latent direction: ||z_diff|| = {z_diff.norm():.2f}, dim = {z_diff.shape[0]}")
    return z_dir


# ── Generation ───────────────────────────────────────────────────────────

def generate_with_spherical_steering(model, tokenizer, prompt, direction,
                                     angle_deg, layer, device, n_tokens=30):
    """Raw spherical steering on last token only."""
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    dir_gpu = direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            act = hidden[0, -1, :]

            R = act.norm()
            a_unit = act / (R + 1e-10)
            proj = (a_unit * dir_gpu).sum()
            v_perp = dir_gpu - proj * a_unit
            v_perp = v_perp / (v_perp.norm() + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)

            hidden = hidden.clone()
            hidden[0, -1, :] = steered
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        next_id = outputs.logits[0, -1, :].argmax().item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


def generate_with_latent_steering(model, tokenizer, prompt, latent_dir,
                                  angle_deg, layer, device, ae, n_tokens=30):
    """Steer in AE latent space: encode → spherical rotate → decode.

    At each step:
      1. Get last-token activation at layer
      2. Encode to 30-dim
      3. Spherical rotation in 30-dim toward latent_dir
      4. Decode back to 2048-dim
      5. Replace activation
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    latent_dir_gpu = latent_dir.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            act = hidden[0, -1, :].float()  # [d]

            z = ae.encode(act.unsqueeze(0)).squeeze(0)  # [k]

            # Spherical rotation in latent space
            R = z.norm()
            z_unit = z / (R + 1e-10)
            proj = (z_unit * latent_dir_gpu).sum()
            v_perp = latent_dir_gpu - proj * z_unit
            v_perp = v_perp / (v_perp.norm() + 1e-10)
            z_steered = R * (cos_theta * z_unit + sin_theta * v_perp)

            recon = ae.decode(z_steered.unsqueeze(0))  # [1, d]

            hidden = hidden.clone()
            hidden[0, -1, :] = recon.squeeze(0).to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        next_id = outputs.logits[0, -1, :].argmax().item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


# ── Metrics ──────────────────────────────────────────────────────────────

def cjk_fraction(text):
    if not text:
        return 0.0
    n_cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    n_total = sum(1 for ch in text
                  if not ch.isspace() and ch not in '.,;:!?()[]{}"\'-/')
    return n_cjk / max(n_total, 1)


FEMALE_WORDS = {'she', 'her', 'hers', 'herself', 'woman', 'women', 'girl',
                'mother', 'daughter', 'sister', 'wife', 'queen', 'princess',
                'actress', 'waitress', 'hostess', 'heroine', 'bride', 'aunt',
                'grandmother', 'niece', 'lady', 'female', 'mrs', 'ms'}
MALE_WORDS = {'he', 'him', 'his', 'himself', 'man', 'men', 'boy',
              'father', 'son', 'brother', 'husband', 'king', 'prince',
              'actor', 'waiter', 'host', 'hero', 'groom', 'uncle',
              'grandfather', 'nephew', 'lord', 'gentleman', 'male', 'mr'}


def female_fraction(text):
    """Fraction of gendered words that are female."""
    if not text:
        return 0.5
    words = text.lower().split()
    n_f = sum(1 for w in words if w.strip('.,;:!?()') in FEMALE_WORDS)
    n_m = sum(1 for w in words if w.strip('.,;:!?()') in MALE_WORDS)
    total = n_f + n_m
    return n_f / total if total > 0 else 0.5


def compute_coherence(model, tokenizer, prompt, generated_ids, device):
    if len(generated_ids) < 2:
        return float('inf')
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    gen_ids = torch.tensor([generated_ids])
    all_ids = torch.cat([prompt_ids, gen_ids], dim=1).to(device)
    with torch.no_grad():
        outputs = model(all_ids)
    prompt_len = prompt_ids.shape[1]
    logits = outputs.logits[0, prompt_len - 1:-1]
    labels = all_ids[0, prompt_len:]
    return F.cross_entropy(logits, labels, reduction='mean').item()


# ── Plotting ─────────────────────────────────────────────────────────────

def find_breakdown(x_values, median_ppls, threshold_mult=3.0):
    baseline = median_ppls[0]
    if baseline <= 0 or not np.isfinite(baseline):
        return None
    threshold = baseline * threshold_mult
    for x, ppl in zip(x_values, median_ppls):
        if ppl > threshold:
            return x
    return None


def plot_results(results, angles, output_dir, model_name, layer, metric_name):
    """Plot effectiveness vs steering angle for both methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [
        ('raw', '#2196F3', 'Raw spherical'),
        ('latent', '#E91E63', 'Latent spherical (encode\u2192rotate\u2192decode)'),
    ]

    for method, color, label in configs:
        fracs = results[method]['effect_fracs']
        median_frac = np.median(fracs, axis=1)
        q25 = np.percentile(fracs, 25, axis=1)
        q75 = np.percentile(fracs, 75, axis=1)

        ax.plot(angles, median_frac, '-o', color=color, label=label,
                linewidth=2, markersize=4)
        ax.fill_between(angles, q25, q75, color=color, alpha=0.1)

        med_ppl = np.median(results[method]['perplexities'], axis=1)
        bd = find_breakdown(angles, med_ppl)
        if bd is not None:
            ax.axvline(bd, color=color, linestyle='--', alpha=0.7,
                       label=f'{label} coherence breakdown ({bd}\u00b0)')

    ax.set_xlabel('Steering angle (degrees)', fontsize=12)
    ax.set_ylabel(f'{metric_name} (effectiveness)', fontsize=12)
    ax.set_title(f'Raw vs latent-space spherical steering\n'
                 f'{model_name}, layer {layer}', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(angles[0], angles[-1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'latent_vs_raw_steering.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {path}")


def save_generations(results, angles, prompts, output_dir, metric_label):
    path = os.path.join(output_dir, 'sample_generations.txt')
    with open(path, 'w') as f:
        for si, angle in enumerate(angles):
            f.write(f"\n{'='*70}\n")
            f.write(f"ANGLE: {angle} degrees\n")
            f.write(f"{'='*70}\n")
            for pi, prompt in enumerate(prompts):
                f.write(f"\nPrompt: \"{prompt}\"\n")
                for method, label in [('raw', 'Raw'), ('latent', 'Latent')]:
                    text = results[method]['texts'][si][pi]
                    frac = results[method]['effect_fracs'][si][pi]
                    ppl = results[method]['perplexities'][si][pi]
                    f.write(f"  {label:7s} ({metric_label}={frac:.2f}, ppl={ppl:.2f}): "
                            f"{text[:150]}\n")
    print(f"Generations saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--ae_weights', required=True)
    parser.add_argument('--angles', type=str, default=None,
                        help='Comma-separated angles for raw (default: 0-60 step 2)')
    parser.add_argument('--n_tokens', type=int, default=30)
    parser.add_argument('--n_prompts', type=int, default=5)
    parser.add_argument('--prompts', default='language',
                        choices=list(PROMPT_SETS.keys()),
                        help='Which prompt pairs to use')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Import prompt pairs
    import importlib
    prompt_mod = importlib.import_module(PROMPT_SETS[args.prompts])
    prompt_pairs = prompt_mod.PROMPT_PAIRS

    if args.output_dir is None:
        args.output_dir = f'results/latent_steering/qwen3-1.7b_layer2_{args.prompts}'
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Metric setup
    if args.prompts == 'gender':
        effect_metric = female_fraction
        metric_name = 'Female word fraction'
        metric_label = 'FEM'
    else:
        effect_metric = cjk_fraction
        metric_name = 'CJK character fraction'
        metric_label = 'CJK'

    if args.angles:
        angles = [float(a) for a in args.angles.split(',')]
    else:
        angles = list(range(0, 61, 2))

    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16}

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    # Load AE
    print(f"Loading AE from {args.ae_weights}...")
    checkpoint = torch.load(args.ae_weights, weights_only=True, map_location=device)
    config = checkpoint['config']
    ae = Autoencoder(config['d_in'], config['k'], config.get('hidden_dims')).to(device)
    ae.load_state_dict(checkpoint['state_dict'])
    ae.eval()
    print(f"  d_in={config['d_in']}, k={config['k']}")

    # Get paired activations (source=male/EN, target=female/ZH)
    src_acts, tgt_acts = get_paired_acts(model, tokenizer, device, args.layer, prompt_pairs)

    # Compute directions
    print("\nComputing directions:")
    raw_dir = compute_raw_direction(src_acts, tgt_acts)
    latent_dir = compute_latent_direction(src_acts, tgt_acts, ae, device)

    # Select prompts — use source prompts for generation
    src_prompts = [p[0] for p in prompt_pairs][:args.n_prompts]
    print(f"\nPrompts ({len(src_prompts)}):")
    for p in src_prompts:
        print(f"  \"{p}\"")

    # Run generation
    methods = ['raw', 'latent']
    results = {
        m: {'texts': [], 'effect_fracs': [], 'perplexities': [], 'ids': []}
        for m in methods
    }

    total = len(angles) * len(src_prompts) * 2
    done = 0
    t_start = time.time()

    print(f"\nRunning: {len(angles)} steps × {len(src_prompts)} prompts × 2 methods "
          f"= {total} generations")
    print(f"{'='*70}")

    for ai, angle in enumerate(angles):
        step_results = {m: {'fracs': [], 'ppls': [], 'texts': [], 'ids': []}
                        for m in methods}

        for pi, prompt in enumerate(src_prompts):
            # Raw: spherical steering in activation space
            text, ids = generate_with_spherical_steering(
                model, tokenizer, prompt, raw_dir, angle,
                args.layer, device, args.n_tokens)
            frac = effect_metric(text)
            ppl = compute_coherence(model, tokenizer, prompt, ids, device)
            step_results['raw']['fracs'].append(frac)
            step_results['raw']['ppls'].append(ppl)
            step_results['raw']['texts'].append(text)
            step_results['raw']['ids'].append(ids)
            done += 1

            # Latent: encode → spherical rotate in 30-dim → decode
            text, ids = generate_with_latent_steering(
                model, tokenizer, prompt, latent_dir, angle,
                args.layer, device, ae, args.n_tokens)
            frac = effect_metric(text)
            ppl = compute_coherence(model, tokenizer, prompt, ids, device)
            step_results['latent']['fracs'].append(frac)
            step_results['latent']['ppls'].append(ppl)
            step_results['latent']['texts'].append(text)
            step_results['latent']['ids'].append(ids)
            done += 1

        for m in methods:
            results[m]['effect_fracs'].append(step_results[m]['fracs'])
            results[m]['perplexities'].append(step_results[m]['ppls'])
            results[m]['texts'].append(step_results[m]['texts'])
            results[m]['ids'].append(step_results[m]['ids'])

        med_raw = np.median(step_results['raw']['fracs'])
        med_lat = np.median(step_results['latent']['fracs'])
        ppl_raw = np.median(step_results['raw']['ppls'])
        ppl_lat = np.median(step_results['latent']['ppls'])
        elapsed = time.time() - t_start
        eta = elapsed / done * (total - done) if done > 0 else 0
        print(f"  {angle:5.1f}\u00b0: raw {metric_label}={med_raw:.2f} ppl={ppl_raw:.1f} | "
              f"latent {metric_label}={med_lat:.2f} ppl={ppl_lat:.1f}  "
              f"[{done}/{total}, ETA {eta:.0f}s]")

    total_time = time.time() - t_start
    print(f"\nGeneration done in {total_time:.0f}s")

    for m in methods:
        results[m]['effect_fracs'] = np.array(results[m]['effect_fracs'])
        results[m]['perplexities'] = np.array(results[m]['perplexities'])

    # Plot
    plot_results(results, angles, args.output_dir, args.model, args.layer, metric_name)

    # Save
    save_generations(results, angles, src_prompts, args.output_dir, metric_label)

    save_data = {
        'args': vars(args),
        'angles': angles,
        'raw_direction': raw_dir.cpu(),
        'latent_direction': latent_dir.cpu(),
        'prompts': src_prompts,
        'results': {
            m: {
                'effect_fracs': results[m]['effect_fracs'].tolist(),
                'perplexities': results[m]['perplexities'].tolist(),
                'texts': results[m]['texts'],
            }
            for m in methods
        },
    }
    results_path = os.path.join(args.output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Results saved: {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()
