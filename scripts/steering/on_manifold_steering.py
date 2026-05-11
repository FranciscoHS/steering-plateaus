#!/usr/bin/env python3
"""On-manifold steering: compare raw vs autoencoder-projected spherical steering.

Steers English prompts toward Mandarin using a contrastive direction at layer 2
of Qwen3-1.7B. Compares two approaches:
  1. Raw spherical steering (norm-preserving rotation)
  2. On-manifold steering: steer then project through trained autoencoder

Measures:
  - Effectiveness: CJK character fraction
  - Coherence: perplexity of generated text under unsteered model

Usage:
    python -u scripts/steering/on_manifold_steering.py

    # With pre-trained AE:
    python -u scripts/steering/on_manifold_steering.py \
        --ae_weights results/on_manifold_steering/qwen3-1.7b_layer2/ae_weights.pt

    # Custom angles:
    python -u scripts/steering/on_manifold_steering.py --angles 0,2,4,6,8,10,12,14,16,18,20
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
from src.data import load_fineweb_fixed_length
from src.autoencoders import Autoencoder
from data.bilingual_prompts_hq import PROMPT_PAIRS


# ── Contrastive direction ────────────────────────────────────────────────

def compute_contrastive_direction(model, tokenizer, device, layer):
    """Compute Mandarin - English mean-difference direction at given layer.

    Returns unit vector pointing toward Mandarin activations.
    """
    en_prompts = [p[0] for p in PROMPT_PAIRS]
    zh_prompts = [p[1] for p in PROMPT_PAIRS]

    def get_acts(prompts):
        acts = []
        for p in prompts:
            toks = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**toks, output_hidden_states=True)
            acts.append(out.hidden_states[layer + 1][:, -1, :].squeeze(0))
        return torch.stack(acts).float().cpu()

    print(f"Computing contrastive direction at layer {layer} ({len(PROMPT_PAIRS)} pairs)...")
    en_acts = get_acts(en_prompts)
    zh_acts = get_acts(zh_prompts)

    diff = (zh_acts - en_acts).mean(dim=0)
    diff_norm = diff.norm().item()
    direction = F.normalize(diff, dim=0)

    print(f"  ||mean diff|| = {diff_norm:.2f}")
    return direction


# ── AE training ──────────────────────────────────────────────────────────

def cache_activations(model, tokenizer, device, layer, n_samples=50000,
                      seq_len=10, seed=42):
    """Cache last-token activations at given layer from FineWeb."""
    print(f"Caching {n_samples} activations at layer {layer}...")
    token_ids_list = load_fineweb_fixed_length(
        n_samples, tokenizer, seq_len=seq_len, seed=seed)

    all_acts = []
    batch_size = 64
    t0 = time.time()
    for i in range(0, n_samples, batch_size):
        batch = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_acts.append(hidden[:, -1, :].float().cpu())
        if i % (batch_size * 100) == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  {i}/{n_samples} ({rate:.0f} samples/s)")
    print(f"  Done in {time.time() - t0:.0f}s")

    acts = torch.cat(all_acts, dim=0)
    print(f"  Shape: {acts.shape}, mean norm={acts.norm(dim=-1).mean():.2f}")
    return acts


def train_ae(activations, d_model, k=30, hidden_dims=None, n_steps=10000,
             batch_size=4096, lr=1e-3, device='cuda'):
    """Train autoencoder on cached activations.

    Uses best config from hillclimb: GELU + batchnorm, warmup 500,
    mse + 5*cosine loss.
    """
    ae = Autoencoder(d_model, k, hidden_dims).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"\nTraining AE: {n_params / 1e6:.1f}M params, k={k}, {n_steps} steps")

    activations_gpu = activations.to(device)
    n_cache = len(activations_gpu)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-7 / lr, total_iters=500)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps - 500)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[500])

    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, n_cache, (batch_size,), device=device)
        batch = activations_gpu[idx]

        recon, z = ae(batch)
        mse = F.mse_loss(recon, batch)
        cos_loss = 1.0 - F.cosine_similarity(recon, batch, dim=-1).mean()
        loss = mse + 5.0 * cos_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 2000 == 0:
            with torch.no_grad():
                cos = F.cosine_similarity(batch, recon, dim=-1).mean()
            print(f"  Step {step:5d}: loss={loss.item():.6f}, cos={cos.item():.4f}")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s")

    # Final eval
    ae.eval()
    with torch.no_grad():
        eval_batch = activations_gpu[:10000]
        recon, _ = ae(eval_batch)
        cos = F.cosine_similarity(eval_batch, recon, dim=-1).mean().item()
        rel_l2 = ((eval_batch - recon).norm(dim=-1) /
                  eval_batch.norm(dim=-1)).mean().item()
    print(f"  Final eval: cos={cos:.4f}, rel_L2={rel_l2:.4f}")

    return ae


# ── Generation ───────────────────────────────────────────────────────────

def generate_with_steering(model, tokenizer, prompt, direction, angle_deg,
                           layer, device, n_tokens=30, ae=None):
    """Generate with spherical steering, optionally projecting through AE.

    At each generation step, hooks apply:
      1. Spherical rotation toward `direction` by `angle_deg`
      2. If ae is provided, pass steered activations through encode→decode

    Steers only the last (current) token position at each generation step.
    """
    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    dir_gpu = direction.to(device)

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)

    generated_ids = []

    for step in range(n_tokens):
        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            act = hidden[0, -1, :]  # [d] last token only

            # 1. Spherical rotation
            R = act.norm()
            a_unit = act / (R + 1e-10)
            proj = (a_unit * dir_gpu).sum()
            v_perp = dir_gpu - proj * a_unit
            v_perp = v_perp / (v_perp.norm() + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)

            # 2. Optional AE projection
            if ae is not None:
                recon, _ = ae(steered.unsqueeze(0).float())
                steered = recon.squeeze(0).to(hidden.dtype)

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


# ── Metrics ──────────────────────────────────────────────────────────────

def cjk_fraction(text):
    """Fraction of non-whitespace/punctuation characters that are CJK."""
    if not text:
        return 0.0
    n_cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    n_total = sum(1 for ch in text
                  if not ch.isspace() and ch not in '.,;:!?()[]{}"\'-/')
    return n_cjk / max(n_total, 1)


def compute_coherence(model, tokenizer, prompt, generated_ids, device):
    """Coherence via perplexity: run generated tokens through unsteered model.

    Computes cross-entropy on generated tokens conditioned on the prompt.
    Lower = more coherent.
    """
    if len(generated_ids) < 2:
        return float('inf')

    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    gen_ids = torch.tensor([generated_ids])
    all_ids = torch.cat([prompt_ids, gen_ids], dim=1).to(device)

    with torch.no_grad():
        outputs = model(all_ids)

    prompt_len = prompt_ids.shape[1]
    # Predictions for generated tokens
    logits = outputs.logits[0, prompt_len - 1:-1]
    labels = all_ids[0, prompt_len:]
    return F.cross_entropy(logits, labels, reduction='mean').item()


# ── Plotting ─────────────────────────────────────────────────────────────

def find_breakdown_angle(angles, median_ppls, threshold_mult=3.0):
    """Find first angle where median perplexity exceeds threshold × baseline."""
    baseline = median_ppls[0]
    if baseline <= 0 or not np.isfinite(baseline):
        return None
    threshold = baseline * threshold_mult
    for angle, ppl in zip(angles, median_ppls):
        if ppl > threshold:
            return angle
    return None


def plot_results(results, angles, output_dir, model_name, layer):
    """Plot effectiveness (CJK fraction) vs steering angle for both methods.

    Dashed vertical lines mark coherence breakdown regions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [
        ('raw', '#2196F3', 'Raw steering'),
        ('ae',  '#E91E63', 'On-manifold (AE)'),
    ]

    for method, color, label in configs:
        fracs = results[method]['cjk_fracs']  # [n_angles, n_prompts]
        median_frac = np.median(fracs, axis=1)
        q25 = np.percentile(fracs, 25, axis=1)
        q75 = np.percentile(fracs, 75, axis=1)

        ax.plot(angles, median_frac, '-o', color=color, label=label,
                linewidth=2, markersize=4)
        ax.fill_between(angles, q25, q75, color=color, alpha=0.1)

        # Coherence breakdown vertical line
        med_ppl = np.median(results[method]['perplexities'], axis=1)
        breakdown = find_breakdown_angle(angles, med_ppl)
        if breakdown is not None:
            ax.axvline(breakdown, color=color, linestyle='--', alpha=0.7,
                       label=f'{label} coherence breakdown ({breakdown}\u00b0)')
            ax.axvspan(breakdown, max(angles), color=color, alpha=0.04)

    ax.set_xlabel('Steering angle (degrees)', fontsize=12)
    ax.set_ylabel('CJK character fraction (effectiveness)', fontsize=12)
    ax.set_title(f'On-manifold vs raw spherical steering\n'
                 f'{model_name}, layer {layer}, EN\u2192ZH contrastive',
                 fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(angles[0], angles[-1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'on_manifold_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {path}")


def save_generations(results, angles, prompts, output_dir):
    """Save sample generations to a text file for inspection."""
    path = os.path.join(output_dir, 'sample_generations.txt')
    with open(path, 'w') as f:
        for ai, angle in enumerate(angles):
            f.write(f"\n{'='*70}\n")
            f.write(f"ANGLE: {angle} degrees\n")
            f.write(f"{'='*70}\n")
            for pi, prompt in enumerate(prompts):
                f.write(f"\nPrompt: \"{prompt}\"\n")
                raw_text = results['raw']['texts'][ai][pi]
                ae_text = results['ae']['texts'][ai][pi]
                raw_cjk = results['raw']['cjk_fracs'][ai][pi]
                ae_cjk = results['ae']['cjk_fracs'][ai][pi]
                raw_ppl = results['raw']['perplexities'][ai][pi]
                ae_ppl = results['ae']['perplexities'][ai][pi]
                f.write(f"  Raw   (CJK={raw_cjk:.2f}, ppl={raw_ppl:.2f}): "
                        f"{raw_text[:150]}\n")
                f.write(f"  AE    (CJK={ae_cjk:.2f}, ppl={ae_ppl:.2f}): "
                        f"{ae_text[:150]}\n")
    print(f"Generations saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='On-manifold steering: raw vs AE-projected')
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--ae_weights', default=None,
                        help='Path to pre-trained AE weights (trains new if absent)')
    parser.add_argument('--ae_k', type=int, default=30,
                        help='AE bottleneck dimension')
    parser.add_argument('--ae_hidden_dims', type=int, nargs='+', default=None,
                        help='AE hidden dims (default: 4096 4096 2048)')
    parser.add_argument('--ae_steps', type=int, default=10000)
    parser.add_argument('--ae_cache_size', type=int, default=50000)
    parser.add_argument('--angles', type=str, default=None,
                        help='Comma-separated angles in degrees (default: 0-25)')
    parser.add_argument('--n_tokens', type=int, default=30,
                        help='Tokens to generate per prompt')
    parser.add_argument('--n_prompts', type=int, default=5)
    parser.add_argument('--output_dir',
                        default='results/on_manifold_steering/qwen3-1.7b_layer2')
    parser.add_argument('--dtype', default='bfloat16',
                        choices=['float32', 'bfloat16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.angles:
        angles = [float(a) for a in args.angles.split(',')]
    else:
        angles = list(range(0, 26))

    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16}

    # ── 1. Load model ────────────────────────────────────────────────────
    print(f"Loading {args.model} ({args.dtype})...")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    print(f"  d_model={d_model}, device={device}")

    # ── 2. Contrastive direction ─────────────────────────────────────────
    direction = compute_contrastive_direction(
        model, tokenizer, device, args.layer)

    # ── 3. Train or load AE ──────────────────────────────────────────────
    if args.ae_weights and os.path.exists(args.ae_weights):
        print(f"\nLoading AE from {args.ae_weights}...")
        checkpoint = torch.load(
            args.ae_weights, weights_only=True, map_location=device)
        config = checkpoint['config']
        ae = Autoencoder(
            config['d_in'], config['k'],
            config.get('hidden_dims')).to(device)
        ae.load_state_dict(checkpoint['state_dict'])
        ae.eval()
        print(f"  Loaded: d_in={config['d_in']}, k={config['k']}")
    else:
        print("\nNo pre-trained AE found, training from scratch...")
        activations = cache_activations(
            model, tokenizer, device, args.layer,
            n_samples=args.ae_cache_size)

        hidden_dims = args.ae_hidden_dims or [4096, 4096, 2048]
        ae = train_ae(activations, d_model, k=args.ae_k,
                      hidden_dims=hidden_dims, n_steps=args.ae_steps,
                      device=device)

        # Save AE weights
        ae_path = os.path.join(args.output_dir, 'ae_weights.pt')
        torch.save({
            'state_dict': ae.state_dict(),
            'config': {'d_in': d_model, 'k': args.ae_k,
                       'hidden_dims': hidden_dims},
        }, ae_path)
        print(f"  Saved AE: {ae_path}")

        del activations
        torch.cuda.empty_cache()

    ae.eval()

    # ── 4. Select prompts ────────────────────────────────────────────────
    en_prompts = [p[0] for p in PROMPT_PAIRS][:args.n_prompts]
    print(f"\nPrompts ({len(en_prompts)}):")
    for p in en_prompts:
        print(f"  \"{p}\"")

    # ── 5. Run generation ────────────────────────────────────────────────
    results = {
        method: {'texts': [], 'cjk_fracs': [], 'perplexities': [], 'ids': []}
        for method in ['raw', 'ae']
    }

    total = len(angles) * len(en_prompts) * 2
    done = 0
    t_start = time.time()

    print(f"\nRunning: {len(angles)} angles × {len(en_prompts)} prompts × 2 methods "
          f"= {total} generations")
    print(f"{'='*70}")

    for ai, angle in enumerate(angles):
        angle_results = {m: {'fracs': [], 'ppls': [], 'texts': [], 'ids': []}
                         for m in ['raw', 'ae']}

        for pi, prompt in enumerate(en_prompts):
            for method, ae_arg in [('raw', None), ('ae', ae)]:
                text, ids = generate_with_steering(
                    model, tokenizer, prompt, direction, angle,
                    args.layer, device, args.n_tokens, ae=ae_arg)

                frac = cjk_fraction(text)
                ppl = compute_coherence(model, tokenizer, prompt, ids, device)

                angle_results[method]['fracs'].append(frac)
                angle_results[method]['ppls'].append(ppl)
                angle_results[method]['texts'].append(text)
                angle_results[method]['ids'].append(ids)
                done += 1

        for method in ['raw', 'ae']:
            results[method]['cjk_fracs'].append(angle_results[method]['fracs'])
            results[method]['perplexities'].append(angle_results[method]['ppls'])
            results[method]['texts'].append(angle_results[method]['texts'])
            results[method]['ids'].append(angle_results[method]['ids'])

        # Progress
        med_raw = np.median(angle_results['raw']['fracs'])
        med_ae = np.median(angle_results['ae']['fracs'])
        ppl_raw = np.median(angle_results['raw']['ppls'])
        ppl_ae = np.median(angle_results['ae']['ppls'])
        elapsed = time.time() - t_start
        eta = elapsed / done * (total - done) if done > 0 else 0
        print(f"  {angle:5.1f}\u00b0: raw CJK={med_raw:.2f} ppl={ppl_raw:.1f} | "
              f"ae CJK={med_ae:.2f} ppl={ppl_ae:.1f}  "
              f"[{done}/{total}, ETA {eta:.0f}s]")

    total_time = time.time() - t_start
    print(f"\nGeneration done in {total_time:.0f}s")

    # Convert to numpy arrays
    for method in ['raw', 'ae']:
        results[method]['cjk_fracs'] = np.array(results[method]['cjk_fracs'])
        results[method]['perplexities'] = np.array(results[method]['perplexities'])

    # ── 6. Plot ──────────────────────────────────────────────────────────
    plot_results(results, angles, args.output_dir, args.model, args.layer)

    # ── 7. Save ──────────────────────────────────────────────────────────
    save_generations(results, angles, en_prompts, args.output_dir)

    save_data = {
        'args': vars(args),
        'angles': angles,
        'direction': direction.cpu(),
        'prompts': en_prompts,
        'results': {
            method: {
                'cjk_fracs': results[method]['cjk_fracs'].tolist(),
                'perplexities': results[method]['perplexities'].tolist(),
                'texts': results[method]['texts'],
            }
            for method in ['raw', 'ae']
        },
    }
    results_path = os.path.join(args.output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Results saved: {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()
