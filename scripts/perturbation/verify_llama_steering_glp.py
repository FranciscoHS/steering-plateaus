#!/usr/bin/env python3
"""Verify whether GLP on-manifold projection improves steering coherence.

Generates text with English→Chinese steering at various angles,
with and without GLP projection after steering. Compares:
- % Mandarin characters in output
- Qualitative coherence

Usage:
    python scripts/perturbation/verify_llama_steering_glp.py
"""

import os, sys, unicodedata, pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Patch for torch < 2.5
if not hasattr(nn, 'Buffer'):
    nn.Buffer = lambda x: nn.Parameter(x, requires_grad=False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS
from glp.denoiser import load_glp
from glp import flow_matching

LAYER = 7
STEER_ANGLES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
TEST_PROMPTS = ENGLISH_PROMPTS[:10]
MAX_NEW_TOKENS = 50
GLP_U = 0.5
GLP_TIMESTEPS = 50
RESULTS_DIR = 'results/llama_plateau'


def is_cjk(char):
    try:
        return 'CJK' in unicodedata.name(char, '')
    except ValueError:
        return False


def pct_mandarin(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return 100.0 * sum(1 for c in chars if is_cjk(c)) / len(chars)


def project_on_manifold(glp_model, activations, u=0.5, num_timesteps=50):
    """Project activations onto learned manifold via GLP denoising.

    Args:
        glp_model: loaded GLP model
        activations: [..., d_model] tensor (any leading dims)
        u: noise level
        num_timesteps: diffusion steps

    Returns:
        projected: same shape as activations
    """
    orig_shape = activations.shape
    orig_dtype = activations.dtype
    device = activations.device

    # GLP expects [batch, 1, d_model]
    flat = activations.reshape(-1, orig_shape[-1]).float().unsqueeze(1)

    latents = glp_model.normalizer.normalize(flat)
    noise = torch.randn_like(latents)
    noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
        glp_model.scheduler, latents, noise,
        u=torch.ones(flat.shape[0]) * u,
    )
    projected = flow_matching.sample_on_manifold(
        glp_model, noisy_latents,
        start_timestep=timesteps[0].item(),
        num_timesteps=num_timesteps,
    )
    projected = glp_model.normalizer.denormalize(projected)
    projected = projected.squeeze(1).to(dtype=orig_dtype).reshape(orig_shape)
    return projected


def make_hook(steer_vec_unit, angle_rad, glp_model=None, glp_u=0.5, glp_timesteps=50):
    """Create a hook that steers (spherical rotation) and optionally projects via GLP."""

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
        else:
            hidden = output[0].clone()

        if angle_rad != 0:
            d = steer_vec_unit.to(hidden.device, hidden.dtype)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            R = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a_unit = hidden / R
            proj = (a_unit * d).sum(dim=-1, keepdim=True)
            v_perp = d.unsqueeze(0).unsqueeze(0) - proj * a_unit
            v_perp = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden = R * (cos_a * a_unit + sin_a * v_perp)

        if glp_model is not None:
            # Project each position through GLP
            orig_shape = hidden.shape
            flat = hidden.reshape(-1, hidden.shape[-1])
            with torch.no_grad():
                flat = project_on_manifold(
                    glp_model, flat,
                    u=glp_u, num_timesteps=glp_timesteps
                )
            hidden = flat.reshape(orig_shape)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            output[0] = hidden
            return output

    return hook


def run_generation(model, tokenizer, block, steer_vec_unit, angles, prompts,
                   glp_model=None, label=""):
    """Run generation at each angle, return dict of angle -> list of (pct, sample)."""
    results = {}
    for angle_deg in angles:
        angle_rad = np.deg2rad(angle_deg)
        hook = make_hook(steer_vec_unit, angle_rad, glp_model=glp_model,
                         glp_u=GLP_U, glp_timesteps=GLP_TIMESTEPS)
        handle = block.register_forward_hook(hook)

        pcts = []
        samples = []
        ppls = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors='pt').to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(**tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated = tokenizer.decode(out[0][tokens['input_ids'].shape[1]:], skip_special_tokens=True)
            pcts.append(pct_mandarin(generated))
            samples.append(generated)

            # Compute perplexity of generated text under the model (no hooks)
            handle.remove()
            gen_tokens = tokenizer(generated, return_tensors='pt').to(next(model.parameters()).device)
            if gen_tokens['input_ids'].shape[1] > 1:
                with torch.no_grad():
                    logits = model(**gen_tokens).logits
                # Shift: predict token i+1 from position i
                shift_logits = logits[:, :-1, :].float()
                shift_labels = gen_tokens['input_ids'][:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                )
                ppls.append(loss.exp().item())
            else:
                ppls.append(float('nan'))
            # Re-register hook for next prompt
            handle = block.register_forward_hook(hook)

        handle.remove()
        results[angle_deg] = {'pcts': pcts, 'samples': samples, 'ppls': ppls}
        mean_pct = np.mean(pcts)
        mean_ppl = np.nanmean(ppls)
        print(f"  [{label}] {angle_deg:3d}°: {mean_pct:5.1f}% ZH, ppl={mean_ppl:7.1f} | {samples[0][:70]}")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Llama-3.2-1B...")
    model, tokenizer, device = load_model('meta-llama/Llama-3.2-1B', device=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading GLP...")
    glp = load_glp('generative-latent-prior/glp-llama1b-d6', device=device, checkpoint='final')
    print("GLP loaded.")

    # Compute steering vector
    print("Computing steering vector...")
    en_acts, zh_acts = [], []
    for en_text, zh_text in zip(ENGLISH_PROMPTS, CHINESE_PROMPTS):
        for text, acts_list in [(en_text, en_acts), (zh_text, zh_acts)]:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts_list.append(outputs.hidden_states[LAYER + 1][:, -1, :].squeeze(0))

    en_mean = torch.stack(en_acts).mean(dim=0)
    zh_mean = torch.stack(zh_acts).mean(dim=0)
    steer_vec = zh_mean - en_mean
    steer_vec_unit = steer_vec / steer_vec.norm()
    print(f"Steering vector norm: {steer_vec.norm():.3f}")

    block = model.model.layers[LAYER]

    # Run without GLP
    print("\n=== Steering only (no GLP) ===")
    baseline_results = run_generation(
        model, tokenizer, block, steer_vec_unit, STEER_ANGLES, TEST_PROMPTS,
        glp_model=None, label="Baseline"
    )

    # Run with GLP
    print("\n=== Steering + GLP projection ===")
    glp_results = run_generation(
        model, tokenizer, block, steer_vec_unit, STEER_ANGLES, TEST_PROMPTS,
        glp_model=glp, label="GLP"
    )

    # Save
    save_data = {
        'steer_angles': STEER_ANGLES,
        'baseline': baseline_results,
        'glp': glp_results,
        'config': {
            'layer': LAYER,
            'glp_u': GLP_U,
            'glp_timesteps': GLP_TIMESTEPS,
            'n_prompts': len(TEST_PROMPTS),
            'max_new_tokens': MAX_NEW_TOKENS,
        }
    }
    pkl_path = os.path.join(RESULTS_DIR, 'steering_glp_comparison.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: 2 panels — % Mandarin and perplexity
    angles = np.array(STEER_ANGLES)

    b_pct_mean = np.array([np.mean(baseline_results[a]['pcts']) for a in STEER_ANGLES])
    b_pct_std = np.array([np.std(baseline_results[a]['pcts']) for a in STEER_ANGLES])
    g_pct_mean = np.array([np.mean(glp_results[a]['pcts']) for a in STEER_ANGLES])
    g_pct_std = np.array([np.std(glp_results[a]['pcts']) for a in STEER_ANGLES])

    b_ppl_mean = np.array([np.nanmean(baseline_results[a]['ppls']) for a in STEER_ANGLES])
    b_ppl_std = np.array([np.nanstd(baseline_results[a]['ppls']) for a in STEER_ANGLES])
    g_ppl_mean = np.array([np.nanmean(glp_results[a]['ppls']) for a in STEER_ANGLES])
    g_ppl_std = np.array([np.nanstd(glp_results[a]['ppls']) for a in STEER_ANGLES])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Panel 1: % Mandarin
    ax1.fill_between(angles, np.maximum(b_pct_mean - b_pct_std, 0),
                     np.minimum(b_pct_mean + b_pct_std, 100), color='C0', alpha=0.15)
    ax1.plot(angles, b_pct_mean, 'C0-o', lw=2, label='Steer only')
    ax1.fill_between(angles, np.maximum(g_pct_mean - g_pct_std, 0),
                     np.minimum(g_pct_mean + g_pct_std, 100), color='C1', alpha=0.15)
    ax1.plot(angles, g_pct_mean, 'C1-s', lw=2, label='Steer + GLP projection')
    ax1.set_ylabel('% Mandarin characters')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Steering effectiveness: with vs without GLP projection\nLlama-3.2-1B, English→Chinese, layer 7')

    # Panel 2: Perplexity
    ax2.fill_between(angles, np.maximum(b_ppl_mean - b_ppl_std, 0),
                     b_ppl_mean + b_ppl_std, color='C0', alpha=0.15)
    ax2.plot(angles, b_ppl_mean, 'C0-o', lw=2, label='Steer only')
    ax2.fill_between(angles, np.maximum(g_ppl_mean - g_ppl_std, 0),
                     g_ppl_mean + g_ppl_std, color='C1', alpha=0.15)
    ax2.plot(angles, g_ppl_mean, 'C1-s', lw=2, label='Steer + GLP projection')
    ax2.set_xlabel('Steering angle (degrees)')
    ax2.set_ylabel('Perplexity')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'steering_glp_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == '__main__':
    main()
