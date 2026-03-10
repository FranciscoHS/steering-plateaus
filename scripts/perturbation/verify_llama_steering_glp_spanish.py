#!/usr/bin/env python3
"""Verify whether GLP on-manifold projection improves steering coherence.

English→Spanish steering with GLP projection. Qualitative comparison only
(no automatic language metric since Spanish uses Latin script).

Usage:
    python scripts/perturbation/verify_llama_steering_glp_spanish.py
"""

import os, sys, pickle
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
from data.bilingual_prompts_spanish import ENGLISH_PROMPTS, SPANISH_PROMPTS
from glp.denoiser import load_glp
from glp import flow_matching

LAYER = 7
STEER_ANGLES = [0, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30]
TEST_PROMPTS = ENGLISH_PROMPTS[:10]
MAX_NEW_TOKENS = 50
GLP_U = 0.5
GLP_TIMESTEPS = 50
RESULTS_DIR = 'results/llama_plateau'


def project_on_manifold(glp_model, activations, u=0.5, num_timesteps=50):
    """Project activations onto learned manifold via GLP denoising."""
    orig_shape = activations.shape
    orig_dtype = activations.dtype

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
    """Run generation at each angle, return dict of angle -> samples."""
    results = {}
    for angle_deg in angles:
        angle_rad = np.deg2rad(angle_deg)
        hook = make_hook(steer_vec_unit, angle_rad, glp_model=glp_model,
                         glp_u=GLP_U, glp_timesteps=GLP_TIMESTEPS)
        handle = block.register_forward_hook(hook)

        samples = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors='pt').to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(**tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated = tokenizer.decode(out[0][tokens['input_ids'].shape[1]:], skip_special_tokens=True)
            samples.append(generated)

        handle.remove()
        results[angle_deg] = {'samples': samples}
        print(f"  [{label}] {angle_deg:3d}°: {samples[0][:80]}")

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

    # Compute English→Spanish steering vector
    print("Computing steering vector...")
    en_acts, es_acts = [], []
    for en_text, es_text in zip(ENGLISH_PROMPTS, SPANISH_PROMPTS):
        for text, acts_list in [(en_text, en_acts), (es_text, es_acts)]:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts_list.append(outputs.hidden_states[LAYER + 1][:, -1, :].squeeze(0))

    en_mean = torch.stack(en_acts).mean(dim=0)
    es_mean = torch.stack(es_acts).mean(dim=0)
    steer_vec = es_mean - en_mean
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
            'language': 'en→es',
        }
    }
    pkl_path = os.path.join(RESULTS_DIR, 'steering_glp_comparison_spanish_fine.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")


if __name__ == '__main__':
    main()
