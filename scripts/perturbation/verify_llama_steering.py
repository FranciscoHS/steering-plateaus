#!/usr/bin/env python3
"""Verify that the en→zh diff-of-means steering vector works on Llama-3.2-1B.

For various steering magnitudes, generate text and measure:
- % Mandarin characters in output
- Qualitative coherence (print samples)

Usage:
    python scripts/perturbation/verify_llama_steering.py
"""

import os, sys, unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
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

LAYER = 7
STEER_ANGLES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
TEST_PROMPTS = ENGLISH_PROMPTS[:10]
MAX_NEW_TOKENS = 50
RESULTS_DIR = 'results/llama_plateau'


def is_cjk(char):
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    try:
        return 'CJK' in unicodedata.name(char, '')
    except ValueError:
        return False


def pct_mandarin(text):
    """Percentage of non-whitespace characters that are CJK."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return 100.0 * sum(1 for c in chars if is_cjk(c)) / len(chars)


def spherical_steer_hook_fn(steer_vec_unit, angle_rad):
    """Create a hook that spherically rotates activations toward steer direction.

    Vectorized over all positions. Skips positions with near-zero norm.
    """
    def hook(module, input, output):
        if angle_rad == 0:
            return output

        if isinstance(output, tuple):
            hidden = output[0].clone()
        else:
            hidden = output[0].clone()

        d = steer_vec_unit.to(hidden.device, hidden.dtype)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # hidden: [batch, seq, d_model]
        R = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a_unit = hidden / R
        proj = (a_unit * d).sum(dim=-1, keepdim=True)
        v_perp = d.unsqueeze(0).unsqueeze(0) - proj * a_unit
        v_perp = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        hidden = R * (cos_a * a_unit + sin_a * v_perp)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            output[0] = hidden
            return output

    return hook


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Llama-3.2-1B...")
    model, tokenizer, device = load_model('meta-llama/Llama-3.2-1B', device=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    d_model = model._plateau_info['d_model']

    # Compute diff-of-means steering vector
    print("Computing steering vector...")
    en_acts, zh_acts = [], []

    for en_text, zh_text in zip(ENGLISH_PROMPTS, CHINESE_PROMPTS):
        for text, acts_list in [(en_text, en_acts), (zh_text, zh_acts)]:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[LAYER + 1]
            acts_list.append(hidden[:, -1, :].squeeze(0))

    en_mean = torch.stack(en_acts).mean(dim=0)
    zh_mean = torch.stack(zh_acts).mean(dim=0)
    steer_vec = zh_mean - en_mean
    steer_vec_unit = steer_vec / steer_vec.norm()
    print(f"Steering vector norm: {steer_vec.norm():.3f}")

    # Measure activation norms for context
    en_norms = torch.stack(en_acts).norm(dim=-1)
    print(f"English activation norms: mean={en_norms.mean():.2f}, std={en_norms.std():.2f}")

    # Generate at each angle
    print(f"\n{'Angle':>6} | {'%ZH':>5} | Sample output")
    print("-" * 90)

    all_pcts = {}  # angle -> list of pcts per prompt
    all_samples = {}

    for angle_deg in STEER_ANGLES:
        angle_rad = np.deg2rad(angle_deg)

        # Register hook
        hook = spherical_steer_hook_fn(steer_vec_unit, angle_rad)
        block = model.model.layers[LAYER]
        handle = block.register_forward_hook(hook)

        pcts = []
        samples = []
        for pi, prompt in enumerate(TEST_PROMPTS):
            tokens = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    **tokens,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            generated = tokenizer.decode(out[0][tokens['input_ids'].shape[1]:], skip_special_tokens=True)
            pcts.append(pct_mandarin(generated))
            samples.append(generated[:100].replace('\n', ' '))

        handle.remove()

        all_pcts[angle_deg] = pcts
        all_samples[angle_deg] = samples
        mean_pct = np.mean(pcts)
        print(f"{angle_deg:5d}° | {mean_pct:5.1f} | {samples[0][:80]}")

    # Plot: %ZH vs steering angle
    angles = np.array(STEER_ANGLES)
    mean_zh = np.array([np.mean(all_pcts[a]) for a in STEER_ANGLES])
    std_zh = np.array([np.std(all_pcts[a]) for a in STEER_ANGLES])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(angles, mean_zh - std_zh, np.minimum(mean_zh + std_zh, 100),
                    color='C0', alpha=0.15)
    ax.plot(angles, mean_zh, 'C0-o', lw=2, label='% Mandarin characters')
    ax.set_xlabel('Steering angle (degrees)')
    ax.set_ylabel('% Mandarin characters in output')
    ax.set_title('Llama-3.2-1B: English→Chinese steering verification\n(spherical rotation at layer 7)')
    ax.set_xlim(0, max(STEER_ANGLES))
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'steering_verification_enzh.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")


if __name__ == '__main__':
    main()
