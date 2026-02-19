#!/usr/bin/env python3
"""Quick check: which language directions reliably steer away from baseline?"""

import sys, os, pickle, numpy as np
sys.path.insert(0, '.')
import torch
from src.model import load_model

def steer_and_generate(model, tokenizer, device, input_ids, layer, direction,
                       magnitude_deg, max_new_tokens=40):
    direction_unit = (direction / direction.norm()).to(device)
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        a = hidden[:, -1, :]
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / (a_norm + 1e-10)
        proj = (a_unit * direction_unit).sum(dim=-1, keepdim=True)
        u_perp = direction_unit.unsqueeze(0) - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
        angle = np.radians(magnitude_deg)
        a_steered = a_norm * (np.cos(angle) * a_unit + np.sin(angle) * u_perp)
        hidden = hidden.clone()
        hidden[:, -1, :] = a_steered
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    blocks = model.model.layers
    h = blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(input_ids.to(device), max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        h.remove()

# Language detection heuristics
LANG_KEYWORDS = {
    'python': ['def ', 'import ', 'print(', 'elif ', 'self.', '    return'],
    'javascript': ['var ', 'const ', 'console.', '===', 'function '],
    'java': ['public ', 'private ', 'import java', 'System.out', 'class '],
    'c': ['#include', 'printf', 'malloc', 'int main('],
    'go': ['func ', 'fmt.', ':= ', 'package '],
    'rust': ['fn ', 'let mut', 'println!', '-> ', 'impl '],
    'typescript': ['interface ', ': number', ': string', ': boolean'],
}

def detect_language(text):
    """Return detected language or 'unknown'."""
    scores = {}
    for lang, keywords in LANG_KEYWORDS.items():
        scores[lang] = sum(1 for k in keywords if k in text)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else 'unknown'

def main():
    with open('results/programming_contrastive/qwen3-0.6b/contrastive_results.pkl', 'rb') as f:
        results = pickle.load(f)
    languages = results['languages']
    f_hat = results['f_hat']

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    prompts = [
        "The following code",
        "The code below",
        "Here is some code",
        "Consider the following program",
        "The following function",
        "Below is an implementation",
    ]

    angles = [15, 20, 30]

    # Track: for each direction × angle, how often does it produce its target language?
    results_table = {lang: {a: [] for a in angles} for lang in languages}

    for prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids']

        # Baseline
        with torch.no_grad():
            out = model.generate(input_ids.to(device), max_new_tokens=40,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        baseline = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        baseline_lang = detect_language(baseline)

        for angle in angles:
            for lang in languages:
                comp = steer_and_generate(model, tokenizer, device, input_ids, 14,
                                          f_hat[lang], angle, 40)
                detected = detect_language(comp)
                shifted = (detected != baseline_lang)
                target_hit = (detected == lang)
                results_table[lang][angle].append((detected, shifted, target_hit, prompt_text))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEERING RELIABILITY (across {len(prompts)} prompts)")
    print(f"{'='*70}")
    for lang in languages:
        print(f"\n  {lang}:")
        for angle in angles:
            entries = results_table[lang][angle]
            n = len(entries)
            shifted = sum(1 for _, s, _, _ in entries if s)
            target = sum(1 for _, _, t, _ in entries if t)
            detected_langs = [d for d, _, _, _ in entries]
            lang_counts = {}
            for d in detected_langs:
                lang_counts[d] = lang_counts.get(d, 0) + 1
            dist = ', '.join(f'{l}:{c}' for l, c in sorted(lang_counts.items(), key=lambda x: -x[1]))
            print(f"    {angle:>2d}°: shifted={shifted}/{n}, target_hit={target}/{n}  [{dist}]")

if __name__ == '__main__':
    main()
