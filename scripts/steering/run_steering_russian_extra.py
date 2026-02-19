#!/usr/bin/env python3
"""
Extra Russian→English steering at angles 32, 34, 36, 38 to fill in
the 30-40° range where coherence differences live.

Merges with existing steering_russian.pkl results.

Usage:
    python scripts/run_steering_russian_extra.py
"""

import sys, os, pickle, time, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks
from data.multilingual_prompts import PROMPT_SETS

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

RU_PROMPTS = [pset['ru'] for pset in PROMPT_SETS]
EXTRA_ANGLES = [32, 34, 36, 38]


def is_cyrillic(ch):
    cp = ord(ch)
    if 0x0400 <= cp <= 0x04FF: return True
    if 0x0500 <= cp <= 0x052F: return True
    return False


def is_cjk_char(ch):
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: return True
    if 0x3400 <= cp <= 0x4DBF: return True
    if 0x20000 <= cp <= 0x2A6DF: return True
    if 0xF900 <= cp <= 0xFAFF: return True
    if 0x3000 <= cp <= 0x303F: return True
    if 0xFF00 <= cp <= 0xFFEF: return True
    return False


def cyrillic_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if is_cyrillic(ch)) / len(chars)


def cjk_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if is_cjk_char(ch)) / len(chars)


def english_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if ch.isascii() and ch.isalpha()) / len(chars)


def compute_perplexity(model, tokenizer, device, text, max_len=128):
    if not text.strip():
        return float('nan')
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
    input_ids = tokens['input_ids'].to(device)
    if input_ids.shape[1] < 2:
        return float('nan')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1))
    return loss.exp().item()


def repetition_score(text, n=3):
    chars = list(text.replace(' ', ''))
    if len(chars) < n: return 0.0
    ngrams = [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]
    if len(ngrams) == 0: return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    if angle_deg == 0:
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        generated_ids = []
        for step in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    cos_theta = np.cos(np.radians(angle_deg))
    sin_theta = np.sin(np.radians(angle_deg))
    d = steer_direction

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    for step in range(n_tokens):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            steer_slice = hidden[0, first_steer_pos:, :]
            if steer_slice.shape[0] == 0:
                return output
            R = steer_slice.norm(dim=-1, keepdim=True)
            a_unit = steer_slice / (R + 1e-10)
            proj = (a_unit * d).sum(dim=-1, keepdim=True)
            v_perp = d.unsqueeze(0) - proj * a_unit
            v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
            steered = R * (cos_theta * a_unit + sin_theta * v_perp)
            hidden[0, first_steer_pos:, :] = steered
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        handle.remove()

        logits = outputs.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    results_dir = 'results/scaling_test/qwen3-0.6b'

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    # Load directions
    with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
        scaling = pickle.load(f)
    with open(os.path.join(results_dir, 'validation_results.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open('results/multi_contrastive/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
        old_mc = pickle.load(f)

    directions = {
        'd_ER': scaling['pairwise_dirs']['ru'].to(device),
        'd_EM': old_mc['directions']['d_EM'].to(device),
        'k=10 SVD': val['svd_k10'].to(device),
    }

    # === Generate at extra angles ===
    print(f"Extra angles: {EXTRA_ANGLES}")
    print(f"Russian → English ({len(RU_PROMPTS)} prompts, {len(EXTRA_ANGLES)} angles, {len(directions)} dirs)")
    extra_results = []
    total = len(RU_PROMPTS) * len(EXTRA_ANGLES) * len(directions)
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(RU_PROMPTS):
        for angle in EXTRA_ANGLES:
            for dir_name, direction in directions.items():
                gen_seed = SEED + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                extra_results.append({
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'cyrillic_fraction': cyrillic_fraction(text),
                    'cjk_fraction': cjk_fraction(text),
                    'english_fraction': english_fraction(text),
                })

                count += 1
                if count % 30 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    print(f"  Generations done in {time.time()-t0:.0f}s")

    # === Perplexity ===
    print(f"\nComputing perplexity for {len(extra_results)} generations...")
    for i, r in enumerate(extra_results):
        r['perplexity'] = compute_perplexity(model, tokenizer, device, r['text'])
        r['repetition'] = repetition_score(r['text'])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(extra_results)}")

    # === Merge with existing results ===
    with open(os.path.join(results_dir, 'steering_russian.pkl'), 'rb') as f:
        existing = pickle.load(f)

    merged = existing['ru_results'] + extra_results
    all_angles = sorted(set(existing['angles'] + EXTRA_ANGLES))

    # === Summary ===
    print(f"\n{'Direction':>15} |", end='')
    for a in EXTRA_ANGLES:
        print(f" {a:>4}°", end='')
    print()
    print("-" * (17 + 6 * len(EXTRA_ANGLES)))
    for dir_name in directions:
        print(f"{'CYR '+dir_name:>15} |", end='')
        for a in EXTRA_ANGLES:
            vals = [r['cyrillic_fraction'] for r in extra_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()
    for dir_name in directions:
        print(f"{'ENG '+dir_name:>15} |", end='')
        for a in EXTRA_ANGLES:
            vals = [r['english_fraction'] for r in extra_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()
    for dir_name in directions:
        print(f"{'PPL '+dir_name:>15} |", end='')
        for a in EXTRA_ANGLES:
            vals = [r['perplexity'] for r in extra_results
                    if r['direction'] == dir_name and r['angle'] == a
                    and not np.isnan(r['perplexity'])]
            print(f" {np.median(vals):>4.0f}", end='')
        print()

    # Save merged
    save_data = {
        'ru_results': merged,
        'angles': all_angles,
        'directions': existing['directions'],
    }
    with open(os.path.join(results_dir, 'steering_russian.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved merged: {results_dir}/steering_russian.pkl ({len(merged)} total results)")
    print("Done!")


if __name__ == '__main__':
    main()
