#!/usr/bin/env python3
"""
Steering evaluation on held-out language (Spanish).

Tests whether multi-contrastive directions (f_E, f_M, f_D) steer more cleanly
than pairwise diff-of-means (d_EM, d_ED, d_MD) when applied to a language
that wasn't used in direction estimation.

Usage:
    python scripts/run_steering_multi_contrastive.py
"""

import sys
import pickle
import json
import os
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks
from data.spanish_prompts import PROMPTS as SPANISH_PROMPTS

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

ANGLES = [0, 5, 10, 15, 20, 25, 30, 45]


def is_cjk_char(ch):
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: return True
    if 0x3400 <= cp <= 0x4DBF: return True
    if 0x20000 <= cp <= 0x2A6DF: return True
    if 0xF900 <= cp <= 0xFAFF: return True
    if 0x3000 <= cp <= 0x303F: return True
    if 0xFF00 <= cp <= 0xFFEF: return True
    return False


def cjk_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if is_cjk_char(ch)) / len(chars)


def is_latin_extended(ch):
    """Check for characters common in Dutch/German but not English/Spanish."""
    return ch in 'ïëüöäàèéêîôûùâ'


def detect_language_simple(text):
    """Simple heuristic language detection. Returns dict of scores."""
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return {'english': 0, 'mandarin': 0, 'dutch': 0, 'spanish': 0}

    cjk_frac = cjk_fraction(text)

    # Dutch indicators
    dutch_words = {'de', 'het', 'een', 'van', 'en', 'in', 'is', 'dat', 'op',
                   'te', 'voor', 'met', 'niet', 'zijn', 'die', 'hij', 'aan',
                   'er', 'maar', 'ook', 'dan', 'nog', 'uit', 'bij', 'om',
                   'ze', 'dit', 'kan', 'hun', 'ik', 'wij', 'zij', 'mijn',
                   'naar', 'haar', 'meer', 'heel', 'veel', 'goed', 'groot',
                   'heeft', 'wordt', 'werd', 'waren', 'hebben', 'deze',
                   'wat', 'waar', 'hoe', 'welke', 'omdat', 'wanneer'}
    dutch_count = sum(1 for w in words if w in dutch_words)

    # Spanish indicators
    spanish_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'y',
                     'que', 'es', 'por', 'con', 'para', 'del', 'al', 'se',
                     'su', 'como', 'pero', 'más', 'muy', 'fue', 'este',
                     'esta', 'son', 'ser', 'hay', 'tiene', 'puede', 'entre',
                     'todos', 'también', 'desde', 'donde', 'porque', 'cuando',
                     'otro', 'sobre', 'después', 'estos', 'ella', 'nos',
                     'están', 'hasta', 'sino', 'cada', 'cual', 'yo', 'así'}
    spanish_count = sum(1 for w in words if w in spanish_words)

    # English indicators
    english_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                     'into', 'through', 'during', 'before', 'after', 'that',
                     'this', 'these', 'those', 'it', 'its', 'they', 'them',
                     'their', 'we', 'our', 'you', 'your', 'he', 'she',
                     'him', 'her', 'his', 'not', 'but', 'and', 'or', 'if'}
    english_count = sum(1 for w in words if w in english_words)

    n = max(len(words), 1)
    return {
        'english': english_count / n,
        'mandarin': cjk_frac,
        'dutch': dutch_count / n,
        'spanish': spanish_count / n,
    }


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
    results_dir = 'results/multi_contrastive/qwen3-0.6b'
    os.makedirs(results_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    # Load directions from multi-contrastive experiment
    with open(os.path.join(results_dir, 'optimization_results.pkl'), 'rb') as f:
        saved = pickle.load(f)

    directions = {}
    for name in ['d_EM', 'd_ED', 'd_MD', 'f_E', 'f_M', 'f_D',
                  'MELBO_joint_1', 'MELBO_EM']:
        directions[name] = saved['directions'][name].to(device)
    # Also test negatives for steering (toward the language, not away)
    # d_EM points English→Mandarin, so -d_EM steers toward Mandarin
    # f_M already points toward Mandarin

    # Directions to test with their semantic meaning for steering
    steer_configs = {
        # Toward Mandarin
        '-d_EM': -directions['d_EM'],          # pairwise: EN-ZH
        'f_M': directions['f_M'],              # multi-contrastive Mandarin
        '-MELBO_EM': -directions['MELBO_EM'],   # MELBO on EN-ZH span
        '-MELBO_joint_1': -directions['MELBO_joint_1'],  # joint MELBO
        # Toward English
        'd_EM': directions['d_EM'],            # pairwise: EN-ZH
        'f_E': directions['f_E'],              # multi-contrastive English
        # Toward Dutch
        '-d_ED': -directions['d_ED'],          # pairwise: EN-NL
        'f_D': directions['f_D'],              # multi-contrastive Dutch
    }

    prompts = SPANISH_PROMPTS
    dir_names = list(steer_configs.keys())
    total = len(prompts) * len(ANGLES) * len(dir_names)
    print(f"Spanish prompts: {len(prompts)}, Angles: {ANGLES}, Directions: {len(dir_names)}")
    print(f"Total generations: {total}")

    all_results = []
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        for angle in ANGLES:
            for dir_name in dir_names:
                gen_seed = SEED + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                direction = steer_configs[dir_name]
                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                lang_scores = detect_language_simple(text)

                result = {
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'cjk_fraction': cjk_fraction(text),
                    'lang_scores': lang_scores,
                    'n_chars': len([ch for ch in text if not ch.isspace()]),
                }
                all_results.append(result)

                count += 1
                if count % 30 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nAll {total} generations complete in {elapsed:.0f}s")

    with open(os.path.join(results_dir, 'steering_spanish.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {results_dir}/steering_spanish.json")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Language scores by direction and angle (from Spanish prompts)")
    print(f"{'='*80}")

    for dir_name in dir_names:
        print(f"\n  {dir_name}:")
        print(f"    {'Angle':>6s}  {'CJK':>6s}  {'English':>8s}  {'Dutch':>6s}  {'Spanish':>8s}")
        for angle in ANGLES:
            results_at = [r for r in all_results
                         if r['direction'] == dir_name and r['angle'] == angle]
            cjk_mean = np.mean([r['cjk_fraction'] for r in results_at])
            en_mean = np.mean([r['lang_scores']['english'] for r in results_at])
            nl_mean = np.mean([r['lang_scores']['dutch'] for r in results_at])
            es_mean = np.mean([r['lang_scores']['spanish'] for r in results_at])
            print(f"    {angle:>5d}°  {cjk_mean:>5.1%}  {en_mean:>7.1%}  {nl_mean:>5.1%}  {es_mean:>7.1%}")

    print("\nDone!")


if __name__ == '__main__':
    main()
