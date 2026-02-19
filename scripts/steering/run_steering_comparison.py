#!/usr/bin/env python3
"""
Steering comparison: can multi-contrastive English features steer
Mandarin prompts into English? Compare d_EM, k=10 mean, k=10 SVD.

Then optionally test on Spanish prompts too (same English feature
should work if it's truly language-general).

Produces KL + CJK fraction plots like kl_vs_cjk_v2.png.

Usage:
    python scripts/run_steering_comparison.py [--spanish]
"""

import sys, os, pickle, json, time, argparse
import numpy as np
sys.path.insert(0, '.')

import torch
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

# Mandarin prompts (from bilingual_prompts.py)
ZH_PROMPTS = [
    "太阳从", "水在温度达到", "太阳系中最大的行星是",
    "树木通过", "光速大约是", "地震是由",
    "人类大脑大约包含", "钻石是由", "世界上最深的海洋是",
    "光合作用是", "日本的首都是", "长城的建造是为了",
    "非洲最长的河流是", "印度的人口大约是", "珠穆朗玛峰位于",
    "古埃及人建造金字塔是为了", "太平洋是", "第一个登上月球的人是",
    "澳大利亚以其", "撒哈拉沙漠的面积约为",
    "早餐是一天中最重要的", "很多人喜欢在",
    "运动很重要，因为它有助于", "读书可以提高你的",
    "均衡的饮食应该包含", "充足的睡眠对于",
    "学习一门新语言需要", "在家做饭比",
    "音乐可以帮助人们感到更", "去新的地方旅行帮助我们",
]

# Spanish prompts for transfer test
ES_PROMPTS = [
    "El sol sale por", "El agua hierve a una temperatura de",
    "El planeta más grande del sistema solar es",
    "Los árboles producen oxígeno mediante", "La velocidad de la luz es aproximadamente",
    "Los terremotos son causados por", "El cerebro humano contiene aproximadamente",
    "Los diamantes están hechos de", "El océano más profundo del mundo es",
    "La fotosíntesis es el proceso por el cual",
    "La capital de Japón es", "La Gran Muralla China fue construida para",
    "El río más largo de África es", "La población de la India es aproximadamente",
    "El Monte Everest se encuentra en",
    "Los antiguos egipcios construyeron pirámides para",
    "El océano Pacífico es", "La primera persona en caminar en la luna fue",
    "Australia es conocida por", "El desierto del Sahara cubre un área de",
    "El desayuno es la comida más importante del",
    "Muchas personas disfrutan de tomar café por",
    "El ejercicio es importante porque ayuda a",
    "Leer libros puede mejorar tu",
    "Una dieta equilibrada debe incluir",
    "Dormir lo suficiente es esencial para",
    "Aprender un nuevo idioma requiere",
    "Cocinar en casa es más saludable que",
    "La música puede ayudar a las personas",
    "Viajar a nuevos lugares nos ayuda a",
]

ANGLES = [0, 3, 5, 7, 10, 15, 20, 25, 30, 45]


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
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if is_cjk_char(ch)) / len(chars)


def english_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if ch.isascii() and ch.isalpha()) / len(chars)


def generate_steered(model, tokenizer, device, prompt, steer_direction, angle_deg,
                     n_tokens, temperature, layer):
    """Generate with spherical steering at every token."""
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


def run_steering_experiment(model, tokenizer, device, prompts, directions,
                           angles, source_lang, results_dir):
    """Run steering experiment for a set of prompts."""
    all_results = []
    total = len(prompts) * len(angles) * len(directions)
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        for angle in angles:
            for dir_name, direction in directions.items():
                gen_seed = SEED + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                result = {
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'cjk_fraction': cjk_fraction(text),
                    'english_fraction': english_fraction(text),
                    'source_lang': source_lang,
                }
                all_results.append(result)

                count += 1
                if count % 30 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  All {total} generations complete in {elapsed:.0f}s")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spanish', action='store_true',
                        help='Also run Spanish→English steering test')
    args = parser.parse_args()

    results_dir = 'results/scaling_test/qwen3-0.6b'
    os.makedirs(results_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    # Load directions
    with open(os.path.join(results_dir, 'scaling_results.pkl'), 'rb') as f:
        scaling = pickle.load(f)
    with open(os.path.join(results_dir, 'validation_results.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open('results/multi_contrastive/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
        old_mc = pickle.load(f)

    # Positive directions = toward English (for steering ZH/ES → EN)
    directions = {
        'd_EM': old_mc['directions']['d_EM'].to(device),
        'k=10 mean': scaling['ref_direction'].to(device),
        'k=10 SVD': val['svd_k10'].to(device),
    }

    print(f"Angles: {ANGLES}")
    print(f"Directions: {list(directions.keys())}")

    # === Mandarin → English ===
    print(f"\n{'='*60}")
    print("Mandarin → English steering")
    print(f"{'='*60}")
    zh_results = run_steering_experiment(
        model, tokenizer, device, ZH_PROMPTS, directions,
        ANGLES, 'zh', results_dir)

    # Print summary
    print(f"\n{'Direction':>15} |", end='')
    for a in ANGLES:
        print(f" {a:>4}°", end='')
    print()
    print("-" * (17 + 6 * len(ANGLES)))
    for dir_name in directions:
        print(f"{'CJK '+dir_name:>15} |", end='')
        for a in ANGLES:
            vals = [r['cjk_fraction'] for r in zh_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()
    for dir_name in directions:
        print(f"{'ENG '+dir_name:>15} |", end='')
        for a in ANGLES:
            vals = [r['english_fraction'] for r in zh_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()

    # === Spanish → English (optional) ===
    es_results = None
    if args.spanish:
        print(f"\n{'='*60}")
        print("Spanish → English steering")
        print(f"{'='*60}")
        es_results = run_steering_experiment(
            model, tokenizer, device, ES_PROMPTS, directions,
            ANGLES, 'es', results_dir)

        print(f"\n{'Direction':>15} |", end='')
        for a in ANGLES:
            print(f" {a:>4}°", end='')
        print()
        print("-" * (17 + 6 * len(ANGLES)))
        for dir_name in directions:
            print(f"{'ENG '+dir_name:>15} |", end='')
            for a in ANGLES:
                vals = [r['english_fraction'] for r in es_results
                        if r['direction'] == dir_name and r['angle'] == a]
                print(f" {np.mean(vals):>4.0%}", end='')
            print()

    # Save
    save = {
        'zh_results': zh_results,
        'es_results': es_results,
        'angles': ANGLES,
        'directions': list(directions.keys()),
    }
    save_path = os.path.join(results_dir, 'steering_comparison.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save, f)
    print(f"\nSaved: {save_path}")
    print("Done!")


if __name__ == '__main__':
    main()
