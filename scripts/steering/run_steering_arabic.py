#!/usr/bin/env python3
"""
Arabic → English steering: out-of-distribution test.

Arabic was NOT in the k=10 language set used to estimate the English
feature direction. This tests whether k=10 SVD generalizes to a
truly unseen language (different family: Semitic, different script).

Compares k=10 SVD vs d_EM (EN-ZH pairwise) — neither has seen Arabic.

Usage:
    python scripts/run_steering_arabic.py
"""

import sys, os, pickle, time, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks

LAYER = 14
N_TOKENS = 50
TEMPERATURE = 0.7
SEED = 42

# Arabic prompts (semantically matched to our multilingual set)
AR_PROMPTS = [
    "الشمس دافئة",            # The sun is warm
    "المطر ينزل بقوة",         # The rain falls hard
    "الثلج أبيض",              # The snow is white
    "الريح تهب بقوة",          # The wind blows hard
    "البحر عميق",              # The sea is deep
    "الشجرة عالية",            # The tree is tall
    "الزهرة حمراء",            # The flower is red
    "الليل بارد",              # The night is cold
    "الكلب كبير",              # The dog is big
    "القطة تنام جيداً",        # The cat sleeps well
    "الطائر يطير عالياً",      # The bird flies high
    "السمكة تسبح بسرعة",       # The fish swims fast
    "الخبز طازج",              # The bread is fresh
    "الماء بارد",              # The water is cold
    "الطعام لذيذ",             # The food is delicious
    "البيت قديم",              # The house is old
    "الكتاب طويل",             # The book is long
    "القهوة ساخنة",            # The coffee is hot
    "الطفل يركض بسرعة",        # The child runs fast
    "المعلم يتكلم جيداً",      # The teacher speaks well
    "الطبيب يساعد الناس",      # The doctor helps people
    "أمي تطبخ جيداً",          # My mom cooks well
    "المدينة كبيرة",           # The city is big
    "الجبل عالٍ",              # The mountain is tall
    "النهر يجري ببطء",         # The river flows slowly
    "المدرسة قريبة",           # The school is near
    "الموسيقى صاخبة",          # The music is loud
    "المشكلة صعبة",            # The problem is hard
    "القصة طويلة",             # The story is long
    "النوم مهم جداً",          # Sleep is very important
    "الوقت يمر بسرعة",         # Time passes quickly
    "الطريق طويل",             # The road is long
    "الحديقة جميلة",           # The garden is beautiful
    "السماء زرقاء",            # The sky is blue
    "الباب مفتوح",             # The door is open
    "الضوء ساطع",              # The light is bright
]

ANGLES = [0, 3, 5, 7, 10, 15, 20, 25, 30, 32, 34, 36, 38, 45]


def is_arabic(ch):
    cp = ord(ch)
    if 0x0600 <= cp <= 0x06FF: return True   # Arabic
    if 0x0750 <= cp <= 0x077F: return True   # Arabic Supplement
    if 0xFB50 <= cp <= 0xFDFF: return True   # Arabic Presentation Forms-A
    if 0xFE70 <= cp <= 0xFEFF: return True   # Arabic Presentation Forms-B
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


def arabic_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if is_arabic(ch)) / len(chars)


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
    with open(os.path.join(results_dir, 'validation_results.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open('results/multi_contrastive/qwen3-0.6b/optimization_results.pkl', 'rb') as f:
        old_mc = pickle.load(f)

    # Neither direction has seen Arabic
    directions = {
        'd_EM': old_mc['directions']['d_EM'].to(device),      # EN-ZH pairwise
        'k=10 SVD': val['svd_k10'].to(device),                # multi-contrastive (no Arabic)
    }

    print("\nDirection cosines:")
    for n1, d1 in directions.items():
        for n2, d2 in directions.items():
            if n1 < n2:
                cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
                print(f"  cos({n1}, {n2}) = {cos:.4f}")

    # === Steering from Arabic ===
    print(f"\nArabic → English steering (OOD test)")
    print(f"  {len(AR_PROMPTS)} prompts, {len(ANGLES)} angles, {len(directions)} dirs")
    all_results = []
    total = len(AR_PROMPTS) * len(ANGLES) * len(directions)
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(AR_PROMPTS):
        for angle in ANGLES:
            for dir_name, direction in directions.items():
                gen_seed = SEED + prompt_idx * 1000 + angle * 10 + hash(dir_name) % 100
                torch.manual_seed(gen_seed)

                text = generate_steered(
                    model, tokenizer, device, prompt, direction, angle,
                    N_TOKENS, TEMPERATURE, LAYER)

                all_results.append({
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': dir_name,
                    'angle': angle,
                    'text': text,
                    'arabic_fraction': arabic_fraction(text),
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
    print(f"\nComputing perplexity for {len(all_results)} generations...")
    for i, r in enumerate(all_results):
        r['perplexity'] = compute_perplexity(model, tokenizer, device, r['text'])
        r['repetition'] = repetition_score(r['text'])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_results)}")

    # === Sample generations ===
    for dir_name in directions:
        print(f"\n{'='*60}")
        print(f"Sample generations: {dir_name}")
        print(f"{'='*60}")
        for angle in [0, 5, 10, 15, 25, 36, 45]:
            samples = [r for r in all_results
                       if r['direction'] == dir_name and r['angle'] == angle]
            if not samples:
                continue
            print(f"\n--- {angle}° ---")
            for s in samples[:3]:
                ppl = s['perplexity']
                ar = s['arabic_fraction']
                eng = s['english_fraction']
                text_preview = s['text'][:120].replace('\n', ' ')
                print(f"  AR={ar:.0%} ENG={eng:.0%} PPL={ppl:.0f} | {text_preview}")

    # === Summary ===
    print(f"\n{'Direction':>15} |", end='')
    for a in ANGLES:
        print(f" {a:>4}°", end='')
    print()
    print("-" * (17 + 6 * len(ANGLES)))
    for dir_name in directions:
        print(f"{'AR '+dir_name:>15} |", end='')
        for a in ANGLES:
            vals = [r['arabic_fraction'] for r in all_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()
    for dir_name in directions:
        print(f"{'ENG '+dir_name:>15} |", end='')
        for a in ANGLES:
            vals = [r['english_fraction'] for r in all_results
                    if r['direction'] == dir_name and r['angle'] == a]
            print(f" {np.mean(vals):>4.0%}", end='')
        print()
    for dir_name in directions:
        print(f"{'PPL '+dir_name:>15} |", end='')
        for a in ANGLES:
            vals = [r['perplexity'] for r in all_results
                    if r['direction'] == dir_name and r['angle'] == a
                    and not np.isnan(r['perplexity'])]
            print(f" {np.median(vals):>4.0f}", end='')
        print()

    # Save
    save_data = {
        'ar_results': all_results,
        'angles': ANGLES,
        'directions': list(directions.keys()),
    }
    with open(os.path.join(results_dir, 'steering_arabic.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {results_dir}/steering_arabic.pkl")
    print("Done!")


if __name__ == '__main__':
    main()
