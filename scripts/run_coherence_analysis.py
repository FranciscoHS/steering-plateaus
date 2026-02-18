#!/usr/bin/env python3
"""
Coherence analysis for steered generations.

For each generated text from the steering comparison, compute:
1. Model perplexity (proxy for coherence)
2. CJK fraction
3. English fraction
4. Repetition score (fraction of repeated n-grams)

Produces a multi-panel plot like kl_vs_cjk_v2.png for d_EM and k=10 SVD.

Usage:
    python scripts/run_coherence_analysis.py
"""

import sys, os, pickle, json, numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model

results_dir = 'results/scaling_test/qwen3-0.6b'


def compute_perplexity(model, tokenizer, device, text, max_len=128):
    """Compute model perplexity on a text string."""
    if not text.strip():
        return float('nan')
    tokens = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=max_len)
    input_ids = tokens['input_ids'].to(device)
    if input_ids.shape[1] < 2:
        return float('nan')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab]
        targets = input_ids[:, 1:]  # [1, seq_len-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1))
    return loss.exp().item()


def repetition_score(text, n=3):
    """Fraction of n-grams that are repeated."""
    chars = list(text.replace(' ', ''))
    if len(chars) < n:
        return 0.0
    ngrams = [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)


def is_cjk_char(ch):
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: return True
    if 0x3400 <= cp <= 0x4DBF: return True
    if 0x20000 <= cp <= 0x2A6DF: return True
    if 0xF900 <= cp <= 0xFAFF: return True
    if 0x3000 <= cp <= 0x303F: return True
    if 0xFF00 <= cp <= 0xFFEF: return True
    return False


def main():
    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-0.6b')

    with open(os.path.join(results_dir, 'steering_comparison.pkl'), 'rb') as f:
        data = pickle.load(f)

    zh_results = data['zh_results']
    angles = data['angles']

    # Compute perplexity for each generation
    print(f"\nComputing perplexity for {len(zh_results)} generations...")
    for i, r in enumerate(zh_results):
        # Perplexity of the generated text (not the prompt)
        r['perplexity'] = compute_perplexity(model, tokenizer, device, r['text'])
        r['repetition'] = repetition_score(r['text'])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(zh_results)}")

    # Also print some sample generations at key angles
    for dir_name in ['d_EM', 'k=10 SVD']:
        print(f"\n{'='*60}")
        print(f"Sample generations: {dir_name}")
        print(f"{'='*60}")
        for angle in [0, 5, 10, 15, 25, 45]:
            samples = [r for r in zh_results
                       if r['direction'] == dir_name and r['angle'] == angle]
            if not samples:
                continue
            # Pick 3 representative samples
            print(f"\n--- {angle}° ---")
            for s in samples[:3]:
                ppl = s['perplexity']
                cjk = s['cjk_fraction']
                rep = s['repetition']
                text_preview = s['text'][:100].replace('\n', ' ')
                print(f"  CJK={cjk:.0%} PPL={ppl:.0f} REP={rep:.2f} | {text_preview}")

    # Save enriched results
    save_data = {
        'zh_results': zh_results,
        'angles': angles,
        'directions': data['directions'],
    }
    with open(os.path.join(results_dir, 'coherence_results.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {results_dir}/coherence_results.pkl")
    print("Done!")


if __name__ == '__main__':
    main()
