#!/usr/bin/env python3
"""
Quick check: does Qwen3 0.6B speak Dutch?

Feeds a few Dutch prompts and checks if completions are coherent Dutch.
Also reports token counts for our bilingual prompts to verify ~5 tokens each.
"""

import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model
from data.bilingual_prompts_dutch import PROMPT_PAIRS, ENGLISH_PROMPTS, DUTCH_PROMPTS


def main():
    print("Loading Qwen3-0.6B...")
    model, tokenizer, device = load_model('qwen3-0.6b')
    print(f"Device: {device}\n")

    # 1. Check Dutch generation
    test_prompts = [
        "De zon schijnt op",
        "Het weer vandaag is",
        "Ik drink altijd koffie",
        "De kinderen spelen buiten",
        "Nederland is een klein",
    ]

    print("=== Dutch generation test ===")
    for prompt in test_prompts:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **tokens, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        completion = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  {prompt!r}")
        print(f"    -> {completion!r}")
        print()

    # 2. Token counts for all prompt pairs
    print("=== Token counts for prompt pairs ===")
    en_counts = []
    nl_counts = []
    for i, (en, nl) in enumerate(PROMPT_PAIRS):
        en_tok = tokenizer(en)['input_ids']
        nl_tok = tokenizer(nl)['input_ids']
        en_counts.append(len(en_tok))
        nl_counts.append(len(nl_tok))
        print(f"  {i+1:2d}. EN ({len(en_tok)} tok): {en!r}")
        print(f"      NL ({len(nl_tok)} tok): {nl!r}")

    print(f"\nEN tokens: min={min(en_counts)}, max={max(en_counts)}, mean={sum(en_counts)/len(en_counts):.1f}")
    print(f"NL tokens: min={min(nl_counts)}, max={max(nl_counts)}, mean={sum(nl_counts)/len(nl_counts):.1f}")


if __name__ == '__main__':
    main()
