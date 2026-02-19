#!/usr/bin/env python3
"""
Test which programming languages Qwen3-0.6B can reliably autocomplete.

Feeds each prompt from data/programming_prompts.py to the model, generates
30-50 tokens, and reports results grouped by language. Includes heuristic
checks for language-specific syntax markers.

Usage:
    python scripts/directions/test_programming_autocompletion.py [--model qwen3-0.6b] [--max_tokens 50]
"""

import sys
import argparse

sys.path.insert(0, '.')

import torch
from src.model import load_model
from data.programming_prompts import PROMPT_SETS, LANGUAGES

# Heuristic markers: if a completion contains these, it's likely the right language.
# We check for presence of language-specific syntax in the *completion* (not the prompt).
LANGUAGE_MARKERS = {
    'python': ['def ', 'return ', 'if ', 'elif ', 'for ', 'import ', '    ', 'print(', 'self.', 'None'],
    'javascript': ['const ', 'let ', 'var ', 'return ', 'console.', '===', '=>', 'null', 'undefined'],
    'java': ['return ', 'int ', 'String ', 'public ', 'private ', 'new ', 'this.', 'null', 'void '],
    'c': ['return ', 'int ', 'printf', 'malloc', 'NULL', 'sizeof', 'void ', '#include'],
    'go': ['return ', 'fmt.', 'if ', 'for ', 'err ', ':= ', 'nil', 'range '],
    'rust': ['let ', 'return ', 'match ', 'Some(', 'None', 'mut ', '-> ', '&', 'unwrap('],
    'php': ['$', 'return ', 'echo ', 'array(', '=> ', 'foreach', 'isset(', 'null'],
    'typescript': ['const ', 'let ', 'return ', 'interface ', ': number', ': string', ': boolean', '=> '],
}

# Negative markers: if these appear, the completion likely switched to the wrong language
CONFUSION_MARKERS = {
    'python': {'function ', 'const ', 'public static', 'fn ', 'func '},
    'javascript': {'def ', 'fn ', 'func ', 'public static'},
    'java': {'def ', 'fn ', 'func ', 'function '},
    'c': {'def ', 'fn ', 'func ', 'function ', 'public static'},
    'go': {'def ', 'fn ', 'function ', 'public static'},
    'rust': {'def ', 'function ', 'func ', 'public static'},
    'php': {'def ', 'fn ', 'func ', 'public static', 'const ', 'let '},
    'typescript': {'def ', 'fn ', 'func ', 'public static'},
}


def generate_completion(model, tokenizer, device, prompt, max_tokens=50):
    """Generate a completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_completion(completion, lang):
    """Heuristic score: how many positive markers vs negative markers appear."""
    positive = sum(1 for m in LANGUAGE_MARKERS.get(lang, []) if m in completion)
    negative = sum(1 for m in CONFUSION_MARKERS.get(lang, set()) if m in completion)
    return positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--n_prompts', type=int, default=None,
                        help='Number of prompt sets to test (default: all)')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Device: {device}")

    prompt_sets = PROMPT_SETS[:args.n_prompts] if args.n_prompts else PROMPT_SETS
    n_sets = len(prompt_sets)

    # Track per-language stats
    stats = {lang: {'positive': 0, 'negative': 0, 'total': 0} for lang in LANGUAGES}

    # Generate completions grouped by language
    for lang in LANGUAGES:
        print(f"\n{'='*70}")
        print(f"  LANGUAGE: {lang.upper()}")
        print(f"{'='*70}")

        for i, pset in enumerate(prompt_sets):
            prompt = pset[lang]
            completion = generate_completion(model, tokenizer, device, prompt, args.max_tokens)
            pos, neg = score_completion(completion, lang)
            stats[lang]['positive'] += pos
            stats[lang]['negative'] += neg
            stats[lang]['total'] += 1

            # Show first 5 and last 2 for each language (enough for inspection)
            if i < 5 or i >= n_sets - 2:
                # Truncate long completions for display
                comp_display = completion.replace('\n', '\\n')
                if len(comp_display) > 120:
                    comp_display = comp_display[:120] + '...'
                marker = '+' if pos > neg else ('?' if pos == neg else '-')
                print(f"  [{marker}] {prompt}")
                print(f"       -> {comp_display}")
            elif i == 5:
                print(f"  ... ({n_sets - 7} more prompts) ...")

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for lang in LANGUAGES:
        s = stats[lang]
        avg_pos = s['positive'] / s['total']
        avg_neg = s['negative'] / s['total']
        verdict = 'GOOD' if avg_pos > 2 * avg_neg and avg_pos >= 1.5 else 'WEAK' if avg_pos > avg_neg else 'BAD'
        print(f"  {lang:>12s}: avg_positive={avg_pos:.1f}, avg_negative={avg_neg:.1f}  [{verdict}]")

    print(f"\nTested {n_sets} prompt sets across {len(LANGUAGES)} languages.")
    print("Inspect completions above. Languages marked WEAK or BAD may need to be dropped.")


if __name__ == '__main__':
    main()
