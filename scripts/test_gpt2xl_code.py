#!/usr/bin/env python3
"""Quick test: can GPT-2 XL complete code in multiple programming languages?"""

import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model

prompts = {
    'Python': [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
        "class LinkedList:\n    def __init__(self):\n        ",
    ],
    'C++': [
        "int fibonacci(int n) {\n    ",
        "int binary_search(vector<int>& arr, int target) {\n    ",
        "class LinkedList {\npublic:\n    LinkedList() {\n        ",
    ],
    'Java': [
        "public static int fibonacci(int n) {\n    ",
        "public static int binarySearch(int[] arr, int target) {\n    ",
        "public class LinkedList {\n    public LinkedList() {\n        ",
    ],
    'Rust': [
        "fn fibonacci(n: u64) -> u64 {\n    ",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    ",
        "struct LinkedList {\n    head: Option<Box<Node>>,\n}\n\nimpl LinkedList {\n    fn new() -> Self {\n        ",
    ],
}

def main():
    print("Loading GPT-2 XL...")
    model, tokenizer, device = load_model('gpt2-xl')

    for lang, lang_prompts in prompts.items():
        print(f"\n{'='*60}")
        print(f"  {lang}")
        print(f"{'='*60}")
        for prompt in lang_prompts:
            tokens = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    **tokens,
                    max_new_tokens=40,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(out[0], skip_special_tokens=True)
            new_text = completion[len(tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)):]
            print(f"\n  Prompt: {prompt.rstrip()}")
            print(f"  Completion: {new_text}")

    print("\nDone!")

if __name__ == '__main__':
    main()
