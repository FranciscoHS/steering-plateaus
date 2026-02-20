#!/usr/bin/env python3
"""Quick test: can GPT-2 XL complete Haskell code?"""

import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model

PROMPTS = [
    "fibonacci :: Int -> Int\nfibonacci n =",
    "isPrime :: Int -> Bool\nisPrime n =",
    "reverseString :: String -> String\nreverseString s =",
    "sortList :: Ord a => [a] -> [a]\nsortList xs =",
    "binarySearch :: Ord a => [a] -> a -> Maybe Int\nbinarySearch xs target =",
    "calculateDistance :: Double -> Double -> Double -> Double -> Double\ncalculateDistance x1 y1 x2 y2 =",
    "myGcd :: Int -> Int -> Int\nmyGcd a b =",
    "maxValue :: [Int] -> Int\nmaxValue xs =",
    "validateEmail :: String -> Bool\nvalidateEmail email =",
    "matrixMultiply :: [[Int]] -> [[Int]] -> [[Int]]\nmatrixMultiply a b =",
]

model, tokenizer, device = load_model('gpt2-xl')

print(f"\n{'='*60}")
print(f"  HASKELL ({len(PROMPTS)} prompts)")
print(f"{'='*60}")

for prompt in PROMPTS:
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **tokens, max_new_tokens=60,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][tokens['input_ids'].shape[1]:]
    comp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    comp_display = comp.replace('\n', '\\n')
    if len(comp_display) > 140:
        comp_display = comp_display[:140] + '...'
    print(f"  {prompt.replace(chr(10), ' | ')}")
    print(f"    -> {comp_display}")
