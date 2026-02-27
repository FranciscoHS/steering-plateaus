#!/usr/bin/env python3
"""Quick check: does GPT-2 Large reliably predict Python-related tokens?"""
import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model

model, tokenizer, device = load_model('gpt2-large')

prompts = [
    'def fibonacci(n):\n    if n <= 1:\n        return n',
    'import numpy as np\ndata = np.array([1, 2, 3])',
    'for i in range(100):\n    if i % 2 == 0:',
    'x = [i**2 for i in range(10)]\nprint(x',
    'Python is a high-level interpreted programming language known',
    'Many data scientists prefer using Python because of its',
    'List comprehensions in Python provide a concise way to',
    'Python uses indentation instead of braces to define code',
    'The def keyword in Python is used to define a',
    'In Python, you can iterate over a dictionary using',
]

for p in prompts:
    tokens = tokenizer(p, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**tokens).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_ids = probs.topk(10)
    top_tokens = [(tokenizer.decode([tid]).strip(), f"{v:.3f}") for tid, v in zip(top_ids, top_vals)]
    print(f"{p[:55]:<55} -> {top_tokens[:5]}")
