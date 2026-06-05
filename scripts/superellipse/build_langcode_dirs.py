#!/usr/bin/env python3
"""Build the language (en->lang) and code (en->lang) contrastive directions
for qwen3-1.7b at a given layer and save them as a dictionary.pt, so the
dict_* diagnostics (e.g. SAE alignment) can run on them.

These were dropped from the dictionary-build semantic scaffold but still exist
in the fsec registry (LANG_KEYS / CODE_KEYS).
"""
import argparse
import sys
sys.path.insert(0, '.')
import torch
from src.model import load_model
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold,
)

LANG = ['Portuguese', 'Chinese', 'Spanish', 'French', 'German', 'Dutch',
        'Italian', 'Japanese', 'Russian', 'Arabic']
CODE = ['Python', 'JavaScript', 'Java', 'Rust', 'Haskell', 'Cpp', 'Go',
        'Lisp', 'TypeScript']

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='qwen3-1.7b')
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--out', required=True)
ap.add_argument('--dtype', default='bfloat16')
args = ap.parse_args()

dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
names = LANG + CODE
print(f'Building {len(names)} lang+code dirs at L{args.layer}: {names}',
      flush=True)
D = load_or_build_scaffold(model, tokenizer, device, args.model, args.layer,
                           names)
D = (D / D.norm(dim=-1, keepdim=True)).cpu().float()
torch.save({'dictionary': D, 'n_done': D.shape[0], 'names': names,
            'n_lang': len(LANG), 'n_code': len(CODE)}, args.out)
print(f'Saved {args.out}  shape={tuple(D.shape)}', flush=True)
