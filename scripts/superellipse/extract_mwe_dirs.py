#!/usr/bin/env python3
"""Extract DoM directions for all MWE persona concepts and export them as a
contrastive-style dictionary.pt ({'dictionary': [N,d], 'names': [...]}) so
they can be fed to dict_vs_contrastive_align.py / the dashboard.

Uses the exp_map machinery (registry + directions cache): only names missing
from results/exp_map/data/directions/dirs_<target>_L<layer>.pkl are computed.

    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/extract_mwe_dirs.py \\
        --target qwen --layer 14 --out_pt results/mwe_qwen_L14/dictionary.pt
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import torch

from src.model import load_model
from scripts.exp_map.lib import directions as dirlib, registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='qwen')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--out_pt', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    model_name = registry.TARGETS[args.target]['model']
    print(f'Loading {model_name}', flush=True)
    model, tok, dev = load_model(model_name, dtype=dtype_map[args.dtype])

    names = list(registry.MWE_KEYS)
    print(f'{len(names)} MWE concepts', flush=True)
    blob = dirlib.extract_all(model, tok, dev, args.target, args.layer,
                              names=names)
    missing = [n for n in names if n not in blob['directions']]
    if missing:
        print(f'WARNING: {len(missing)} concepts missing: {missing[:5]}...',
              flush=True)
    have = [n for n in names if n in blob['directions']]
    D = torch.stack([blob['directions'][n].float() for n in have])
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save({'dictionary': D, 'names': have,
                'source': f'MWE DoM, {model_name} L{args.layer}'}, args.out_pt)
    print(f'Saved {args.out_pt}  ({D.shape[0]} dirs, d={D.shape[1]})',
          flush=True)


if __name__ == '__main__':
    main()
