#!/usr/bin/env python3
"""Dump each concept's matched holdout source prompts (from the fsec-anonymous
registry) to a pkl, so the holdout sweep can run without importing fsec.

Usage (on the pod, fsec on path):
    FSEC_PATH=/workspace/fsec-anonymous python -u \
        scripts/superellipse/dump_holdout_prompts.py \
        --names_from results/scaffold_qwen_L14/dictionary.pt \
        --out results/holdout_prompts_qwen.pkl
"""
import argparse
import os
import pickle
import sys

import torch

# Resolve fsec-anonymous and import its registry helper.
FSEC = os.environ.get('FSEC_PATH', '/workspace/fsec-anonymous')
sys.path.insert(0, FSEC)
from scripts.lib import registry as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names_from', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    names = torch.load(args.names_from, map_location='cpu', weights_only=False)['names']
    prompts = {}
    for n in names:
        try:
            prompts[n] = list(R.get_source_prompts(n))
        except Exception as e:
            print(f"  {n}: NO holdout source ({str(e)[:50]})")
    for n in prompts:
        print(f"  {n:14s} n={len(prompts[n])}  e.g. {prompts[n][0]!r}")
    with open(args.out, 'wb') as f:
        pickle.dump(prompts, f)
    print(f"saved {len(prompts)} concepts -> {args.out}")


if __name__ == '__main__':
    main()
