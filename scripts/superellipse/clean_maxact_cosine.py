#!/usr/bin/env python3
"""Strip a dict_max_activating.py report down to just the top-N COSINE hits per
direction (the raw-dot and mean-removed blocks are dropped). Much easier to
skim than the 3-metric interleaved dump.

Usage:
    python scripts/superellipse/clean_maxact_cosine.py \\
        --src .../max_activating/max_activating.txt \\
        --out .../max_activating/cosine_top15.txt --topn 15
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--topn', type=int, default=15)
args = ap.parse_args()

lines = open(args.src).read().splitlines()
hdr = re.compile(r'^=== atom\s+(\d+)\s+\|\s+cosine\s+\|')
row = re.compile(r'^\s*(\d+)\.\s+([\-0-9.]+)\s+(.*)$')

# Collect cosine block for each atom (atom_id -> list of (score, text)).
blocks = {}
i = 0
while i < len(lines):
    m = hdr.match(lines[i])
    if not m:
        i += 1
        continue
    aid = int(m.group(1))
    hits = []
    i += 1
    while i < len(lines) and not lines[i].startswith('==='):
        r = row.match(lines[i])
        if r:
            hits.append((float(r.group(2)), r.group(3).rstrip()))
        i += 1
    blocks[aid] = hits

with open(args.out, 'w') as f:
    f.write(f'Top-{args.topn} corpus hits by COSINE per direction '
            f'(from {args.src})\n')
    f.write('=' * 70 + '\n\n')
    for aid in sorted(blocks):
        hits = blocks[aid][:args.topn]
        mx = hits[0][0] if hits else float('nan')
        f.write(f'atom {aid:3d}   (max cos {mx:.3f})\n')
        for score, text in hits:
            f.write(f'   {score:.3f}  {text}\n')
        f.write('\n')

print(f'wrote {args.out}  ({len(blocks)} directions)')
