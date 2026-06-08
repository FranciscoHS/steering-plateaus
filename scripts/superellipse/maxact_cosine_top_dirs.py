"""Cosine-only max-activating report for the top directions by max cosine.

Reads a dict_max_activating.py report, keeps ONLY the cosine blocks (drops the
raw-dot and mean-removed blocks), ranks directions by their top-1 cosine hit,
and writes a clean report for just the top-N directions.

Usage:
    python scripts/superellipse/maxact_cosine_top_dirs.py \\
        --src .../max_activating/max_activating.txt \\
        --out .../max_activating/cosine_top10_dirs.txt \\
        --top_dirs 10 --topn 15
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--top_dirs', type=int, default=10,
                help='# of directions to keep, ranked by max cosine')
ap.add_argument('--topn', type=int, default=15,
                help='# of cosine hits to show per direction')
args = ap.parse_args()

lines = open(args.src).read().splitlines()
hdr = re.compile(r'^=== atom\s+(\d+)\s+\|\s+cosine\s+\|')
row = re.compile(r'^\s*(\d+)\.\s+([\-0-9.]+)\s+(.*)$')

# Collect the cosine block (list of (score, text)) for each atom.
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

# Rank directions by top-1 cosine hit, keep the top_dirs.
ranked = sorted(blocks.items(), key=lambda kv: -(kv[1][0][0] if kv[1] else -1))
ranked = ranked[:args.top_dirs]

with open(args.out, 'w') as f:
    f.write(f'Top {args.top_dirs} directions by max cosine (cosine hits only)\n')
    f.write(f'source: {args.src}\n')
    f.write('=' * 70 + '\n\n')
    for rank, (aid, hits) in enumerate(ranked):
        maxc = hits[0][0] if hits else float('nan')
        f.write(f'#{rank+1}  atom {aid}   (max cos {maxc:.3f})\n')
        for score, text in hits[:args.topn]:
            f.write(f'   {score:6.3f}  {text}\n')
        f.write('\n')
print(f'Saved {args.out}  ({len(ranked)} directions, top {args.topn} hits each)')
