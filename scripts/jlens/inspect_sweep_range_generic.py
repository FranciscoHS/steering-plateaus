#!/usr/bin/env python3
"""Print the L2 response range of each J-Lens iso-plateau sweep in a results
dir (generic version of inspect_sweep_range.py, parameterized by --dir).
Used to pick fit thresholds: they must sit well above the noise floor
(L2 min) and below the axis maxes.

Usage: python scripts/jlens/inspect_sweep_range_generic.py \
           --dir results/jlens/qwen3.6-27b
"""
import argparse, glob, os, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True,
                    help='results dir containing sweeps/ subdir')
    args = ap.parse_args()
    out = os.path.join(args.dir, 'sweep_ranges.txt')

    lines = [f"{'file':<18s} {'tokens':<28s} {'L2 min':>8s} {'axis1 max':>9s} "
             f"{'axis2 max':>9s} {'grid max':>8s}"]
    for f in sorted(glob.glob(os.path.join(args.dir, 'sweeps', 'sweep_*.pkl'))):
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        med = np.median(r['l2'], axis=0)
        lines.append(f"{os.path.basename(f):<18s} "
                     f"{r['tokens'][0]!r}x{r['tokens'][1]!r:<14s} "
                     f"{med.min():>8.3f} {med[:, 0].max():>9.3f} "
                     f"{med[0, :].max():>9.3f} {med.max():>8.3f}")

    txt = '\n'.join(lines)
    with open(out, 'w') as fh:
        fh.write(txt)
    print(txt)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
