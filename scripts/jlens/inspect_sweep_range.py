#!/usr/bin/env python3
"""Print the L2 response range of each J-Lens iso-plateau sweep, to diagnose
why superellipse fits found no valid contours at the chosen thresholds."""
import glob, os, pickle
import numpy as np

OUT = 'results/jlens/qwen3-8b/sweep_ranges.txt'

lines = [f"{'file':<18s} {'tokens':<28s} {'L2 min':>8s} {'axis1 max':>9s} "
         f"{'axis2 max':>9s} {'grid max':>8s}"]
for f in sorted(glob.glob('results/jlens/qwen3-8b/sweeps/sweep_*.pkl')):
    with open(f, 'rb') as fh:
        r = pickle.load(fh)
    med = np.median(r['l2'], axis=0)
    lines.append(f"{os.path.basename(f):<18s} "
                 f"{r['tokens'][0]!r}x{r['tokens'][1]!r:<14s} "
                 f"{med.min():>8.3f} {med[:, 0].max():>9.3f} "
                 f"{med[0, :].max():>9.3f} {med.max():>8.3f}")

txt = '\n'.join(lines)
with open(OUT, 'w') as fh:
    fh.write(txt)
print(txt)
print(f"\nSaved: {OUT}")
