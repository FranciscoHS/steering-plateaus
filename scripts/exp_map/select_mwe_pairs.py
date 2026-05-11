"""Pick the bottom-K lowest-overlap pairs (subject to overlap < tau)
involving at least one MWE direction, and write them out as a pair list
for sweep_2d.py.

Pipeline (Phase 3 of Plans/exp_map_mwe_concepts.md):
  1. Load dirs_<target>_L<layer>.pkl.
  2. Compute |<d_i, d_j>| over every pair.
  3. Restrict to "candidate" pairs: at least one side is an MWE direction
     AND the pair has not yet been swept canonically (i.e. it's NOT in
     the existing all_pairs() over sem+lang+code, since those pairs are
     already in fits_<target>_L<layer>.pkl).
  4. Filter to overlap < tau (default 0.15) and take the bottom K
     (default 1000) by overlap.
  5. Write the result as a CSV (one pair per line) and a sweep_2d
     --pairs string (semicolon-separated) so the sweep can be launched
     on a pod without re-running this script.

Output:
  results/exp_map/data/pair_lists/mwe_pairs_<target>_L<layer>_ov<tau>_top<K>.csv
  results/exp_map/data/pair_lists/mwe_pairs_<target>_L<layer>_ov<tau>_top<K>.argv
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.exp_map.lib import registry

OUT_DIR = "results/exp_map/data/pair_lists"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.15,
                    help="overlap threshold |<d1,d2>|. Pairs above this "
                         "are dropped before bottom-K selection.")
    ap.add_argument("--k", type=int, default=1000,
                    help="how many lowest-overlap pairs to keep. Plan "
                         "default 1000 (~6 GPU-hr at ~21 s/pair).")
    ap.add_argument("--include_mwe_only", action="store_true",
                    help="if set, also include MWExMWE pairs. Default "
                         "policy keeps mwe×<existing> AND mwe×mwe so "
                         "every MWE direction appears in many pairs.")
    args = ap.parse_args()

    pkl = f"results/exp_map/data/directions/dirs_{args.target}_L{args.layer}.pkl"
    if not os.path.exists(pkl):
        print(f"missing {pkl} — run dirlib.extract_all first."); return
    with open(pkl, "rb") as f: blob = pickle.load(f)
    dirs = {k: v.cpu().numpy().astype(np.float64)
            for k, v in blob["directions"].items()}

    mwe_names = set(registry.MWE_KEYS)
    existing_names = (set(registry.DIRECTIONS) | set(registry.LANG_KEYS)
                      | set(registry.CODE_KEYS))

    all_names = sorted(dirs)
    print(f"[select] {len(all_names)} directions in cache "
          f"({len(set(all_names) & mwe_names)} MWE, "
          f"{len(set(all_names) & existing_names)} existing)")

    # Build the candidate list and their overlaps.
    cand: list[tuple[float, str, str]] = []
    for i, a in enumerate(all_names):
        for j in range(i + 1, len(all_names)):
            b = all_names[j]
            in_mwe = (a in mwe_names, b in mwe_names)
            if not any(in_mwe): continue        # at least one MWE side
            ov = float(abs(dirs[a] @ dirs[b]))
            cand.append((ov, a, b))

    print(f"[select] {len(cand)} candidate MWE-touching pairs")

    cand.sort(key=lambda t: t[0])
    kept = [(o, a, b) for (o, a, b) in cand if o < args.tau][: args.k]
    if not kept:
        print("nothing passed overlap filter — relax tau?"); return

    print(f"[select] kept {len(kept)} pairs "
          f"(overlap range {kept[0][0]:.4f} .. {kept[-1][0]:.4f})")

    tag = (f"mwe_pairs_{args.target}_L{args.layer}"
           f"_ov{args.tau:g}_top{len(kept)}")
    csv_path = os.path.join(OUT_DIR, tag + ".csv")
    with open(csv_path, "w") as f:
        f.write("overlap,a,b\n")
        for o, a, b in kept:
            f.write(f"{o:.6f},{a},{b}\n")

    # Sweep launcher form. semicolon-separated pairs match sweep_2d.py's
    # --pairs convention. Multi-line file so it's diff-friendly.
    argv_path = os.path.join(OUT_DIR, tag + ".argv")
    with open(argv_path, "w") as f:
        for i, (_, a, b) in enumerate(kept):
            f.write(f"{a},{b}")
            f.write(";\n" if i < len(kept) - 1 else "\n")
    print(f"saved {csv_path}\nsaved {argv_path}")


if __name__ == "__main__":
    main()
