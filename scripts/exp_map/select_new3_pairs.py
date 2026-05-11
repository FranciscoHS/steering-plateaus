"""Generate the 93-pair list for adding Wealth/Health/Certainty to a
canonical (target, layer) sweep.

Pair set:
  (W, x), (H, x), (C, x) for x in 30 kept_dirs
  + (W, H), (W, C), (H, C)

kept_dirs = 11 retained semantic + 10 lang + 9 code = 30. (We hard-
code the names rather than re-deriving from the registry so the prune
list — drop HonestyShort, TensePresent, Formal — stays explicit.)

Output: results/exp_map/data/pair_lists/new3_pairs_<target>_L<layer>.csv
        with columns (a, b). sweep_2d.py --pairs_file consumes it.

With --overlap_max <thr>, pairs with |<d_a, d_b>| >= thr are dropped
using the per-(target, layer) dirs cache. Skips writing rows where the
overlap exceeds the plot-time filter so we don't sweep pairs we'll
discard. Requires the W/H/C DoMs to already be in the dirs pkl.
"""
from __future__ import annotations
import os, pickle, sys, argparse
sys.path.insert(0, ".")

KEPT_SEM  = ["Age", "Era", "Gender", "Honesty", "Literary", "Number",
             "Person", "Refusal", "Sentiment", "Status", "Tense"]
LANG_KEYS = ["Portuguese", "Chinese", "Spanish", "French", "German",
             "Dutch", "Italian", "Japanese", "Russian", "Arabic"]
CODE_KEYS = ["Python", "JavaScript", "Java", "Rust", "Haskell", "Cpp",
             "Go", "Lisp", "TypeScript"]
NEW3      = ["Wealth", "Health", "Certainty"]

OUT_DIR  = "results/exp_map/data/pair_lists"
DIRS_DIR = "results/exp_map/data/directions"


def _load_dirs(target: str, layer: int) -> dict:
    pkl = os.path.join(DIRS_DIR, f"dirs_{target}_L{layer}.pkl")
    with open(pkl, "rb") as f:
        return pickle.load(f)["directions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--overlap_max", type=float, default=None,
                    help="If set, drop pairs with |<d_a, d_b>| >= this "
                         "threshold using the dirs cache.")
    args = ap.parse_args()

    others = KEPT_SEM + LANG_KEYS + CODE_KEYS
    pairs: list[tuple[str, str]] = []
    for n in NEW3:
        for o in others:
            pairs.append((n, o))
    # internal new x new
    pairs += [("Wealth", "Health"), ("Wealth", "Certainty"),
              ("Health", "Certainty")]
    assert len(pairs) == 3 * len(others) + 3 == 93

    if args.overlap_max is not None:
        dirs = _load_dirs(args.target, args.layer)
        missing = [n for n in NEW3 + others if n not in dirs]
        if missing:
            raise SystemExit(f"dirs cache missing {missing}; "
                             f"run _extract_new3.py first")
        kept, dropped = [], []
        for a, b in pairs:
            ov = abs(float((dirs[a] * dirs[b]).sum().item()))
            if ov < args.overlap_max:
                kept.append((a, b))
            else:
                dropped.append((a, b, ov))
        print(f"[overlap_max={args.overlap_max}] kept {len(kept)}/"
              f"{len(pairs)} pairs; dropped {len(dropped)}")
        for a, b, ov in dropped:
            print(f"  drop {a:10s} x {b:14s}  |<,>|={ov:.3f}")
        pairs = kept

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                        f"new3_pairs_{args.target}_L{args.layer}.csv")
    with open(out, "w") as f:
        f.write("a,b\n")
        for a, b in pairs:
            f.write(f"{a},{b}\n")
    print(f"saved {out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
