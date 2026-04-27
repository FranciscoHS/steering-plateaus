"""E2: 2D iso-plateau pair sweeps in the exp-map chart.

Per (model, pair=(d1, d2)): 31x31 grid in (alpha_1, alpha_2) up to 60 deg,
with 30 FineWeb anchors per pair (Plan D2/D3/D4).

Output: results/exp_map/data/sweeps_2d/sweep2d_<target>_L<layer>_<d1>__<d2>_fineweb_60deg.pkl
{
  'angles_deg':   (n,),
  'l2':           (n_anchors, n, n),
  ... full PKL header (see Plans/exp_map_workspace.md)
}

Usage:
  python scripts/exp_map/sweep_2d.py --target gemma --layer 2          # all 20 Tier-1 pairs
  python scripts/exp_map/sweep_2d.py --target gemma --layer 2 \
      --pairs Gender,Tense                                              # one pair
"""
from __future__ import annotations
import os, sys, pickle, argparse, time, subprocess
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib, activations as actlib
from scripts.exp_map.lib.sweep_core import run_2d_per_anchors
from scripts.exp_map.lib.parametrize import eps_thresholds, CHART

OUT_DIR = "results/exp_map/data/sweeps_2d"
os.makedirs(OUT_DIR, exist_ok=True)

# locked params (Plan E2, D2/D3/D4)
N_STEPS_2D = 31
MAX_ANGLE_2D = 60.0
N_ANCHORS = 30


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", path],
            text=True).strip()
    except Exception:
        return "unknown"


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(registry.TARGETS))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--pairs", default=None,
                    help="optional override; comma-separated single pair "
                         "like 'Gender,Tense' OR semicolon-separated multiple "
                         "pairs like 'Gender,Tense;Gender,Sentiment'")
    ap.add_argument("--pair_set", default="all",
                    choices=["all", "tier1"],
                    help="default 'all' = C(N,2) over semantic+lang+code; "
                         "'tier1' = the 20-pair v1 list")
    ap.add_argument("--shard", default=None,
                    help="optional 'i/N' shard selector; e.g. '0/4' runs the "
                         "first quarter of the pair list (after sorting). "
                         "Use to parallelize a model across multiple pods.")
    ap.add_argument("--max_angle", type=float, default=MAX_ANGLE_2D)
    ap.add_argument("--n_steps", type=int, default=N_STEPS_2D)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfg = registry.TARGETS[args.target]
    perturb_layer = args.layer
    print(f"[{ts()}] target={args.target} model={cfg['model']} "
          f"L={perturb_layer}", flush=True)
    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers} measure_layer={measure_layer}", flush=True)

    if args.pairs:
        pair_list = []
        for chunk in args.pairs.split(";"):
            a, b = chunk.split(",")
            pair_list.append((a.strip(), b.strip()))
    elif args.pair_set == "tier1":
        pair_list = registry.TIER1_PAIRS
    else:
        pair_list = registry.all_pairs()
    if args.shard:
        i, N = (int(x) for x in args.shard.split("/"))
        pair_list = sorted(pair_list)[i::N]
    print(f"[{ts()}] {len(pair_list)} pairs (set={args.pair_set}, "
          f"shard={args.shard})", flush=True)

    # Decide which directions we actually need based on pair_list (semantic
    # + language + code as needed); extends an existing cache rather than
    # rebuilding it from scratch.
    needed = sorted({n for ab in pair_list for n in ab})
    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, perturb_layer, names=needed)
    all_dirs = dirs_blob["directions"]
    signs = dirs_blob["signs"]

    # FineWeb anchors (shared across pairs)
    print(f"[{ts()}] loading FineWeb anchors", flush=True)
    fw = actlib.fineweb_acts(model, tokenizer, device, args.target,
                              perturb_layer, n=N_ANCHORS)
    contexts = fw["contexts"]
    activations = fw["activations"]
    print(f"  N={len(activations)}, ||a||={activations.norm(dim=-1).mean():.2f}",
          flush=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    parametrize_sha = _git_sha("scripts/exp_map/lib/parametrize.py")

    for (a, b) in pair_list:
        if a not in all_dirs or b not in all_dirs:
            print(f"  miss {a}x{b}"); continue
        out_path = os.path.join(
            OUT_DIR,
            f"sweep2d_{args.target}_L{perturb_layer}_{a}__{b}_fineweb_"
            f"{int(args.max_angle)}deg.pkl")
        if os.path.exists(out_path) and args.skip_existing:
            print(f"  skip {a}x{b} (exists)"); continue

        d1 = all_dirs[a].float()
        d2 = all_dirs[b].float()
        t0 = time.time()
        r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                                perturb_layer, measure_layer, device,
                                angles_rad)
        l2 = r["l2"]
        print(f"  {a:>10s} x {b:<10s} {time.time()-t0:.1f}s "
              f"l2_max={l2.max():.2f}", flush=True)

        out = {
            "chart": CHART,
            "parametrize_git_sha": parametrize_sha,
            "prompt_set_hashes": {
                a: dirs_blob["prompt_set_hashes"][a],
                b: dirs_blob["prompt_set_hashes"][b],
                "source": None,  # FineWeb
            },
            "seed_anchors": fw.get("seed", 42),
            "seed_random_dirs": None,
            "direction_signs": {a: signs[a], b: signs[b]},
            "eps_thresholds": eps_thresholds(),
            "model": args.target,
            "model_full": cfg["model"],
            "perturb_layer": perturb_layer,
            "measure_layer": measure_layer,
            "direction_labels": (a, b),
            "source_name": "fineweb",
            "angles_deg": angles_deg,
            "n_anchors": len(activations),
            "l2": l2,
            "anchor_norm_mean": float(activations.norm(dim=-1).mean()),
        }
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(out, f)
        os.replace(tmp, out_path)

    print(f"[{ts()}] done", flush=True)


if __name__ == "__main__":
    main()
