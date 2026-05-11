"""Run 1D exp-map sweeps for arbitrary directions at a cached anchor set.

Anchors loaded from results/exp_map/data/activations/acts_<target>_L<layer>_<anchor_source>.pkl
(must already exist; re-uses the exact same activations as a prior sweep).
Directions loaded from the cached dirs_<target>_L<layer>.pkl via dirlib.extract_all.

Output: results/exp_map/data/sweeps_1d/sweep1d_<target>_L<layer>_<dir>_<anchor_source>_<max>deg.pkl
(no random baseline; baselines live in the original MWE pkl with the same anchor_source.)

Usage:
  python scripts/exp_map/sweep_1d_at_anchors.py --target gemma --layer 2 \
      --anchor_source mwe_persona_psychopathy_match \
      --directions Gender,French
"""
from __future__ import annotations
import os, sys, pickle, argparse, time, subprocess
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib
from scripts.exp_map.lib.sweep_core import run_1d_per_anchors
from scripts.exp_map.lib.parametrize import eps_thresholds, CHART

ACTS_DIR = "results/exp_map/data/activations"
OUT_DIR = "results/exp_map/data/sweeps_1d"
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 51
MAX_ANGLE = 75.0


def _git_sha(path: str) -> str:
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
    ap.add_argument("--anchor_source", required=True,
                    help="comma-separated cached anchor source names, e.g. "
                         "mwe_persona_psychopathy_match,mwe_persona_psychopathy_match")
    ap.add_argument("--directions", required=True,
                    help="comma-separated direction names from dirs cache")
    ap.add_argument("--max_angle", type=float, default=MAX_ANGLE)
    ap.add_argument("--n_steps", type=int, default=N_STEPS)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfg = registry.TARGETS[args.target]
    perturb_layer = args.layer
    print(f"[{ts()}] target={args.target} L={perturb_layer}", flush=True)

    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2

    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, perturb_layer)
    all_dirs = dirs_blob["directions"]
    signs = dirs_blob["signs"]

    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    parametrize_sha = _git_sha("scripts/exp_map/lib/parametrize.py")

    for anchor_source in args.anchor_source.split(","):
        acts_path = os.path.join(
            ACTS_DIR, f"acts_{args.target}_L{perturb_layer}_{anchor_source}.pkl")
        if not os.path.exists(acts_path):
            print(f"  MISS anchors {acts_path}"); continue
        with open(acts_path, "rb") as f:
            acts_blob = pickle.load(f)
        contexts = acts_blob["contexts"]
        activations = acts_blob["activations"]
        print(f"  anchors={anchor_source} N={len(activations)}", flush=True)

        for name in args.directions.split(","):
            if name not in all_dirs:
                print(f"    miss {name}"); continue
            out_path = os.path.join(
                OUT_DIR,
                f"sweep1d_{args.target}_L{perturb_layer}_{name}_"
                f"{anchor_source}_{int(args.max_angle)}deg.pkl")
            if os.path.exists(out_path) and args.skip_existing:
                print(f"    skip {name} (exists)"); continue
            d = all_dirs[name].float()
            t0 = time.time()
            l2 = run_1d_per_anchors(model, contexts, activations, d,
                                     perturb_layer, measure_layer, device,
                                     angles_rad)
            print(f"    {name:30s} {time.time()-t0:.1f}s", flush=True)
            out = {
                "chart": CHART,
                "parametrize_git_sha": parametrize_sha,
                "prompt_set_hashes": {
                    **dirs_blob["prompt_set_hashes"][name],
                    "source": acts_blob["prompt_set_hash"],
                },
                "direction_signs": {name: signs[name]},
                "eps_thresholds": eps_thresholds(),
                "model": args.target,
                "model_full": cfg["model"],
                "perturb_layer": perturb_layer,
                "measure_layer": measure_layer,
                "direction_name": name,
                "source_name": anchor_source,
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
