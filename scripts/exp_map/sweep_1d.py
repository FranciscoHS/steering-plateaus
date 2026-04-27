"""E1: 1D single-direction exp-map sweeps.

Per (model, direction): rotate from 0 to MAX_ANGLE degrees along the
DoM direction, anchored at matched-source holdout activations. Also
sweep N_RAND random unit vectors at the same anchors as a baseline.

Output: results/exp_map/data/sweeps_1d/sweep1d_<target>_L<layer>_<dir>_<source>_<max>deg.pkl
{
  'angles_deg':   (N_STEPS,),
  'l2':           (N_ANCHORS, N_STEPS),     contrastive direction
  'l2_random':    (N_RAND, N_ANCHORS, N_STEPS),
  ... full PKL header (see Plans/exp_map_workspace.md)
}

Usage:
  python scripts/exp_map/sweep_1d.py --target gemma --layer 2 \
      --directions Gender,Sentiment    # comma-separated; default = all 14
"""
from __future__ import annotations
import os, sys, pickle, argparse, time, hashlib, subprocess
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib, activations as actlib
from scripts.exp_map.lib.sweep_core import run_1d_per_anchors
from scripts.exp_map.lib.parametrize import eps_thresholds, CHART

OUT_DIR = "results/exp_map/data/sweeps_1d"
os.makedirs(OUT_DIR, exist_ok=True)

# locked params (Plan E1)
N_STEPS = 51
MAX_ANGLE = 75.0
N_RAND = 10
SEED_RAND = 42


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
    ap.add_argument("--directions", default=None,
                    help="comma-separated; default = all 14")
    ap.add_argument("--max_angle", type=float, default=MAX_ANGLE)
    ap.add_argument("--n_steps", type=int, default=N_STEPS)
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

    direction_names = (args.directions.split(",") if args.directions
                       else list(registry.DIRECTIONS))

    # Build (or load) all DoM directions for this target/layer
    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, perturb_layer)
    all_dirs = dirs_blob["directions"]
    signs = dirs_blob["signs"]

    angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)

    parametrize_sha = _git_sha("scripts/exp_map/lib/parametrize.py")

    for name in direction_names:
        if name not in all_dirs:
            print(f"  miss {name}"); continue
        spec = registry.DIRECTIONS[name]
        src_mod, src_var = spec["source"]
        source_name = src_var.lower()
        out_path = os.path.join(
            OUT_DIR,
            f"sweep1d_{args.target}_L{perturb_layer}_{name}_{source_name}_"
            f"{int(args.max_angle)}deg.pkl")
        if os.path.exists(out_path) and args.skip_existing:
            print(f"  skip {name} (exists)"); continue

        # source activations
        prompts = registry.get_source_prompts(name)
        acts_blob = actlib.holdout_acts(
            model, tokenizer, device, args.target, perturb_layer,
            source_name, prompts)
        contexts = acts_blob["contexts"]
        activations = acts_blob["activations"]

        # contrastive direction sweep
        d = all_dirs[name].float()
        t0 = time.time()
        l2 = run_1d_per_anchors(model, contexts, activations, d,
                                 perturb_layer, measure_layer, device,
                                 angles_rad)
        # random baseline (N_RAND directions, same anchors)
        rng = np.random.RandomState(SEED_RAND)
        D = activations.shape[-1]
        rand_dirs = F.normalize(
            torch.tensor(rng.randn(N_RAND, D), dtype=torch.float32), dim=-1)
        l2_rand = np.zeros((N_RAND, len(activations), args.n_steps),
                           dtype=np.float32)
        for ri in range(N_RAND):
            l2_rand[ri] = run_1d_per_anchors(
                model, contexts, activations, rand_dirs[ri],
                perturb_layer, measure_layer, device, angles_rad)
        print(f"  {name:14s} {time.time()-t0:.1f}s "
              f"src={source_name} N={len(activations)}", flush=True)

        out = {
            "chart": CHART,
            "parametrize_git_sha": parametrize_sha,
            "prompt_set_hashes": {
                **dirs_blob["prompt_set_hashes"][name],
                "source": acts_blob["prompt_set_hash"],
            },
            "seed_anchors": SEED_RAND,
            "seed_random_dirs": SEED_RAND,
            "direction_signs": {name: signs[name]},
            "eps_thresholds": eps_thresholds(),
            "model": args.target,
            "model_full": cfg["model"],
            "perturb_layer": perturb_layer,
            "measure_layer": measure_layer,
            "direction_name": name,
            "source_name": source_name,
            "angles_deg": angles_deg,
            "n_anchors": len(activations),
            "l2": l2,
            "l2_random": l2_rand,
            "n_random": N_RAND,
            "anchor_norm_mean": float(activations.norm(dim=-1).mean()),
        }
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(out, f)
        os.replace(tmp, out_path)

    print(f"[{ts()}] done", flush=True)


if __name__ == "__main__":
    main()
