"""Single-pair pilot at high anchor count (default N=100) to find the
empirical noise floor of the per-pair fitted superellipse exponent n.

Pipeline:
  1. Load gemma-2-9b, build wiki_en anchors at N=100 (separate cache from
     the production N=30 anchors so we don't clobber).
  2. Run the standard 2D sweep for one pair (default Gender x Tense).
  3. Bootstrap with replacement at sizes 10, 20, 30, 50, 70, 100 (B=200)
     and report median per-pair std at each — this is the empirical analog
     of the 1/sqrt(N) prediction in scripts/exp_map/anchor_subsample_stability.py.

Output: results/exp_map/data/anchor_floor_pilot_<pair>_<source>_n<N>.pkl
        with the per-anchor grid + bootstrap stats.

Usage:
  python scripts/exp_map/anchor_floor_pilot.py \
    --pair Gender,Tense --anchor_source wiki_en --n_anchors 100
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib
from scripts.exp_map.lib.activations import _tokenize_fixed_length
from scripts.exp_map.lib.sweep_core import run_2d_per_anchors
from scripts.exp_map.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse,
)

OUT_DIR = "results/exp_map/data"
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS_2D = 31
MAX_ANGLE_2D = 60.0
SEQ_LEN = 5
SEED = 42


def fit_n_from_grid(angles, grid, threshold_frac=0.5,
                     max_alpha_deg=60.0) -> float:
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = threshold_frac * base
    contour = extract_contour(angles, grid, thresh)
    if contour.size == 0: return np.nan
    edge_eps = 1.0
    grid_max = float(angles.max())
    contour = contour[(contour[:, 0] < grid_max - edge_eps)
                       & (contour[:, 1] < grid_max - edge_eps)]
    contour = contour[(contour[:, 0] <= max_alpha_deg)
                       & (contour[:, 1] <= max_alpha_deg)]
    if len(contour) < 3: return np.nan
    t1 = axis_intercept(angles, grid[:, 0], thresh)
    t2 = axis_intercept(angles, grid[0, :], thresh)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return np.nan
    return fit_superellipse(contour[:, 0] / t1, contour[:, 1] / t2)["p"]


def load_wiki_anchors(model, tokenizer, device, target, layer, n,
                       language="en", seed=SEED, seq_len=SEQ_LEN):
    """Standalone wiki anchor loader that respects n in cache filename so
    we don't conflict with the canonical N=30 cache."""
    from datasets import load_dataset
    cache_dir = f"/workspace/wiki_{language}_cache_{target}"
    cache_file = os.path.join(
        cache_dir,
        f"wiki_{language}_n{n}_seq{seq_len}_seed{seed}.pt")
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{language}",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    token_lists = _tokenize_fixed_length(
        (s["text"] for s in ds), n, tokenizer, seq_len, cache_file)
    contexts, acts = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="Gender,Tense",
                    help="comma-separated pair, e.g. 'Gender,Tense'")
    ap.add_argument("--anchor_source", default="wiki_en",
                    choices=["wiki_en", "wiki_zh"])
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--n_anchors", type=int, default=100)
    ap.add_argument("--B", type=int, default=200,
                    help="bootstrap resamples per N")
    ap.add_argument("--Ns", default="10,20,30,50,70,100")
    args = ap.parse_args()
    Ns = [int(x) for x in args.Ns.split(",")]
    a, b = args.pair.split(",")
    a, b = a.strip(), b.strip()

    cfg = registry.TARGETS[args.target]
    print(f"[{time.strftime('%H:%M:%S')}] loading model {cfg['model']}",
          flush=True)
    model, tokenizer, device = load_model(cfg["model"], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2

    print(f"[{time.strftime('%H:%M:%S')}] extracting DoM directions for "
          f"{a}, {b}", flush=True)
    dirs_blob = dirlib.extract_all(model, tokenizer, device, args.target,
                                    args.layer, names=[a, b])
    d1 = dirs_blob["directions"][a].float()
    d2 = dirs_blob["directions"][b].float()

    language = args.anchor_source.split("_")[1]
    print(f"[{time.strftime('%H:%M:%S')}] loading wiki_{language} anchors "
          f"(N={args.n_anchors})", flush=True)
    contexts, activations = load_wiki_anchors(
        model, tokenizer, device, args.target, args.layer,
        n=args.n_anchors, language=language)
    print(f"  N={len(activations)}, ||a||={activations.norm(dim=-1).mean():.2f}",
          flush=True)

    angles_deg = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles_deg)

    print(f"[{time.strftime('%H:%M:%S')}] running 2D sweep on {a} x {b}",
          flush=True)
    t0 = time.time()
    r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                            args.layer, measure_layer, device, angles_rad,
                            mode="geodesic")
    print(f"  took {time.time()-t0:.1f}s, l2 shape={r['l2'].shape}, "
          f"l2_max={r['l2'].max():.2f}", flush=True)

    # Bootstrap: for each N, B subsamples (with replacement), refit n.
    rng = np.random.default_rng(0)
    per_N_n = {N: [] for N in Ns}
    n_anchors = r["l2"].shape[0]
    for N in Ns:
        ns = []
        for _ in range(args.B):
            idx = rng.choice(n_anchors, size=N, replace=True)
            grid = np.median(r["l2"][idx], axis=0)
            v = fit_n_from_grid(angles_deg, grid)
            if np.isfinite(v): ns.append(v)
        per_N_n[N] = ns

    print(f"\n{a} x {b}  source={args.anchor_source}  N_pool={n_anchors}  "
          f"B={args.B} bootstraps")
    print(f"{'N':>4s}  {'#fit':>5s}  {'mean':>6s}  {'std':>6s}  "
          f"{'p05':>6s}  {'p95':>6s}")
    for N in Ns:
        ns = np.array(per_N_n[N])
        if ns.size < 2:
            print(f"{N:>4d}  no data"); continue
        print(f"{N:>4d}  {ns.size:>5d}  {ns.mean():>6.3f}  {ns.std():>6.3f}  "
              f"{np.percentile(ns,5):>6.3f}  {np.percentile(ns,95):>6.3f}")

    # Save full per-anchor grid for later re-analysis.
    out = {"pair": (a, b), "anchor_source": args.anchor_source,
           "n_anchors": n_anchors, "angles_deg": angles_deg,
           "l2": r["l2"], "per_N_n": per_N_n,
           "Ns": Ns, "B": args.B}
    out_pkl = os.path.join(
        OUT_DIR,
        f"anchor_floor_pilot_{a}_{b}_{args.anchor_source}_n{n_anchors}.pkl")
    with open(out_pkl, "wb") as f: pickle.dump(out, f)
    print(f"\nsaved {out_pkl}")


if __name__ == "__main__":
    main()
