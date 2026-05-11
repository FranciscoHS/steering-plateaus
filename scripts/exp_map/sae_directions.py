"""Build a cache of 33 SAE-decoder directions at L=2 by sampling random
latent indices uniformly from a Gemma Scope SAE (default width 16k,
canonical L0).

Each direction is `W_dec[k]` L2-normalized — no reference to FineWeb or
contrastive prompts (fully blind sampling). Tests "feature-basis aligned
but not necessarily active at our anchors".

Loads the SAE directly from
  google/gemma-scope-9b-pt-res :: layer_<L>/width_<W>/average_l0_<L0>/params.npz
without sae_lens (which depends on transformer_lens, which has chronic
incompatibility with the latest transformers releases on RunPod).

Output:
  results/exp_map/data/directions/dirs_gemma_L2_sae.pkl
  schema: see top of file (matches the DoM cache schema).

Usage:
  python scripts/exp_map/sae_directions.py
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = "results/exp_map/data/directions"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIRS = 33
SEED = 42

# Canonical L0 mapping for gemma-scope-9b-pt-res. These are the values
# sae_lens uses for the "canonical" sae_id at each (layer, width). For
# layer 2 width 16k the canonical L0 is 67.
CANONICAL_L0 = {
    (2, "16k"): 67,
}


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DIRS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--repo",
                    default="google/gemma-scope-9b-pt-res")
    ap.add_argument("--l0", type=int, default=None,
                    help="average_l0 of the SAE variant; default = "
                         "canonical L0 from CANONICAL_L0[(layer, width)]")
    args = ap.parse_args()

    l0 = args.l0
    if l0 is None:
        key = (args.layer, args.width)
        if key not in CANONICAL_L0:
            raise SystemExit(f"no canonical L0 mapped for {key}; pass --l0")
        l0 = CANONICAL_L0[key]

    rel_path = f"layer_{args.layer}/width_{args.width}/average_l0_{l0}/params.npz"
    print(f"[{ts()}] downloading {args.repo} :: {rel_path}", flush=True)
    from huggingface_hub import hf_hub_download
    fpath = hf_hub_download(repo_id=args.repo, filename=rel_path)
    print(f"  -> {fpath}", flush=True)

    npz = np.load(fpath)
    print(f"[{ts()}] keys: {list(npz.keys())}", flush=True)
    W_dec = npz["W_dec"]   # expected (d_sae, d_in)
    print(f"  W_dec shape: {W_dec.shape}", flush=True)
    n_latents, d_model = W_dec.shape

    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(n_latents, size=args.n, replace=False).tolist())
    print(f"[{ts()}] sampled {args.n} latent indices "
          f"(seed={args.seed}): {indices[:5]} ... {indices[-3:]}",
          flush=True)

    raw = torch.from_numpy(W_dec[indices].copy()).float()   # (n, d_in)
    dirs = F.normalize(raw, dim=-1)

    sae_id = f"layer_{args.layer}/width_{args.width}/average_l0_{l0}"
    names = [f"sae_{k:06d}" for k in indices]
    out = {
        "directions": {n: dirs[i].clone() for i, n in enumerate(names)},
        "family": "sae_random",
        "signs": {n: "sae_decoder" for n in names},
        "prompt_set_hashes": {n: {"family": "sae_random",
                                   "sae_repo": args.repo,
                                   "sae_id": sae_id,
                                   "latent_index": int(idx),
                                   "seed": args.seed}
                              for n, idx in zip(names, indices)},
        "sae_repo": args.repo,
        "sae_id": sae_id,
        "width": args.width,
        "average_l0": l0,
        "latent_indices": indices,
        "seed": args.seed,
        "layer": args.layer,
        "model": args.target,
        "d_model": d_model,
        "n_latents": n_latents,
        "n_dirs": args.n,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}_sae.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    print(f"[{ts()}] saved -> {out_path}", flush=True)
    print(f"  ||d|| range: "
          f"[{dirs.norm(dim=-1).min():.6f}, {dirs.norm(dim=-1).max():.6f}]")
    print(f"  pairwise |cos| max (off-diag): "
          f"{(dirs @ dirs.T).fill_diagonal_(0).abs().max():.4f}")


if __name__ == "__main__":
    main()
