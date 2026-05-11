"""Extract Wealth/Health/Certainty DoMs at (target, layer) and merge
into the existing dirs cache. Idempotent."""
from __future__ import annotations
import sys, argparse
sys.path.insert(0, ".")
from src.model import load_model
from scripts.exp_map.lib import directions as dirlib, registry
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(registry.TARGETS))
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()
    cfg = registry.TARGETS[args.target]
    m, tok, dev = load_model(cfg["model"], dtype=torch.bfloat16)
    blob = dirlib.extract_all(m, tok, dev, args.target, args.layer,
                               names=["Wealth", "Health", "Certainty"])
    for n in ("Wealth", "Health", "Certainty"):
        d = blob["directions"][n]
        print(f"  {n:10s}  ||d||={d.norm().item():.4f}")


if __name__ == "__main__":
    main()
