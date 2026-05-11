"""Dump full bisection traces (prompts + continuations + judge verdicts)
for the outlier rays of a coherence_boundary.py run, into a plain text
file suitable for human inspection.
"""
from __future__ import annotations
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coherence_json", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phis", type=float, nargs="+",
                    default=[0.0, 18.0, 24.0, 30.0, 36.0, 42.0],
                    help="phi_deg values of rays to include")
    args = ap.parse_args()

    with open(args.coherence_json) as f:
        d = json.load(f)
    prompts = d["prompts"]

    out_path = args.out or args.coherence_json.replace(".json", "_outliers.txt")
    by_phi = {round(r["phi_deg"], 3): r for r in d["rays"]}

    with open(out_path, "w") as f:
        f.write(f"# Coherence outliers from {args.coherence_json}\n")
        f.write(f"# {d['target']} L{d['layer']} {d['d1']}x{d['d2']} "
                f"judge={d['judge_model']} threshold={d['coherent_threshold']}/{d['n_prompts']}\n\n")
        for phi in args.phis:
            ray = by_phi.get(round(phi, 3))
            if ray is None:
                f.write(f"==== phi={phi} not found ====\n\n")
                continue
            f.write("=" * 72 + "\n")
            f.write(f"RAY phi={ray['phi_deg']:.2f} deg   "
                    f"boundary_r = {ray['boundary_r']:.3f} deg   "
                    f"(r_high after bisection = {ray['r_high']:.3f})\n")
            f.write("=" * 72 + "\n\n")
            for ev in ray["trace"]:
                f.write(f"--- iter {ev['iter']}   r={ev['r']:.3f}   "
                        f"alpha1={ev['alpha1_deg']:.3f} "
                        f"alpha2={ev['alpha2_deg']:.3f}   "
                        f"yes={ev['n_yes']}/{ev['n']}   "
                        f"{'COHERENT' if ev['coherent'] else 'incoherent'}"
                        "\n")
                for k, (p, c, j) in enumerate(zip(prompts,
                                                  ev["continuations"],
                                                  ev["judgments"])):
                    mark = "Y" if j["yes"] else "N"
                    f.write(f"  [{k}] {mark}  prompt: {p!r}\n")
                    f.write(f"       cont:   {c!r}\n")
                    f.write(f"       judge:  {j['raw']!r}\n")
                    if j["error"]:
                        f.write(f"       ERROR:  {j['error']}\n")
                f.write("\n")
            f.write("\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
