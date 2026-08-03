#!/usr/bin/env python3
"""1D response-curve comparison for Qwen3.6-27B L36->L62.

The responsiveness question (distinct from the superellipse-shape p): do the
J-Lens directions elicit a stronger downstream L2 response as you rotate along
them than random directions do, and how does that compare to contrastive
directions? Perturb a single direction on the sphere by angle alpha and
measure L2 at the penultimate layer, over 30 FineWeb anchors, for three
families:
  - J-Lens ZH   : individual token J-Lens vectors (from jlens_zh100 pairs)
  - Contrastive : diff-of-means directions (dirs_qwen36_L36.pkl)
  - Random      : isotropic unit vectors

Median +- IQR band per family, overlaid on one axis. Run on the pod.
"""
import os, sys, pickle, argparse, json
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.exp_map.lib.sweep_core import run_1d_per_anchors

MODEL_NAME = 'qwen3.6-27b'
LAYER = 36
N_ACT = 30
SEED = 123
SEQ_LEN = 5
MAX_ANGLE = 60.0
N_STEPS = 46
N_PER_FAMILY = 20
RESULTS_DIR = 'results/jlens/qwen3.6-27b'
OUT_PKL = os.path.join(RESULTS_DIR, 'response_1d_comparison_L36.pkl')
OUT_PNG = os.path.join(RESULTS_DIR, 'response_1d_comparison_L36.png')

COLORS = {'Contrastive': '#e41a1c', 'J-Lens ZH': '#377eb8',
          'Random': '#4daf4a'}


def load_families():
    """Return {family: [np.ndarray[D], ...]} unit directions."""
    fam = {}
    # J-Lens ZH: first token vec of the first N pairs
    with open(os.path.join(RESULTS_DIR,
                           'jlens_zh100_pairs_L36.pkl'), 'rb') as f:
        zh = pickle.load(f)
    jl = []
    for pair in zh['zh_pairs']:
        jl.append(np.asarray(pair['vecs'][0], dtype=np.float32))
        if len(jl) >= N_PER_FAMILY:
            break
    fam['J-Lens ZH'] = jl
    # Contrastive DoM
    with open('results/exp_map/data/directions/dirs_qwen36_L36.pkl', 'rb') as f:
        dirs = pickle.load(f)['directions']
    names = sorted(dirs)[:N_PER_FAMILY]
    fam['Contrastive'] = [dirs[n].float().numpy() for n in names]
    # Random isotropic (seed 42)
    rng = np.random.default_rng(42)
    d = len(fam['J-Lens ZH'][0])
    R = rng.standard_normal((N_PER_FAMILY, d)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    fam['Random'] = list(R)
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    angles_deg = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles_deg)

    if not args.plot_only:
        torch.manual_seed(SEED)
        print(f"Loading {MODEL_NAME}...")
        model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.bfloat16)
        measure_layer = len(_get_blocks(model)) - 2
        print(f"  perturb@L{LAYER}, measure@L{measure_layer}")

        ids = load_fineweb_fixed_length(N_ACT, tokenizer, seq_len=SEQ_LEN,
                                        seed=SEED)
        contexts, activations = [], []
        for t in ids:
            with torch.no_grad():
                out = model(t.unsqueeze(0).to(device), output_hidden_states=True)
            h = out.hidden_states[LAYER + 1]
            contexts.append(h[:, :-1, :].float().cpu())
            activations.append(h[:, -1, :].squeeze(0).float().cpu())

        fam = load_families()
        results = {}
        for label, dirs in fam.items():
            grids = []
            for j, d in enumerate(dirs):
                g = run_1d_per_anchors(
                    model, contexts, activations,
                    torch.tensor(d, dtype=torch.float32),
                    LAYER, measure_layer, device, angles_rad)
                grids.append(g)                       # (n_anchors, n_steps)
            results[label] = np.stack(grids)          # (n_dir, n_anchors, n_steps)
            print(f"  {label}: {results[label].shape} "
                  f"max L2={results[label].max():.1f}", flush=True)
        with open(OUT_PKL, 'wb') as f:
            pickle.dump({'angles_deg': angles_deg, 'results': results,
                         'layer': LAYER, 'measure_layer': measure_layer,
                         'model': MODEL_NAME}, f)
        print(f"Saved: {OUT_PKL}")

    with open(OUT_PKL, 'rb') as f:
        blob = pickle.load(f)
    angles_deg = blob['angles_deg']
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, arr in blob['results'].items():
        flat = arr.reshape(-1, arr.shape[-1])         # (n_dir*n_anchor, n_steps)
        med = np.median(flat, axis=0)
        lo = np.percentile(flat, 25, axis=0)
        hi = np.percentile(flat, 75, axis=0)
        c = COLORS.get(label, None)
        ax.plot(angles_deg, med, color=c, lw=2, label=f"{label} (median)")
        ax.fill_between(angles_deg, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel('downstream L2 response (L36→L62)')
    ax.set_title('Qwen3.6-27B — 1D response vs angle by direction family\n'
                 '(median, IQR band; 30 anchors × 20 dirs)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}")


if __name__ == '__main__':
    main()
