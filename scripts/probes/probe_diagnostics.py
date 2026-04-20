#!/usr/bin/env python3
"""Probe diagnostics: train accuracy + C sweep for raw / SAE / AE reconstructions.

Reuses cached activations from a previous run_probe_sae_vs_ae.py run.
"""

import argparse
import json
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reuse the AE/SAE classes + loaders from the main probe script
from scripts.probes.run_probe_sae_vs_ae import load_sae, load_ae, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True,
                        help='Path to cached activations.pt from run_probe_sae_vs_ae.py')
    parser.add_argument('--sae_weights', required=True)
    parser.add_argument('--ae_weights', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--cs', type=float, nargs='+',
                        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading activations: {args.activations}")
    cache = torch.load(args.activations, weights_only=False)
    train_acts = cache['train_acts']
    train_labels = cache['train_labels']
    id_test_acts = cache['id_test_acts']
    id_test_labels = cache['id_test_labels']
    ood_acts = cache['ood_acts']
    ood_labels = cache['ood_labels']
    print(f"  train={train_acts.shape}, id={id_test_acts.shape}, ood={ood_acts.shape}")

    print(f"\nLoading SAE: {args.sae_weights}")
    sae = load_sae(args.sae_weights, device)
    print(f"Loading AE: {args.ae_weights}")
    ae = load_ae(args.ae_weights, device)

    print("\nReconstructing (SAE)...")
    train_sae = reconstruct(sae, train_acts, device)
    id_sae = reconstruct(sae, id_test_acts, device)
    ood_sae = reconstruct(sae, ood_acts, device)
    print("Reconstructing (AE)...")
    train_ae = reconstruct(ae, train_acts, device)
    id_ae = reconstruct(ae, id_test_acts, device)
    ood_ae = reconstruct(ae, ood_acts, device)

    conditions = [
        ('raw', train_acts, id_test_acts, ood_acts),
        ('sae', train_sae, id_sae, ood_sae),
        ('ae', train_ae, id_ae, ood_ae),
    ]

    results = {}  # results[condition][C] = {train, id, ood, w_norm}
    print("\nC sweep:")
    print(f"  {'cond':4s} {'C':>10s} {'train':>7s} {'id':>7s} {'ood':>7s} "
          f"{'gap':>7s} {'|w|':>8s}")

    for name, tr, idt, oodt in conditions:
        results[name] = {}
        for C in args.cs:
            clf = LogisticRegression(max_iter=5000, random_state=args.seed, C=C)
            clf.fit(tr.numpy(), train_labels)
            tr_acc = accuracy_score(train_labels, clf.predict(tr.numpy()))
            id_acc = accuracy_score(id_test_labels, clf.predict(idt.numpy()))
            ood_acc = accuracy_score(ood_labels, clf.predict(oodt.numpy()))
            w_norm = float(np.linalg.norm(clf.coef_))
            gap = id_acc - ood_acc
            results[name][C] = {
                'train_acc': tr_acc, 'id_acc': id_acc, 'ood_acc': ood_acc,
                'gap': gap, 'w_norm': w_norm,
            }
            print(f"  {name:4s} {C:10.4g} {tr_acc:7.4f} {id_acc:7.4f} "
                  f"{ood_acc:7.4f} {gap:7.4f} {w_norm:8.2f}")

    # ── Save raw results ─────────────────────────────────────────────
    out_pkl = os.path.join(args.out_dir, 'diagnostics.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'args': vars(args), 'results': results}, f)
    print(f"\nSaved: {out_pkl}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    colors = {'raw': '#555', 'sae': '#2196F3', 'ae': '#E91E63'}
    cs = args.cs

    # Panel 1: train/id/ood vs C (one panel per condition would be cluttered;
    # instead overlay all three with linestyle = train/id/ood)
    ax = axes[0]
    for name in ['raw', 'sae', 'ae']:
        tr = [results[name][C]['train_acc'] for C in cs]
        idd = [results[name][C]['id_acc'] for C in cs]
        ood = [results[name][C]['ood_acc'] for C in cs]
        c = colors[name]
        ax.plot(cs, tr, '--', color=c, alpha=0.6, label=f'{name} train')
        ax.plot(cs, idd, '-o', color=c, label=f'{name} ID')
        ax.plot(cs, ood, '-s', color=c, alpha=0.8, label=f'{name} OOD',
                markerfacecolor='none')
    ax.set_xscale('log')
    ax.set_xlabel('C (probe regularization, larger = less reg)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train / ID / OOD accuracy vs C')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc='lower right')

    # Panel 2: ID-OOD gap vs C
    ax = axes[1]
    for name in ['raw', 'sae', 'ae']:
        gaps = [results[name][C]['gap'] for C in cs]
        ax.plot(cs, gaps, '-o', color=colors[name], label=name)
    ax.set_xscale('log')
    ax.set_xlabel('C')
    ax.set_ylabel('ID − OOD accuracy')
    ax.set_title('Generalization gap vs C')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: weight norm vs C
    ax = axes[2]
    for name in ['raw', 'sae', 'ae']:
        wns = [results[name][C]['w_norm'] for C in cs]
        ax.plot(cs, wns, '-o', color=colors[name], label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('C')
    ax.set_ylabel('||probe weights||_2')
    ax.set_title('Probe weight norm vs C')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'diagnostics.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
