#!/usr/bin/env python3
"""Check alignment between MLP neuron directions and probe directions in OthelloGPT.

For each layer, computes cosine similarity between each W_out column (neuron's
write direction into residual stream) and each probe direction (class weight
for each square/state). Reports max alignment statistics.
"""

import os, pickle
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

PROBE_DIR = '/workspace/mech_int_othelloGPT/mech_int/probes/linear'
RESULTS_DIR = 'results/othello_plateau'
# Center squares (27,28,35,36) are not playable
CENTER = {27, 28, 35, 36}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = HookedTransformer.from_pretrained("othello-gpt")
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp
    state_names = ['empty', 'mine', 'opp']

    print(f"Model: {n_layers} layers, d_model={d_model}, d_mlp={d_mlp}")

    all_results = {}

    for layer in range(n_layers):
        print(f"\n=== Layer {layer} ===")

        # Neuron write directions: columns of W_out [d_mlp, d_model] -> normalize each row
        W_out = model.blocks[layer].mlp.W_out.detach().cpu().float()  # [d_mlp, d_model]
        neuron_dirs = F.normalize(W_out, dim=1)  # [d_mlp, d_model]

        # Probe directions
        probe = torch.load(f'{PROBE_DIR}/resid_{layer}_linear.pth',
                           map_location='cpu', weights_only=True).float()
        # probe shape: [1, d_model, 8, 8, 3]
        # For each (row, col, state), the probe direction is probe[0, :, r, c, s]

        probe_dirs = []
        probe_labels = []
        for r in range(8):
            for c in range(8):
                sq = r * 8 + c
                if sq in CENTER:
                    continue
                for s in range(3):
                    d = probe[0, :, r, c, s]
                    probe_dirs.append(d)
                    probe_labels.append(f"({r},{c})_{state_names[s]}")

        probe_dirs = torch.stack(probe_dirs)  # [n_probes, d_model]
        probe_dirs = F.normalize(probe_dirs, dim=1)

        # Cosine similarity matrix: [n_probes, d_mlp]
        cos_sim = probe_dirs @ neuron_dirs.T  # [n_probes, d_mlp]

        # For each probe direction, find max |cos_sim| with any neuron
        max_abs_cos = cos_sim.abs().max(dim=1).values  # [n_probes]
        best_neuron = cos_sim.abs().max(dim=1).indices

        # For each neuron, find max |cos_sim| with any probe direction
        neuron_max_abs_cos = cos_sim.abs().max(dim=0).values  # [d_mlp]

        print(f"  Probe→neuron max |cos|: mean={max_abs_cos.mean():.4f}, "
              f"median={max_abs_cos.median():.4f}, max={max_abs_cos.max():.4f}")
        print(f"  Neuron→probe max |cos|: mean={neuron_max_abs_cos.mean():.4f}, "
              f"median={neuron_max_abs_cos.median():.4f}, max={neuron_max_abs_cos.max():.4f}")

        # Top 10 most aligned probe-neuron pairs
        top_k = 10
        top_vals, top_idx = max_abs_cos.topk(top_k)
        print(f"  Top {top_k} probe-neuron alignments:")
        for i in range(top_k):
            pi = top_idx[i].item()
            ni = best_neuron[pi].item()
            print(f"    {probe_labels[pi]}: neuron {ni}, |cos|={top_vals[i]:.4f}")

        # Random baseline: same number of random directions
        n_rand = len(probe_dirs)
        rand_dirs = F.normalize(torch.randn(n_rand, d_model), dim=1)
        rand_cos_sim = rand_dirs @ neuron_dirs.T
        rand_max_abs_cos = rand_cos_sim.abs().max(dim=1).values

        print(f"  Random baseline max |cos|: mean={rand_max_abs_cos.mean():.4f}, "
              f"median={rand_max_abs_cos.median():.4f}, max={rand_max_abs_cos.max():.4f}")

        all_results[layer] = {
            'probe_max_cos': max_abs_cos.detach().numpy(),
            'rand_max_cos': rand_max_abs_cos.detach().numpy(),
            'neuron_max_cos': neuron_max_abs_cos.detach().numpy(),
            'probe_labels': probe_labels,
            'best_neuron_per_probe': best_neuron.detach().numpy(),
        }

    # Save
    with open(os.path.join(RESULTS_DIR, 'neuron_probe_alignment.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    # Plot: distribution of max |cos| per probe direction, per layer
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for layer in range(n_layers):
        ax = axes[layer // 4][layer % 4]
        bins = np.linspace(0, 1, 31)
        ax.hist(all_results[layer]['probe_max_cos'], bins=bins, edgecolor='black',
                alpha=0.7, label='Probe', color='C3')
        ax.hist(all_results[layer]['rand_max_cos'], bins=bins, edgecolor='black',
                alpha=0.5, label='Random', color='C0')
        ax.set_title(f'Layer {layer}')
        ax.set_xlabel('Max |cos| with any neuron')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle('OthelloGPT: Probe vs random direction alignment with MLP neurons (W_out)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'neuron_probe_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {RESULTS_DIR}/neuron_probe_alignment.png")


if __name__ == '__main__':
    main()
