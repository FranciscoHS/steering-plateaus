#!/usr/bin/env python3
"""
Analyze the optimized delta vector from run_delta_optimization.py.

1. Sanity check: what do shifted activations predict? (logit entropy, top tokens)
2. Projection analysis: project many real activations onto delta direction
3. Sort by projection, print top/bottom sequences
4. Distribution of projections

Usage:
    python scripts/analyze_delta.py
    python scripts/analyze_delta.py --n_sequences 1000
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length


def main():
    parser = argparse.ArgumentParser(description='Analyze optimized delta vector')
    parser.add_argument('--results_file', type=str, default='results/delta_opt/results_r66.pkl')
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_sequences', type=int, default=500,
                        help='Number of FineWeb sequences for projection analysis')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top/bottom sequences to show')
    parser.add_argument('--seed', type=int, default=123,
                        help='Different seed from optimization to get fresh sequences')
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)

    delta = results['delta']  # [d_model] tensor
    r = results['r']
    ref_norms = results['ref_norms']
    shifted_norms = results['shifted_norms']

    delta_hat = delta / delta.norm()
    print(f"Delta norm: {delta.norm().item():.1f} (r={r:.1f})")
    print(f"Ref norms: mean={ref_norms.mean():.1f}, std={ref_norms.std():.1f}")
    print(f"Shifted norms: mean={shifted_norms.mean():.1f}, std={shifted_norms.std():.1f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    delta = delta.to(device)
    delta_hat = delta_hat.to(device)

    d_model = delta.shape[0]

    # =========================================================
    # Part 1: Sanity check — what do shifted activations predict?
    # =========================================================
    print(f"\n=== Part 1: Shifted activation predictions ===")
    print(f"Using the {len(results['ref_norms'])} activations from the optimization run")

    # Load the checkpoint to get the original activations and contexts
    checkpoint_path = os.path.join(os.path.dirname(args.results_file), 'checkpoint.pkl')
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    activations = checkpoint['activations']  # [100, d_model]
    contexts = checkpoint['contexts']  # list of [1, seq_len-1, d_model]

    # For a handful of activations, compare predictions before and after shifting
    print(f"\nSample predictions (original vs shifted):")
    n_show = 10
    entropies_orig = []
    entropies_shifted = []

    for i in range(n_show):
        ctx = contexts[i].to(device)
        a_i = activations[i].to(device)

        # Original predictions
        point_orig = a_i.reshape(1, 1, d_model)
        with torch.no_grad():
            logits_orig = forward_from_layer(model, ctx, point_orig, args.layer)
            probs_orig = torch.softmax(logits_orig.squeeze(), dim=-1)
            entropy_orig = -(probs_orig * probs_orig.log()).sum().item()
            top5_orig = torch.topk(probs_orig, 5)

        # Shifted predictions
        point_shifted = (a_i + delta).reshape(1, 1, d_model)
        with torch.no_grad():
            logits_shifted = forward_from_layer(model, ctx, point_shifted, args.layer)
            probs_shifted = torch.softmax(logits_shifted.squeeze(), dim=-1)
            entropy_shifted = -(probs_shifted * probs_shifted.log()).sum().item()
            top5_shifted = torch.topk(probs_shifted, 5)

        entropies_orig.append(entropy_orig)
        entropies_shifted.append(entropy_shifted)

        orig_tokens = [tokenizer.decode([idx]) for idx in top5_orig.indices]
        shifted_tokens = [tokenizer.decode([idx]) for idx in top5_shifted.indices]

        print(f"\n  Sample {i}:")
        print(f"    Original:  H={entropy_orig:.2f}  top5={orig_tokens}")
        print(f"    Shifted:   H={entropy_shifted:.2f}  top5={shifted_tokens}")

    # Compute entropies for all 100
    print(f"\nComputing entropies for all {len(activations)} activations...")
    all_entropies_orig = []
    all_entropies_shifted = []
    for i in range(len(activations)):
        ctx = contexts[i].to(device)
        a_i = activations[i].to(device)

        with torch.no_grad():
            logits_o = forward_from_layer(model, ctx, a_i.reshape(1,1,d_model), args.layer)
            probs_o = torch.softmax(logits_o.squeeze(), dim=-1)
            all_entropies_orig.append(-(probs_o * probs_o.log()).sum().item())

            logits_s = forward_from_layer(model, ctx, (a_i+delta).reshape(1,1,d_model), args.layer)
            probs_s = torch.softmax(logits_s.squeeze(), dim=-1)
            all_entropies_shifted.append(-(probs_s * probs_s.log()).sum().item())

    eo = np.array(all_entropies_orig)
    es = np.array(all_entropies_shifted)
    print(f"Logit entropy:")
    print(f"  Original:  mean={eo.mean():.2f}, std={eo.std():.2f}")
    print(f"  Shifted:   mean={es.mean():.2f}, std={es.std():.2f}")

    # =========================================================
    # Part 2: Projection analysis
    # =========================================================
    print(f"\n=== Part 2: Projection analysis ({args.n_sequences} sequences) ===")

    token_ids_list = load_fineweb_fixed_length(
        args.n_sequences, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    # Extract activations
    print("Extracting activations...")
    all_acts = []
    all_texts = []

    for i in range(0, len(token_ids_list), 64):
        batch_end = min(i + 64, len(token_ids_list))
        batch_tokens = torch.stack(token_ids_list[i:batch_end]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_acts.append(hidden[:, -1, :].cpu())

        # Decode texts for display
        for j in range(i, batch_end):
            text = tokenizer.decode(token_ids_list[j])
            all_texts.append(text)

        if (i // 64) % 10 == 0:
            print(f"  {batch_end}/{len(token_ids_list)}", end='\r')

    print()
    all_acts = torch.cat(all_acts, dim=0)  # [n_sequences, d_model]
    print(f"Activations shape: {all_acts.shape}")

    # Compute projections and cosine similarities
    delta_hat_cpu = delta_hat.cpu()
    projections = (all_acts * delta_hat_cpu).sum(dim=1).numpy()
    act_norms = all_acts.norm(dim=1).numpy()
    cosine_sims = projections / act_norms

    print(f"\nProjection onto delta_hat (a · delta_hat):")
    print(f"  Mean: {projections.mean():.2f}")
    print(f"  Std: {projections.std():.2f}")
    print(f"  Min: {projections.min():.2f}, Max: {projections.max():.2f}")

    print(f"\nCosine similarity cos(a, delta):")
    print(f"  Mean: {cosine_sims.mean():.4f}")
    print(f"  Std: {cosine_sims.std():.4f}")
    print(f"  Min: {cosine_sims.min():.4f}, Max: {cosine_sims.max():.4f}")

    # Sort by projection
    sorted_indices = np.argsort(projections)

    print(f"\n--- Top {args.top_k} (highest projection onto delta) ---")
    for rank, idx in enumerate(sorted_indices[-args.top_k:][::-1]):
        print(f"  #{rank+1} proj={projections[idx]:.2f} cos={cosine_sims[idx]:.4f} "
              f"||a||={act_norms[idx]:.1f}  text={repr(all_texts[idx])}")

    print(f"\n--- Bottom {args.top_k} (lowest projection onto delta) ---")
    for rank, idx in enumerate(sorted_indices[:args.top_k]):
        print(f"  #{rank+1} proj={projections[idx]:.2f} cos={cosine_sims[idx]:.4f} "
              f"||a||={act_norms[idx]:.1f}  text={repr(all_texts[idx])}")

    # Distribution plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(projections, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Projection (a · delta_hat)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Projection Distribution')
    axes[0].axvline(projections.mean(), color='red', linestyle='--', label=f'mean={projections.mean():.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(cosine_sims, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Cosine Similarity cos(a, delta)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Cosine Similarity Distribution')
    axes[1].axvline(cosine_sims.mean(), color='red', linestyle='--', label=f'mean={cosine_sims.mean():.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(args.results_file), 'projection_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {save_path}")


if __name__ == '__main__':
    main()
