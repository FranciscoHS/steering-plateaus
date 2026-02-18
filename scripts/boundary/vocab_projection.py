#!/usr/bin/env python3
"""
Project boundary MELBO directions onto the vocabulary to see which tokens
they're most aligned with.

Computes cosine similarity between each direction and:
  - lm_head weights (unembedding: what tokens does this direction predict?)
  - embed_tokens weights (embedding: what input tokens align with this direction?)

Usage:
    python scripts/boundary/vocab_projection.py --model qwen3-0.6b --from_lang en
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_lm_head, _get_final_norm


def top_bottom_tokens(direction, weight_matrix, tokenizer, k=30):
    """Compute cosine similarity between direction and all rows of weight_matrix.

    Returns top-k and bottom-k tokens with their cosine similarities.
    """
    # weight_matrix: [vocab_size, d_model]
    # direction: [d_model]
    cos_sims = F.cosine_similarity(
        weight_matrix, direction.unsqueeze(0), dim=1)  # [vocab_size]

    topk = cos_sims.topk(k)
    bottomk = cos_sims.topk(k, largest=False)

    top_tokens = [(tokenizer.decode([idx]), cos_sims[idx].item())
                  for idx in topk.indices]
    bottom_tokens = [(tokenizer.decode([idx]), cos_sims[idx].item())
                     for idx in bottomk.indices]

    return top_tokens, bottom_tokens, cos_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--from_lang', default='en')
    parser.add_argument('--bnd_melbo_path', default=None)
    parser.add_argument('--orig_melbo_path', default=None)
    parser.add_argument('--iterative_path', default=None,
                        help='Path to iterative results.pkl. If set, projects all iterative directions.')
    parser.add_argument('--k', type=int, default=30, help='Number of top/bottom tokens')
    args = parser.parse_args()

    if args.bnd_melbo_path is None:
        args.bnd_melbo_path = f'results/boundary_melbo/{args.model}/from_{args.from_lang}/optimization_results.pkl'
    if args.orig_melbo_path is None:
        args.orig_melbo_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.iterative_path is None:
        args.iterative_path = f'results/iterative_boundary_melbo/{args.model}/results.pkl'

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # Get weight matrices
    lm_head_weight = _get_lm_head(model).weight.data  # [vocab_size, d_model]
    embed_weight = model.model.embed_tokens.weight.data  # [vocab_size, d_model]
    final_norm = _get_final_norm(model)
    print(f"Vocab size: {lm_head_weight.shape[0]}, d_model: {lm_head_weight.shape[1]}")

    # Load original MELBO
    print(f"Loading original MELBO from {args.orig_melbo_path}")
    with open(args.orig_melbo_path, 'rb') as f:
        orig_saved = pickle.load(f)
    melbo_1 = orig_saved['melbo_directions'][0]
    mean_dir = orig_saved['mean_dir']

    # Ensure MELBO_1 points English-ward
    if (melbo_1 @ mean_dir) < 0:
        melbo_1 = -melbo_1

    # Build directions list
    directions = [
        ('MELBO_1 (->En)', melbo_1),
        ('MELBO_1 (->Zh)', -melbo_1),
    ]

    # Load iterative directions if available
    iterative_mode = os.path.exists(args.iterative_path)
    if iterative_mode:
        print(f"Loading iterative results from {args.iterative_path}")
        with open(args.iterative_path, 'rb') as f:
            iter_results = pickle.load(f)
        for it in iter_results['iterations']:
            i = it['iteration']
            for j, d in enumerate(it['bnd_melbo_directions']):
                directions.append((f'iter{i}_bndM{j+1} (+)', d))
                directions.append((f'iter{i}_bndM{j+1} (-)', -d))
    else:
        # Fall back to single boundary MELBO
        print(f"Loading boundary MELBO from {args.bnd_melbo_path}")
        with open(args.bnd_melbo_path, 'rb') as f:
            bnd_saved = pickle.load(f)
        bnd_dirs = bnd_saved['bnd_melbo_directions']
        for j, d in enumerate(bnd_dirs):
            directions.append((f'bnd_MELBO_{j+1} (+)', d))
            directions.append((f'bnd_MELBO_{j+1} (-)', -d))

    output_lines = []

    for matrix_name, weight_matrix in [
        ('lm_head (unembedding)', lm_head_weight),
        ('embed_tokens (embedding)', embed_weight),
    ]:
        header = f"\n{'='*70}\n{matrix_name}\n{'='*70}"
        print(header)
        output_lines.append(header)

        for dir_name, direction in directions:
            d = direction.to(device)

            # Also try applying final norm to the direction
            # (since logits = lm_head(norm(hidden)), the relevant projection
            # for lm_head is really against norm-transformed directions)
            top, bottom, cos_sims = top_bottom_tokens(
                d, weight_matrix, tokenizer, k=args.k)

            dir_header = f"\n  --- {dir_name} ---"
            print(dir_header)
            output_lines.append(dir_header)

            # Stats
            stats = (f"  cos_sim stats: mean={cos_sims.mean():.4f}, "
                     f"std={cos_sims.std():.4f}, "
                     f"max={cos_sims.max():.4f}, min={cos_sims.min():.4f}")
            print(stats)
            output_lines.append(stats)

            print(f"  Top {args.k}:")
            output_lines.append(f"  Top {args.k}:")
            for token, sim in top:
                token_repr = repr(token)
                line = f"    {sim:+.4f}  {token_repr}"
                print(line)
                output_lines.append(line)

            print(f"  Bottom {args.k}:")
            output_lines.append(f"  Bottom {args.k}:")
            for token, sim in bottom:
                token_repr = repr(token)
                line = f"    {sim:+.4f}  {token_repr}"
                print(line)
                output_lines.append(line)

    # Save
    if iterative_mode:
        results_dir = os.path.dirname(args.iterative_path)
    else:
        results_dir = os.path.dirname(args.bnd_melbo_path)
    output_path = os.path.join(results_dir, 'vocab_projection.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
