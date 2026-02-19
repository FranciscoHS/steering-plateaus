#!/usr/bin/env python3
"""
Perturbation sweeps for all combinations of genre directions.

For each subset size k=1..K, iterates over all C(11,k) subsets of genre
directions, sums the (normalized) directions, re-normalizes, and runs a
perturbation sweep measuring KL divergence and L2 distance.

This tests whether composite directions (sum of multiple genre features)
produce different perturbation profiles than individual directions.

Usage:
    python scripts/directions/run_combo_sweeps.py [--model qwen3-0.6b] [--max_k 11]
"""

import sys
import argparse
import pickle
import os
import time
import numpy as np
from itertools import combinations

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks, _is_rope_model, _get_position_embeddings, _run_block, _get_final_norm, _get_lm_head
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device,
                                   batch_size=64):
    """Extract layer-L context and final-token activation for fixed-length sequences."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].cpu())
        all_activations.append(hidden[:, -1, :].cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def forward_to_hidden_and_logits(model, hidden_states, start_layer):
    """Forward pass returning both final-block hidden state and logits (last token)."""
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return final_hidden, logits


def build_perturbed_hidden(contexts, activations, direction, angle_rad):
    """Construct hidden states after spherical perturbation along direction."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    """Perturbation sweep for one fixed direction."""
    N = activations.shape[0]
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles))
    l2_dists = np.zeros((N, n_angles))
    direction_dev = direction.to(device)
    for ang_idx, theta in enumerate(angles_rad):
        with torch.no_grad():
            hidden = build_perturbed_hidden(contexts, activations, direction_dev, theta)
            pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            l2 = (pert_final - ref_final).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            l2_dists[:, ang_idx] = l2.cpu().numpy()
            kl_divs[:, ang_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def sweep_random_directions(model, contexts, activations, ref_final, ref_log_probs,
                            angles_rad, layer, device, n_dirs=10):
    """Perturbation sweep with random directions (different per activation)."""
    N, d = activations.shape
    n_angles = len(angles_rad)
    kl_divs = np.zeros((N, n_angles, n_dirs))
    l2_dists = np.zeros((N, n_angles, n_dirs))
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    v_perps = torch.zeros(N, n_dirs, d, device=device)
    for i in range(N):
        v = torch.randn(n_dirs, d, device=device)
        v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)
    for ang_idx, theta in enumerate(angles_rad):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for dir_idx in range(n_dirs):
            v = v_perps[:, dir_idx, :]
            a_pert = a_norms * (cos_t * a_unit + sin_t * v)
            with torch.no_grad():
                hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
                pert_final, pert_logits = forward_to_hidden_and_logits(model, hidden, layer)
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                l2 = (pert_final - ref_final).norm(dim=-1)
                kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
                l2_dists[:, ang_idx, dir_idx] = l2.cpu().numpy()
                kl_divs[:, ang_idx, dir_idx] = kl.cpu().numpy()
    return kl_divs, l2_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--max_angle', type=float, default=60,
                        help='Maximum steering angle in degrees')
    parser.add_argument('--n_angles', type=int, default=20,
                        help='Number of angle steps')
    parser.add_argument('--max_k', type=int, default=11,
                        help='Maximum subset size (default: all 11 genres)')
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of FineWeb activations')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_random_dirs', type=int, default=10)
    args = parser.parse_args()

    results_dir = f'results/genre_contrastive/{args.model}'

    # Load directions
    directions_path = os.path.join(results_dir, 'direction_results.pkl')
    print("Loading directions...")
    with open(directions_path, 'rb') as f:
        saved = pickle.load(f)

    genres = saved['registers']
    k = len(genres)
    print(f"Genres ({k}): {genres}")

    # Get normalized all-pairwise directions
    genre_dirs = {}
    for g in genres:
        genre_dirs[g] = saved['f_allpair'][g]['normalized']
    print(f"Loaded {len(genre_dirs)} normalized all-pairwise directions")

    # Count total combinations
    max_k = min(args.max_k, k)
    total_combos = sum(len(list(combinations(range(k), r))) for r in range(1, max_k + 1))
    print(f"Subset sizes 1..{max_k}: {total_combos} total combinations to sweep")

    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(args.model)
    torch.manual_seed(args.seed)

    # Extract FineWeb activations
    print(f"Loading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    contexts = contexts.to(device)
    activations = activations.to(device)

    # Reference logits
    with torch.no_grad():
        ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
        ref_final, ref_logits = forward_to_hidden_and_logits(model, ref_hidden, args.layer)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Angles
    angles_deg = np.linspace(3, args.max_angle, args.n_angles)
    angles_rad = np.radians(angles_deg)
    print(f"Angles: {args.n_angles} steps from 3° to {args.max_angle}°")

    # Checkpoint path
    out_path = os.path.join(results_dir, 'combo_sweep_results.pkl')

    # Load checkpoint if it exists (resume support)
    results_by_k = {}
    completed_k = set()
    if os.path.exists(out_path):
        print(f"Found existing checkpoint: {out_path}")
        with open(out_path, 'rb') as f:
            checkpoint = pickle.load(f)
        results_by_k = checkpoint.get('results_by_k', {})
        completed_k = set(results_by_k.keys())
        print(f"  Already completed k={sorted(completed_k)}")

    count = 0
    t0 = time.time()

    for r in range(1, max_k + 1):
        if r in completed_k:
            n_skip = len(list(combinations(range(k), r)))
            print(f"\nSubset size k={r}: already done ({n_skip} combos), skipping")
            count += n_skip
            continue

        combos = list(combinations(range(k), r))
        print(f"\nSubset size k={r}: {len(combos)} combinations")
        combo_results = []

        for combo in combos:
            # Sum normalized directions and re-normalize
            combo_names = tuple(genres[i] for i in combo)
            dir_sum = sum(genre_dirs[genres[i]] for i in combo)
            dir_normalized = dir_sum / dir_sum.norm()

            kl, l2 = sweep_single_direction(
                model, contexts, activations, ref_final, ref_log_probs,
                dir_normalized, angles_rad, args.layer, device)

            combo_results.append({
                'genres': combo_names,
                'kl_mean': kl.mean(axis=0),
                'l2_mean': l2.mean(axis=0),
                'kl_std': kl.std(axis=0),
                'l2_std': l2.std(axis=0),
            })

            count += 1
            if count % 50 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  [{count}/{total_combos}] {elapsed:.0f}s elapsed, {rate:.1f} combos/s", flush=True)

        results_by_k[r] = combo_results

        # Checkpoint after each k
        save_data = {
            'args': vars(args),
            'genres': genres,
            'angles_deg': angles_deg,
            'results_by_k': results_by_k,
            'random': None,  # filled in at the end
        }
        with open(out_path, 'wb') as f:
            pickle.dump(save_data, f)
        elapsed = time.time() - t0
        print(f"  Checkpointed k={r} ({elapsed:.0f}s total)", flush=True)

    # Random baseline
    print("\nSweeping random directions...")
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts, activations, ref_final, ref_log_probs,
        angles_rad, args.layer, device, n_dirs=args.n_random_dirs)

    elapsed = time.time() - t0
    print(f"\nAll {total_combos} sweeps + random complete in {elapsed:.0f}s")

    # Final save with random baseline
    save_data = {
        'args': vars(args),
        'genres': genres,
        'angles_deg': angles_deg,
        'results_by_k': results_by_k,
        'random': {
            'kl': rand_kl,
            'l2': rand_l2,
        },
    }
    with open(out_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {out_path}")

    # Print summary: median KL at key angles for each subset size
    print(f"\n{'='*60}")
    print("SUMMARY: median KL divergence by subset size")
    print(f"{'='*60}")
    key_angles = [15, 30, 45, 60]
    print(f"  {'k':>3s}", end='')
    for a in key_angles:
        print(f"  {a:>6d}°", end='')
    print(f"  {'n_combos':>10s}")

    for r in range(1, max_k + 1):
        combo_kls = np.array([c['kl_mean'] for c in results_by_k[r]])  # (n_combos, n_angles)
        median_kl = np.median(combo_kls, axis=0)
        print(f"  {r:>3d}", end='')
        for a in key_angles:
            ai = np.argmin(np.abs(angles_deg - a))
            print(f"  {median_kl[ai]:>7.3f}", end='')
        print(f"  {len(results_by_k[r]):>10d}")

    # Random
    rand_kl_mean = rand_kl.mean(axis=(0, 2)) if rand_kl.ndim == 3 else rand_kl.mean(axis=0)
    print(f"  {'rnd':>3s}", end='')
    for a in key_angles:
        ai = np.argmin(np.abs(angles_deg - a))
        print(f"  {rand_kl_mean[ai]:>7.3f}", end='')
    print(f"  {args.n_random_dirs:>10d}")

    print(f"\nSUMMARY: median L2 distance by subset size")
    print(f"{'='*60}")
    print(f"  {'k':>3s}", end='')
    for a in key_angles:
        print(f"  {a:>6d}°", end='')
    print(f"  {'n_combos':>10s}")

    for r in range(1, max_k + 1):
        combo_l2s = np.array([c['l2_mean'] for c in results_by_k[r]])
        median_l2 = np.median(combo_l2s, axis=0)
        print(f"  {r:>3d}", end='')
        for a in key_angles:
            ai = np.argmin(np.abs(angles_deg - a))
            print(f"  {median_l2[ai]:>7.1f}", end='')
        print(f"  {len(results_by_k[r]):>10d}")

    rand_l2_mean = rand_l2.mean(axis=(0, 2)) if rand_l2.ndim == 3 else rand_l2.mean(axis=0)
    print(f"  {'rnd':>3s}", end='')
    for a in key_angles:
        ai = np.argmin(np.abs(angles_deg - a))
        print(f"  {rand_l2_mean[ai]:>7.1f}", end='')
    print(f"  {args.n_random_dirs:>10d}")

    print("\nDone!")


if __name__ == '__main__':
    main()
