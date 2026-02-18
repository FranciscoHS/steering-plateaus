#!/usr/bin/env python3
"""
Boundary MELBO: discover high-response directions at the plateau boundary.

Starting from activations in one language:
  1. Walk along the original MELBO_1 direction toward the other language
     to find the boundary (point of maximum sensitivity / steepest KL change)
  2. From the boundary, run unconstrained MELBO to find new directions
     of highest response, orthogonal to MELBO_1 and to each other
  3. Evaluate found directions with perturbation sweeps

Usage:
    python scripts/run_boundary_melbo.py --model qwen3-0.6b --layer 14 --from_lang en
    python scripts/run_boundary_melbo.py --model qwen3-0.6b --layer 14 --from_lang zh
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_allpos,
    _get_blocks, _is_rope_model, _get_position_embeddings, _run_block,
    _get_final_norm, _get_lm_head,
)
from data.bilingual_prompts import CHINESE_PROMPTS, ENGLISH_PROMPTS


# ── Reused infrastructure (same as other MELBO scripts) ──

def forward_to_hidden_and_logits(model, hidden_states, start_layer):
    """Forward pass returning both final hidden state and logits (last token)."""
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
    """Spherical perturbation along direction. Preserves activation norm."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    proj = (direction.unsqueeze(0) * a_unit).sum(dim=-1, keepdim=True)
    u_perp = direction.unsqueeze(0) - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    a_pert = a_norms * (cos_t * a_unit + sin_t * u_perp)
    return torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)


def compute_boundary_activations(activations, direction, angles_rad):
    """Compute boundary activations with per-sample angles.

    Args:
        activations: [N, d] on device
        direction: [d] unit vector on device
        angles_rad: [N] per-sample angles in radians

    Returns: [N, d] boundary activations (same norms as originals)
    """
    cos_t = torch.cos(angles_rad).unsqueeze(1)  # [N, 1]
    sin_t = torch.sin(angles_rad).unsqueeze(1)  # [N, 1]
    a_norms = activations.norm(dim=-1, keepdim=True)
    a_unit = activations / a_norms
    d = direction.unsqueeze(0)  # [1, d]
    proj = (d * a_unit).sum(dim=-1, keepdim=True)
    u_perp = d - proj * a_unit
    u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
    return a_norms * (cos_t * a_unit + sin_t * u_perp)


def sweep_single_direction(model, contexts, activations, ref_final, ref_log_probs,
                           direction, angles_rad, layer, device):
    """Perturbation sweep for one fixed direction. Returns kl [N, n_angles], l2 [N, n_angles]."""
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


# ── Fixed-length activation extraction ──

def extract_prompts_fixed_length(tokenizer, prompts, seq_len):
    """Tokenize prompts and filter/truncate to fixed length.

    Returns list of [seq_len] token tensors (same format as load_fineweb_fixed_length).
    """
    token_ids_list = []
    used_prompts = []
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if len(ids) >= seq_len:
            token_ids_list.append(torch.tensor(ids[:seq_len]))
            used_prompts.append(prompt)
    return token_ids_list, used_prompts


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


# ── Boundary finding ──

def find_boundary_angles(kl_sweep, angles_deg):
    """Find per-sample boundary angle (angle of steepest KL increase).

    Args:
        kl_sweep: [N, n_angles] KL divergences
        angles_deg: [n_angles] angles in degrees

    Returns: [N] boundary angles in degrees
    """
    N = kl_sweep.shape[0]
    boundary_angles = np.zeros(N)
    for i in range(N):
        dkl = np.diff(kl_sweep[i]) / np.diff(angles_deg)
        max_idx = np.argmax(dkl)
        boundary_angles[i] = (angles_deg[max_idx] + angles_deg[max_idx + 1]) / 2
    return boundary_angles


# ── MELBO at boundary ──

def optimize_direction_at_boundary(model, contexts, boundary_acts, ref_log_probs,
                                   angle_rad, layer, device, d_model,
                                   n_steps=300, lr=0.01, n_restarts=5,
                                   prev_directions=None):
    """Unconstrained MELBO at boundary activations.

    Finds direction in full R^d that maximizes KL when perturbing from boundary.
    Enforces orthogonality to prev_directions via projection.

    Returns: best_direction, best_kl, best_history
    """
    best_kl = -float('inf')
    best_direction = None
    best_history = None

    for restart in range(n_restarts):
        direction = torch.randn(d_model, device=device)
        if prev_directions:
            for pd in prev_directions:
                direction = direction - (direction @ pd) * pd
        direction = F.normalize(direction, dim=0)
        direction = direction.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([direction], lr=lr)
        history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # Normalize + orthogonalize
            d_normed = direction / direction.norm()
            if prev_directions:
                for pd in prev_directions:
                    d_normed = d_normed - (d_normed @ pd) * pd
                d_normed = d_normed / (d_normed.norm() + 1e-10)

            # Perturbed forward pass from boundary
            hidden = build_perturbed_hidden(contexts, boundary_acts, d_normed, angle_rad)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_log_probs = F.log_softmax(logits, dim=-1)

            # KL(ref || pert) — how much does perturbing from boundary change output?
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1).mean()

            loss = -kl
            loss.backward()
            optimizer.step()

            # Project to orthogonal subspace after Adam step
            if prev_directions:
                with torch.no_grad():
                    for pd in prev_directions:
                        direction.data -= (direction.data @ pd) * pd

            kl_val = kl.item()
            history.append(kl_val)

            if step % 50 == 0 or step == n_steps - 1:
                print(f"    restart {restart+1}/{n_restarts}, step {step}: "
                      f"KL = {kl_val:.6f}")

        final_kl = history[-1]
        if final_kl > best_kl:
            best_kl = final_kl
            with torch.no_grad():
                d_final = direction / direction.norm()
                if prev_directions:
                    for pd in prev_directions:
                        d_final = d_final - (d_final @ pd) * pd
                    d_final = d_final / d_final.norm()
                best_direction = d_final.clone()
            best_history = history

    return best_direction, best_kl, best_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--from_lang', default='zh', choices=['en', 'zh'],
                        help='Starting language: en=English (walk toward Zh), zh=Mandarin (walk toward En)')
    parser.add_argument('--seq_len', type=int, default=5,
                        help='Fixed token length for prompts')
    parser.add_argument('--opt_angle', type=float, default=15.0,
                        help='Perturbation angle for MELBO optimization (degrees)')
    parser.add_argument('--n_directions', type=int, default=2,
                        help='Number of MELBO directions to find at boundary')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_boundary_angles', type=int, default=60,
                        help='Number of angles for boundary-finding sweep (0-90 deg)')
    parser.add_argument('--n_eval_angles', type=int, default=60,
                        help='Number of angles for evaluation sweep (0-90 deg)')
    parser.add_argument('--n_random_dirs', type=int, default=10)
    parser.add_argument('--melbo_path', default=None,
                        help='Path to saved MELBO optimization results')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.melbo_path is None:
        args.melbo_path = f'results/melbo_constrained/{args.model}/optimization_results.pkl'
    if args.results_dir is None:
        args.results_dir = f'results/boundary_melbo/{args.model}/from_{args.from_lang}'
    os.makedirs(args.results_dir, exist_ok=True)

    # ===== Phase 1: Load model and saved MELBO directions =====
    print("=" * 60)
    print("PHASE 1: Load model and saved MELBO directions")
    print("=" * 60)

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    print(f"Device: {device}, d_model: {d_model}")

    print(f"Loading MELBO directions from {args.melbo_path}")
    with open(args.melbo_path, 'rb') as f:
        saved_melbo = pickle.load(f)
    melbo_1 = saved_melbo['melbo_directions'][0]  # [d_model] unit vector
    mean_dir = saved_melbo['mean_dir']
    melbo_1_dev = melbo_1.to(device)
    print(f"cos(MELBO_1, mean_dir): {(melbo_1 @ mean_dir).abs().item():.4f}")

    # ===== Phase 2: Extract fixed-length activations =====
    if args.from_lang == 'en':
        src_prompts = ENGLISH_PROMPTS
        src_label = 'English'
        # Walk toward Mandarin: use -MELBO_1 (MELBO_1 points English-ward)
        walk_dir = -melbo_1_dev if (melbo_1 @ mean_dir).item() > 0 else melbo_1_dev
        walk_label = 'MELBO_1 (->Zh)'
    else:
        src_prompts = CHINESE_PROMPTS
        src_label = 'Mandarin'
        # Walk toward English: use +MELBO_1 (MELBO_1 points English-ward)
        walk_dir = melbo_1_dev if (melbo_1 @ mean_dir).item() > 0 else -melbo_1_dev
        walk_label = 'MELBO_1 (->En)'

    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Extract {src_label} activations (seq_len={args.seq_len})")
    print(f"Walking direction: {walk_label}")
    print("=" * 60)

    token_ids_list, used_prompts = extract_prompts_fixed_length(
        tokenizer, src_prompts, args.seq_len)
    print(f"Prompts with >= {args.seq_len} tokens: {len(token_ids_list)}/{len(src_prompts)}")

    if len(token_ids_list) < 5:
        print("WARNING: Very few prompts match. Try reducing --seq_len.")

    for i, (tids, p) in enumerate(zip(token_ids_list[:5], used_prompts[:5])):
        decoded = tokenizer.decode(tids)
        print(f"  [{i}] {p} -> tokens: {tids.tolist()} -> decoded: '{decoded}'")

    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    N = activations.shape[0]
    print(f"Extracted {N} activations, shape: {activations.shape}")
    print(f"Mean activation norm: {activations.norm(dim=-1).mean():.2f}")

    contexts_dev = contexts.to(device)
    activations_dev = activations.to(device)
    walk_dir_cpu = walk_dir.cpu()

    # ===== Phase 3: Find boundary along walk direction =====
    print(f"\n{'=' * 60}")
    print(f"PHASE 3: Find boundary along {walk_label} for each {src_label} activation")
    print("=" * 60)

    # Compute reference at original activations
    with torch.no_grad():
        ref_hidden_orig = torch.cat(
            [contexts_dev, activations_dev.unsqueeze(1)], dim=1)
        ref_final_orig, ref_logits_orig = forward_to_hidden_and_logits(
            model, ref_hidden_orig, args.layer)
        ref_lp_orig = F.log_softmax(ref_logits_orig, dim=-1)

    # Sweep MELBO_1 from original activations to find boundary
    boundary_angles_deg = np.linspace(0, 90, args.n_boundary_angles + 1)[1:]
    boundary_angles_rad = np.deg2rad(boundary_angles_deg)

    print(f"Sweeping {len(boundary_angles_deg)} angles (0-90 deg) along {walk_label}...")
    t0 = time.time()
    kl_boundary_sweep, _ = sweep_single_direction(
        model, contexts_dev, activations_dev, ref_final_orig, ref_lp_orig,
        walk_dir_cpu, boundary_angles_rad, args.layer, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Find per-prompt boundary angles
    per_prompt_boundary_deg = find_boundary_angles(kl_boundary_sweep, boundary_angles_deg)
    per_prompt_boundary_rad = torch.tensor(
        np.deg2rad(per_prompt_boundary_deg), device=device, dtype=torch.float32)

    median_boundary = np.median(per_prompt_boundary_deg)
    print(f"Boundary angles (degrees):")
    print(f"  Median: {median_boundary:.1f}")
    print(f"  Mean:   {np.mean(per_prompt_boundary_deg):.1f}")
    print(f"  Std:    {np.std(per_prompt_boundary_deg):.1f}")
    print(f"  Min:    {np.min(per_prompt_boundary_deg):.1f}")
    print(f"  Max:    {np.max(per_prompt_boundary_deg):.1f}")

    # ===== Phase 4: Compute boundary activations + references =====
    print(f"\n{'=' * 60}")
    print("PHASE 4: Compute boundary activations and references")
    print("=" * 60)

    # Per-prompt boundary activations
    boundary_acts = compute_boundary_activations(
        activations_dev, walk_dir, per_prompt_boundary_rad)
    print(f"Boundary activations shape: {boundary_acts.shape}")
    print(f"  Norm preservation: orig={activations_dev.norm(dim=-1).mean():.2f}, "
          f"boundary={boundary_acts.norm(dim=-1).mean():.2f}")

    cos_sim = F.cosine_similarity(activations_dev, boundary_acts, dim=-1)
    print(f"  Cos(original, boundary): mean={cos_sim.mean():.4f}")

    # Reference at boundary
    with torch.no_grad():
        ref_hidden_bnd = torch.cat(
            [contexts_dev, boundary_acts.unsqueeze(1)], dim=1)
        ref_final_bnd, ref_logits_bnd = forward_to_hidden_and_logits(
            model, ref_hidden_bnd, args.layer)
        ref_lp_bnd = F.log_softmax(ref_logits_bnd, dim=-1)

    # KL between original and boundary (sanity check)
    kl_orig_to_bnd = (ref_lp_orig.exp() * (ref_lp_orig - ref_lp_bnd)).sum(dim=-1)
    print(f"  KL(original || boundary): mean={kl_orig_to_bnd.mean():.4f}, "
          f"median={kl_orig_to_bnd.median():.4f}")

    # ===== Phase 5: MELBO at boundary =====
    opt_angle_rad = np.deg2rad(args.opt_angle)
    print(f"\n{'=' * 60}")
    print(f"PHASE 5: Unconstrained MELBO at boundary ({args.n_directions} directions)")
    print(f"Perturbation angle: {args.opt_angle} deg, steps: {args.n_steps}, "
          f"restarts: {args.n_restarts}")
    print(f"Orthogonal to: original MELBO_1 + previously found directions")
    print("=" * 60)

    # The directions we enforce orthogonality against
    # (walk_dir is ±MELBO_1, so being ⊥ to walk_dir == ⊥ to MELBO_1)
    orth_constraints = [walk_dir]  # always ⊥ to the walking direction

    bnd_melbo_directions = []
    bnd_melbo_kls = []
    bnd_melbo_histories = []

    for d_idx in range(args.n_directions):
        print(f"\n--- Boundary MELBO direction {d_idx+1}/{args.n_directions} ---")
        print(f"  Orthogonal to: MELBO_1" +
              (f" + {d_idx} previous boundary directions" if d_idx > 0 else ""))
        t0 = time.time()

        direction, best_kl, history = optimize_direction_at_boundary(
            model, contexts_dev, boundary_acts, ref_lp_bnd,
            opt_angle_rad, args.layer, device, d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            prev_directions=orth_constraints
        )
        elapsed = time.time() - t0

        bnd_melbo_directions.append(direction.cpu())
        bnd_melbo_kls.append(best_kl)
        bnd_melbo_histories.append(history)

        # Add to orthogonality constraints for next direction
        orth_constraints.append(direction)

        # Print diagnostics
        cos_melbo1 = (direction @ melbo_1_dev).abs().item()
        cos_mean = (direction @ mean_dir.to(device)).abs().item()
        print(f"  Best KL: {best_kl:.6f}")
        print(f"  cos(bnd_MELBO_{d_idx+1}, MELBO_1): {cos_melbo1:.4f}")
        print(f"  cos(bnd_MELBO_{d_idx+1}, mean_dir): {cos_mean:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        for prev_idx in range(d_idx):
            cos_prev = (direction @ bnd_melbo_directions[prev_idx].to(device)).abs().item()
            print(f"  cos(bnd_MELBO_{d_idx+1}, bnd_MELBO_{prev_idx+1}): {cos_prev:.4f}")

    # Save optimization checkpoint
    opt_checkpoint = {
        'args': vars(args),
        'bnd_melbo_directions': bnd_melbo_directions,
        'bnd_melbo_kls': bnd_melbo_kls,
        'bnd_melbo_histories': bnd_melbo_histories,
        'melbo_1': melbo_1.cpu(),
        'walk_dir': walk_dir.cpu(),
        'walk_label': walk_label,
        'mean_dir': mean_dir.cpu(),
        'per_prompt_boundary_deg': per_prompt_boundary_deg,
        'kl_boundary_sweep': kl_boundary_sweep,
        'boundary_angles_deg': boundary_angles_deg,
        'used_prompts': used_prompts,
    }
    ckpt_path = os.path.join(args.results_dir, 'optimization_results.pkl')
    with open(ckpt_path, 'wb') as f:
        pickle.dump(opt_checkpoint, f)
    print(f"\nSaved optimization checkpoint to {ckpt_path}")

    # ===== Phase 6: Evaluation sweeps from boundary =====
    eval_angles_deg = np.linspace(0, 90, args.n_eval_angles + 1)[1:]
    eval_angles_rad = np.deg2rad(eval_angles_deg)

    print(f"\n{'=' * 60}")
    print(f"PHASE 6: Evaluation sweeps from boundary ({len(eval_angles_deg)} angles, 0-90 deg)")
    print("=" * 60)

    sweep_results = {}

    # Named directions to sweep
    named_dirs = [(f'{walk_label} (walk dir)', walk_dir_cpu)]
    for i, d in enumerate(bnd_melbo_directions):
        named_dirs.append((f'bnd_MELBO_{i+1}', d))

    for name, direction in named_dirs:
        print(f"  Sweeping: {name}...", end=' ', flush=True)
        t0 = time.time()
        kl, l2 = sweep_single_direction(
            model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
            direction, eval_angles_rad, args.layer, device)
        sweep_results[name] = {'kl': kl, 'l2': l2}
        print(f"{time.time()-t0:.1f}s")

    # Random directions
    print(f"  Sweeping: random ({args.n_random_dirs} dirs/act)...", end=' ', flush=True)
    t0 = time.time()
    rand_kl, rand_l2 = sweep_random_directions(
        model, contexts_dev, boundary_acts, ref_final_bnd, ref_lp_bnd,
        eval_angles_rad, args.layer, device, n_dirs=args.n_random_dirs)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2}
    print(f"{time.time()-t0:.1f}s")

    # ===== Phase 7: Save everything =====
    all_results = {
        **opt_checkpoint,
        'eval_angles_deg': eval_angles_deg,
        'sweep_results': sweep_results,
    }
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved full results to {results_path}")

    # ===== Phase 8: Plots =====
    print(f"\n{'=' * 60}")
    print("PHASE 8: Plotting")
    print("=" * 60)

    walk_key = f'{walk_label} (walk dir)'
    colors = {
        walk_key: '#2ca02c',              # green
        'bnd_MELBO_1': '#d62728',          # red
        'bnd_MELBO_2': '#ff7f0e',          # orange
        'random': 'gray',
    }

    # --- Plot 1: Boundary finding (KL along MELBO_1 from Mandarin activations) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for i in range(min(N, 10)):
        ax.plot(boundary_angles_deg, kl_boundary_sweep[i], alpha=0.3, color='gray')
    median_kl = np.median(kl_boundary_sweep, axis=0)
    ax.plot(boundary_angles_deg, median_kl, color='red', lw=2.5, label='Median')
    ax.axvline(median_boundary, color='blue', ls='--', lw=1.5,
               label=f'Median boundary ({median_boundary:.0f} deg)')
    ax.set_xlabel(f'Perturbation angle along {walk_label} (degrees)')
    ax.set_ylabel(f'KL divergence (vs {src_label} output)')
    ax.set_title(f'KL from {src_label} activations along {walk_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(per_prompt_boundary_deg, bins=15, color='steelblue', edgecolor='white')
    ax.axvline(median_boundary, color='red', ls='--', lw=2,
               label=f'Median ({median_boundary:.0f} deg)')
    ax.set_xlabel('Boundary angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of boundary angles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Boundary finding: {args.model}, layer {args.layer}\n'
                 f'{N} {src_label} prompts, walking along {walk_label}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'boundary_finding.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 2: MELBO sweeps from boundary ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for metric_idx, (key, ylabel) in enumerate(
            [('kl', 'KL divergence'), ('l2', 'L2 distance (final hidden)')]):
        ax = axes[metric_idx]

        # Random baseline
        rand_data = sweep_results['random'][key]
        rand_per_act = np.median(rand_data, axis=2)
        rand_median = np.median(rand_per_act, axis=0)
        rand_q25 = np.percentile(rand_per_act, 25, axis=0)
        rand_q75 = np.percentile(rand_per_act, 75, axis=0)
        ax.fill_between(eval_angles_deg, rand_q25, rand_q75, color='gray', alpha=0.2)
        ax.plot(eval_angles_deg, rand_median, color='gray', lw=2,
                label='Random', linestyle='--')

        # Named directions
        for name in [walk_key, 'bnd_MELBO_1', 'bnd_MELBO_2']:
            if name not in sweep_results:
                continue
            data = sweep_results[name][key]
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            color = colors.get(name, 'black')
            ax.plot(eval_angles_deg, median, color=color, lw=2, label=name)
            ax.fill_between(eval_angles_deg, q25, q75, color=color, alpha=0.1)

        ax.set_xlabel('Perturbation angle from boundary (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 90)

    fig.suptitle(f'Boundary MELBO: {args.model}, layer {args.layer}\n'
                 f'Directions found at boundary (MELBO at {args.opt_angle} deg), '
                 f'orthogonal to MELBO_1', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'boundary_melbo_sweeps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 3: Optimization convergence ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(bnd_melbo_histories):
        color = colors.get(f'bnd_MELBO_{i+1}', 'black')
        ax.plot(history, label=f'bnd_MELBO_{i+1} (final={history[-1]:.4f})', color=color)
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('KL divergence')
    ax.set_title(f'Boundary MELBO optimization ({args.opt_angle} deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.results_dir, 'optimization_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print(f"Model: {args.model}, layer {args.layer}")
    print(f"From: {src_label}, walking: {walk_label}")
    print(f"Prompts used: {N} (seq_len={args.seq_len})")
    print(f"Boundary angle (median): {median_boundary:.1f} deg")

    print(f"\nDirections found at boundary:")
    for i in range(len(bnd_melbo_directions)):
        d = bnd_melbo_directions[i].to(device)
        cos_m1 = (d @ melbo_1_dev).abs().item()
        cos_mean = (d @ mean_dir.to(device)).abs().item()
        print(f"  bnd_MELBO_{i+1}: KL={bnd_melbo_kls[i]:.6f}, "
              f"|cos(MELBO_1)|={cos_m1:.4f}, |cos(mean)|={cos_mean:.4f}")

    # KL at reference angles
    for ref_angle in [15, 30]:
        idx = np.argmin(np.abs(eval_angles_deg - ref_angle))
        print(f"\nMedian KL at {ref_angle} deg from boundary:")
        for name in [walk_key, 'bnd_MELBO_1', 'bnd_MELBO_2']:
            if name not in sweep_results:
                continue
            med_kl = np.median(sweep_results[name]['kl'][:, idx])
            print(f"  {name}: {med_kl:.4f}")
        rand_med = np.median(sweep_results['random']['kl'][:, idx, :].mean(axis=1))
        print(f"  random: {rand_med:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
