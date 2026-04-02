#!/usr/bin/env python3
"""
Cross-prompt feature recovery for 'John reads the' vs 'Mary reads the'.

Replicates the full workflow from the He/She reads experiment:
1. Load MELBO directions from both prompts
2. Compute pairwise cosine similarity
3. Top-k token boosting analysis at 20 and 40 deg
4. Orthogonal decomposition of best-matching pair
5. Slerp rotation between matched directions

Usage:
    python scripts/melbo/cross_prompt_john_mary.py \
        --results_a results/iterative_melbo_hf/qwen3-1.7b_L7_meas26_angle0.1_prompt-John_reads_the/results.pkl \
        --results_b results/iterative_melbo_hf/qwen3-1.7b_L7_meas26_angle0.1_prompt-Mary_reads_the/results.pkl
"""

import argparse, os, pickle, sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F

from src.model import load_model, _get_blocks


def get_topk_boosted(model, tokenizer, prompt, direction, angle_deg, layer, device, k=5):
    """Get top-k most boosted tokens by probability change."""
    angle_rad = np.deg2rad(angle_deg)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    # Get baseline logits
    with torch.no_grad():
        base_out = model(input_ids)
        base_probs = torch.softmax(base_out.logits[0, -1, :], dim=-1)

    # Get steered logits via hook
    block = _get_blocks(model)[layer]
    d = direction.to(device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        a = hidden[:, -1, :]
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / (a_norm + 1e-10)
        d_expanded = d.unsqueeze(0).to(a.dtype)
        proj = (d_expanded * a_unit).sum(dim=-1, keepdim=True)
        u_perp = d_expanded - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
        a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
        hidden = hidden.clone()
        hidden[:, -1, :] = a_pert
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = block.register_forward_hook(hook_fn)
    with torch.no_grad():
        steer_out = model(input_ids)
        steer_probs = torch.softmax(steer_out.logits[0, -1, :], dim=-1)
    handle.remove()

    # Top-k by probability boost
    boosts = (steer_probs - base_probs).cpu()
    topk_idx = boosts.argsort(descending=True)[:k]
    results = []
    for idx in topk_idx:
        tok = tokenizer.decode([idx.item()])
        results.append((tok, boosts[idx].item(), base_probs[idx].cpu().item(), steer_probs[idx].cpu().item()))
    return results


def slerp(v0, v1, t):
    """Spherical linear interpolation between unit vectors."""
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    dot = (v0 * v1).sum().clamp(-1, 1)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return F.normalize((1 - t) * v0 + t * v1, dim=-1)
    return (torch.sin((1 - t) * omega) * v0 + torch.sin(t * omega) * v1) / torch.sin(omega)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_a', type=str, required=True)
    parser.add_argument('--results_b', type=str, required=True)
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--output_dir', type=str,
                        default='results/cross_prompt_feature_recovery/john_mary')
    args = parser.parse_args()

    # Load results
    with open(args.results_a, 'rb') as f:
        res_a = pickle.load(f)
    with open(args.results_b, 'rb') as f:
        res_b = pickle.load(f)

    dirs_a = res_a['directions']
    dirs_b = res_b['directions']
    prompt_a = res_a['args'].get('custom_prompt', 'prompt_A')
    prompt_b = res_b['args'].get('custom_prompt', 'prompt_B')

    n_a, n_b = len(dirs_a), len(dirs_b)
    print(f"Prompt A: {prompt_a!r} — {n_a} directions")
    print(f"Prompt B: {prompt_b!r} — {n_b} directions")

    # Stack and normalize
    D_a = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d) for d in dirs_a]).float()
    D_b = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d) for d in dirs_b]).float()
    D_a = F.normalize(D_a, dim=-1)
    D_b = F.normalize(D_b, dim=-1)

    # Cosine similarity matrix
    cos_sim = (D_a @ D_b.T).numpy()
    print(f"\nCosine similarity matrix: {cos_sim.shape}")
    print(f"Max |cos_sim|: {np.abs(cos_sim).max():.4f}")
    print(f"Mean |cos_sim|: {np.abs(cos_sim).mean():.4f}")

    # Top matches
    print("\nTop 10 matches (by |cosine similarity|):")
    flat_idx = np.argsort(np.abs(cos_sim).ravel())[::-1]
    for rank in range(min(10, len(flat_idx))):
        i, j = np.unravel_index(flat_idx[rank], cos_sim.shape)
        print(f"  A[{i}] — B[{j}]: cos_sim = {cos_sim[i, j]:+.4f}")

    # Find best match pair
    best_flat = np.argmax(np.abs(cos_sim))
    best_i, best_j = np.unravel_index(best_flat, cos_sim.shape)
    best_cos = cos_sim[best_i, best_j]
    print(f"\nBest match: A[{best_i}] — B[{best_j}], cos_sim = {best_cos:+.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    D_a_dev = D_a.to(device)
    D_b_dev = D_b.to(device)

    # === Top-k token boosting at 20 and 40 deg ===
    lines = [f"Top cosine match: A[{best_i}] -- B[{best_j}] = {best_cos:.4f}\n"]

    for angle in [20, 40]:
        lines.append(f"=== Steering at {angle} deg ===")
        for prompt_label, prompt, dirs_dev, di_a, di_b in [
            ('A', prompt_a, D_a_dev, best_i, best_j),
            ('B', prompt_b, D_b_dev, best_i, best_j),
        ]:
            lines.append(f"Prompt {prompt_label}: {prompt!r}")

            # Same-prompt direction
            if prompt_label == 'A':
                same_dir, same_idx, cross_dir, cross_idx = D_a_dev[di_a], f'A[{di_a}]', D_b_dev[di_b], f'B[{di_b}]'
            else:
                same_dir, same_idx, cross_dir, cross_idx = D_b_dev[di_b], f'B[{di_b}]', D_a_dev[di_a], f'A[{di_a}]'

            for dir_vec, dir_label, kind in [
                (same_dir, same_idx, 'same-prompt'),
                (cross_dir, cross_idx, 'CROSS-prompt'),
            ]:
                topk = get_topk_boosted(model, tokenizer, prompt, dir_vec, angle, args.inject_layer, device)
                lines.append(f"  {dir_label} ({kind}) — top 5 most boosted:")
                for tok, boost, base_p, steer_p in topk:
                    lines.append(f"    {boost:+.4f}  {tok!r}  ({base_p:.4f} -> {steer_p:.4f})")

    topk_path = os.path.join(args.output_dir, 'topk_analysis.txt')
    report = '\n'.join(lines)
    with open(topk_path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {topk_path}")
    print(report)

    # === Orthogonal decomposition ===
    dir_a = D_a_dev[best_i]
    dir_b = D_b_dev[best_j]

    # B orth A: component of B orthogonal to A
    proj_ba = (dir_b * dir_a).sum() * dir_a
    b_orth_a = F.normalize(dir_b - proj_ba, dim=-1)
    # A orth B: component of A orthogonal to B
    proj_ab = (dir_a * dir_b).sum() * dir_b
    a_orth_b = F.normalize(dir_a - proj_ab, dim=-1)

    lines_orth = []
    for angle in [20, 40]:
        lines_orth.append(f"=== Steering at {angle} deg ===")
        for prompt_label, prompt in [('A', prompt_a), ('B', prompt_b)]:
            lines_orth.append(f"Prompt {prompt_label}: {prompt!r}")
            for dir_vec, dir_label in [
                (dir_a, f'A[{best_i}]'),
                (dir_b, f'B[{best_j}]'),
                (b_orth_a, f'B[{best_j}] orth A[{best_i}]'),
                (a_orth_b, f'A[{best_i}] orth B[{best_j}]'),
            ]:
                topk = get_topk_boosted(model, tokenizer, prompt, dir_vec, angle, args.inject_layer, device)
                lines_orth.append(f"  {dir_label} — top 5 most boosted:")
                for tok, boost, base_p, steer_p in topk:
                    lines_orth.append(f"    {boost:+.4f}  {tok!r}  ({base_p:.4f} -> {steer_p:.4f})")

    orth_path = os.path.join(args.output_dir, 'orthogonal_decomposition.txt')
    orth_report = '\n'.join(lines_orth)
    with open(orth_path, 'w') as f:
        f.write(orth_report)
    print(f"\nSaved: {orth_path}")
    print(orth_report)

    # === Slerp rotation ===
    cos_ab = (dir_a * dir_b).sum().item()
    angle_ab = np.rad2deg(np.arccos(np.clip(cos_ab, -1, 1)))
    print(f"\nSlerp: A[{best_i}] → B[{best_j}], cos={cos_ab:.4f}, angle={angle_ab:.1f} deg")

    steer_angle = 40  # steering angle for slerp
    t_values = [-0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]

    lines_slerp = [
        f"Rotating B[{best_j}] towards A[{best_i}] (slerp), steering at {steer_angle} deg",
        f"cos(A[{best_i}], B[{best_j}]) = {cos_ab:.4f}, angle = {angle_ab:.1f} deg",
        "",
    ]

    def get_steered_probs(model, tokenizer, prompt, direction, steer_angle_deg, layer, device):
        """Get full steered probability vector."""
        angle_rad = np.deg2rad(steer_angle_deg)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        toks_in = tokenizer(prompt, return_tensors='pt').to(device)
        block = _get_blocks(model)[layer]
        d = direction.to(device)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            a = hidden[:, -1, :]
            a_norm = a.norm(dim=-1, keepdim=True)
            a_unit = a / (a_norm + 1e-10)
            dd = d.unsqueeze(0).to(a.dtype)
            proj = (dd * a_unit).sum(dim=-1, keepdim=True)
            u_perp = dd - proj * a_unit
            u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)
            a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
            hidden = hidden.clone()
            hidden[:, -1, :] = a_pert
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = block.register_forward_hook(hook_fn)
        with torch.no_grad():
            out = model(toks_in['input_ids'])
            probs = torch.softmax(out.logits[0, -1, :], dim=-1).cpu()
        handle.remove()
        return probs

    # Identify token to track: most boosted by A[best_i] at steer_angle on prompt A
    topk_a = get_topk_boosted(model, tokenizer, prompt_a, dir_a, steer_angle, args.inject_layer, device, k=1)
    track_token = topk_a[0][0]
    # Find token ID
    track_id = None
    for tid in range(tokenizer.vocab_size):
        if tokenizer.decode([tid]) == track_token:
            track_id = tid
            break
    print(f"Tracking token: {track_token!r} (id={track_id})")

    for prompt_label, prompt in [('A', prompt_a), ('B', prompt_b)]:
        # Base probs
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            base_out = model(tokens['input_ids'])
            base_probs = torch.softmax(base_out.logits[0, -1, :], dim=-1).cpu()

        base_p = base_probs[track_id].item()
        lines_slerp.append(f"=== Prompt {prompt_label}: {prompt!r} (base {track_token!r} prob: {base_p:.4f}) ===")
        lines_slerp.append(f"  {'t':>7s}  {'cos_w_A':>9s}  {track_token.strip()+'_boost':>12s}  {'top_boosted_token':>18s}")

        for t in t_values:
            # slerp: t=0 is B[best_j], t=1 is A[best_i]
            d_interp = slerp(dir_b.float(), dir_a.float(), t).to(device)
            cos_w_a = F.cosine_similarity(d_interp.unsqueeze(0), dir_a.unsqueeze(0)).item()

            steer_probs = get_steered_probs(model, tokenizer, prompt, d_interp, steer_angle, args.inject_layer, device)
            boosts = steer_probs - base_probs
            track_boost = boosts[track_id].item()
            top_idx = boosts.argmax().item()
            top_boost = boosts[top_idx].item()
            top_tok = tokenizer.decode([top_idx])

            lines_slerp.append(f"  {t:+.3f}  {cos_w_a:+.4f}    {track_boost:+.4f}      {top_boost:+.4f} {top_tok!r}")

        lines_slerp.append("")

    # Find optimal t for each prompt (fine grid)
    lines_slerp.append("=== Optimal directions ===")
    opt_ts = {}
    for prompt_label, prompt in [('A', prompt_a), ('B', prompt_b)]:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            base_out = model(tokens['input_ids'])
            base_probs = torch.softmax(base_out.logits[0, -1, :], dim=-1).cpu()

        best_t = None
        best_boost_val = -1
        for t in np.linspace(-0.5, 1.5, 41):
            d_interp = slerp(dir_b.float(), dir_a.float(), t).to(device)
            steer_probs = get_steered_probs(model, tokenizer, prompt, d_interp, steer_angle, args.inject_layer, device)
            boost = (steer_probs[track_id] - base_probs[track_id]).item()
            if boost > best_boost_val:
                best_boost_val = boost
                best_t = t

        opt_ts[prompt_label] = best_t
        d_opt = slerp(dir_b.float(), dir_a.float(), best_t).to(device)
        cos_opt_a = F.cosine_similarity(d_opt.unsqueeze(0), dir_a.unsqueeze(0)).item()
        cos_opt_b = F.cosine_similarity(d_opt.unsqueeze(0), dir_b.unsqueeze(0)).item()
        lines_slerp.append(f"Prompt {prompt_label}: optimal t={best_t:.3f}, boost={best_boost_val:+.4f} ({track_token!r})")
        lines_slerp.append(f"  cos(opt, A[{best_i}])={cos_opt_a:.4f}, cos(opt, B[{best_j}])={cos_opt_b:.4f}")

    # Cosine sim between optimal directions
    d_opt_a = slerp(dir_b.float(), dir_a.float(), opt_ts['A']).to(device)
    d_opt_b = slerp(dir_b.float(), dir_a.float(), opt_ts['B']).to(device)
    cos_opts = F.cosine_similarity(d_opt_a.unsqueeze(0), d_opt_b.unsqueeze(0)).item()
    lines_slerp.append(f"\ncos(opt_A, opt_B) = {cos_opts:.4f}")

    slerp_path = os.path.join(args.output_dir, 'slerp_rotation.txt')
    slerp_report = '\n'.join(lines_slerp)
    with open(slerp_path, 'w') as f:
        f.write(slerp_report)
    print(f"\nSaved: {slerp_path}")
    print(slerp_report)

    # Save raw data
    raw = {
        'prompt_a': prompt_a,
        'prompt_b': prompt_b,
        'dirs_a': D_a.cpu(),
        'dirs_b': D_b.cpu(),
        'cos_sim': cos_sim,
        'best_match': (best_i, best_j, best_cos),
        'args': vars(args),
    }
    raw_path = os.path.join(args.output_dir, 'results.pkl')
    with open(raw_path, 'wb') as f:
        pickle.dump(raw, f)
    print(f"\nSaved: {raw_path}")


if __name__ == '__main__':
    main()
