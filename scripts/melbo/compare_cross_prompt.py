#!/usr/bin/env python3
"""
Cross-prompt feature recovery: compare MELBO directions from two prompts.

1. Loads directions from two MELBO results.pkl files
2. Computes pairwise cosine similarity matrix
3. Steers both prompts with each direction, generates completions (greedy, 40 tokens)
4. Saves: cosine similarity matrix, all steered completions, raw data

Usage:
    python scripts/melbo/compare_cross_prompt.py \
        --results_a results/iterative_melbo_hf/run_a/results.pkl \
        --results_b results/iterative_melbo_hf/run_b/results.pkl \
        --model qwen3-1.7b --inject_layer 7
"""

import argparse, os, pickle, sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks


def make_steering_hook(direction, angle_rad):
    """Hook that spherically perturbs the last-token activation at each generation step."""
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        a = hidden[:, -1, :]
        a_norm = a.norm(dim=-1, keepdim=True)
        a_unit = a / (a_norm + 1e-10)

        d = direction.unsqueeze(0).to(a.dtype)
        proj = (d * a_unit).sum(dim=-1, keepdim=True)
        u_perp = d - proj * a_unit
        u_perp = u_perp / (u_perp.norm(dim=-1, keepdim=True) + 1e-10)

        a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
        hidden = hidden.clone()
        hidden[:, -1, :] = a_pert

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook_fn


def generate_steered(model, tokenizer, prompt, direction, angle_deg, layer,
                     device, n_tokens=40):
    """Generate greedy with steering hook at given layer."""
    angle_rad = np.deg2rad(angle_deg)
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    block = _get_blocks(model)[layer]
    hook = block.register_forward_hook(make_steering_hook(direction, angle_rad))

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=n_tokens,
                pad_token_id=tokenizer.eos_token_id, do_sample=False)
        new_ids = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids)
    finally:
        hook.remove()


def generate_unsteered(model, tokenizer, prompt, device, n_tokens=40):
    """Normal greedy generation."""
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(
            tokens['input_ids'], max_new_tokens=n_tokens,
            pad_token_id=tokenizer.eos_token_id, do_sample=False)
    new_ids = output_ids[0, tokens['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_a', type=str, required=True,
                        help='Path to first MELBO results.pkl')
    parser.add_argument('--results_b', type=str, required=True,
                        help='Path to second MELBO results.pkl')
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--inject_layer', type=int, default=7)
    parser.add_argument('--steer_angle', type=float, default=20.0,
                        help='Steering angle in degrees for generation')
    parser.add_argument('--n_tokens', type=int, default=40)
    parser.add_argument('--max_dirs', type=int, default=0,
                        help='Max directions to use from each set (0=all)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None)
    args = parser.parse_args()

    # Load results
    print(f"Loading A: {args.results_a}")
    with open(args.results_a, 'rb') as f:
        res_a = pickle.load(f)
    print(f"Loading B: {args.results_b}")
    with open(args.results_b, 'rb') as f:
        res_b = pickle.load(f)

    dirs_a = res_a['directions']
    dirs_b = res_b['directions']
    prompt_a = res_a['args'].get('custom_prompt', 'prompt_A')
    prompt_b = res_b['args'].get('custom_prompt', 'prompt_B')

    if args.max_dirs > 0:
        dirs_a = dirs_a[:args.max_dirs]
        dirs_b = dirs_b[:args.max_dirs]

    n_a, n_b = len(dirs_a), len(dirs_b)
    print(f"Prompt A: {prompt_a!r} — {n_a} directions")
    print(f"Prompt B: {prompt_b!r} — {n_b} directions")

    # Stack into matrices
    D_a = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d) for d in dirs_a]).float()
    D_b = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d) for d in dirs_b]).float()
    D_a = F.normalize(D_a, dim=-1)
    D_b = F.normalize(D_b, dim=-1)

    # Pairwise cosine similarity: [n_a, n_b]
    cos_sim = (D_a @ D_b.T).numpy()
    print(f"\nCosine similarity matrix shape: {cos_sim.shape}")
    print(f"Max |cos_sim|: {np.abs(cos_sim).max():.4f}")
    print(f"Mean |cos_sim|: {np.abs(cos_sim).mean():.4f}")

    # Find top matches
    print("\nTop matches (by |cosine similarity|):")
    flat_idx = np.argsort(np.abs(cos_sim).ravel())[::-1]
    for rank in range(min(10, len(flat_idx))):
        i, j = np.unravel_index(flat_idx[rank], cos_sim.shape)
        print(f"  A[{i}] — B[{j}]: cos_sim = {cos_sim[i, j]:+.4f}")

    # Output dir
    if args.output_dir is None:
        args.output_dir = 'results/cross_prompt_feature_recovery'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model for steering
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)

    # Move directions to device
    D_a_dev = D_a.to(device)
    D_b_dev = D_b.to(device)

    prompts = {'A': prompt_a, 'B': prompt_b}
    dir_sets = {'A': D_a_dev, 'B': D_b_dev}

    # Generate unsteered baselines
    print("\nGenerating unsteered baselines...")
    baselines = {}
    for label, prompt in prompts.items():
        baselines[label] = generate_unsteered(model, tokenizer, prompt, device, args.n_tokens)
        print(f"  {label} ({prompt!r}): {baselines[label]}")

    # Steer each prompt with each direction from both sets
    print(f"\nSteering at {args.steer_angle}° with {n_a + n_b} directions × 2 prompts...")
    completions = {}  # (dir_source, dir_idx, target_prompt) -> completion

    for src_label, src_dirs in dir_sets.items():
        n_dirs = src_dirs.shape[0]
        for di in range(n_dirs):
            direction = src_dirs[di]
            for tgt_label, prompt in prompts.items():
                key = (src_label, di, tgt_label)
                comp = generate_steered(
                    model, tokenizer, prompt, direction,
                    args.steer_angle, args.inject_layer, device, args.n_tokens)
                completions[key] = comp
                print(f"  {src_label}[{di}] → {tgt_label}: {comp[:80]}")

    # Save text report
    lines = []
    lines.append(f"Cross-Prompt Feature Recovery")
    lines.append(f"Prompt A: {prompt_a!r}")
    lines.append(f"Prompt B: {prompt_b!r}")
    lines.append(f"Steer angle: {args.steer_angle}°")
    lines.append(f"Directions: A={n_a}, B={n_b}")
    lines.append("")

    lines.append("=== BASELINES (unsteered) ===")
    for label, prompt in prompts.items():
        lines.append(f"  {label} ({prompt!r}): {baselines[label]}")
    lines.append("")

    # Group by direction, showing cross-prompt steering
    for src_label in ['A', 'B']:
        n_dirs = dir_sets[src_label].shape[0]
        lines.append(f"=== Directions from prompt {src_label} ({prompts[src_label]!r}) ===")
        for di in range(n_dirs):
            lines.append(f"\n  --- {src_label}[{di}] ---")
            for tgt_label in ['A', 'B']:
                comp = completions[(src_label, di, tgt_label)]
                same = "same" if src_label == tgt_label else "CROSS"
                lines.append(f"    → {tgt_label} ({same}): {prompts[tgt_label]}{comp}")
            # Show cosine sim to best match in the other set
            if src_label == 'A':
                sims = cos_sim[di, :]
            else:
                sims = cos_sim[:, di]
            best_idx = np.argmax(np.abs(sims))
            lines.append(f"    Best match in {'B' if src_label == 'A' else 'A'}: "
                         f"[{best_idx}] (cos_sim={sims[best_idx]:+.4f})")
        lines.append("")

    report = '\n'.join(lines)
    report_path = os.path.join(args.output_dir, 'steering_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")

    # Plot cosine similarity matrix
    fig, ax = plt.subplots(figsize=(max(6, n_b * 0.4 + 2), max(5, n_a * 0.4 + 2)))
    im = ax.imshow(np.abs(cos_sim), cmap='RdBu_r', vmin=0, vmax=max(0.3, np.abs(cos_sim).max()),
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='|cosine similarity|')
    ax.set_xlabel(f'Directions from B ({prompt_b!r})')
    ax.set_ylabel(f'Directions from A ({prompt_a!r})')
    ax.set_title(f'Cross-prompt direction similarity\nmax |cos|={np.abs(cos_sim).max():.4f}')
    ax.set_xticks(range(n_b))
    ax.set_yticks(range(n_a))
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'cosine_similarity_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Save raw data
    raw = {
        'prompt_a': prompt_a,
        'prompt_b': prompt_b,
        'dirs_a': D_a.cpu(),
        'dirs_b': D_b.cpu(),
        'cos_sim': cos_sim,
        'baselines': baselines,
        'completions': completions,
        'steer_angle': args.steer_angle,
        'args': vars(args),
    }
    raw_path = os.path.join(args.output_dir, 'results.pkl')
    with open(raw_path, 'wb') as f:
        pickle.dump(raw, f)
    print(f"Saved data: {raw_path}")


if __name__ == '__main__':
    main()
