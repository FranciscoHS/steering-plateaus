#!/usr/bin/env python3
"""Compare perturbation plateaus: FineWeb vs cov-matched random vs nonsense prompts.

Four conditions:
1. FineWeb (coherent text)
2. Gibberish (random character strings like "aiusfsafhiso")
3. Word salad (real words, random order like "tiger golden carpet lemon")
4. Covariance-matched random points

Usage:
    python scripts/perturbation/sweep_nonsense_vs_real.py [--layer 2] [--n_act 50]
"""
import argparse, os, sys, pickle, time, random
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, _get_blocks,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from src.data import load_fineweb_fixed_length, sample_random_activations_v2
from scripts.perturbation.sweep_random_points import extract_context_and_activation


def generate_gibberish_prompts(n, tokenizer, seq_len=5, seed=42):
    """Generate gibberish prompts: random character strings like 'aiusfsafhiso'.

    Args:
        n: Number of sequences to return
        tokenizer: Model tokenizer
        seq_len: Exact token length for each sequence
        seed: Random seed

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    rng = random.Random(seed)
    letters = 'abcdefghijklmnopqrstuvwxyz'

    token_ids_list = []
    attempts = 0
    max_attempts = n * 50

    while len(token_ids_list) < n and attempts < max_attempts:
        attempts += 1
        str_len = rng.randint(10, 40)
        gibberish = ''.join(rng.choice(letters) for _ in range(str_len))

        ids = tokenizer.encode(gibberish, add_special_tokens=False)
        if len(ids) >= seq_len:
            token_ids_list.append(torch.tensor(ids[:seq_len], dtype=torch.long))

    print(f"Generated {len(token_ids_list)} gibberish sequences (each {seq_len} tokens)")
    for i in range(min(5, len(token_ids_list))):
        text = tokenizer.decode(token_ids_list[i])
        print(f"  Example {i}: '{text}'")

    return token_ids_list


def generate_word_salad_prompts(n, tokenizer, seq_len=5, seed=42):
    """Generate word salad: real words in random order like 'tiger golden carpet lemon'.

    Args:
        n: Number of sequences to return
        tokenizer: Model tokenizer
        seq_len: Exact token length for each sequence
        seed: Random seed

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    rng = random.Random(seed + 777)  # different seed from gibberish

    word_pool = [
        "the", "cat", "dog", "run", "big", "small", "blue", "red", "green", "house",
        "tree", "fish", "bird", "book", "water", "fire", "sky", "moon", "sun", "star",
        "rock", "dust", "cold", "warm", "fast", "slow", "loud", "soft", "dark", "light",
        "king", "door", "wall", "road", "hill", "lake", "rain", "snow", "wind", "leaf",
        "chair", "table", "glass", "stone", "metal", "paper", "clock", "train", "ship",
        "bridge", "tower", "river", "cloud", "flame", "ocean", "plant", "earth", "bread",
        "knife", "wheel", "chain", "storm", "paint", "music", "dance", "horse", "mouse",
        "snake", "tiger", "eagle", "shark", "pearl", "sword", "crown", "ghost", "dream",
        "night", "morning", "garden", "forest", "desert", "mountain", "valley", "island",
        "castle", "village", "market", "winter", "summer", "spring", "autumn", "silver",
        "golden", "purple", "orange", "yellow", "frozen", "broken", "ancient", "modern",
        "quiet", "gentle", "bitter", "sweet", "angry", "happy", "tired", "hungry",
    ]

    token_ids_list = []
    attempts = 0
    max_attempts = n * 50

    while len(token_ids_list) < n and attempts < max_attempts:
        attempts += 1
        n_words = rng.randint(3, 10)
        words = [rng.choice(word_pool) for _ in range(n_words)]
        text = " ".join(words)

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= seq_len:
            token_ids_list.append(torch.tensor(ids[:seq_len], dtype=torch.long))

    print(f"Generated {len(token_ids_list)} word salad sequences (each {seq_len} tokens)")
    for i in range(min(5, len(token_ids_list))):
        text = tokenizer.decode(token_ids_list[i])
        print(f"  Example {i}: '{text}'")

    return token_ids_list


def sweep_one_activation(model, ctx, act, n_dirs, layer, measure_layer, device,
                         angles_rad, metric='l2', batch_size=64):
    """Sweep random directions for one activation. Returns [n_dirs, n_angles] distances."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R

    n_angles = len(angles_rad)
    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos()
    sin_t = thetas.sin()

    # Reference output
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d.unsqueeze(0) if ctx_d.dim() == 2 else ctx_d,
            a_d[None, None].to(model_dtype), layer, measure_layer).float()

    curves = np.zeros((n_dirs, n_angles))
    d_model = act.shape[-1]

    for j in range(n_dirs):
        d = torch.randn(d_model, device=device)
        d = d - (d @ a_unit) * a_unit
        d = F.normalize(d, dim=0)

        # Build perturbed points
        all_points = (R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])).to(model_dtype)

        # Batch over angles
        for start in range(0, n_angles, batch_size):
            end = min(start + batch_size, n_angles)
            batch_points = all_points[start:end].unsqueeze(1)  # [batch, 1, d]
            ctx_for_batch = ctx_d.unsqueeze(0) if ctx_d.dim() == 2 else ctx_d

            with torch.no_grad():
                measured = forward_from_layer_to_layer(
                    model, ctx_for_batch, batch_points, layer, measure_layer).float()
                if metric == 'l2':
                    dist = (measured - ref).norm(dim=-1)
                elif metric == 'cosine':
                    dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                curves[j, start:end] = dist.cpu().numpy()

    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer to measure at (default: n_layers - 2)')
    parser.add_argument('--metric', default='l2', choices=['l2', 'cosine'],
                        help='Distance metric (default: l2)')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=50, help='Activations per condition')
    parser.add_argument('--n_dirs', type=int, default=10, help='Random directions per activation')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=120)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--conditions', default='fineweb,gibberish,word_salad,random',
                        help='Comma-separated conditions to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()
    args.conditions_list = [c.strip() for c in args.conditions.split(',')]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/nonsense_vs_real/{args.model}'
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"d_model={d_model}, n_layers={n_layers}, perturb@L{args.layer}, measure@L{measure_layer}")

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # === 1. FineWeb activations ===
    print(f"\n{'='*60}")
    print("Condition 1: FineWeb (coherent text)")
    print(f"{'='*60}")
    # Load extra for covariance fitting
    n_for_cov = max(args.n_act, d_model + 100)  # need N > D for full-rank cov
    n_load = max(args.n_act, n_for_cov)
    fineweb_tokens = load_fineweb_fixed_length(n_load, tokenizer, seq_len=args.seq_len, seed=args.seed)
    fineweb_ctx, fineweb_acts = extract_context_and_activation(
        model, tokenizer, fineweb_tokens[:n_load], args.layer, device)
    print(f"FineWeb activation norms: mean={fineweb_acts.norm(dim=-1).mean():.2f}, "
          f"std={fineweb_acts.norm(dim=-1).std():.2f}")

    # === 2. Gibberish prompts ===
    gibberish_ctx, gibberish_acts = None, None
    if 'gibberish' in args.conditions_list:
        print(f"\n{'='*60}")
        print("Condition: Gibberish (random characters)")
        print(f"{'='*60}")
        gibberish_tokens = generate_gibberish_prompts(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
        gibberish_ctx, gibberish_acts = extract_context_and_activation(
            model, tokenizer, gibberish_tokens, args.layer, device)
        print(f"Gibberish activation norms: mean={gibberish_acts.norm(dim=-1).mean():.2f}, "
              f"std={gibberish_acts.norm(dim=-1).std():.2f}")

    # === 3. Word salad prompts ===
    salad_ctx, salad_acts = None, None
    if 'word_salad' in args.conditions_list:
        print(f"\n{'='*60}")
        print("Condition: Word salad (random real words)")
        print(f"{'='*60}")
        salad_tokens = generate_word_salad_prompts(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
        salad_ctx, salad_acts = extract_context_and_activation(
            model, tokenizer, salad_tokens, args.layer, device)
        print(f"Word salad activation norms: mean={salad_acts.norm(dim=-1).mean():.2f}, "
              f"std={salad_acts.norm(dim=-1).std():.2f}")

    # === 4. Covariance-matched random ===
    random_acts = None
    if 'random' in args.conditions_list:
        print(f"\n{'='*60}")
        print("Condition: Covariance-matched random points")
        print(f"{'='*60}")
        random_acts, cov_stats = sample_random_activations_v2(
            fineweb_acts[:n_for_cov], args.n_act, mode='full', seed=args.seed, device=device)
        print(f"Random activation norms: mean={random_acts.norm(dim=-1).mean():.2f}, "
              f"std={random_acts.norm(dim=-1).std():.2f}")

    # === Build conditions dict ===
    conditions = {}
    if 'fineweb' in args.conditions_list:
        conditions['fineweb'] = (fineweb_ctx[:args.n_act], fineweb_acts[:args.n_act])
    if 'gibberish' in args.conditions_list:
        conditions['gibberish'] = (gibberish_ctx, gibberish_acts)
    if 'word_salad' in args.conditions_list:
        conditions['word_salad'] = (salad_ctx, salad_acts)
    if 'random' in args.conditions_list:
        conditions['random'] = (fineweb_ctx[:args.n_act], random_acts)

    all_curves = {}
    for cond_name, (ctxs, acts) in conditions.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {cond_name} ({len(acts)} acts x {args.n_dirs} dirs)")
        print(f"{'='*60}")
        curves = np.zeros((len(acts), args.n_dirs, len(angles_deg)))
        t0 = time.time()
        for i in range(len(acts)):
            curves[i] = sweep_one_activation(
                model, ctxs[i], acts[i], args.n_dirs, args.layer, measure_layer,
                device, angles_rad, metric=args.metric)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(acts)}  ({(i+1)/elapsed:.1f} act/s)")
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        all_curves[cond_name] = curves

    # === Save results ===
    results = {
        'config': vars(args),
        'angles_deg': angles_deg,
        'measure_layer': measure_layer,
    }
    for cond_name, curves in all_curves.items():
        results[f'{cond_name}_curves'] = curves  # [n_act, n_dirs, n_angles]
    if 'fineweb' in conditions:
        results['fineweb_norms'] = fineweb_acts[:args.n_act].norm(dim=-1).numpy()
    if 'gibberish' in conditions:
        results['gibberish_norms'] = gibberish_acts.norm(dim=-1).numpy()
    if 'word_salad' in conditions:
        results['word_salad_norms'] = salad_acts.norm(dim=-1).numpy()
    if 'random' in conditions:
        results['random_norms'] = random_acts.norm(dim=-1).numpy()

    pkl_path = os.path.join(args.results_dir, 'nonsense_vs_real.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # === Plot ===
    cond_order = [c for c in ['fineweb', 'gibberish', 'word_salad', 'random'] if c in conditions]
    colors = {'fineweb': 'C0', 'gibberish': 'C3', 'word_salad': 'C2', 'random': 'C1'}
    labels = {
        'fineweb': 'FineWeb (coherent)',
        'gibberish': 'Gibberish (random chars)',
        'word_salad': 'Word salad (random words)',
        'random': 'Cov-matched random',
    }

    # Median plot (median over directions per activation, then median + IQR over activations)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond_name in cond_order:
        curves = all_curves[cond_name]
        # Median over directions for each activation
        per_act = np.median(curves, axis=1)  # [n_act, n_angles]
        med = np.median(per_act, axis=0)
        q25 = np.percentile(per_act, 25, axis=0)
        q75 = np.percentile(per_act, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color=colors[cond_name], alpha=0.15)
        ax.plot(angles_deg, med, color=colors[cond_name], lw=2.5, label=labels[cond_name])

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ylabel = f'L2 distance (layer {measure_layer})' if args.metric == 'l2' else f'Cosine distance (layer {measure_layer})'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'{args.model}: nonsense vs real vs random, '
                 f'inject@L{args.layer} measure@L{measure_layer}', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'nonsense_vs_real_median.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Individual curves plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond_name in cond_order:
        curves = all_curves[cond_name]
        per_act = np.median(curves, axis=1)  # [n_act, n_angles]
        for i in range(min(20, len(per_act))):
            ax.plot(angles_deg, per_act[i], color=colors[cond_name], alpha=0.3, lw=0.8)
        ax.plot([], [], color=colors[cond_name], lw=2, label=labels[cond_name])

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=12)
    ylabel = f'L2 distance (layer {measure_layer})' if args.metric == 'l2' else f'Cosine distance (layer {measure_layer})'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'{args.model}: individual curves, '
                 f'inject@L{args.layer} measure@L{measure_layer}', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'nonsense_vs_real_individual.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    print("\n=== Breaking angles (threshold crossings) ===")
    for cond_name in cond_order:
        curves = all_curves[cond_name]
        per_act = np.median(curves, axis=1)
        med = np.median(per_act, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"  {cond_name}: median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"  {cond_name}: median never crosses {thresh}")

    print("\nDone!")


if __name__ == '__main__':
    main()
