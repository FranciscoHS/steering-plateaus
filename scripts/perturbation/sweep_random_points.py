#!/usr/bin/env python3
"""Perturbation sweep on randomly sampled points vs real activations.

Generalized version of sweep_random_points_chess.py that works with any model.
Samples random points (Gaussian-matched to real activations), perturbs them
along random directions on the sphere, and measures cosine distance at a
downstream layer.

Usage:
    python scripts/perturbation/sweep_random_points.py --model csp_yolo1 --layer 2 --measure_layer -2
    python scripts/perturbation/sweep_random_points.py --model gpt2 --layer 2 --measure_layer -2
    python scripts/perturbation/sweep_random_points.py --model tinystories-28m --layer 2 --measure_layer -2
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import (
    load_fineweb_fixed_length, generate_random_token_sequences,
    load_python_code_fixed_length, generate_chess_games,
    sample_random_activations_v2,
)


def extract_context_and_activation(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """Extract context and final-token activation at a given layer.

    Handles both standard HF models (output_hidden_states) and circuitgpt models
    (manual forward through embedding + blocks).
    """
    all_contexts = []
    all_activations = []
    is_circuitgpt = model._plateau_info['model_type'] == 'circuitgpt'

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)
        with torch.no_grad():
            if is_circuitgpt:
                # CSP doesn't support output_hidden_states; run embedding + blocks manually
                gpt = model.circuit_model
                tok_emb = gpt.transformer.wte(batch_tokens)
                x = gpt.transformer.drop(tok_emb)
                seq_len = batch_tokens.shape[1]
                has_cat_pos = getattr(gpt.config, 'd_pos_emb', None) is not None
                if has_cat_pos:
                    pos_emb = gpt.transformer.wpe.weight[:seq_len].unsqueeze(0)
                else:
                    # Dense CSP: add position embeddings to residual stream (GPT-2 style)
                    positions = torch.arange(seq_len, device=device).unsqueeze(0)
                    x = x + gpt.transformer.wpe(positions)
                for block in gpt.transformer.h[:layer + 1]:
                    x = block(x, pos_emb) if has_cat_pos else block(x)
                hidden = x
            else:
                outputs = model(batch_tokens, output_hidden_states=True)
                hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def sweep_direction(model, ctx, act, direction, layer, measure_layer, device, angles_rad,
                    batch_size=64):
    """Sweep a single direction, return cosine distance array. Batched over angles."""
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = direction.to(device=device, dtype=torch.float32)
    # Project out component along activation to get orthogonal direction
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    # Build all perturbed points at once: p[j] = R * (cos(theta_j) * a_unit + sin(theta_j) * d_orth)
    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos().unsqueeze(1)  # [n_angles, 1]
    sin_t = thetas.sin().unsqueeze(1)  # [n_angles, 1]
    all_points = R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_orth.unsqueeze(0))  # [n_angles, d]
    all_points = all_points.to(model_dtype)

    cos_arr = np.zeros(len(angles_rad))
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None], layer, measure_layer).float()
        # Process in batches to avoid OOM
        for start in range(0, len(angles_rad), batch_size):
            end = min(start + batch_size, len(angles_rad))
            batch_points = all_points[start:end].unsqueeze(1)  # [bs, 1, d]
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch_points, layer, measure_layer).float()
            cos_arr[start:end] = (1.0 - F.cosine_similarity(ref, pf, dim=-1)).cpu().numpy()
    return cos_arr


def resolve_data_source(model_name, model):
    """Auto-detect the appropriate data source for a model."""
    if model_name.startswith('chess'):
        return 'chess'
    elif model._plateau_info['model_type'] == 'circuitgpt':
        return 'python_code'
    elif 'tinystories' in model_name.lower():
        return 'fineweb'  # TinyStories uses GPT-Neo tokenizer, fineweb works
    else:
        return 'fineweb'


def load_token_sequences(data_source, n, model, model_name, tokenizer, seq_len, seed):
    """Load token sequences from the appropriate data source."""
    if data_source == 'chess':
        print(f"Generating {n} chess game sequences (seq_len={seq_len})...")
        return generate_chess_games(n, seq_len, seed=seed)
    elif data_source == 'random':
        vocab_size = getattr(model.config, 'vocab_size', None) or model.circuit_model.config.vocab_size
        print(f"Generating {n} random token sequences (vocab_size={vocab_size})...")
        return generate_random_token_sequences(n, vocab_size, seq_len, seed=seed)
    elif data_source == 'python_code':
        from transformers import AutoTokenizer
        print(f"Loading CSP tokenizer...")
        csp_tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
        print(f"Loading {n} Python code sequences...")
        return load_python_code_fixed_length(n, csp_tokenizer, seq_len=seq_len, seed=seed)
    else:  # fineweb
        print(f"Loading {n} FineWeb sequences...")
        return load_fineweb_fixed_length(n, tokenizer, seq_len=seq_len, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description='Perturbation sweep: real vs random points for any model')
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--data_source', default='auto',
                        choices=['auto', 'fineweb', 'random', 'python_code', 'chess'],
                        help='Data source (default: auto-detect from model)')
    parser.add_argument('--layer', type=int, default=2,
                        help='Layer to inject activations at')
    parser.add_argument('--measure_layer', type=int, default=-2,
                        help='Layer to measure cosine distance at (default: 2nd-to-last)')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=50,
                        help='Number of random points to sweep')
    parser.add_argument('--n_dirs', type=int, default=20,
                        help='Random directions per point')
    parser.add_argument('--n_fit', type=int, default=200,
                        help='Number of real activations for covariance fitting')
    parser.add_argument('--n_real', type=int, default=50,
                        help='Number of real activations to sweep for comparison')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--n_angles', type=int, default=None,
                        help='Number of angle steps (default: 4 per degree)')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Model dtype (default: model default)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/random_points_sweep/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    load_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}" + (f" (dtype={args.dtype})" if args.dtype else ""))
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']

    # Resolve negative measure_layer
    if args.measure_layer < 0:
        args.measure_layer = n_layers + args.measure_layer
    print(f"d_model={d_model}, n_layers={n_layers}, "
          f"inject@layer {args.layer}, measure@layer {args.measure_layer}")

    # Resolve data source
    data_source = args.data_source
    if data_source == 'auto':
        data_source = resolve_data_source(args.model, model)
    print(f"Data source: {data_source}")

    # Load token sequences and extract activations
    n_load = max(args.n_fit, args.n_real)
    token_ids_list = load_token_sequences(
        data_source, n_load, model, args.model, tokenizer, args.seq_len, args.seed)

    print(f"Extracting layer-{args.layer} activations from {len(token_ids_list)} sequences...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Real activation norms: mean={all_activations.norm(dim=-1).mean():.2f}, "
          f"std={all_activations.norm(dim=-1).std():.2f}")

    # Sample random points from fitted Gaussian
    print(f"\nFitting covariance from {all_activations.shape[0]} activations...")
    random_acts, cov_stats = sample_random_activations_v2(
        all_activations, args.n_act, mode='full', seed=args.seed, device=device
    )
    print(f"Random point norms: mean={random_acts.norm(dim=-1).mean():.2f}, "
          f"std={random_acts.norm(dim=-1).std():.2f}")

    # Use real contexts as stand-ins for the random points
    real_contexts = all_contexts[:max(args.n_act, args.n_real)]
    real_activations = all_activations[:args.n_real]

    # Angle grid
    n_angles = args.n_angles if args.n_angles else int(args.max_angle * 4)
    angles_deg = np.linspace(0, args.max_angle, n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Sweep random points
    print(f"\n=== Random points ({args.n_act} points x {args.n_dirs} dirs) ===")
    rand_curves = np.zeros((args.n_act, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_act):
        ctx = real_contexts[i % len(real_contexts)]
        for j in range(args.n_dirs):
            d = torch.randn(d_model)
            rand_curves[i, j] = sweep_direction(
                model, ctx, random_acts[i], d,
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_act}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Random points done in {time.time()-t0:.1f}s")

    # Sweep real activations for comparison
    print(f"\n=== Real activations ({args.n_real} points x {args.n_dirs} dirs) ===")
    real_curves = np.zeros((args.n_real, args.n_dirs, len(angles_deg)))
    t0 = time.time()
    for i in range(args.n_real):
        for j in range(args.n_dirs):
            d = torch.randn(d_model)
            real_curves[i, j] = sweep_direction(
                model, real_contexts[i], real_activations[i], d,
                args.layer, args.measure_layer, device, angles_rad)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_real}  ({(i+1)/elapsed:.1f} act/s)")
    print(f"Real activations done in {time.time()-t0:.1f}s")

    # Save results
    results = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'rand_curves': rand_curves,   # [n_act, n_dirs, n_angles]
        'real_curves': real_curves,    # [n_real, n_dirs, n_angles]
        'rand_norms': random_acts.norm(dim=-1).numpy(),
        'real_norms': real_activations.norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(args.results_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: median + IQR for both groups
    rand_per_act = np.median(rand_curves, axis=1)  # [n_act, n_angles]
    real_per_act = np.median(real_curves, axis=1)   # [n_real, n_angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Real activations
    ax.fill_between(angles_deg,
                    np.percentile(real_per_act, 25, axis=0),
                    np.percentile(real_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(real_per_act, axis=0),
            color='C0', lw=2.5, label=f'Real activations (n={args.n_real})')

    # Random points
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C1', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C1', lw=2.5, label=f'Random points (n={args.n_act})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel('Cosine distance (at layer {})'.format(args.measure_layer))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.max_angle)
    ax.set_title(f'{args.model}: real vs random points, '
                 f'inject@L{args.layer} measure@L{args.measure_layer}')
    plt.tight_layout()

    plot_path = os.path.join(args.results_dir, 'real_vs_random_points.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for name, curves in [('Real', real_per_act), ('Random', rand_per_act)]:
        med = np.median(curves, axis=0)
        crossings = np.where(med > 0.1)[0]
        if len(crossings) > 0:
            print(f"{name}: median crosses 0.1 at {angles_deg[crossings[0]]:.1f}°")
        else:
            print(f"{name}: median never crosses 0.1")


if __name__ == '__main__':
    main()
