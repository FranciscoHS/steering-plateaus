"""Robustness checks for TWO-NN intrinsic dimensionality estimate.

Tests whether the ID estimate is robust to:
1. Context length (seq_len): 3, 5, 7, 10, 32, 64, 128, 256
2. Token position: positions 1, 2, 4, 8, 16, 32, 64, -1 within seq_len=128
   (Position 0 is skipped as it's usually weird.)

Each condition: collect activations from a pool of 5000 texts, then subsample
2000 (without replacement) for each of 10 TWO-NN runs. Reports mean ± SEM.

Uses Qwen3-1.7B layer 2 to match the main analysis.
"""

import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import skdim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_activations(model, tokenizer, device, texts, layer, seq_len, positions=(-1,)):
    """Extract activations at a given layer for multiple token positions in one forward pass.

    Returns dict mapping position -> (n_samples, d_model) tensor.
    """
    result = {p: [] for p in positions}
    batch_size = min(128, max(32, 4096 // seq_len))
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        h = out.hidden_states[layer + 1]  # (batch, seq_len, d_model)
        for p in positions:
            result[p].append(h[:, p, :].float().cpu())
    return {p: torch.cat(result[p], dim=0) for p in positions}


def load_texts(n_texts, min_chars, seed=42):
    """Load FineWeb texts with minimum character count."""
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT', split='train', streaming=True)
    ds = ds.shuffle(seed=seed)
    texts = []
    for ex in ds:
        if len(ex['text']) >= min_chars:
            texts.append(ex['text'])
        if len(texts) >= n_texts:
            break
    return texts


def run_twonn(X_np):
    """Run TWO-NN with tiny jitter to handle near-duplicates."""
    X_np = X_np + np.random.randn(*X_np.shape) * 1e-7 * X_np.std()
    est = skdim.id.TwoNN()
    est.fit(X_np)
    return est.dimension_


def subsample_and_estimate(X_np, n_samples, n_runs):
    """Subsample n_samples from X_np (without replacement) n_runs times, return ID estimates."""
    ids = []
    for run in range(n_runs):
        rng = np.random.RandomState(run)
        idx = rng.choice(len(X_np), size=n_samples, replace=False)
        ids.append(run_twonn(X_np[idx]))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=2000, help='Subsample size per TWO-NN run')
    parser.add_argument('--n_pool', type=int, default=5000, help='Total activations to collect per condition')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--outdir', type=str, default='results/jacobian')
    args = parser.parse_args()

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device

    min_chars = 256 * 4  # enough for longest seq_len
    print(f"\nLoading texts ({args.n_pool} texts, min {min_chars} chars)...")
    all_texts = load_texts(args.n_pool, min_chars)
    print(f"Collected {len(all_texts)} texts")

    results = {
        'model': args.model, 'layer': args.layer,
        'n_samples': args.n_samples, 'n_pool': args.n_pool, 'n_runs': args.n_runs,
    }

    # ── Check 1: Context length robustness ──────────────────────────────
    # One forward pass per seq_len, extract last-token activations
    seq_lens = [3, 5, 7, 10, 32, 64, 128, 256]
    pos_seq_len = 128
    positions = [1, 2, 4, 8, 16, 32, 64, -1]

    print(f"\n=== Check 1: Context length robustness ({args.n_runs} runs each) ===")
    ctx_results = {}
    cached_128 = None  # cache seq_len=128 last-token for reuse in check 2
    for sl in seq_lens:
        acts_dict = collect_activations(model, tokenizer, device, all_texts, args.layer, sl, positions=(-1,))
        X_np = acts_dict[-1].numpy()
        if sl == pos_seq_len:
            cached_128 = X_np  # save for check 2
        ids = subsample_and_estimate(X_np, args.n_samples, args.n_runs)
        mean, sem = np.mean(ids), np.std(ids, ddof=1) / np.sqrt(len(ids))
        ctx_results[str(sl)] = {'mean': round(mean, 2), 'sem': round(sem, 2), 'runs': [round(x, 2) for x in ids]}
        print(f"  seq_len={sl:>4d}: ID = {mean:.2f} ± {sem:.2f}")
    results['context_length_sweep'] = ctx_results

    # ── Check 2: Token position robustness ──────────────────────────────
    # Single forward pass at seq_len=128, extract all positions at once
    print(f"\n=== Check 2: Token position robustness ({args.n_runs} runs each, seq_len={pos_seq_len}) ===")
    acts_dict = collect_activations(model, tokenizer, device, all_texts, args.layer, pos_seq_len, positions=tuple(positions))
    pos_results = {}
    for pos in positions:
        X_np = acts_dict[pos].numpy()
        ids = subsample_and_estimate(X_np, args.n_samples, args.n_runs)
        mean, sem = np.mean(ids), np.std(ids, ddof=1) / np.sqrt(len(ids))
        pos_results[str(pos)] = {'mean': round(mean, 2), 'sem': round(sem, 2), 'runs': [round(x, 2) for x in ids]}
        print(f"  position={pos:>4d}: ID = {mean:.2f} ± {sem:.2f}")
    results['position_sweep'] = pos_results

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, 'intrinsic_dim_robustness.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # ── Plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: ID vs context length
    ax = axes[0]
    sls = [int(k) for k in ctx_results]
    means = [ctx_results[k]['mean'] for k in ctx_results]
    sems = [ctx_results[k]['sem'] for k in ctx_results]
    ax.errorbar(sls, means, yerr=sems, fmt='o-', capsize=4, color='C0')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sls)
    ax.set_xticklabels([str(s) for s in sls])
    ax.set_xlabel('Context length (seq_len)')
    ax.set_ylabel('TWO-NN intrinsic dimension')
    ax.set_title('ID vs context length (last token)')
    ax.grid(True, alpha=0.3)

    # Plot 2: ID vs token position
    ax = axes[1]
    pos_labels = list(pos_results.keys())
    pos_x = list(range(len(pos_labels)))
    means = [pos_results[k]['mean'] for k in pos_labels]
    sems = [pos_results[k]['sem'] for k in pos_labels]
    ax.errorbar(pos_x, means, yerr=sems, fmt='o-', capsize=4, color='C1')
    ax.set_xticks(pos_x)
    ax.set_xticklabels(pos_labels)
    ax.set_xlabel('Token position')
    ax.set_ylabel('TWO-NN intrinsic dimension')
    ax.set_title(f'ID vs token position (seq_len={pos_seq_len})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plotpath = os.path.join(args.outdir, 'intrinsic_dim_robustness.png')
    plt.savefig(plotpath, dpi=150)
    print(f"Plot saved to {plotpath}")


if __name__ == '__main__':
    main()
