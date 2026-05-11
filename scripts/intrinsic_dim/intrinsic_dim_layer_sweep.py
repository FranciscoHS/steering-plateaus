"""TWO-NN intrinsic dimensionality across residual stream layers.

Fixes seq_len=10 and last token position; varies the layer index. Mirrors
`intrinsic_dim_robustness.py` but sweeping layers rather than context length
or position.
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


def collect_all_layers(model, tokenizer, device, texts, seq_len, n_layers):
    """One forward pass per batch; return list of (n_samples, d_model) tensors,
    one per layer (indices 0..n_layers-1, using hidden_states[layer+1])."""
    result = [[] for _ in range(n_layers)]
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
        for layer in range(n_layers):
            h = out.hidden_states[layer + 1]  # (batch, seq_len, d_model)
            result[layer].append(h[:, -1, :].float().cpu())
    return [torch.cat(xs, dim=0) for xs in result]


def load_texts(n_texts, min_chars, seed=42):
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
    X_np = X_np + np.random.randn(*X_np.shape) * 1e-7 * X_np.std()
    est = skdim.id.TwoNN()
    est.fit(X_np)
    return est.dimension_


def subsample_and_estimate(X_np, n_samples, n_runs):
    ids = []
    for run in range(n_runs):
        rng = np.random.RandomState(run)
        idx = rng.choice(len(X_np), size=n_samples, replace=False)
        ids.append(run_twonn(X_np[idx]))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_pool', type=int, default=5000)
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
    n_layers = model.config.num_hidden_layers
    print(f"  n_layers = {n_layers}, d_model = {model.config.hidden_size}")

    min_chars = args.seq_len * 10  # cushion so tokenization yields enough
    print(f"\nLoading texts ({args.n_pool} texts, min {min_chars} chars)...")
    all_texts = load_texts(args.n_pool, min_chars)
    print(f"Collected {len(all_texts)} texts")

    results = {
        'model': args.model, 'seq_len': args.seq_len,
        'n_samples': args.n_samples, 'n_pool': args.n_pool, 'n_runs': args.n_runs,
        'n_layers': n_layers,
    }

    print(f"\n=== Collecting activations across {n_layers} layers (one forward pass per batch) ===")
    acts_per_layer = collect_all_layers(model, tokenizer, device, all_texts, args.seq_len, n_layers)
    print(f"  Got {acts_per_layer[0].shape[0]} activations per layer")

    # Free the model before the (CPU-only) TwoNN loop.
    del model
    torch.cuda.empty_cache()

    print(f"\n=== TWO-NN per layer ({args.n_runs} runs each) ===")
    layer_results = {}
    for layer in range(n_layers):
        X_np = acts_per_layer[layer].numpy()
        ids = subsample_and_estimate(X_np, args.n_samples, args.n_runs)
        mean = float(np.mean(ids))
        sem = float(np.std(ids, ddof=1) / np.sqrt(len(ids)))
        layer_results[str(layer)] = {
            'mean': round(mean, 2),
            'sem': round(sem, 2),
            'runs': [round(x, 2) for x in ids],
        }
        print(f"  layer={layer:>3d}: ID = {mean:.2f} ± {sem:.2f}")
    results['layer_sweep'] = layer_results

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, 'intrinsic_dim_layer_sweep.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = [int(k) for k in layer_results]
    means = [layer_results[k]['mean'] for k in layer_results]
    sems = [layer_results[k]['sem'] for k in layer_results]
    ax.errorbar(layers, means, yerr=sems, fmt='o-', capsize=4, color='C2')
    ax.set_xlabel('Residual stream layer')
    ax.set_ylabel('TWO-NN intrinsic dimension')
    ax.set_title(f'ID vs layer (last token, seq_len={args.seq_len})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plotpath = os.path.join(args.outdir, 'intrinsic_dim_layer_sweep.png')
    plt.savefig(plotpath, dpi=150)
    print(f"Plot saved to {plotpath}")


if __name__ == '__main__':
    main()
