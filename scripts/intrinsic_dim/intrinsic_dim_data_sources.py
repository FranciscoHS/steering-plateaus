"""TWO-NN intrinsic dimensionality across data sources.

Fixes layer, seq_len, last-token position. Compares ID across:
  - FineWeb-Edu (baseline)
  - Wikipedia: en, fr, zh, ar
  - Python code (the-stack-smol)
  - arXiv (RedPajama-1T-Sample)

Each source is processed with the same protocol (n_pool texts → tokenize at
seq_len → last-token activation at the chosen layer → n_runs subsampled TwoNN
estimates of size n_samples).
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import skdim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loaders + SOURCE_LOADERS now live in src.data_sources so probe scripts
# can share them without reaching into scripts/. Re-exported here for
# backward-compat with any caller that still imports from this module.
from src.data_sources import (
    SOURCE_LOADERS,
    load_fineweb,
    load_wikipedia,
    load_code_python,
    load_arxiv,
    _stream_filter_text,
)


# ── Activations ─────────────────────────────────────────────────────────

def collect_layer(model, tokenizer, device, texts, seq_len, layer):
    """One forward pass per batch; return (n_samples, d_model) tensor of last-token
    activations at the chosen layer."""
    out_chunks = []
    batch_size = min(128, max(32, 4096 // seq_len))
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors='pt', padding='max_length', truncation=True,
            max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        h = out.hidden_states[layer + 1]  # (batch, seq_len, d_model)
        out_chunks.append(h[:, -1, :].float().cpu())
    return torch.cat(out_chunks, dim=0)


# ── TwoNN ───────────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_pool', type=int, default=5000)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sources', nargs='+',
                        default=list(SOURCE_LOADERS.keys()),
                        choices=list(SOURCE_LOADERS.keys()))
    parser.add_argument('--outdir', default='results/jacobian/id_data_sources')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                 'float32': torch.float32}

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
    print(f"  n_layers = {n_layers}, d_model = {model.config.hidden_size}, "
          f"using layer = {args.layer}")

    min_chars = args.seq_len * 10

    results = {
        'model': args.model, 'layer': args.layer, 'seq_len': args.seq_len,
        'n_samples': args.n_samples, 'n_pool': args.n_pool, 'n_runs': args.n_runs,
        'sources': {},
    }

    outpath = os.path.join(args.outdir, f'id_data_sources_layer{args.layer}.json')

    # Per source: load → activations → TwoNN → save partial JSON.
    # That way a later failure preserves earlier sources' results.
    for name in args.sources:
        print(f"\n=== {name}: loading {args.n_pool} texts (min {min_chars} chars) ===")
        try:
            texts = SOURCE_LOADERS[name](args.n_pool, min_chars, args.seed)
        except Exception as e:
            print(f"  FAILED to load {name}: {e!r}")
            results['sources'][name] = {'error': repr(e)}
            with open(outpath, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        print(f"  got {len(texts)} texts")
        acts = collect_layer(model, tokenizer, device, texts, args.seq_len,
                             args.layer)
        print(f"  activations: {tuple(acts.shape)}")
        X_np = acts.numpy()
        n_eff = min(args.n_samples, len(X_np))
        ids = subsample_and_estimate(X_np, n_eff, args.n_runs)
        mean = float(np.mean(ids))
        sem = float(np.std(ids, ddof=1) / np.sqrt(len(ids)))
        results['sources'][name] = {
            'mean': round(mean, 2),
            'sem': round(sem, 2),
            'runs': [round(x, 2) for x in ids],
            'n_samples_used': n_eff,
            'n_pool_actual': len(X_np),
        }
        print(f"  {name:>12s}: ID = {mean:.2f} ± {sem:.2f}  (n={n_eff})")
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  saved partial: {outpath}")

    del model
    torch.cuda.empty_cache()
    print(f"\nFinal results: {outpath}")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [n for n, v in results['sources'].items() if 'mean' in v]
    means = [results['sources'][n]['mean'] for n in names]
    sems = [results['sources'][n]['sem'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sems, capsize=4, color='C2', alpha=0.85)
    for i, m in enumerate(means):
        ax.text(i, m + sems[i] + 0.2, f'{m:.2f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('TWO-NN intrinsic dimension')
    ax.set_title(f'ID across data sources (layer {args.layer}, '
                 f'seq_len={args.seq_len}, last token)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plotpath = os.path.join(args.outdir, f'id_data_sources_layer{args.layer}.png')
    plt.savefig(plotpath, dpi=150)
    print(f"Plot saved: {plotpath}")


if __name__ == '__main__':
    main()
