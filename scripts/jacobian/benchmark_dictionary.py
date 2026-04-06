#!/usr/bin/env python3
"""
Benchmark pruned dictionary vs SAE reconstruction quality.

Compares three methods across varying sparsity levels:
  1. Our MP-based dictionary (on resid_post)
  2. EleutherAI's SAE (on mlp_out) — apples-to-oranges comparison
  3. Our trained SAE (on resid_post) — apples-to-apples comparison

Metrics: relative L2 error, cosine similarity, and KL-divergence of
downstream logits.

Usage:
    python -u scripts/jacobian/benchmark_dictionary.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --dictionary_path results/pruned_dictionary/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --our_sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/sae_weights.pt \
        --output_dir results/benchmark_dictionary/llama-3.2-1b_layer2
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer
from src.data import load_fineweb_fixed_length


# ── Matching pursuit reconstruction ──────────────────────────────────────

def compute_frame_potential(dictionary, chunk_size=1000):
    """Compute frame potential and ratio to tight-frame bound.

    Returns (fp, fp_ratio) where fp_ratio = FP / (n²/d).
    A tight frame has fp_ratio = 1.0.
    """
    n, d = dictionary.shape
    device = dictionary.device
    fp_sum = 0.0
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        gram_chunk = dictionary[i:end] @ dictionary.T
        fp_sum += (gram_chunk ** 2).sum().item()
    fp_tight = n ** 2 / d
    return fp_sum, fp_sum / fp_tight


def matching_pursuit_reconstruct(dictionary, activations, n_nonzero_coefs):
    """Reconstruct activations via matching pursuit with n atoms.

    Args:
        dictionary: (n_atoms, d_model) on GPU, unit-normalized
        activations: (n_val, d_model) on GPU
        n_nonzero_coefs: number of atoms to select per activation

    Returns:
        recon: (n_val, d_model) reconstructed activations
    """
    n_val = activations.shape[0]
    device = activations.device
    residuals = activations.clone()
    recon = torch.zeros_like(activations)

    for step in range(n_nonzero_coefs):
        corr = residuals @ dictionary.T  # (n_val, n_atoms)
        best = corr.abs().argmax(dim=1)  # (n_val,)
        coefs = corr[torch.arange(n_val, device=device), best]
        contribution = coefs.unsqueeze(1) * dictionary[best]
        recon = recon + contribution
        residuals = residuals - contribution

    return recon


# ── SAE top-n reconstruction ────────────────────────────────────────────

def sae_reconstruct_topn(sae, activations, n, batch_size=256, sae_type='sparsify'):
    """Reconstruct activations using SAE with top-n features.

    Args:
        sae: SAE object (sparsify Sae or our TopKSAE)
        activations: (n_val, d_model) on GPU
        n: number of top features to keep
        batch_size: process in chunks to manage memory
        sae_type: 'sparsify' or 'ours' — controls how to get pre_acts

    Returns:
        recon: (n_val, d_model) reconstructed activations
    """
    n_val = activations.shape[0]
    recon_chunks = []

    for i in range(0, n_val, batch_size):
        batch = activations[i:i+batch_size]
        enc_out = sae.encode(batch)

        if sae_type == 'sparsify':
            pre_acts = enc_out.pre_acts  # namedtuple
        else:
            pre_acts = enc_out[2]  # plain tuple: (top_vals, top_idx, pre_acts)

        topn_vals, topn_idx = pre_acts.topk(n, dim=-1)
        recon_batch = sae.decode(topn_vals, topn_idx)
        recon_chunks.append(recon_batch)

    return torch.cat(recon_chunks, dim=0)


# ── Load our trained TopK SAE ───────────────────────────────────────────

def load_our_sae(path, device):
    """Load a TopKSAE from a saved weights file."""
    from scripts.jacobian.train_sae import TopKSAE

    data = torch.load(path, map_location=device, weights_only=False)
    config = data['config']
    sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
    sae.load_state_dict(data['state_dict'])
    sae.to(device)
    sae.eval()
    print(f"  Our SAE: d_in={config['d_in']}, d_sae={config['d_sae']}, k={config['k']}")
    return sae


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_reconstruction_metrics(original, recon):
    """Compute relative L2 error and cosine similarity.

    Returns:
        rel_l2: mean relative L2 error (scalar)
        cos_sim: mean cosine similarity (scalar)
    """
    # Relative L2: ||orig - recon|| / ||orig||
    errors = (original - recon).norm(dim=-1)
    norms = original.norm(dim=-1)
    rel_l2 = (errors / (norms + 1e-8)).mean().item()

    cos_sim = F.cosine_similarity(original, recon, dim=-1).mean().item()

    return rel_l2, cos_sim


def compute_kl_div_batched(model, contexts, patched_points, clean_logits,
                           start_layer, batch_size=64):
    """Compute KL(clean || patched) for each activation, batched.

    Args:
        model: loaded model
        contexts: (n_val, seq_len-1, d_model) prefix hidden states at start_layer
        patched_points: (n_val, d_model) patched last-token activations
        clean_logits: (n_val, vocab_size) reference logits
        start_layer: layer index for forward_from_layer
        batch_size: forward pass batch size

    Returns:
        mean_kl: scalar, mean KL-divergence
    """
    n_val = patched_points.shape[0]
    device = patched_points.device
    all_kl = []

    clean_log_probs = F.log_softmax(clean_logits, dim=-1)

    for i in range(0, n_val, batch_size):
        ctx = contexts[i:i+batch_size]  # (B, seq-1, d)
        pts = patched_points[i:i+batch_size].unsqueeze(1)  # (B, 1, d)

        # Each activation has its own context, so we can't share context
        # across the batch. Process individually.
        batch_logits = []
        for j in range(ctx.shape[0]):
            with torch.no_grad():
                logits_j = forward_from_layer(
                    model, ctx[j:j+1], pts[j:j+1], start_layer
                )  # (1, vocab_size)
            batch_logits.append(logits_j)
        patched_logits = torch.cat(batch_logits, dim=0)  # (B, vocab)

        patched_log_probs = F.log_softmax(patched_logits, dim=-1)
        clean_batch = clean_log_probs[i:i+batch_size].to(device)

        # KL(clean || patched) = sum clean_prob * (log clean_prob - log patched_prob)
        kl = F.kl_div(patched_log_probs, clean_batch.exp(), reduction='none').sum(dim=-1)
        all_kl.append(kl.cpu())

    return torch.cat(all_kl).mean().item()


# ── Extraction ───────────────────────────────────────────────────────────

def extract_activations(model, token_ids_list, layer, device, eval_start, eval_end,
                        batch_size=64):
    """Extract resid_post, mlp_out, clean logits, and contexts for eval set.

    Returns dict with keys: resid_post, mlp_out, clean_logits, contexts
    All tensors are float32 on CPU.
    """
    eval_tokens = token_ids_list[eval_start:eval_end]
    n_eval = len(eval_tokens)

    all_resid = []
    all_mlp = []
    all_logits = []
    all_contexts = []

    # Hook to capture MLP output
    blocks = None
    # Find the right module path for the MLP
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        mlp_module = model.model.layers[layer].mlp
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        mlp_module = model.transformer.h[layer].mlp
    else:
        raise RuntimeError(f"Cannot find MLP module for layer {layer}")

    print(f"Extracting {n_eval} activations (resid_post, mlp_out, logits, contexts)...")
    for i in range(0, n_eval, batch_size):
        batch = torch.stack(eval_tokens[i:i+batch_size]).to(device)

        # Set up MLP hook
        mlp_output = {}
        def mlp_hook(module, input, output):
            # Llama MLP returns a tensor directly
            out = output[0] if isinstance(output, tuple) else output
            mlp_output['out'] = out.detach()

        handle = mlp_module.register_forward_hook(mlp_hook)

        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)

        handle.remove()

        # resid_post at layer: hidden_states[layer+1]
        hidden = outputs.hidden_states[layer + 1]  # (B, seq, d)
        all_resid.append(hidden[:, -1, :].float().cpu())
        all_contexts.append(hidden[:, :-1, :].float().cpu())

        # mlp_out at layer
        all_mlp.append(mlp_output['out'][:, -1, :].float().cpu())

        # Clean logits
        all_logits.append(outputs.logits[:, -1, :].float().cpu())

        done = min(i + batch_size, n_eval)
        if (i // batch_size) % 5 == 0:
            print(f"  {done}/{n_eval}", end='\r')

    print()

    result = {
        'resid_post': torch.cat(all_resid, dim=0),
        'mlp_out': torch.cat(all_mlp, dim=0),
        'clean_logits': torch.cat(all_logits, dim=0),
        'contexts': torch.cat(all_contexts, dim=0),
    }
    print(f"  resid_post: {result['resid_post'].shape}")
    print(f"  mlp_out: {result['mlp_out'].shape}")
    print(f"  clean_logits: {result['clean_logits'].shape}")
    print(f"  contexts: {result['contexts'].shape}")

    return result


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plot(results, output_path):
    """3-panel figure: relative L2, cosine sim, KL-div vs n."""
    n_values = results['n_values']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('rel_l2', 'Relative L2 Error', True),
        ('cos_sim', 'Cosine Similarity', False),
        ('kl_div', 'KL Divergence', True),
    ]

    # Methods to plot: (results_key, label, marker, linestyle, color)
    # Only include methods that have data
    all_methods = [
        ('dictionary', 'Dictionary (MP)', 'o', '-', 'C0'),
        ('our_sae', 'SAE 8x (resid)', '^', '-.', 'C2'),
        ('sae', 'EleutherAI SAE (mlp)', 's', '--', 'C1'),
        ('random', 'Random (MP)', 'x', ':', 'C3'),
    ]
    methods = [(k, l, m, ls, c) for k, l, m, ls, c in all_methods
               if k in results and results[k]]

    for ax, (key, title, use_log) in zip(axes, metrics):
        for method_key, label, marker, ls, color in methods:
            vals = [results[method_key][n][key] for n in n_values]
            if vals[0] is None:
                continue
            ax.plot(n_values, vals, marker=marker, linestyle=ls,
                    label=label, color=color)

        ax.set_xlabel('n (active features)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if use_log:
            ax.set_yscale('log')

    fig.suptitle('Reconstruction Quality Comparison', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark dictionary vs SAE')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--dictionary_path', required=True,
                        help='Path to pruned_dictionary.pkl')
    parser.add_argument('--sae_repo', default='EleutherAI/sae-Llama-3.2-1B-131k')
    parser.add_argument('--our_sae_path', default=None,
                        help='Path to our trained SAE weights (sae_weights.pt)')
    parser.add_argument('--skip_eleuther', action='store_true',
                        help='Skip EleutherAI SAE (mlp_out) benchmark')
    parser.add_argument('--add_random', action='store_true',
                        help='Add random dictionary baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=8000)
    parser.add_argument('--eval_start', type=int, default=7000)
    parser.add_argument('--eval_end', type=int, default=8000)
    parser.add_argument('--n_values', default='1,2,5,10,15,20,30,50',
                        help='Comma-separated list of sparsity levels')
    parser.add_argument('--output_dir',
                        default='results/benchmark_dictionary/llama-3.2-1b_layer2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--skip_kl', action='store_true',
                        help='Skip KL-divergence computation (faster)')
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print("Benchmark: Dictionary vs SAEs")
    print(f"  model: {args.model}, layer: {args.layer}")
    print(f"  dictionary: {args.dictionary_path}")
    if not args.skip_eleuther:
        print(f"  EleutherAI SAE: {args.sae_repo}")
    print(f"  Our SAE: {args.our_sae_path or 'not provided'}")
    print(f"  Random baseline: {args.add_random}")
    print(f"  n values: {n_values}")
    print(f"  eval set: [{args.eval_start}, {args.eval_end})")
    print()

    # ── Load dictionary ──────────────────────────────────────────────────
    print("Loading dictionary...")
    with open(args.dictionary_path, 'rb') as f:
        dict_data = pickle.load(f)
    dictionary = dict_data['dictionary']  # (n_atoms, d_model)
    # Normalize (should already be unit-norm, but be safe)
    dictionary = F.normalize(dictionary, dim=-1)
    fp, fp_ratio = compute_frame_potential(dictionary)
    print(f"  Dictionary: {dictionary.shape}")
    print(f"  Frame potential: {fp:.1f} (ratio to tight frame: {fp_ratio:.4f})")

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer, device = load_model(args.model)

    # ── Load EleutherAI SAE ──────────────────────────────────────────────
    sae = None
    if not args.skip_eleuther:
        print(f"\nLoading EleutherAI SAE from {args.sae_repo}...")
        from sparsify import Sae
        sae = Sae.load_from_hub(args.sae_repo,
                                hookpoint=f"layers.{args.layer}.mlp",
                                device=device)
        print(f"  SAE latents: {sae.encoder.weight.shape[0]}, "
              f"d_in: {sae.encoder.weight.shape[1]}")

    # ── Load our SAE (optional) ──────────────────────────────────────────
    our_sae = None
    if args.our_sae_path:
        print(f"\nLoading our SAE from {args.our_sae_path}...")
        our_sae = load_our_sae(args.our_sae_path, device)

    # ── Create random dictionary baseline ────────────────────────────────
    random_dictionary = None
    if args.add_random:
        d_model = dictionary.shape[1]
        n_atoms = dictionary.shape[0]
        print(f"\nGenerating random dictionary ({n_atoms} x {d_model})...")
        torch.manual_seed(args.seed + 999)
        random_dictionary = torch.randn(n_atoms, d_model, device='cpu')
        random_dictionary = F.normalize(random_dictionary, dim=-1)

    # ── Extract eval activations ─────────────────────────────────────────
    print(f"\nLoading {args.n_total} FineWeb samples...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed
    )

    data = extract_activations(
        model, token_ids_list, args.layer, device,
        args.eval_start, args.eval_end, args.batch_size
    )

    # Move to GPU for benchmarking
    resid_post = data['resid_post'].to(device)
    mlp_out = data['mlp_out'].to(device)
    clean_logits = data['clean_logits']  # keep on CPU, moved per-batch in KL
    contexts = data['contexts']  # keep on CPU, moved per-batch in KL
    dictionary = dictionary.to(device)
    if random_dictionary is not None:
        random_dictionary = random_dictionary.to(device)

    # ── Benchmark loop ───────────────────────────────────────────────────
    results = {
        'n_values': n_values,
        'dictionary': {},
        'dictionary_fp': fp,
        'dictionary_fp_ratio': fp_ratio,
        'sae': {},
        'our_sae': {},
        'random': {},
        'args': vars(args),
    }

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"n = {n}")
        print(f"{'='*60}")

        # ── Dictionary (MP on resid_post) ────────────────────────────────
        t0 = time.time()
        dict_recon = matching_pursuit_reconstruct(dictionary, resid_post, n)
        dict_time = time.time() - t0

        dict_rel_l2, dict_cos = compute_reconstruction_metrics(resid_post, dict_recon)
        print(f"  Dictionary:    rel_L2={dict_rel_l2:.4f}, cos_sim={dict_cos:.4f} "
              f"({dict_time:.1f}s)")

        results['dictionary'][n] = {
            'rel_l2': dict_rel_l2,
            'cos_sim': dict_cos,
            'time_s': dict_time,
        }

        # ── EleutherAI SAE (top-n on mlp_out) ───────────────────────────
        sae_recon = None
        if sae is not None:
            t0 = time.time()
            sae_recon = sae_reconstruct_topn(sae, mlp_out, n, sae_type='sparsify')
            sae_time = time.time() - t0

            sae_rel_l2, sae_cos = compute_reconstruction_metrics(mlp_out, sae_recon)
            print(f"  EleutherAI:    rel_L2={sae_rel_l2:.4f}, cos_sim={sae_cos:.4f} "
                  f"({sae_time:.1f}s)")

            results['sae'][n] = {
                'rel_l2': sae_rel_l2,
                'cos_sim': sae_cos,
                'time_s': sae_time,
            }

        # ── Our SAE (top-n on resid_post) ────────────────────────────────
        our_recon = None
        if our_sae is not None:
            t0 = time.time()
            our_recon = sae_reconstruct_topn(our_sae, resid_post, n, sae_type='ours')
            our_time = time.time() - t0

            our_rel_l2, our_cos = compute_reconstruction_metrics(resid_post, our_recon)
            print(f"  Our SAE:       rel_L2={our_rel_l2:.4f}, cos_sim={our_cos:.4f} "
                  f"({our_time:.1f}s)")

            results['our_sae'][n] = {
                'rel_l2': our_rel_l2,
                'cos_sim': our_cos,
                'time_s': our_time,
            }

        # ── Random dictionary (MP on resid_post) ────────────────────────
        rand_recon = None
        if random_dictionary is not None:
            t0 = time.time()
            rand_recon = matching_pursuit_reconstruct(random_dictionary, resid_post, n)
            rand_time = time.time() - t0

            rand_rel_l2, rand_cos = compute_reconstruction_metrics(resid_post, rand_recon)
            print(f"  Random:        rel_L2={rand_rel_l2:.4f}, cos_sim={rand_cos:.4f} "
                  f"({rand_time:.1f}s)")

            results['random'][n] = {
                'rel_l2': rand_rel_l2,
                'cos_sim': rand_cos,
                'time_s': rand_time,
            }

        # ── KL-divergence ───────────────────────────────────────────────
        if not args.skip_kl:
            # Dictionary
            t0 = time.time()
            dict_kl = compute_kl_div_batched(
                model, contexts.to(device), dict_recon,
                clean_logits, args.layer
            )
            dict_kl_time = time.time() - t0
            print(f"  Dict KL:       {dict_kl:.6f} ({dict_kl_time:.1f}s)")
            results['dictionary'][n]['kl_div'] = dict_kl

            # EleutherAI SAE
            if sae is not None and sae_recon is not None:
                sae_patched = resid_post + (sae_recon - mlp_out)
                t0 = time.time()
                sae_kl = compute_kl_div_batched(
                    model, contexts.to(device), sae_patched,
                    clean_logits, args.layer
                )
                sae_kl_time = time.time() - t0
                print(f"  Eleuther KL:   {sae_kl:.6f} ({sae_kl_time:.1f}s)")
                results['sae'][n]['kl_div'] = sae_kl

            # Our SAE
            if our_recon is not None:
                t0 = time.time()
                our_kl = compute_kl_div_batched(
                    model, contexts.to(device), our_recon,
                    clean_logits, args.layer
                )
                our_kl_time = time.time() - t0
                print(f"  Our SAE KL:    {our_kl:.6f} ({our_kl_time:.1f}s)")
                results['our_sae'][n]['kl_div'] = our_kl

            # Random
            if rand_recon is not None:
                t0 = time.time()
                rand_kl = compute_kl_div_batched(
                    model, contexts.to(device), rand_recon,
                    clean_logits, args.layer
                )
                rand_kl_time = time.time() - t0
                print(f"  Random KL:     {rand_kl:.6f} ({rand_kl_time:.1f}s)")
                results['random'][n]['kl_div'] = rand_kl
        else:
            results['dictionary'][n]['kl_div'] = None
            if sae is not None:
                results['sae'][n]['kl_div'] = None
            if our_sae is not None:
                results['our_sae'][n]['kl_div'] = None
            if random_dictionary is not None:
                results['random'][n]['kl_div'] = None

    # ── Save results ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "benchmark_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "benchmark_plot.png")
    make_plot(results, plot_path)


if __name__ == '__main__':
    main()
