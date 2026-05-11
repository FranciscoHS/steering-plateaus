#!/usr/bin/env python3
"""
Evaluate reconstruction quality via CE loss (KL divergence).

Loads a trained autoencoder or SAE, reconstructs held-out activations at a
given layer, splices them back into the model, and compares output logits
against the unmodified model.

Reports: KL divergence, CE loss increase, cosine similarity, MSE.

Usage:
    python -u scripts/autoencoder/eval_ce_loss.py \
        --model Qwen/Qwen3-1.7B \
        --layer 2 \
        --ae_path results/.../best_model.pt \
        --ae_type autoencoder \
        --n_eval 1000 \
        --eval_start 600000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_model(model_name):
    """Load HuggingFace model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    print(f"  Loaded on {device}, dtype={model.dtype}")
    return model, tokenizer, device


def get_eval_data(tokenizer, n_eval, eval_start, seq_len, seed):
    """Load FineWeb samples for evaluation (held-out from training)."""
    from datasets import load_dataset

    print(f"Loading FineWeb eval data (streaming): starting at {eval_start}, need {n_eval}")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    if eval_start > 0:
        ds = ds.skip(eval_start)

    tokens_list = []
    for i, sample in enumerate(ds):
        toks = tokenizer.encode(sample['text'], add_special_tokens=False)[:seq_len]
        if len(toks) == seq_len:
            tokens_list.append(toks)
        if len(tokens_list) >= n_eval:
            break
        if i > n_eval * 3:
            break  # safety limit

    tokens = torch.tensor(tokens_list)
    print(f"  Got {tokens.shape[0]} sequences of length {tokens.shape[1]}")
    return tokens


def load_autoencoder(path, device):
    """Load a trained autoencoder from checkpoint."""
    data = torch.load(path, weights_only=False, map_location=device)

    # Handle different save formats
    if 'config' in data:
        config = data['config']
    else:
        config = data.get('args', {})

    state_dict = data['state_dict']

    # Detect if it's a FiLM autoencoder
    is_film = any('film' in k for k in state_dict.keys())

    if is_film:
        from scripts.autoencoder.hillclimb_autoencoder import FiLMAutoencoder
        ae = FiLMAutoencoder(
            d_in=config['d_in'], k=config['k'],
            hidden_dims=config['hidden_dims'],
            activation=config.get('activation', 'gelu'),
            norm=config.get('norm', 'batchnorm'),
        )
    else:
        from src.autoencoders import FlexAutoencoder
        ae = FlexAutoencoder(
            d_in=config['d_in'], k=config['k'],
            hidden_dims=config['hidden_dims'],
            activation=config.get('activation', 'gelu'),
            norm=config.get('norm', 'batchnorm'),
            dropout=config.get('dropout', 0.0),
        )

    ae.load_state_dict(state_dict)
    ae = ae.to(device).eval()
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"  Loaded autoencoder: {n_params/1e6:.1f}M params, k={config['k']}")
    return ae


def load_sae(path, device):
    """Load a trained SAE from checkpoint."""
    from src.autoencoders import TopKSAE

    data = torch.load(path, weights_only=False, map_location=device)
    config = data['config']
    sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
    sae.load_state_dict(data['state_dict'])
    sae = sae.to(device).eval()
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"  Loaded SAE: {n_params/1e6:.1f}M params, d_sae={config['d_sae']}, k={config['k']}")
    return sae


def evaluate(model, tokenizer, device, reconstruct_fn, tokens, layer, batch_size=32):
    """Evaluate reconstruction quality via CE loss and other metrics.

    Args:
        model: HuggingFace causal LM
        reconstruct_fn: callable(activations) -> reconstructed activations
        tokens: (n_eval, seq_len) input token ids
        layer: which layer to splice at
        batch_size: eval batch size

    Returns:
        dict with metrics
    """
    all_kl = []
    all_ce_orig = []
    all_ce_recon = []
    all_cos = []
    all_mse = []

    n_eval = tokens.shape[0]

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch = tokens[start:end].to(device)

        # Original forward pass — get logits and layer activations
        with torch.no_grad():
            outputs_orig = model(batch, output_hidden_states=True)
            logits_orig = outputs_orig.logits[:, -1, :]  # last token logits
            # hidden_states[0] = embeddings, [1] = after layer 0, [layer+1] = after target layer
            act_orig = outputs_orig.hidden_states[layer + 1][:, -1, :]  # (batch, d_model)

        # Reconstruct
        with torch.no_grad():
            act_recon = reconstruct_fn(act_orig)

        # Cosine and MSE on activations
        cos = F.cosine_similarity(act_orig, act_recon, dim=-1)
        mse = (act_orig - act_recon).pow(2).mean(dim=-1)
        all_cos.append(cos.cpu())
        all_mse.append(mse.cpu())

        # Forward pass with reconstructed activations spliced in
        # We need to hook the model to replace the activation at the target layer
        hook_handle = None
        replaced = [False]

        def splice_hook(module, input, output):
            if replaced[0]:
                return output
            replaced[0] = True
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[:, -1, :] = act_recon
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                hs[:, -1, :] = act_recon
                return hs

        # Register hook on the target layer
        target_layer = model.model.layers[layer]
        hook_handle = target_layer.register_forward_hook(splice_hook)

        with torch.no_grad():
            outputs_recon = model(batch, output_hidden_states=False)
            logits_recon = outputs_recon.logits[:, -1, :]

        hook_handle.remove()

        # Compute KL divergence: KL(p_orig || p_recon)
        log_p_orig = F.log_softmax(logits_orig, dim=-1)
        log_p_recon = F.log_softmax(logits_recon, dim=-1)
        p_orig = F.softmax(logits_orig, dim=-1)

        kl = F.kl_div(log_p_recon, p_orig, reduction='none').sum(dim=-1)  # (batch,)
        all_kl.append(kl.cpu())

        # CE losses
        # We don't have labels, but we can compute CE of the original distribution
        # under the reconstructed model (cross-entropy of p_orig w.r.t. p_recon)
        ce_orig = -(p_orig * log_p_orig).sum(dim=-1)  # entropy of original
        ce_recon = -(p_orig * log_p_recon).sum(dim=-1)  # CE of orig under recon
        all_ce_orig.append(ce_orig.cpu())
        all_ce_recon.append(ce_recon.cpu())

        done = min(end, n_eval)
        if (done // batch_size) % 10 == 0 or done == n_eval:
            print(f"  Eval {done}/{n_eval}: "
                  f"KL={torch.cat(all_kl).mean():.6f}, "
                  f"cos={torch.cat(all_cos).mean():.4f}")

    kl = torch.cat(all_kl)
    cos = torch.cat(all_cos)
    mse = torch.cat(all_mse)
    ce_orig = torch.cat(all_ce_orig)
    ce_recon = torch.cat(all_ce_recon)

    results = {
        'kl_div_mean': kl.mean().item(),
        'kl_div_std': kl.std().item(),
        'kl_div_median': kl.median().item(),
        'ce_increase_mean': (ce_recon - ce_orig).mean().item(),
        'cos_sim_mean': cos.mean().item(),
        'cos_sim_std': cos.std().item(),
        'mse_mean': mse.mean().item(),
        'n_eval': n_eval,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction via CE loss')
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--model_path', required=True,
                        help='Path to trained AE or SAE weights')
    parser.add_argument('--model_type', choices=['autoencoder', 'sae'], required=True)
    parser.add_argument('--n_eval', type=int, default=1000)
    parser.add_argument('--eval_start', type=int, default=0,
                        help='FineWeb index offset for held-out data (training used [8000, 508000))')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', default=None,
                        help='Where to save results JSON')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Load eval data
    tokens = get_eval_data(tokenizer, args.n_eval, args.eval_start,
                           args.seq_len, args.seed)

    # Load reconstruction model
    print(f"\nLoading {args.model_type} from {args.model_path}")
    if args.model_type == 'autoencoder':
        ae = load_autoencoder(args.model_path, device)
        def reconstruct_fn(act):
            recon, _ = ae(act)
            return recon
    elif args.model_type == 'sae':
        sae = load_sae(args.model_path, device)
        def reconstruct_fn(act):
            recon, _, _ = sae(act)
            return recon

    # Evaluate
    print(f"\nEvaluating on {tokens.shape[0]} held-out samples...")
    t0 = time.time()
    results = evaluate(model, tokenizer, device, reconstruct_fn,
                       tokens, args.layer, args.batch_size)
    elapsed = time.time() - t0

    results['model_path'] = args.model_path
    results['model_type'] = args.model_type
    results['eval_time'] = elapsed

    print(f"\n{'='*60}")
    print(f"Results for {args.model_type}: {args.model_path}")
    print(f"{'='*60}")
    print(f"  KL divergence:  {results['kl_div_mean']:.6f} ± {results['kl_div_std']:.6f}")
    print(f"  KL median:      {results['kl_div_median']:.6f}")
    print(f"  CE increase:    {results['ce_increase_mean']:.6f}")
    print(f"  Cosine sim:     {results['cos_sim_mean']:.4f} ± {results['cos_sim_std']:.4f}")
    print(f"  MSE:            {results['mse_mean']:.6f}")
    print(f"  Eval time:      {elapsed:.0f}s")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_path}")


if __name__ == '__main__':
    main()
