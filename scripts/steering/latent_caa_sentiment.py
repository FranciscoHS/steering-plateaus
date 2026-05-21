#!/usr/bin/env python3
"""Latent-space CAA for sentiment — additive steering.

Smallest-possible first test of on-manifold steering done entirely in the
autoencoder's bottleneck space:

  1. For each (positive, negative) sentiment prompt pair, extract the
     last-token activation at layer l, encode it via the trained AE.
  2. Steering vector = unit-normalized mean difference of encoded
     activations (positive - negative).
  3. At every generation step on a held-out prompt, encode the last-token
     activation, ADD alpha * ||z_ref|| * v_unit, decode, patch back.
     ||z_ref|| is the latent norm of the held-out prompt's last-token
     activation, computed once and held fixed across generation steps.
  4. Sweep alpha in {0.1, 1, 10}, plus a fully unsteered baseline
     (no AE involvement at all). alpha=0 is treated as unsteered (no AE
     roundtrip), so it collapses to the baseline if explicitly passed.

Eyeball the generations to see whether sentiment shifts.

Default config: Qwen3-1.7B, layer 10, the best layer-10 AE
(`l10_doubled_k32_2M_model.pt`, KL=0.137, cos_sim=0.91).

Usage (on RunPod, where the AE weights live):
    python -u scripts/steering/latent_caa_sentiment.py \\
        --ae_weights /workspace/steering-plateaus/results/trained_autoencoder/l10_scaling/l10_doubled_k32_2M_model.pt
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from src.autoencoders import Autoencoder, FlexAutoencoder
from data.sentiment_prompts import PROMPT_PAIRS


# ── AE loader (auto-detect Flex vs vanilla) ──────────────────────────────

def load_autoencoder(path, device):
    print(f"Loading AE from {path}...")
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    config = checkpoint['config']
    is_flex = any(k.startswith('encoder_blocks.') for k in checkpoint['state_dict'])
    if is_flex:
        ae = FlexAutoencoder(
            d_in=config['d_in'], k=config['k'],
            hidden_dims=config.get('hidden_dims', [4096, 4096, 2048]),
            activation=config.get('activation', 'gelu'),
            norm=config.get('norm', 'batchnorm'),
            residual=config.get('residual', False),
            encoder_dims=config.get('encoder_dims'),
            decoder_dims=config.get('decoder_dims'),
        ).to(device)
        arch = 'FlexAE'
    else:
        ae = Autoencoder(config['d_in'], config['k'],
                         config.get('hidden_dims')).to(device)
        arch = 'AE'
    ae.load_state_dict(checkpoint['state_dict'])
    ae.eval()
    print(f"  arch={arch}, d_in={config['d_in']}, k={config['k']}, "
          f"hidden={config.get('hidden_dims')}, "
          f"encoder_dims={config.get('encoder_dims')}, "
          f"decoder_dims={config.get('decoder_dims')}, "
          f"act={config.get('activation')}")
    return ae, config


# ── Activation extraction ────────────────────────────────────────────────

def last_token_act(model, tokenizer, device, prompt, layer):
    """Get last-token residual activation at hidden_states[layer+1]."""
    toks = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, -1, :].float().cpu()


def compute_activation_direction(model, tokenizer, device, layer, prompt_pairs):
    """Mean-difference direction in activation space (d_in-dim), unit-normalized.

    Mirror of `compute_latent_direction` but skips the AE. Used as a baseline
    to check whether the latent-space steering's behaviour can be attributed
    to the AE or to the contrastive direction itself.
    """
    pos_prompts = [p[0] for p in prompt_pairs]
    neg_prompts = [p[1] for p in prompt_pairs]

    def get_acts(prompts):
        return torch.stack(
            [last_token_act(model, tokenizer, device, p, layer) for p in prompts])

    print(f"\nExtracting {len(pos_prompts)} positive + {len(neg_prompts)} "
          f"negative activations at layer {layer}...")
    a_pos = get_acts(pos_prompts)
    a_neg = get_acts(neg_prompts)
    print(f"  a_pos shape={tuple(a_pos.shape)}, mean ||a||={a_pos.norm(dim=1).mean():.3f}")
    print(f"  a_neg shape={tuple(a_neg.shape)}, mean ||a||={a_neg.norm(dim=1).mean():.3f}")

    a_pos_mean = a_pos.mean(dim=0)
    a_neg_mean = a_neg.mean(dim=0)
    diff = a_pos_mean - a_neg_mean  # +alpha shifts toward positive sentiment
    raw_diff_norm = diff.norm().item()
    v_unit = F.normalize(diff, dim=0)

    print(f"  ||a_pos_mean - a_neg_mean|| = {raw_diff_norm:.3f}")
    return v_unit, raw_diff_norm, a_pos_mean, a_neg_mean


def compute_latent_direction(model, tokenizer, device, ae, layer, prompt_pairs):
    """Mean-difference direction in AE latent space, normalized to unit length.

    Returns:
        v_unit:       unit vector in latent space, pointing positive -> negative... wait,
                      we return pos - neg, so steering with +alpha moves toward positive.
        raw_diff_norm: ||mean(z_pos) - mean(z_neg)|| (for logging)
        z_pos_mean, z_neg_mean (for logging)
    """
    pos_prompts = [p[0] for p in prompt_pairs]
    neg_prompts = [p[1] for p in prompt_pairs]

    def encode_all(prompts):
        zs = []
        for p in prompts:
            a = last_token_act(model, tokenizer, device, p, layer).to(device)
            with torch.no_grad():
                z = ae.encode(a.unsqueeze(0)).squeeze(0)
            zs.append(z.cpu())
        return torch.stack(zs)

    print(f"\nEncoding {len(pos_prompts)} positive + {len(neg_prompts)} negative prompts...")
    z_pos = encode_all(pos_prompts)
    z_neg = encode_all(neg_prompts)
    print(f"  z_pos shape={tuple(z_pos.shape)}, mean ||z||={z_pos.norm(dim=1).mean():.3f}")
    print(f"  z_neg shape={tuple(z_neg.shape)}, mean ||z||={z_neg.norm(dim=1).mean():.3f}")

    z_pos_mean = z_pos.mean(dim=0)
    z_neg_mean = z_neg.mean(dim=0)
    diff = z_pos_mean - z_neg_mean  # +alpha shifts toward positive sentiment
    raw_diff_norm = diff.norm().item()
    v_unit = F.normalize(diff, dim=0)

    cos_with_pos = F.cosine_similarity(diff.unsqueeze(0), z_pos_mean.unsqueeze(0)).item()
    cos_with_neg = F.cosine_similarity(diff.unsqueeze(0), z_neg_mean.unsqueeze(0)).item()
    print(f"  ||z_pos_mean - z_neg_mean|| = {raw_diff_norm:.3f}")
    print(f"  cos(diff, z_pos_mean) = {cos_with_pos:+.3f}, "
          f"cos(diff, z_neg_mean) = {cos_with_neg:+.3f}")

    return v_unit, raw_diff_norm, z_pos_mean, z_neg_mean


# ── Generation ───────────────────────────────────────────────────────────

def generate_unsteered(model, tokenizer, prompt, device, n_tokens):
    """Plain greedy generation, no hooks."""
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generated = []
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(tokens, use_cache=False)
        nxt = out.logits[0, -1, :].argmax().item()
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        tokens = torch.cat(
            [tokens, torch.tensor([[nxt]], device=device)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def generate_with_activation_additive(model, tokenizer, prompt, steering_vec,
                                      layer, device, n_tokens):
    """Additive steering directly in activation space (no AE).

    At every generation step, on the last-token activation at layer:
        act' = act + steering_vec
    """
    sv = steering_vec.to(device)  # [d_in]
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generated = []

    for _ in range(n_tokens):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[0, -1, :] = hidden[0, -1, :] + sv.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(tokens, use_cache=False)
        handle.remove()

        nxt = out.logits[0, -1, :].argmax().item()
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        tokens = torch.cat(
            [tokens, torch.tensor([[nxt]], device=device)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True), generated


def generate_with_latent_additive(model, tokenizer, prompt, latent_steering_vec,
                                  layer, device, ae, n_tokens):
    """At every generation step, on the last token only:
        encode -> add latent_steering_vec -> decode -> patch.
    """
    sv = latent_steering_vec.to(device)  # [k]
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generated = []

    for _ in range(n_tokens):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            act = hidden[0, -1, :].float()
            z = ae.encode(act.unsqueeze(0)).squeeze(0)
            z_new = z + sv
            recon = ae.decode(z_new.unsqueeze(0)).squeeze(0)
            hidden = hidden.clone()
            hidden[0, -1, :] = recon.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layers = _get_blocks(model)
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(tokens, use_cache=False)
        handle.remove()

        nxt = out.logits[0, -1, :].argmax().item()
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        tokens = torch.cat(
            [tokens, torch.tensor([[nxt]], device=device)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True), generated


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--space', default='latent', choices=['latent', 'activation'],
                        help='Where to steer. "latent" = encode -> add -> decode '
                             'via the AE bottleneck. "activation" = add directly '
                             'in d_in-dim activation space (skips the AE).')
    parser.add_argument('--ae_weights', default=None,
                        help='Path to AE checkpoint (.pt). Required for --space latent.')
    parser.add_argument('--prompt', default='I thought the movie was',
                        help='Held-out generation prompt')
    parser.add_argument('--alphas', default='0.1,1,10',
                        help='Comma-separated steering magnitudes (in units of '
                             '||z_ref|| for latent, ||a_ref|| for activation). '
                             'alpha=0 is treated as unsteered.')
    parser.add_argument('--n_tokens', type=int, default=40)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--dtype', default='bfloat16',
                        choices=['float32', 'bfloat16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.space == 'latent' and args.ae_weights is None:
        parser.error('--ae_weights is required when --space latent')

    if args.output_dir is None:
        tag = (os.path.splitext(os.path.basename(args.ae_weights))[0]
               if args.space == 'latent' else f'actspace_layer{args.layer}')
        args.output_dir = f'results/latent_caa_sentiment/{args.model}_layer{args.layer}_{tag}'
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    alphas = [float(a) for a in args.alphas.split(',')]
    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16}

    # Load model (+ AE if needed)
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    if args.space == 'latent':
        ae, ae_config = load_autoencoder(args.ae_weights, device)
    else:
        ae, ae_config = None, None

    # Compute CAA direction in the appropriate space
    if args.space == 'latent':
        v_unit, raw_diff_norm, pos_mean, neg_mean = compute_latent_direction(
            model, tokenizer, device, ae, args.layer, PROMPT_PAIRS)
        # Reference norm: encode held-out prompt's last-token activation
        ref_vec = ae.encode(
            last_token_act(model, tokenizer, device, args.prompt, args.layer)
            .unsqueeze(0).to(device)).squeeze(0).cpu()
        ref_label = '||z_ref||'
    else:
        v_unit, raw_diff_norm, pos_mean, neg_mean = compute_activation_direction(
            model, tokenizer, device, args.layer, PROMPT_PAIRS)
        # Reference norm: held-out prompt's last-token activation (no encode)
        ref_vec = last_token_act(
            model, tokenizer, device, args.prompt, args.layer)
        ref_label = '||a_ref||'

    ref_norm = ref_vec.norm().item()
    print(f"\nHeld-out prompt: \"{args.prompt}\"")
    print(f"  {ref_label} = {ref_norm:.3f}")

    # Run
    print(f"\n{'='*70}")
    print(f"Space: {args.space}")
    print(f"Running {len(alphas)} steered generations + 1 unsteered baseline")
    print(f"  alphas = {alphas}")
    print(f"  steering vec norm in {args.space} space = alpha * {ref_norm:.3f}")
    print(f"{'='*70}")

    results = {}
    t0 = time.time()

    # Pure unsteered baseline (no hooks)
    text, ids = generate_unsteered(
        model, tokenizer, args.prompt, device, args.n_tokens)
    results['unsteered'] = {'text': text, 'ids': ids}
    print(f"\n[unsteered]               {text[:200]}")

    # Sweep alpha
    for alpha in alphas:
        if alpha == 0.0:
            text, ids = generate_unsteered(
                model, tokenizer, args.prompt, device, args.n_tokens)
            sv_norm = 0.0
        else:
            sv = alpha * ref_norm * v_unit
            if args.space == 'latent':
                text, ids = generate_with_latent_additive(
                    model, tokenizer, args.prompt, sv,
                    args.layer, device, ae, args.n_tokens)
            else:
                text, ids = generate_with_activation_additive(
                    model, tokenizer, args.prompt, sv,
                    args.layer, device, args.n_tokens)
            sv_norm = sv.norm().item()
        key = f'alpha={alpha:+.2f}'
        results[key] = {'text': text, 'ids': ids, 'sv_norm': sv_norm}
        print(f"\n[{key} | sv_norm={sv_norm:.3f}]  {text[:200]}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # Save text output
    txt_path = os.path.join(args.output_dir, 'generations.txt')
    with open(txt_path, 'w') as f:
        f.write(f"# CAA sentiment steering ({args.space} space) — additive\n")
        f.write(f"Model:     {args.model}\n")
        f.write(f"Layer:     {args.layer}\n")
        f.write(f"Space:     {args.space}\n")
        if args.space == 'latent':
            f.write(f"AE:        {args.ae_weights}\n")
            f.write(f"AE config: k={ae_config['k']}, "
                    f"hidden={ae_config.get('hidden_dims')}, "
                    f"act={ae_config.get('activation')}\n")
        f.write(f"Prompt:    \"{args.prompt}\"\n")
        f.write(f"{ref_label}: {ref_norm:.3f}\n")
        f.write(f"||pos_mean - neg_mean||: {raw_diff_norm:.3f}\n")
        f.write(f"Direction: positive sentiment (pos - neg). +alpha => toward positive.\n")
        f.write(f"n_tokens:  {args.n_tokens}\n")
        f.write(f"\n{'='*70}\n\n")
        f.write(f"[unsteered]\n  {results['unsteered']['text']}\n\n")
        for alpha in alphas:
            key = f'alpha={alpha:+.2f}'
            r = results[key]
            f.write(f"[{key} | sv_norm={r['sv_norm']:.3f}]\n")
            f.write(f"  {r['text']}\n\n")
    print(f"Generations saved: {txt_path}")

    # Save metadata pickle
    pkl_path = os.path.join(args.output_dir, 'results.pkl')
    save = {
        'args': vars(args),
        'ae_config': ae_config,
        'alphas': alphas,
        'ref_norm': ref_norm,
        'ref_label': ref_label,
        'raw_diff_norm': raw_diff_norm,
        'v_unit': v_unit,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'ref_vec': ref_vec,
        'results': results,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(save, f)
    print(f"Metadata saved: {pkl_path}")


if __name__ == '__main__':
    main()
