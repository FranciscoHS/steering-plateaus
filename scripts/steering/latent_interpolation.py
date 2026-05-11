#!/usr/bin/env python3
"""Latent-space interpolation between matched EN/ZH prompt activations.

For a matched prompt pair (EN, ZH):
  1. Get layer-2 activations for both
  2. Encode both to 30-dim latent space
  3. Interpolate between z_en and z_zh in steps of 10%
  4. Decode each interpolated z back to activation space
  5. Feed through remaining layers, get logits
  6. Measure relative distance d = ||x-A|| / (||x-A|| + ||x-B||)
     where x = output at interpolation point, A = output at EN end, B = output at ZH end
  7. Also generate text at each step to see qualitative behavior

Usage:
    python -u scripts/steering/latent_interpolation.py \
        --ae_weights results/on_manifold_steering/qwen3-1.7b_layer2/ae_weights.pt

    # Specific prompt pair index:
    python -u scripts/steering/latent_interpolation.py \
        --ae_weights results/on_manifold_steering/qwen3-1.7b_layer2/ae_weights.pt \
        --pair_idx 0
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, _get_final_norm, _get_lm_head, \
    _is_rope_model, _get_position_embeddings, _run_block
PROMPT_SETS = {
    'language': 'data.bilingual_prompts_hq',
    'gender': 'data.gender_prompts',
}


# AE/SAE classes moved to src/autoencoders.py for shared use across
# probes/, steering/, and the training scripts. Re-export here for
# any external importer that may still reference them on this module.
from src.autoencoders import TopKSAE, Autoencoder, FlexAutoencoder


def slerp(a, b, t):
    """Spherical linear interpolation between vectors a and b at fraction t."""
    a_norm = a / (a.norm() + 1e-10)
    b_norm = b / (b.norm() + 1e-10)
    omega = torch.acos(torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0))
    if omega.abs() < 1e-6:
        # Nearly parallel — fall back to lerp
        return (1 - t) * a + t * b
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * a + \
           (torch.sin(t * omega) / sin_omega) * b


# ── Helpers ──────────────────────────────────────────────────────────────

def get_activation_and_context(model, tokenizer, prompt, layer, device):
    """Get last-token activation and prefix context at given layer."""
    toks = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True)
    hidden = out.hidden_states[layer + 1]  # [1, seq_len, d]
    context = hidden[:, :-1, :]  # [1, seq_len-1, d]
    last_tok = hidden[:, -1, :]  # [1, d]
    return last_tok, context


def forward_to_logits(model, context, point, start_layer):
    """Forward from layer through remaining blocks to logits.

    Args:
        context: [1, seq_len-1, d]
        point: [1, d] or [d] last-token activation
        start_layer: layer index

    Returns:
        logits: [vocab_size]
    """
    if point.dim() == 1:
        point = point.unsqueeze(0)  # [d] -> [1, d]
    hidden = torch.cat([context, point.unsqueeze(1)], dim=1)
    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    hidden = _get_final_norm(model)(hidden)
    logits = _get_lm_head(model)(hidden[:, -1, :])
    return logits.squeeze(0)


def generate_from_activation(model, tokenizer, context, activation,
                             start_layer, device, n_tokens=20):
    """Generate tokens starting from a specific activation at layer.

    Replaces the last-token activation at start_layer with the given one,
    then autoregressively generates using hooks.
    """
    # Get the prompt tokens from the context shape
    # We need to reconstruct input_ids — use a dummy forward to get the first token
    logits = forward_to_logits(model, context, activation, start_layer)
    next_id = logits.argmax().item()
    generated_ids = [next_id]

    # For subsequent tokens, we need full forward passes with hooks
    # Build initial input_ids from context length + 1
    seq_len = context.shape[1] + 1
    # We can't easily recover input_ids, so just generate from the logits
    # by doing full forward passes with activation patching

    # Actually, we need input_ids to continue generation. Let's use the hook approach.
    # But we don't have input_ids... Let's just return the first-token prediction
    # and the logits for the relative distance analysis.

    # For generation: do full forward with hooks
    return generated_ids, logits


def cjk_fraction(text):
    if not text:
        return 0.0
    n_cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    n_total = sum(1 for ch in text
                  if not ch.isspace() and ch not in '.,;:!?()[]{}"\'-/')
    return n_cjk / max(n_total, 1)


def generate_with_patched_activation(model, tokenizer, prompt, patched_act,
                                     layer, device, n_tokens=20):
    """Generate by patching the last-token activation at layer on every step.

    On the first step, patches the prompt's last token.
    On subsequent steps, patches the newly generated last token.
    """
    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    generated_ids = []
    patched = patched_act.to(device)

    for step in range(n_tokens):
        def patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            if step == 0:
                # Patch the last prompt token
                hidden[0, -1, :] = patched
            # For subsequent steps, don't patch — let the model generate freely
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        # Only patch on the first step
        if step == 0:
            layers = _get_blocks(model)
            handle = layers[layer].register_forward_hook(patch_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        if step == 0:
            handle.remove()

        next_id = outputs.logits[0, -1, :].argmax().item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--ae_weights', default=None,
                        help='Path to deep AE weights')
    parser.add_argument('--sae_weights', default=None,
                        help='Path to TopK SAE weights (used instead of AE)')
    parser.add_argument('--pair_idx', type=int, default=None,
                        help='Which prompt pair to use (default: run all)')
    parser.add_argument('--n_pairs', type=int, default=5,
                        help='Number of pairs if pair_idx not set')
    parser.add_argument('--n_steps', type=int, default=11,
                        help='Number of interpolation steps (default: 11 = 0,10,...,100%%)')
    parser.add_argument('--n_tokens', type=int, default=20)
    parser.add_argument('--prompts', default='language',
                        choices=list(PROMPT_SETS.keys()),
                        help='Which prompt pairs to use')
    parser.add_argument('--prompt_a', type=str, default=None,
                        help='Custom prompt A (overrides --prompts)')
    parser.add_argument('--prompt_b', type=str, default=None,
                        help='Custom prompt B (overrides --prompts)')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Custom prompt pair or data module
    if args.prompt_a and args.prompt_b:
        PROMPT_PAIRS = [(args.prompt_a, args.prompt_b)]
    else:
        import importlib
        prompt_mod = importlib.import_module(PROMPT_SETS[args.prompts])
        PROMPT_PAIRS = prompt_mod.PROMPT_PAIRS

    if args.output_dir is None:
        args.output_dir = f'results/latent_interpolation/qwen3-1.7b_layer2_{args.prompts}'
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    dtype_map = {'float32': None, 'bfloat16': torch.bfloat16}

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    # Load AE or SAE
    use_sae = args.sae_weights is not None
    if use_sae:
        print(f"Loading SAE from {args.sae_weights}...")
        checkpoint = torch.load(args.sae_weights, weights_only=True, map_location=device)
        config = checkpoint['config']
        ae = TopKSAE(config['d_in'], config['d_sae'], config['k']).to(device)
        ae.load_state_dict(checkpoint['state_dict'])
        ae.eval()
        latent_label = f'SAE (d_sae={config["d_sae"]}, k={config["k"]})'
        print(f"  d_in={config['d_in']}, d_sae={config['d_sae']}, k={config['k']}")
    else:
        assert args.ae_weights is not None, "Must provide --ae_weights or --sae_weights"
        print(f"Loading AE from {args.ae_weights}...")
        checkpoint = torch.load(args.ae_weights, weights_only=True, map_location=device)
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
            arch_name = 'FlexAE'
        else:
            ae = Autoencoder(config['d_in'], config['k'], config.get('hidden_dims')).to(device)
            arch_name = 'AE'
        ae.load_state_dict(checkpoint['state_dict'])
        ae.eval()
        latent_label = f'{arch_name} (k={config["k"]})'
        print(f"  arch={arch_name}, d_in={config['d_in']}, k={config['k']}, "
              f"hidden={config.get('hidden_dims')}, act={config.get('activation')}")

    # Select pairs
    if args.pair_idx is not None:
        pairs = [(args.pair_idx, PROMPT_PAIRS[args.pair_idx])]
    else:
        n = min(args.n_pairs, len(PROMPT_PAIRS))
        pairs = [(i, PROMPT_PAIRS[i]) for i in range(n)]

    alphas = np.linspace(0, 1, args.n_steps)
    print(f"Interpolation steps: {args.n_steps} (alpha = {alphas})")

    all_results = []
    gen_file = os.path.join(args.output_dir, 'sample_generations.txt')
    gen_f = open(gen_file, 'w')

    for pair_idx, (prompt_a, prompt_b) in pairs:
        print(f"\n{'='*70}")
        print(f"Pair {pair_idx}: \"{prompt_a}\" <-> \"{prompt_b}\"")
        print(f"{'='*70}")

        gen_f.write(f"\n{'='*70}\n")
        gen_f.write(f"Pair {pair_idx}: \"{prompt_a}\" <-> \"{prompt_b}\"\n")
        gen_f.write(f"{'='*70}\n")

        # Get activations
        act_a, ctx_a = get_activation_and_context(
            model, tokenizer, prompt_a, args.layer, device)
        act_b, ctx_b = get_activation_and_context(
            model, tokenizer, prompt_b, args.layer, device)

        # ── Raw (activation-space) interpolation ────────────────────────
        print("\n  [RAW] Interpolating in activation space...")
        gen_f.write(f"\n--- RAW (activation space) ---\n")

        # Endpoints for raw: use the activations directly
        with torch.no_grad():
            logits_rawA = forward_to_logits(model, ctx_a, act_a.squeeze(0), args.layer)
            logits_rawB = forward_to_logits(model, ctx_a, act_b.squeeze(0), args.layer)

        raw_results = {'rel_dist': [], 'texts': []}

        for alpha in alphas:
            act_interp = slerp(act_a.squeeze(0), act_b.squeeze(0), alpha).unsqueeze(0)
            with torch.no_grad():
                logits_x = forward_to_logits(
                    model, ctx_a, act_interp.squeeze(0), args.layer)

            dist_A = (logits_x - logits_rawA).norm().item()
            dist_B = (logits_x - logits_rawB).norm().item()
            rel_d = dist_A / (dist_A + dist_B) if (dist_A + dist_B) > 0 else 0.5

            text, _ = generate_with_patched_activation(
                model, tokenizer, prompt_a, act_interp.squeeze(0),
                args.layer, device, args.n_tokens)

            raw_results['rel_dist'].append(rel_d)
            raw_results['texts'].append(text)

            print(f"  alpha={alpha:.2f}: d={rel_d:.3f}, text={text[:80]}")
            gen_f.write(f"\nalpha={alpha:.2f} (d={rel_d:.3f}):\n")
            gen_f.write(f"  {text[:150]}\n")

        # ── Latent interpolation ─────────────────────────────────────────
        print(f"\n  [LATENT] Interpolating in latent space ({latent_label})...")
        gen_f.write(f"\n--- LATENT ({latent_label}) ---\n")

        with torch.no_grad():
            if use_sae:
                z_a = ae.encode_full(act_a.float())  # [1, d_sae]
                z_b = ae.encode_full(act_b.float())  # [1, d_sae]
            else:
                z_a = ae.encode(act_a.float())  # [1, k]
                z_b = ae.encode(act_b.float())  # [1, k]

        print(f"  z_a norm={z_a.norm():.2f}, z_b norm={z_b.norm():.2f}")
        print(f"  cos(z_a, z_b) = {F.cosine_similarity(z_a, z_b).item():.4f}")

        # Endpoints for latent: decode(encode(act))
        with torch.no_grad():
            if use_sae:
                act_latA = ae.decode_from_full(z_a).to(ctx_a.dtype)
                act_latB = ae.decode_from_full(z_b).to(ctx_a.dtype)
            else:
                act_latA = ae.decode(z_a).to(ctx_a.dtype)
                act_latB = ae.decode(z_b).to(ctx_a.dtype)
            logits_latA = forward_to_logits(model, ctx_a, act_latA.squeeze(0), args.layer)
            logits_latB = forward_to_logits(model, ctx_a, act_latB.squeeze(0), args.layer)

        latent_results = {'rel_dist': [], 'texts': []}

        for alpha in alphas:
            z_interp = slerp(z_a.squeeze(0), z_b.squeeze(0), alpha).unsqueeze(0)
            with torch.no_grad():
                if use_sae:
                    act_interp = ae.decode_from_full(z_interp)
                else:
                    act_interp = ae.decode(z_interp)
                logits_x = forward_to_logits(
                    model, ctx_a, act_interp.to(ctx_a.dtype).squeeze(0), args.layer)

            dist_A = (logits_x - logits_latA).norm().item()
            dist_B = (logits_x - logits_latB).norm().item()
            rel_d = dist_A / (dist_A + dist_B) if (dist_A + dist_B) > 0 else 0.5

            text, _ = generate_with_patched_activation(
                model, tokenizer, prompt_a,
                act_interp.squeeze(0).to(ctx_a.dtype),
                args.layer, device, args.n_tokens)

            latent_results['rel_dist'].append(rel_d)
            latent_results['texts'].append(text)

            print(f"  alpha={alpha:.2f}: d={rel_d:.3f}, text={text[:80]}")
            gen_f.write(f"\nalpha={alpha:.2f} (d={rel_d:.3f}):\n")
            gen_f.write(f"  {text[:150]}\n")

        pair_result = {
            'pair_idx': pair_idx,
            'prompt_a': prompt_a,
            'prompt_b': prompt_b,
            'alphas': alphas.tolist(),
            'raw': raw_results,
            'latent': latent_results,
        }
        all_results.append(pair_result)

    gen_f.close()
    print(f"\nGenerations saved: {gen_file}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in all_results:
        label_a = r['prompt_a'][:30]
        label_b = r['prompt_b'][:30]
        pair_label = f"'{label_a}' → '{label_b}'"
        ax.plot(r['alphas'], r['raw']['rel_dist'], '-o', markersize=3,
                color='#2196F3', label=f'Raw (activation space)')
        ax.plot(r['alphas'], r['latent']['rel_dist'], '-s', markersize=3,
                color='#E91E63', label=f'Latent ({latent_label})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Linear')
    ax.set_xlabel('Interpolation alpha (0=A, 1=B)', fontsize=12)
    ax.set_ylabel('Relative distance d', fontsize=12)
    prompt_a_short = all_results[0]['prompt_a'][:35] if all_results else ''
    prompt_b_short = all_results[0]['prompt_b'][:35] if all_results else ''
    ax.set_title(f'Raw vs latent interpolation\n'
                 f'"{prompt_a_short}" → "{prompt_b_short}"', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'interpolation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")

    # Save pickle
    save_data = {
        'args': vars(args),
        'results': all_results,
    }
    pkl_path = os.path.join(args.output_dir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Results saved: {pkl_path}")
    print("Done!")


if __name__ == '__main__':
    main()
