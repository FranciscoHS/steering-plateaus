#!/usr/bin/env python3
"""
Plateau analysis for refusal steering.

Sweeps steering magnitude in one direction and records first-token logits
to analyze whether the transition shows plateau structure.

Usage:
    python scripts/refusal_plateau_analysis.py --model qwen2.5-1.5b-instruct --chat_template --layer 15
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_harmful_prompts(path, n=128, seed=42):
    random.seed(seed)
    with open(path) as f:
        reader = csv.DictReader(f)
        prompts = [row['goal'] for row in reader]
    return random.sample(prompts, min(n, len(prompts)))


def load_harmless_prompts(path, n=128, seed=42):
    random.seed(seed)
    with open(path) as f:
        data = json.load(f)
    prompts = [d['instruction'] for d in data if not d.get('input')]
    return random.sample(prompts, min(n, len(prompts)))


def wrap_chat(text, use_template):
    if use_template:
        return QWEN_CHAT_TEMPLATE.format(text=text)
    return text


def load_model(model_name, device=None, dtype=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} via transformer_lens...")
    kwargs = dict(device=device)
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = HookedTransformer.from_pretrained(model_name, **kwargs)
    model.eval()

    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    return model, device


def compute_steering_vector(model, group_a_prompts, group_b_prompts, layer,
                            use_chat_template=False):
    hook_point = f"blocks.{layer}.hook_resid_post"

    def get_mean_last_activation(prompts, label):
        all_acts = []
        for i, prompt in enumerate(prompts):
            wrapped = wrap_chat(prompt, use_chat_template)
            _, cache = model.run_with_cache(wrapped)
            act = cache[hook_point].squeeze(0)
            last_act = act[-1, :]
            all_acts.append(last_act)
            if (i + 1) % 32 == 0:
                print(f"  {label}: {i+1}/{len(prompts)}")

        return torch.stack(all_acts, dim=0).mean(dim=0)

    print(f"Computing mean activations for {len(group_a_prompts)} harmful prompts...")
    mean_a = get_mean_last_activation(group_a_prompts, "harmful")

    print(f"Computing mean activations for {len(group_b_prompts)} harmless prompts...")
    mean_b = get_mean_last_activation(group_b_prompts, "harmless")

    steering_vec = mean_a - mean_b
    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

    return steering_vec


def normalize_steering_vector(model, test_prompt, steering_vec, layer):
    hook_point = f"blocks.{layer}.hook_resid_post"

    _, cache = model.run_with_cache(test_prompt)
    act = cache[hook_point].squeeze(0)

    act_norm = act.norm(dim=-1).mean().item()
    steer_norm = steering_vec.norm().item()
    steering_vec = steering_vec * (act_norm / steer_norm)

    print(f"Activation norm: {act_norm:.4f}")
    print(f"Normalized steering norm: {steering_vec.norm().item():.4f}")

    return steering_vec, act_norm


def get_first_token_logits(model, prompt, steering_vec, layer, magnitude):
    """Get logits for the first generated token with steering."""
    hook_point = f"blocks.{layer}.hook_resid_post"

    tokens = model.to_tokens(prompt)
    prompt_len = tokens.shape[1]

    def steering_hook(activation, hook):
        if activation.shape[1] > prompt_len:
            activation[:, prompt_len:, :] += magnitude * steering_vec
        return activation

    with torch.no_grad():
        # For first token, we just need one forward pass
        # Steering doesn't apply yet (no generated tokens), but we set it up
        # Actually for first token prediction, there are no post-prompt positions yet
        # So we need to think about this differently...

        # The first token prediction comes from the prompt alone
        # Steering at generation positions won't affect the first token!
        # We need to steer at ALL positions or at least the last prompt position

        # Let me reconsider: steer at all positions for this analysis
        def steering_hook_all(activation, hook):
            activation[:, :, :] += magnitude * steering_vec
            return activation

        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_point, steering_hook_all)]
        )

        first_token_logits = logits[0, -1, :]  # [vocab_size]

    return first_token_logits


def main():
    parser = argparse.ArgumentParser(description='Plateau analysis for refusal steering')
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b-instruct')
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--test_prompt', type=str,
                        default='Explain how photosynthesis works')
    parser.add_argument('--mag_start', type=float, default=0.0)
    parser.add_argument('--mag_end', type=float, default=0.03)
    parser.add_argument('--n_points', type=int, default=50)
    parser.add_argument('--n_prompts', type=int, default=128)
    parser.add_argument('--save_dir', type=str,
                        default='results/refusal_plateau')
    parser.add_argument('--chat_template', action='store_true')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--harmful_path', type=str,
                        default='data/harmful_behaviors.csv')
    parser.add_argument('--harmless_path', type=str,
                        default='data/alpaca_data.json')

    args = parser.parse_args()

    # Load prompts
    print("=" * 60)
    print("Loading prompts")
    print("=" * 60)
    harmful_prompts = load_harmful_prompts(args.harmful_path, args.n_prompts)
    harmless_prompts = load_harmless_prompts(args.harmless_path, args.n_prompts)
    print(f"Loaded {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless prompts")

    # Load model
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype)
    model, device = load_model(args.model, dtype=dtype)

    # Compute steering vector
    print("\n" + "=" * 60)
    print(f"Computing steering vector at layer {args.layer}")
    print("=" * 60)

    steering_vec = compute_steering_vector(
        model, harmful_prompts, harmless_prompts, args.layer,
        use_chat_template=args.chat_template
    )

    # Normalize
    print("\n" + "=" * 60)
    print("Normalizing steering vector")
    print("=" * 60)

    test_prompt = wrap_chat(args.test_prompt, args.chat_template)
    steering_vec, act_norm = normalize_steering_vector(
        model, test_prompt, steering_vec, args.layer
    )

    # Generate magnitude sweep
    magnitudes = torch.linspace(args.mag_start, args.mag_end, args.n_points).tolist()

    print("\n" + "=" * 60)
    print(f"Collecting first-token logits at {args.n_points} magnitudes")
    print(f"Range: {args.mag_start} to {args.mag_end}")
    print("=" * 60)

    all_logits = []
    for i, mag in enumerate(magnitudes):
        logits = get_first_token_logits(model, test_prompt, steering_vec, args.layer, mag)
        all_logits.append(logits)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_points}")

    all_logits = torch.stack(all_logits, dim=0)  # [n_points, vocab_size]

    # Compute relative distance
    # A = logits at mag_start (index 0)
    # B = logits at mag_end (index -1)
    A = all_logits[0]
    B = all_logits[-1]

    dist_to_A = (all_logits - A).norm(dim=-1)  # [n_points]
    dist_to_B = (all_logits - B).norm(dim=-1)  # [n_points]

    relative_dist = dist_to_A / (dist_to_A + dist_to_B)  # [n_points]

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Also get top token at each magnitude for reference
    top_tokens = all_logits.argmax(dim=-1)

    print(f"{'Magnitude':>10} {'Rel. Dist':>10} {'Top Token':>15}")
    print("-" * 40)
    for i in range(0, len(magnitudes), max(1, len(magnitudes)//10)):
        mag = magnitudes[i]
        d = relative_dist[i].item()
        tok = model.to_string(top_tokens[i].item())
        print(f"{mag:>10.4f} {d:>10.4f} {tok:>15}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'magnitudes': magnitudes,
        'relative_dist': relative_dist.cpu().tolist(),
        'dist_to_A': dist_to_A.cpu().tolist(),
        'dist_to_B': dist_to_B.cpu().tolist(),
        'top_tokens': [model.to_string(t.item()) for t in top_tokens],
        'test_prompt': args.test_prompt,
        'layer': args.layer,
        'act_norm': act_norm,
        'mag_start': args.mag_start,
        'mag_end': args.mag_end,
    }

    torch.save(results, save_dir / 'plateau_analysis.pt')
    torch.save(all_logits.cpu(), save_dir / 'all_logits.pt')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(magnitudes, relative_dist.cpu().numpy(), 'b.-', linewidth=1, markersize=3)
    plt.xlabel('Steering magnitude (units of layer norm)')
    plt.ylabel('Relative distance d = ||x-A|| / (||x-A|| + ||x-B||)')
    plt.title(f'Plateau analysis: "{args.test_prompt[:40]}..."\nLayer {args.layer}, mag {args.mag_start} to {args.mag_end}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='d=0.5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'plateau_curve.png', dpi=150)
    print(f"\nSaved plot to {save_dir / 'plateau_curve.png'}")
    print(f"Saved results to {save_dir / 'plateau_analysis.pt'}")


if __name__ == '__main__':
    main()
