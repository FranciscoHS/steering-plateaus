#!/usr/bin/env python3
"""
Plateau analysis for refusal steering - v2.

Measures distance in final-layer activation space (pre-logits) at the last
prompt token position, rather than logit-space distance.

Usage:
    python scripts/refusal_plateau_analysis_v2.py --model qwen2.5-1.5b-instruct --chat_template --layer 15
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

    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

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


def get_final_layer_activation(model, prompt, steering_vec, steer_layer, magnitude):
    """
    Get final-layer activation at the last prompt token position with steering.
    """
    steer_hook_point = f"blocks.{steer_layer}.hook_resid_post"
    final_layer = model.cfg.n_layers - 1
    final_hook_point = f"blocks.{final_layer}.hook_resid_post"

    tokens = model.to_tokens(prompt)

    captured_activation = {}

    def steering_hook(activation, hook):
        activation[:, :, :] += magnitude * steering_vec
        return activation

    def capture_hook(activation, hook):
        captured_activation['act'] = activation[:, -1, :].clone()  # [1, d_model]
        return activation

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (steer_hook_point, steering_hook),
                (final_hook_point, capture_hook)
            ]
        )

    return captured_activation['act'].squeeze(0)  # [d_model]


def main():
    parser = argparse.ArgumentParser(description='Plateau analysis v2 - final layer activations')
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b-instruct')
    parser.add_argument('--layer', type=int, default=15,
                        help='Layer at which to apply steering')
    parser.add_argument('--test_prompt', type=str,
                        default='Explain how photosynthesis works')
    parser.add_argument('--mag_start', type=float, default=0.0)
    parser.add_argument('--mag_end', type=float, default=0.03)
    parser.add_argument('--n_points', type=int, default=50)
    parser.add_argument('--n_prompts', type=int, default=128)
    parser.add_argument('--save_dir', type=str,
                        default='results/refusal_plateau_v2')
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
    print(f"Collecting final-layer activations at {args.n_points} magnitudes")
    print(f"Steering at layer {args.layer}, measuring at layer {model.cfg.n_layers - 1}")
    print(f"Range: {args.mag_start} to {args.mag_end}")
    print("=" * 60)

    all_activations = []
    for i, mag in enumerate(magnitudes):
        act = get_final_layer_activation(model, test_prompt, steering_vec, args.layer, mag)
        all_activations.append(act)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_points}")

    all_activations = torch.stack(all_activations, dim=0)  # [n_points, d_model]

    # Compute relative distance in activation space
    A = all_activations[0]   # activation at mag_start
    B = all_activations[-1]  # activation at mag_end

    dist_to_A = (all_activations - A).norm(dim=-1)  # [n_points]
    dist_to_B = (all_activations - B).norm(dim=-1)  # [n_points]

    relative_dist = dist_to_A / (dist_to_A + dist_to_B)  # [n_points]

    # Also compute logits for reference (to see what token would be predicted)
    # Use the unembedding matrix
    logits = all_activations @ model.W_U  # [n_points, vocab_size]
    if model.cfg.d_vocab_out is not None:
        # Some models have a separate output vocab
        pass
    top_tokens = logits.argmax(dim=-1)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

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
        'steer_layer': args.layer,
        'measure_layer': model.cfg.n_layers - 1,
        'act_norm': act_norm,
        'mag_start': args.mag_start,
        'mag_end': args.mag_end,
    }

    torch.save(results, save_dir / 'plateau_analysis.pt')
    torch.save(all_activations.cpu(), save_dir / 'all_activations.pt')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(magnitudes, relative_dist.cpu().numpy(), 'b.-', linewidth=1, markersize=3)
    plt.xlabel('Steering magnitude (units of layer norm)')
    plt.ylabel('Relative distance d = ||x-A|| / (||x-A|| + ||x-B||)')
    plt.title(f'Plateau analysis (final-layer activations): "{args.test_prompt[:40]}..."\n'
              f'Steer at layer {args.layer}, measure at layer {model.cfg.n_layers - 1}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='d=0.5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'plateau_curve.png', dpi=150)
    print(f"\nSaved plot to {save_dir / 'plateau_curve.png'}")
    print(f"Saved results to {save_dir / 'plateau_analysis.pt'}")


if __name__ == '__main__':
    main()
