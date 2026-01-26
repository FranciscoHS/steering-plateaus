#!/usr/bin/env python3
"""
Gender steering experiment.

Computes a gender steering vector via difference-of-means (male - female) from
the 30 most common US names per gender. Uses last-token activations and applies
steering to all positions (continuous steering).

Usage:
    python scripts/gender_steering.py --model qwen2.5-1.5b-instruct --chat_template --layer 25
    python scripts/gender_steering.py --magnitudes="-0.5,-0.2,0,0.2,0.5"
"""

import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


# 30 most common US names per gender (SSA data)
MALE_NAMES = [
    "James", "Robert", "John", "Michael", "David",
    "William", "Richard", "Joseph", "Thomas", "Christopher",
    "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Kenneth", "Kevin", "Brian", "George", "Timothy",
    "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
]
FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara",
    "Elizabeth", "Susan", "Jessica", "Sarah", "Karen",
    "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
    "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
    "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
]

# Qwen chat template
QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def wrap_chat(text, use_template):
    """Optionally wrap text in Qwen chat template."""
    if use_template:
        return QWEN_CHAT_TEMPLATE.format(text=text)
    return text


def load_model(model_name, device=None, dtype=None):
    """Load model via transformer_lens."""
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
    """
    Compute steering vector as difference-of-means using last-token activations.
    Returns direction (group_a - group_b) as a single [d_model] vector.
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def get_mean_last_activation(prompts):
        all_acts = []
        for prompt in prompts:
            wrapped = wrap_chat(prompt, use_chat_template)
            _, cache = model.run_with_cache(wrapped)
            act = cache[hook_point].squeeze(0)  # [seq_len, d_model]
            last_act = act[-1, :]  # [d_model]
            all_acts.append(last_act)

        return torch.stack(all_acts, dim=0).mean(dim=0)  # [d_model]

    print(f"Computing mean last-token activations for {len(group_a_prompts)} male prompts...")
    mean_a = get_mean_last_activation(group_a_prompts)

    print(f"Computing mean last-token activations for {len(group_b_prompts)} female prompts...")
    mean_b = get_mean_last_activation(group_b_prompts)

    steering_vec = mean_a - mean_b  # [d_model], male - female
    print(f"Steering vector shape: {steering_vec.shape}")
    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

    return steering_vec


def normalize_steering_vector(model, test_prompt, steering_vec, layer):
    """
    Normalize steering vector so ||steering_vec|| = ||layer_activation||.
    Magnitude=1 adds a perturbation equal to the mean activation norm.
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    _, cache = model.run_with_cache(test_prompt)
    act = cache[hook_point].squeeze(0)  # [seq_len, d_model]

    act_norm = act.norm(dim=-1).mean().item()
    steer_norm = steering_vec.norm().item()
    steering_vec = steering_vec * (act_norm / steer_norm)

    print(f"Activation norm (mean across positions): {act_norm:.4f}")
    print(f"Original steering norm: {steer_norm:.4f}")
    print(f"Normalized steering norm: {steering_vec.norm().item():.4f}")

    return steering_vec, act_norm


def generate_with_steering(model, prompt, steering_vec, layer, magnitude,
                           n_tokens=15):
    """
    Generate tokens with continuous steering.

    Steering vector is added only to positions after the prompt (i.e., the
    generated tokens), matching CAA: "steering vectors are added at all token
    positions after the user's prompt".
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    tokens = model.to_tokens(prompt)
    prompt_len = tokens.shape[1]

    def steering_hook(activation, hook):
        # Only steer positions beyond the original prompt
        if activation.shape[1] > prompt_len:
            activation[:, prompt_len:, :] += magnitude * steering_vec
        return activation

    generated_tokens = []

    with torch.no_grad():
        for i in range(n_tokens):
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, steering_hook)]
            )

            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            tokens = torch.cat([tokens, next_token], dim=1)

    generated_text = model.to_string(generated_tokens)
    full_text = model.to_string(tokens[0].tolist())

    return generated_text, full_text


def main():
    parser = argparse.ArgumentParser(description='Gender steering experiment')
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b-instruct')
    parser.add_argument('--layer', type=int, default=25)
    parser.add_argument('--test_prompt', type=str,
                        default="The nurse said that")
    parser.add_argument('--magnitudes', type=str,
                        default='-0.50,-0.30,-0.20,-0.10,-0.05,0,0.05,0.10,0.20,0.30,0.50')
    parser.add_argument('--n_tokens', type=int, default=50)
    parser.add_argument('--save_dir', type=str,
                        default='results/gender_steering')
    parser.add_argument('--chat_template', action='store_true',
                        help='Wrap prompts in Qwen chat template')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype (default: auto)')

    args = parser.parse_args()

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype)

    model, device = load_model(args.model, dtype=dtype)

    # Step 1: Show tokenization of a few prompts
    print("\n" + "=" * 60)
    print("STEP 1: Prompt tokenization (sample)")
    print("=" * 60)

    for label, names in [("Male", MALE_NAMES[:5]), ("Female", FEMALE_NAMES[:5])]:
        print(f"\n{label}:")
        for name in names:
            tokens = model.to_tokens(name, prepend_bos=False).squeeze(0)
            strs = [model.to_string(t.item()) for t in tokens]
            print(f"  '{name}' -> {strs}")

    # Step 2: Compute steering vector
    print("\n" + "=" * 60)
    print(f"STEP 2: Compute steering vector at layer {args.layer}")
    print(f"  Direction: male - female")
    print("=" * 60)

    steering_vec = compute_steering_vector(
        model, MALE_NAMES, FEMALE_NAMES, args.layer,
        use_chat_template=args.chat_template
    )

    # Step 3: Normalize
    print("\n" + "=" * 60)
    print("STEP 3: Normalize steering vector")
    print("=" * 60)

    test_prompt = wrap_chat(args.test_prompt, args.chat_template)

    steering_vec, act_norm = normalize_steering_vector(
        model, test_prompt, steering_vec, args.layer
    )

    # Step 4: Generate at various magnitudes
    print("\n" + "=" * 60)
    print("STEP 4: Generation sweep (continuous steering)")
    print("=" * 60)

    magnitudes = [float(m) for m in args.magnitudes.split(',')]
    print(f"Test prompt: '{args.test_prompt}'")
    if args.chat_template:
        print(f"  (wrapped in Qwen chat template)")
    print(f"Direction: + = male, - = female")
    print(f"Magnitudes: {magnitudes}")
    print(f"Steering at layer {args.layer}, all positions")
    print(f"Tokens to generate: {args.n_tokens}")
    print("-" * 60)

    results = []
    for mag in magnitudes:
        generated, full = generate_with_steering(
            model, test_prompt, steering_vec, args.layer,
            mag, args.n_tokens
        )
        results.append((mag, generated, full))
        if args.chat_template:
            print(f"mag={mag:+.2f}: {generated}")
        else:
            print(f"mag={mag:+.2f}: {full}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    txt_path = save_dir / "generation_sweep.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Test prompt: {args.test_prompt}\n")
        f.write(f"Chat template: {args.chat_template}\n")
        f.write(f"Direction: male - female\n")
        f.write(f"Activation norm: {act_norm:.4f}\n")
        f.write(f"Tokens generated: {args.n_tokens}\n")
        f.write(f"Male names: {MALE_NAMES}\n")
        f.write(f"Female names: {FEMALE_NAMES}\n")
        f.write(f"Method: last-token activation, applied to all positions\n")
        f.write("-" * 60 + "\n")
        for mag, gen, full in results:
            f.write(f"mag={mag:+.2f}: {full}\n")

    print(f"\nSaved results to {txt_path}")

    vec_path = save_dir / f"steering_vec_layer{args.layer}.pt"
    torch.save({
        'steering_vec': steering_vec,
        'act_norm': act_norm,
        'layer': args.layer,
        'male_names': MALE_NAMES,
        'female_names': FEMALE_NAMES,
    }, vec_path)
    print(f"Saved steering vector to {vec_path}")


if __name__ == '__main__':
    main()
