#!/usr/bin/env python3
"""
Sentiment steering sanity check.

Computes a sentiment steering vector via difference-of-means from single-word
prompts. The steering vector is computed from last-token activations (which
aggregate full prompt meaning via causal attention), avoiding token-alignment
issues across prompts of different lengths. The resulting [d_model] vector is
then applied to all token positions at the target layer on every forward pass
(continuous steering).

Supports both base models (qwen3-0.6b) and instruct models (qwen2.5-1.5b-instruct)
via the --chat_template flag.

Usage:
    python scripts/sentiment_sanity_check.py
    python scripts/sentiment_sanity_check.py --model qwen2.5-1.5b-instruct --chat_template
    python scripts/sentiment_sanity_check.py --magnitudes "0,1,2,5,10"
"""

import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


POSITIVE = ["Love", "Happy", "Great", "Wonderful", "Good",
            "Joy", "Amazing", "Excellent", "Beautiful", "Fantastic"]
NEGATIVE = ["Hate", "Sad", "Terrible", "Awful", "Bad",
            "Horrible", "Disgusting", "Dreadful", "Ugly", "Miserable"]

# Qwen chat template (used for instruct models)
QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def wrap_chat(text, use_template):
    """Optionally wrap text in Qwen chat template."""
    if use_template:
        return QWEN_CHAT_TEMPLATE.format(text=text)
    return text


def load_model(model_name, device=None):
    """Load model via transformer_lens."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} via transformer_lens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    return model, device


def compute_steering_vector(model, pos_prompts, neg_prompts, layer,
                            use_chat_template=False):
    """
    Compute steering vector as difference-of-means using last-token activations.

    For each prompt, we extract the activation at the last token position (which
    aggregates the full prompt meaning via causal attention). This avoids
    token-alignment issues across prompts of different lengths.

    Returns a single [d_model] steering vector.
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

    print(f"Computing mean last-token activations for {len(pos_prompts)} positive prompts...")
    mean_pos = get_mean_last_activation(pos_prompts)

    print(f"Computing mean last-token activations for {len(neg_prompts)} negative prompts...")
    mean_neg = get_mean_last_activation(neg_prompts)

    steering_vec = mean_pos - mean_neg  # [d_model]
    print(f"Steering vector shape: {steering_vec.shape}")
    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

    return steering_vec


def normalize_steering_vector(model, test_prompt, steering_vec, layer):
    """
    Normalize steering vector so ||steering_vec|| = ||layer_activation||.

    This means magnitude=1 adds a perturbation equal to the mean activation norm.
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    _, cache = model.run_with_cache(test_prompt)
    act = cache[hook_point].squeeze(0)  # [seq_len, d_model]

    # Mean activation norm across all positions
    act_norm = act.norm(dim=-1).mean().item()

    # Scale steering vector
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

    Steering vector [d_model] is added to all token positions at the target
    layer on every forward pass, scaled by magnitude.
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def steering_hook(activation, hook):
        activation[:, :, :] += magnitude * steering_vec
        return activation

    tokens = model.to_tokens(prompt)  # [1, seq_len]

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
    parser = argparse.ArgumentParser(description='Sentiment steering sanity check')
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--test_prompt', type=str,
                        default="In my opinion, Barack Obama has to be considered")
    parser.add_argument('--magnitudes', type=str, default='0,1,2,5')
    parser.add_argument('--n_tokens', type=int, default=15)
    parser.add_argument('--save_dir', type=str, default='results/sentiment_steering_qwen')
    parser.add_argument('--chat_template', action='store_true',
                        help='Wrap prompts in Qwen chat template (<|im_start|>user\\n...)')

    args = parser.parse_args()

    model, device = load_model(args.model)

    # Step 1: Show tokenization of prompts
    print("\n" + "=" * 60)
    print("STEP 1: Prompt tokenization")
    print("=" * 60)

    for label, prompts in [("Positive", POSITIVE), ("Negative", NEGATIVE)]:
        print(f"\n{label}:")
        for p in prompts:
            tokens = model.to_tokens(p, prepend_bos=False).squeeze(0)
            strs = [model.to_string(t.item()) for t in tokens]
            print(f"  '{p}' -> {strs}")

    # Step 2: Compute steering vector
    print("\n" + "=" * 60)
    print(f"STEP 2: Compute steering vector at layer {args.layer}")
    print("=" * 60)

    steering_vec = compute_steering_vector(model, POSITIVE, NEGATIVE, args.layer,
                                            use_chat_template=args.chat_template)

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
        # Print just the generated part for readability with chat template
        if args.chat_template:
            print(f"mag={mag:+.2f}: {generated}")
        else:
            print(f"mag={mag:+.2f}: {full}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    txt_path = save_dir / "generation_sanity_check.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Test prompt: {args.test_prompt}\n")
        f.write(f"Activation norm: {act_norm:.4f}\n")
        f.write(f"Tokens generated: {args.n_tokens}\n")
        f.write(f"Positive prompts: {POSITIVE}\n")
        f.write(f"Negative prompts: {NEGATIVE}\n")
        f.write(f"Chat template: {args.chat_template}\n")
        f.write(f"Method: last-token activation, applied to all positions\n")
        f.write("-" * 60 + "\n")
        for mag, gen, full in results:
            f.write(f"mag={mag:+.1f}: {full}\n")

    print(f"\nSaved results to {txt_path}")

    # Save steering vector
    vec_path = save_dir / f"steering_vec_layer{args.layer}.pt"
    torch.save({
        'steering_vec': steering_vec,
        'act_norm': act_norm,
        'layer': args.layer,
        'positive_prompts': POSITIVE,
        'negative_prompts': NEGATIVE,
    }, vec_path)
    print(f"Saved steering vector to {vec_path}")


if __name__ == '__main__':
    main()
