#!/usr/bin/env python3
"""
Generate tokens with steering applied at each step.
"""

import argparse
import sys
sys.path.insert(0, '.')

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name='gpt2-large', device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer, device


def get_layer0_output(model, input_ids, device):
    """Get layer 0 output for given input_ids."""
    wte = model.transformer.wte(input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    wpe = model.transformer.wpe(positions)
    hidden_states = wte + wpe
    hidden_states = model.transformer.drop(hidden_states)
    hidden_states = model.transformer.h[0](hidden_states)[0]
    return hidden_states


def forward_with_steering(model, input_ids, steering_vec, alpha, device):
    """
    Forward pass with steering applied at layer 1.

    Steering is applied to the LAST token position only.
    """
    # Get layer 0 output
    hidden = get_layer0_output(model, input_ids, device)

    # Apply steering to last token position
    hidden[:, -1:, :] = hidden[:, -1:, :] + alpha * steering_vec

    # Continue through remaining layers
    for block in model.transformer.h[1:]:
        hidden = block(hidden)[0]

    hidden = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden[:, -1, :])

    return logits


def generate_with_steering(model, tokenizer, device, prompt_a, prompt_b, alpha, n_tokens=10, steer_all=True):
    """
    Generate n_tokens with steering.

    Args:
        prompt_a: Base prompt (e.g., "The house was big")
        prompt_b: Target prompt for steering direction (e.g., "The house was in")
        alpha: Steering magnitude
        n_tokens: Number of tokens to generate
        steer_all: If True, steer at every step. If False, steer only first token.

    Returns:
        generated_tokens: list of token strings
        generated_ids: list of token ids
    """
    # Tokenize prompts
    tokens_a = tokenizer(prompt_a, return_tensors='pt').to(device)
    tokens_b = tokenizer(prompt_b, return_tensors='pt').to(device)

    input_ids_a = tokens_a['input_ids']
    input_ids_b = tokens_b['input_ids']

    # Compute steering vector from last token difference at layer 0
    with torch.no_grad():
        resid_a = get_layer0_output(model, input_ids_a, device)
        resid_b = get_layer0_output(model, input_ids_b, device)

    # Steering vector is the difference at the last token position
    steering_vec = resid_b[:, -1:, :] - resid_a[:, -1:, :]

    # Start generation from prompt_a
    current_ids = input_ids_a.clone()
    generated_tokens = []
    generated_ids = []

    for i in range(n_tokens):
        # Apply steering only if steer_all=True, or if this is the first token
        current_alpha = alpha if (steer_all or i == 0) else 0.0

        with torch.no_grad():
            logits = forward_with_steering(model, current_ids, steering_vec, current_alpha, device)

        # Greedy: pick highest probability token
        next_token_id = torch.argmax(logits, dim=-1).item()
        next_token = tokenizer.decode([next_token_id])

        generated_tokens.append(next_token)
        generated_ids.append(next_token_id)

        # Append to sequence for next iteration
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)

    return generated_tokens, generated_ids


def main():
    parser = argparse.ArgumentParser(description='Generate with steering')
    parser.add_argument('--prompt_a', type=str, default="The house was big")
    parser.add_argument('--prompt_b', type=str, default="The house was in")
    parser.add_argument('--model', type=str, default='gpt2-large')
    parser.add_argument('--alphas', type=str, default="0,1,6,10",
                        help='Comma-separated steering magnitudes')
    parser.add_argument('--n_tokens', type=int, default=10)
    parser.add_argument('--first_only', action='store_true',
                        help='Apply steering only to first token')
    parser.add_argument('--save', type=str, default=None)

    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(',')]

    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    print(f"Using device: {device}")

    steer_mode = "first token only" if args.first_only else "all tokens"
    print(f"\nPrompt A: '{args.prompt_a}'")
    print(f"Prompt B: '{args.prompt_b}'")
    print(f"Generating {args.n_tokens} tokens at α = {alphas}")
    print(f"Steering mode: {steer_mode}")
    print("=" * 80)

    results = []
    for alpha in alphas:
        tokens, ids = generate_with_steering(
            model, tokenizer, device,
            args.prompt_a, args.prompt_b,
            alpha=alpha,
            n_tokens=args.n_tokens,
            steer_all=not args.first_only
        )

        continuation = ''.join(tokens)
        full_text = args.prompt_a + continuation

        print(f"\nα = {alpha}:")
        print(f"  Generated: {continuation}")
        print(f"  Full: {full_text}")

        results.append({
            'alpha': alpha,
            'tokens': tokens,
            'continuation': continuation,
            'full_text': full_text
        })

    if args.save:
        with open(args.save, 'w') as f:
            f.write(f"Steering generation: '{args.prompt_a}' -> '{args.prompt_b}'\n")
            f.write(f"Model: {args.model}\n")
            f.write("=" * 80 + "\n\n")
            for r in results:
                f.write(f"α = {r['alpha']}:\n")
                f.write(f"  Tokens: {r['tokens']}\n")
                f.write(f"  Full: {r['full_text']}\n\n")
        print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
