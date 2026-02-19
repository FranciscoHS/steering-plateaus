#!/usr/bin/env python3
"""
Sentiment steering experiment.

Tests whether steering with a sentiment vector shows plateau-like behavior
by computing a difference-of-means steering vector and interpolating between
steered endpoints.

Usage:
    python scripts/sentiment_steering.py
    python scripts/sentiment_steering.py --layer 15 --n_steps 100 --magnitude 1.0 --test_prompt "The house was"
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Sentiment prompts for computing steering vector (simple single-word prompts)
POSITIVE = ["Love", "Happy", "Great", "Wonderful"]
NEGATIVE = ["Hate", "Sad", "Terrible", "Awful"]


def load_model_tl(model_name="gpt2-xl", device=None):
    """
    Load GPT-2 model via transformer_lens.

    Args:
        model_name: HuggingFace model name
        device: torch device (auto-detected if None)

    Returns:
        model: HookedTransformer model
        device: torch device
    """
    from transformer_lens import HookedTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} via transformer_lens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    return model, device


def compute_steering_vector(model, pos_prompts, neg_prompts, layer):
    """
    Compute steering vector as difference-of-means between positive and negative prompts.

    For each prompt: get activations at layer, average across all token positions.
    Average across all prompts in each category.
    Return mean_positive - mean_negative.

    Args:
        model: HookedTransformer model
        pos_prompts: List of positive sentiment prompts
        neg_prompts: List of negative sentiment prompts
        layer: Layer to extract activations from

    Returns:
        steering_vec: [d_model] tensor
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def get_mean_activation(prompts):
        """Get mean activation across prompts and token positions."""
        all_acts = []
        for prompt in prompts:
            # Run model and cache activations
            _, cache = model.run_with_cache(prompt)
            # Get activation at specified layer: [1, seq_len, d_model]
            act = cache[hook_point]
            # Average across sequence positions
            mean_act = act.mean(dim=1).squeeze(0)  # [d_model]
            all_acts.append(mean_act)

        # Stack and average across prompts
        all_acts = torch.stack(all_acts, dim=0)  # [n_prompts, d_model]
        return all_acts.mean(dim=0)  # [d_model]

    print(f"Computing mean activations for {len(pos_prompts)} positive prompts...")
    mean_pos = get_mean_activation(pos_prompts)

    print(f"Computing mean activations for {len(neg_prompts)} negative prompts...")
    mean_neg = get_mean_activation(neg_prompts)

    # Steering vector: positive - negative
    steering_vec = mean_pos - mean_neg

    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

    return steering_vec


def normalize_steering_vector(model, prompt, steering_vec, layer):
    """
    Normalize steering vector to unit activation norm.

    Scales steering vector so ||steering_vec|| = ||unsteered_act||.
    This means magnitude=1 adds a perturbation equal to the activation norm.

    Args:
        model: HookedTransformer model
        prompt: Test prompt to measure baseline activation
        steering_vec: Raw steering vector [d_model]
        layer: Layer to extract activations from

    Returns:
        unit_vec: Steering vector with norm = activation norm [d_model]
        act_norm: The activation norm used for normalization
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    # Get unsteered activation
    _, cache = model.run_with_cache(prompt)
    unsteered_act = cache[hook_point]  # [1, seq_len, d_model]

    # Compute mean activation norm across positions
    act_norm = unsteered_act.norm(dim=-1).mean().item()

    # Current steering norm
    steer_norm = steering_vec.norm().item()

    # Scale to unit activation norm
    scale = act_norm / (steer_norm + 1e-10)
    unit_vec = steering_vec * scale

    print(f"Unsteered activation norm: {act_norm:.4f}")
    print(f"Original steering norm: {steer_norm:.4f}")
    print(f"Normalized to unit activation norm: {unit_vec.norm().item():.4f}")

    return unit_vec, act_norm


def steer_and_get_logits(model, prompt, steering_vec, layer, magnitude):
    """
    Apply steering and get output logits.

    Adds magnitude * steering_vec to all token positions at specified layer.
    With a unit-normalized steering_vec, magnitude is in units of activation norm.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        steering_vec: Unit-normalized steering vector [d_model]
        layer: Layer to apply steering at
        magnitude: Steering strength (in units of activation norm)
                   Positive = toward positive sentiment
                   Negative = toward negative sentiment

    Returns:
        logits: [vocab_size] tensor (logits for last token)
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def steering_hook(activation, hook):
        activation[:, :, :] = activation + magnitude * steering_vec
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_point, steering_hook)]
        )

    return logits[0, -1, :]


def get_steered_activation(model, prompt, steering_vec, layer, magnitude):
    """
    Get the activation at a layer after applying steering.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        steering_vec: Unit-normalized steering vector [d_model]
        layer: Layer to apply steering and extract from
        magnitude: Steering strength (in units of activation norm)

    Returns:
        activation: [seq_len, d_model] tensor
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    stored_activation = [None]

    def steering_and_capture_hook(activation, hook):
        steered = activation + magnitude * steering_vec
        stored_activation[0] = steered.clone()
        return steered

    with torch.no_grad():
        _ = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_point, steering_and_capture_hook)]
        )

    return stored_activation[0].squeeze(0)


def inject_activation_and_get_logits(model, prompt, injected_act, layer):
    """
    Inject a specific activation at a layer and get output logits.

    Replaces the activation at the specified layer with the injected activation.

    Args:
        model: HookedTransformer model
        prompt: Input prompt (used for token structure)
        injected_act: Activation to inject [seq_len, d_model]
        layer: Layer to inject at

    Returns:
        logits: [vocab_size] tensor (logits for last token)
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def injection_hook(activation, hook):
        # Replace activation with injected one
        activation[:, :, :] = injected_act.unsqueeze(0)
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_point, injection_hook)]
        )

    return logits[0, -1, :]  # [vocab_size]


def run_interpolation(model, prompt, steering_vec, layer, n_steps, mag_start=None, mag_end=None, endpoint_mag=1.0):
    """
    Run interpolation experiment between steered endpoints.

    Gets activations at two magnitudes and interpolates between them,
    measuring relative distance in logit space.

    Args:
        model: HookedTransformer model
        prompt: Test prompt
        steering_vec: Unit-normalized steering vector [d_model]
        layer: Layer to steer at
        n_steps: Number of interpolation steps
        mag_start: Starting magnitude (default: -endpoint_mag)
        mag_end: Ending magnitude (default: +endpoint_mag)
        endpoint_mag: Legacy parameter, used if mag_start/mag_end not specified

    Returns:
        t_values: Array of interpolation parameters [0, 1]
        rel_distances: Array of relative distances
        all_logits: List of logits at each interpolation point
        logits_A: Logits at point A (start)
        logits_B: Logits at point B (end)
    """
    # Handle legacy endpoint_mag parameter
    if mag_start is None:
        mag_start = -endpoint_mag
    if mag_end is None:
        mag_end = +endpoint_mag

    print(f"\nRunning interpolation experiment with {n_steps} steps...")
    print(f"Endpoints: magnitude={mag_start} to magnitude={mag_end}")

    # Get activations at endpoints
    print(f"Getting activation at magnitude={mag_start} (point A)...")
    act_A = get_steered_activation(model, prompt, steering_vec, layer, magnitude=mag_start)

    print(f"Getting activation at magnitude={mag_end} (point B)...")
    act_B = get_steered_activation(model, prompt, steering_vec, layer, magnitude=mag_end)

    # Get logits at endpoints
    logits_A = inject_activation_and_get_logits(model, prompt, act_A, layer)
    logits_B = inject_activation_and_get_logits(model, prompt, act_B, layer)

    print(f"Activation A shape: {act_A.shape}")
    print(f"Activation B shape: {act_B.shape}")
    print(f"Activation distance: {(act_B - act_A).norm().item():.4f}")

    # Interpolation
    t_values = np.linspace(0, 1, n_steps)
    rel_distances = []
    all_logits = []

    print("Interpolating...")
    for i, t in enumerate(t_values):
        # Interpolate activation: act(t) = (1-t)*A + t*B
        interp_act = (1 - t) * act_A + t * act_B

        # Inject and get logits
        logits = inject_activation_and_get_logits(model, prompt, interp_act, layer)

        # Compute relative distance: d = ||logits - A|| / (||logits - A|| + ||logits - B||)
        dist_to_A = (logits - logits_A).norm().item()
        dist_to_B = (logits - logits_B).norm().item()
        rel_dist = dist_to_A / (dist_to_A + dist_to_B + 1e-10)

        rel_distances.append(rel_dist)
        all_logits.append(logits)

        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}/{n_steps}: t={t:.3f}, rel_dist={rel_dist:.4f}")

    rel_distances = np.array(rel_distances)

    return t_values, rel_distances, all_logits, logits_A, logits_B


def plot_results(t_values, rel_distances, prompt, magnitude=None, save_path=None, show_velocity=True):
    """Plot interpolation results: relative distance and optionally velocity."""
    if show_velocity:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    # Plot 1: Relative distance vs interpolation parameter
    ax1.plot(t_values, rel_distances, 'b-', linewidth=2, label='Observed')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Linear (d = t)')
    ax1.set_xlabel("Interpolation parameter t", fontsize=12)
    ax1.set_ylabel("Relative distance d", fontsize=12)
    title = f"Sentiment Steering Interpolation\nPrompt: '{prompt}'"
    if magnitude is not None:
        title += f", magnitude={magnitude}"
    ax1.set_title(title, fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend()

    if show_velocity:
        # Plot 2: Velocity (derivative)
        velocity = np.diff(rel_distances) / np.diff(t_values)
        t_mid = t_values[:-1] + np.diff(t_values) / 2
        ax2.plot(t_mid, velocity, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Linear (=1)')
        ax2.set_xlabel("Interpolation parameter t", fontsize=12)
        ax2.set_ylabel("d(rel_distance)/dt", fontsize=12)
        ax2.set_title("Rate of change (velocity)", fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_top_predictions(model, logits, label, k=5):
    """Print top-k token predictions from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    print(f"\n{label}:")
    for prob, idx in zip(top_probs, top_indices):
        token = model.to_string(idx.item())
        print(f"  '{token}': {prob.item():.4f}")


def generate_with_steering(model, prompt, steering_vec, layer, magnitude, n_tokens=15, single_steer=False):
    """
    Generate multiple tokens with steering.

    Args:
        model: HookedTransformer model
        prompt: Initial prompt
        steering_vec: Unit-normalized steering vector [d_model]
        layer: Layer to apply steering at
        magnitude: Steering strength (in units of activation norm)
        n_tokens: Number of tokens to generate
        single_steer: If True, only steer on first forward pass. If False, steer every pass.

    Returns:
        generated_text: The generated continuation (not including prompt)
        full_text: Prompt + generated continuation
    """
    hook_point = f"blocks.{layer}.hook_resid_post"

    def steering_hook(activation, hook):
        activation[:, :, :] = activation + magnitude * steering_vec
        return activation

    # Tokenize prompt
    tokens = model.to_tokens(prompt)  # [1, seq_len]

    generated_tokens = []

    with torch.no_grad():
        for i in range(n_tokens):
            # Apply steering only on first pass if single_steer, else always
            if single_steer and i > 0:
                # No steering after first token
                logits = model(tokens)
            else:
                # Forward pass with steering
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_point, steering_hook)]
                )

            # Get next token (greedy)
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

    # Decode
    generated_text = model.to_string(generated_tokens)
    full_text = model.to_string(tokens[0].tolist())

    return generated_text, full_text


def run_generation_sweep(model, prompt, steering_vec, layer, magnitudes, n_tokens=15, single_steer=False):
    """
    Run generation at multiple steering magnitudes.

    Args:
        model: HookedTransformer model
        prompt: Initial prompt
        steering_vec: Unit-normalized steering vector
        layer: Layer to steer at
        magnitudes: List of magnitudes to test (in units of activation norm)
        n_tokens: Tokens to generate per magnitude
        single_steer: If True, only steer on first token

    Returns:
        results: List of (magnitude, generated_text, full_text) tuples
    """
    results = []

    for mag in magnitudes:
        generated, full = generate_with_steering(
            model, prompt, steering_vec, layer, mag, n_tokens, single_steer
        )
        results.append((mag, generated, full))
        print(f"mag={mag:+.2f}: {full}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Sentiment steering experiment')
    parser.add_argument('--layer', type=int, default=16,
                        help='Layer to apply steering at')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of interpolation steps')
    parser.add_argument('--test_prompt', type=str, default="The house was",
                        help='Prompt to test steering on')
    parser.add_argument('--model', type=str, default='gpt2-xl',
                        help='Model name (default: gpt2-xl)')
    parser.add_argument('--save_dir', type=str, default='results/sentiment_steering',
                        help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving results')
    parser.add_argument('--no_velocity', action='store_true',
                        help='Skip velocity plot')
    parser.add_argument('--generate', action='store_true',
                        help='Run generation experiment instead of interpolation')
    parser.add_argument('--gen_tokens', type=int, default=15,
                        help='Number of tokens to generate (default: 15)')
    parser.add_argument('--magnitudes', type=str, default='-1,-0.5,0,0.5,1',
                        help='Comma-separated magnitudes for generation (in units of activation norm)')
    parser.add_argument('--endpoint_mag', type=float, default=1.0,
                        help='Endpoint magnitude for interpolation (legacy, use --mag_start/--mag_end)')
    parser.add_argument('--mag_start', type=float, default=None,
                        help='Starting magnitude for interpolation (default: -endpoint_mag)')
    parser.add_argument('--mag_end', type=float, default=None,
                        help='Ending magnitude for interpolation (default: +endpoint_mag)')
    parser.add_argument('--single_steer', action='store_true',
                        help='Only steer on first token (default: steer every token)')

    args = parser.parse_args()

    # Setup
    model, device = load_model_tl(args.model)
    print(f"Using device: {device}")
    print(f"Model config: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # Create save directory
    if not args.no_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "figures").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: Compute steering vector")
    print("=" * 60)

    steering_vec = compute_steering_vector(model, POSITIVE, NEGATIVE, args.layer)

    print("\n" + "=" * 60)
    print("STEP 2: Normalize steering vector to unit activation norm")
    print("=" * 60)

    steering_vec, act_norm = normalize_steering_vector(
        model, args.test_prompt, steering_vec, args.layer
    )
    print(f"Steering vector now has norm = activation norm = {act_norm:.2f}")
    print(f"Magnitude=1.0 means perturbation = 100% of activation norm")

    # Save steering vector
    if not args.no_save:
        vec_path = save_dir / f"steering_vec_layer{args.layer}.pt"
        torch.save({'steering_vec': steering_vec, 'act_norm': act_norm}, vec_path)
        print(f"Saved steering vector to {vec_path}")

    # Branch based on experiment type
    if args.generate:
        # Generation experiment
        print("\n" + "=" * 60)
        print("GENERATION EXPERIMENT")
        print("=" * 60)

        print(f"\nPrompt: '{args.test_prompt}'")
        print(f"Tokens to generate: {args.gen_tokens}")

        # Parse magnitude values
        magnitudes = [float(m) for m in args.magnitudes.split(',')]
        print(f"Magnitudes (in units of activation norm): {magnitudes}")
        steer_mode = "single (first token only)" if args.single_steer else "continuous (every token)"
        print(f"Steering mode: {steer_mode}")
        print("-" * 60)

        results = run_generation_sweep(
            model, args.test_prompt, steering_vec, args.layer,
            magnitudes, n_tokens=args.gen_tokens, single_steer=args.single_steer
        )

        # Save results
        if not args.no_save:
            gen_results = {
                'prompt': args.test_prompt,
                'magnitudes': magnitudes,
                'generations': [(m, gen, full) for m, gen, full in results],
                'args': vars(args)
            }
            results_path = save_dir / "generation_results.pt"
            torch.save(gen_results, results_path)
            print(f"\nSaved generation results to {results_path}")

            # Also save as text file for easy reading
            txt_path = save_dir / "generation_results.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Prompt: {args.test_prompt}\n")
                f.write(f"Layer: {args.layer}\n")
                f.write(f"Tokens: {args.gen_tokens}\n")
                f.write(f"Magnitudes: {magnitudes}\n")
                f.write("-" * 60 + "\n")
                for mag, gen, full in results:
                    f.write(f"mag={mag:+.2f}: {full}\n")
            print(f"Saved text results to {txt_path}")

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)

    else:
        # Interpolation experiment (original behavior)
        print("\n" + "=" * 60)
        print("STEP 3: Verify steering works (endpoint predictions)")
        print("=" * 60)

        print(f"\nTest prompt: '{args.test_prompt}'")
        print(f"Endpoint magnitude: {args.endpoint_mag}")

        # Get predictions at endpoints
        logits_neg = steer_and_get_logits(model, args.test_prompt, steering_vec, args.layer, magnitude=-args.endpoint_mag)
        logits_pos = steer_and_get_logits(model, args.test_prompt, steering_vec, args.layer, magnitude=+args.endpoint_mag)
        logits_neutral = steer_and_get_logits(model, args.test_prompt, steering_vec, args.layer, magnitude=0)

        print_top_predictions(model, logits_neg, f"Predictions at magnitude={-args.endpoint_mag} (negative)")
        print_top_predictions(model, logits_neutral, "Predictions at magnitude=0 (no steering)")
        print_top_predictions(model, logits_pos, f"Predictions at magnitude={+args.endpoint_mag} (positive)")

        print("\n" + "=" * 60)
        print("STEP 4: Run interpolation experiment")
        print("=" * 60)

        t_values, rel_distances, all_logits, logits_A, logits_B = run_interpolation(
            model, args.test_prompt, steering_vec, args.layer, args.n_steps,
            mag_start=args.mag_start, mag_end=args.mag_end, endpoint_mag=args.endpoint_mag
        )

        # Print summary statistics
        velocity = np.diff(rel_distances) / np.diff(t_values)
        print(f"\nInterpolation results:")
        print(f"  d at t=0: {rel_distances[0]:.4f} (should be ~0)")
        print(f"  d at t=1: {rel_distances[-1]:.4f} (should be ~1)")
        print(f"  Max deviation from linear: {np.max(np.abs(rel_distances - t_values)):.4f}")
        print(f"  Mean velocity: {np.mean(velocity):.4f} (linear = 1)")
        print(f"  Max velocity: {np.max(velocity):.4f}")
        print(f"  Min velocity: {np.min(velocity):.4f}")
        print(f"  Velocity std: {np.std(velocity):.4f}")

        # Compute S-curve metric: area between observed curve and linear reference
        area_diff = np.trapz(np.abs(rel_distances - t_values), t_values)
        print(f"  Area between curve and linear: {area_diff:.4f}")

        # Determine actual magnitudes used
        actual_mag_start = args.mag_start if args.mag_start is not None else -args.endpoint_mag
        actual_mag_end = args.mag_end if args.mag_end is not None else +args.endpoint_mag

        # Save interpolation results
        if not args.no_save:
            results = {
                't_values': t_values,
                'rel_distances': rel_distances,
                'all_logits': torch.stack(all_logits),
                'logits_A': logits_A,
                'logits_B': logits_B,
                'mag_start': actual_mag_start,
                'mag_end': actual_mag_end,
                'args': vars(args)
            }
            results_path = save_dir / f"interp_{actual_mag_start}_to_{actual_mag_end}.pt"
            torch.save(results, results_path)
            print(f"Saved interpolation results to {results_path}")

        print("\n" + "=" * 60)
        print("STEP 5: Generate plots")
        print("=" * 60)

        fig_path = None
        if not args.no_save:
            fig_path = save_dir / "figures" / f"interp_{actual_mag_start}_to_{actual_mag_end}.png"

        # Update plot title to show range
        plot_results(t_values, rel_distances, args.test_prompt,
                     magnitude=f"{actual_mag_start} to {actual_mag_end}", save_path=fig_path,
                     show_velocity=not args.no_velocity)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        # Interpretation
        print("\nInterpretation:")
        if area_diff > 0.05:
            print("  -> Significant deviation from linear detected!")
            print("  -> This suggests plateau-like structure in activation space.")
        else:
            print("  -> Curve is close to linear (d = t)")
        print("  -> This suggests flat activation space (no clear plateaus).")


if __name__ == '__main__':
    main()
