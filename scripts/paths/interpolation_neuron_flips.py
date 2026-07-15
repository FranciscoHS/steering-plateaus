#!/usr/bin/env python3
"""
Interpolation neuron-flip experiment.

Question: does a single activation plateau correspond to a *concept*, or is it
just a region of activation space *without neuron flips*?

We slerp-interpolate the last-token activation at inject layer L between two
prompts (e.g. "The house was big" -> "The house was large", and the
known-boundary control "... big" -> "... in"). Along the path we measure:

  (a) relative distance in final-logit space  (the plateau -> boundary -> plateau
      S-curve, as in scripts/paths/interpolation_experiment.py), and
  (b) neuron flips at every intervening MLP layer L+1 .. measure_layer, where a
      "flip" is a sign change of the MLP pre-activation (input to the nonlinearity
      / SwiGLU gate, `blocks.{l}.mlp.hook_pre`) across a threshold tau.

Flips are reported two ways, per-layer and summed:
  - per-step:   fraction of neurons whose gate state differs from the *previous* t
  - cumulative: fraction differing from the *start* (t=0)

The synonym arm (big->large) is the real test: a flip-free stretch there would
say "plateau ~ concept"; flips anyway would say "plateau is just a flip-free
region, not concept-aligned".

Usage (run on a GPU pod, not locally):
    python scripts/paths/interpolation_neuron_flips.py \
        --model qwen3-0.6b --layer 14 --n_steps 300
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


def load_model(model_name, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} via transformer_lens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}, "
          f"d_mlp={model.cfg.d_mlp}")
    return model, device


def get_activations(model, prompt, layer):
    """Last-layer-L residual stream for a prompt. Returns [seq_len, d_model], tokens."""
    _, cache = model.run_with_cache(prompt)
    act = cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)
    tokens = model.to_tokens(prompt).squeeze(0)
    return act, tokens


def slerp_with_linear_magnitude(v0, v1, t):
    """Slerp direction + linear magnitude (Stefan Heimersheim approach)."""
    norm0, norm1 = v0.norm(), v1.norm()
    u0, u1 = v0 / norm0, v1 / norm1
    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        direction = u0
    else:
        sin_theta = torch.sin(theta)
        direction = (torch.sin((1 - t) * theta) / sin_theta) * u0 + \
                    (torch.sin(t * theta) / sin_theta) * u1
    mag = (1 - t) * norm0 + t * norm1
    return direction * mag


def forward_capture(model, prompt, prefix_act, last_token_act, layer, capture_layers):
    """Inject (prefix + last_token) at `layer`, forward, and capture:
        - final last-token logits  [vocab]
        - MLP pre-activations (last token) at each layer in capture_layers
          {l: [d_mlp]}
    """
    inj_hook = f"blocks.{layer}.hook_resid_post"
    full_act = torch.cat([prefix_act, last_token_act.unsqueeze(0)], dim=0).unsqueeze(0)

    def injection_hook(activation, hook):
        activation[:, :, :] = full_act
        return activation

    gates = {}

    def make_gate_hook(l):
        def gate_hook(activation, hook):
            # activation: [batch, pos, d_mlp]; keep last token only
            gates[l] = activation[0, -1, :].detach()
            return activation
        return gate_hook

    fwd_hooks = [(inj_hook, injection_hook)]
    fwd_hooks += [(f"blocks.{l}.mlp.hook_pre", make_gate_hook(l)) for l in capture_layers]

    with torch.no_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
    return logits[0, -1, :], gates


def run_pair(model, prompt_a, prompt_b, layer, measure_layer, n_steps, thresholds):
    """Interpolate A->B, tracking logit rel-distance and neuron flips.

    Returns a results dict.
    """
    print(f"\n=== {prompt_a!r} -> {prompt_b!r} ===")
    act_A, tok_A = get_activations(model, prompt_a, layer)
    act_B, tok_B = get_activations(model, prompt_b, layer)
    if act_A.shape != act_B.shape:
        raise ValueError(
            f"Prompts tokenize to different lengths: "
            f"{[model.to_string(t.item()) for t in tok_A]} vs "
            f"{[model.to_string(t.item()) for t in tok_B]}. "
            f"Final word must be a single token so lengths match.")

    prefix_act = act_A[:-1, :]          # shared prefix, held fixed (from A)
    last_A, last_B = act_A[-1, :], act_B[-1, :]

    angle = torch.acos(torch.clamp(
        torch.dot(last_A / last_A.norm(), last_B / last_B.norm()), -1.0, 1.0)).item()
    print(f"  last-token angle: {np.degrees(angle):.2f} deg, "
          f"||A||={last_A.norm():.2f}, ||B||={last_B.norm():.2f}")

    capture_layers = list(range(layer + 1, measure_layer + 1))
    d_mlp = model.cfg.d_mlp
    print(f"  capturing flips at MLP layers {capture_layers[0]}..{capture_layers[-1]} "
          f"(d_mlp={d_mlp})")

    # Endpoint logits (for relative-distance metric)
    logits_A, gates_A = forward_capture(model, prompt_a, prefix_act, last_A, layer, capture_layers)
    logits_B, _ = forward_capture(model, prompt_a, prefix_act, last_B, layer, capture_layers)

    t_values = np.linspace(0, 1, n_steps)
    rel_distances = np.zeros(n_steps)

    # flip accumulators: per threshold -> array [n_steps, n_capture_layers]
    n_cap = len(capture_layers)
    perstep = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}
    cumulative = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}

    # binary gate state at t=0 (reference) and previous step
    ref_bin = {tau: {l: (gates_A[l] > tau) for l in capture_layers} for tau in thresholds}
    prev_bin = {tau: {l: ref_bin[tau][l].clone() for l in capture_layers} for tau in thresholds}

    for i, t in enumerate(t_values):
        interp = slerp_with_linear_magnitude(last_A, last_B, float(t))
        logits, gates = forward_capture(model, prompt_a, prefix_act, interp, layer, capture_layers)

        dA = (logits - logits_A).norm().item()
        dB = (logits - logits_B).norm().item()
        rel_distances[i] = dA / (dA + dB + 1e-10)

        for tau in thresholds:
            for j, l in enumerate(capture_layers):
                cur = gates[l] > tau
                cumulative[tau][i, j] = (cur != ref_bin[tau][l]).float().mean().item()
                perstep[tau][i, j] = (cur != prev_bin[tau][l]).float().mean().item()
                prev_bin[tau][l] = cur

        if (i + 1) % 50 == 0:
            print(f"  step {i+1}/{n_steps}: t={t:.3f} rel_d={rel_distances[i]:.3f} "
                  f"perstep_flip(tau=0)={perstep[0.0][i].sum():.4f}")

    return {
        'prompt_a': prompt_a, 'prompt_b': prompt_b,
        'layer': layer, 'measure_layer': measure_layer,
        'capture_layers': capture_layers, 'd_mlp': d_mlp,
        't_values': t_values, 'rel_distances': rel_distances,
        'thresholds': thresholds,
        'perstep': perstep, 'cumulative': cumulative,
        'last_token_angle_deg': np.degrees(angle),
    }


def plot_pair(res, model_name, save_path):
    """Per-step total flip COUNT vs t (left axis, with fraction on a secondary
    left axis) overlaid with logit rel-distance (right axis)."""
    taus = res['thresholds']
    t = res['t_values']
    d_mlp = res['d_mlp']
    n_layers = len(res['capture_layers'])
    frac_to_count = d_mlp                      # all captured layers share d_mlp
    max_count = d_mlp * n_layers
    fig, axes = plt.subplots(1, len(taus), figsize=(7.5 * len(taus), 5), squeeze=False)
    for k, tau in enumerate(taus):
        ax = axes[0][k]
        perstep_frac = res['perstep'][tau].sum(axis=1)   # summed over layers
        perstep_count = perstep_frac * frac_to_count
        ax.plot(t, perstep_count, color='C0', lw=2)
        ax.set_xlabel('Interpolation t')
        ax.set_ylabel(f'Per-step flips (count, summed over {n_layers} layers)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(perstep_count.max() * 1.1, 1))
        ax.grid(True, alpha=0.3)

        # Secondary left axis: same data as a fraction
        sec = ax.secondary_yaxis(
            -0.16, functions=(lambda c: c / frac_to_count,
                              lambda f: f * frac_to_count))
        sec.set_ylabel('Per-step flip fraction (summed over layers)', color='C0')
        sec.tick_params(axis='y', labelcolor='C0')

        ax2 = ax.twinx()
        ax2.plot(t, res['rel_distances'], color='C3', lw=2, ls='--',
                 label='logit rel-distance')
        ax2.plot([0, 1], [0, 1], color='gray', lw=1, ls=':', alpha=0.6)
        ax2.set_ylabel('Logit relative distance', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        ax2.set_ylim(-0.02, 1.02)

        ax.set_title(f"tau={tau}  (max possible = {max_count} flips)")
    fig.suptitle(f"{model_name}  L{res['layer']}->{res['measure_layer']}  "
                 f"{res['prompt_a']!r} -> {res['prompt_b']!r}", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved plot: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-0.6b')
    p.add_argument('--layer', type=int, default=14, help='inject layer L')
    p.add_argument('--measure_layer', type=int, default=None,
                   help='last layer to count flips at (default: final layer)')
    p.add_argument('--prefix', default='The house was')
    p.add_argument('--anchor', default='big', help='final word at t=0')
    p.add_argument('--arms', nargs='+', default=['large', 'in'],
                   help='final words to interpolate the anchor toward')
    p.add_argument('--n_steps', type=int, default=300)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.1])
    p.add_argument('--save_dir', default='results/interpolation_flips')
    args = p.parse_args()

    model, device = load_model(args.model)
    measure_layer = args.measure_layer if args.measure_layer is not None else model.cfg.n_layers - 1
    if measure_layer <= args.layer:
        raise ValueError(f"measure_layer ({measure_layer}) must be > inject layer ({args.layer})")
    print(f"Inject L={args.layer}, measure up to L={measure_layer} "
          f"(of {model.cfg.n_layers}), thresholds={args.thresholds}")

    model_short = args.model.replace('/', '_')
    prompt_a = f"{args.prefix} {args.anchor}"
    save_dir = Path(args.save_dir)

    for arm in args.arms:
        prompt_b = f"{args.prefix} {arm}"
        res = run_pair(model, prompt_a, prompt_b, args.layer, measure_layer,
                       args.n_steps, args.thresholds)
        res['model'] = args.model
        tag = f"{args.anchor}_to_{arm}_{model_short}_L{args.layer}"
        torch.save(res, save_dir / f"flips_{tag}.pt")
        print(f"  saved data: {save_dir / f'flips_{tag}.pt'}")
        plot_pair(res, args.model, str(save_dir / f"flips_{tag}.png"))

    print("\nDone.")


if __name__ == '__main__':
    main()
