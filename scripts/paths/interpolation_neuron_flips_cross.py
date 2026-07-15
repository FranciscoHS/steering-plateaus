#!/usr/bin/env python3
"""
Cross-context interpolation neuron-flip experiment.

Like interpolation_neuron_flips.py, but the two prompts have *different
contexts* (and may differ in length). We only transplant the single last-token
activation onto a shared context (prompt A's prefix) and slerp between them.

Motivation: test whether a plateau->boundary->plateau structure can appear even
when both endpoints predict the *same* next token (e.g. both -> " Paris") via
different computational pathways (capital-of vs landmark-location). If so, a
boundary can exist without a next-token-prediction change.

Because both endpoints predict ~the same token, the A-vs-B relative-distance
metric is degenerate. Instead we track, along the path:
  - P(target token)         (e.g. P(" Paris")) -- dips if the path leaves the basin
  - KL(point || endpoint A) -- distance from the start distribution
  - per-step / cumulative neuron flips at MLP layers L+1..measure_layer

Usage:
    python scripts/paths/interpolation_neuron_flips_cross.py --model qwen3-0.6b \
        --layer 14 --n_steps 300 --target " Paris" \
        --prompt_a "The capital of France is" --prompt_b "The Louvre museum is located in" \
        --tag capital_to_louvre
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from transformer_lens import HookedTransformer


def load_model(model_name, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}, d_mlp={model.cfg.d_mlp}")
    return model, device


def get_last_act(model, prompt, layer):
    """Last-token residual at `layer` (computed in the prompt's own context)."""
    _, cache = model.run_with_cache(prompt)
    return cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)[-1, :]   # [d_model]


def get_prefix_act(model, prompt, layer):
    """All-but-last-token residual at `layer` (the shared context)."""
    _, cache = model.run_with_cache(prompt)
    return cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)[:-1, :]  # [seq-1, d_model]


def slerp_with_linear_magnitude(v0, v1, t):
    norm0, norm1 = v0.norm(), v1.norm()
    u0, u1 = v0 / norm0, v1 / norm1
    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        direction = u0
    else:
        st = torch.sin(theta)
        direction = (torch.sin((1 - t) * theta) / st) * u0 + (torch.sin(t * theta) / st) * u1
    return direction * ((1 - t) * norm0 + t * norm1)


def forward_capture(model, ctx_prompt, prefix_act, last_act, layer, capture_layers):
    """Inject (prefix_act + last_act) at `layer` of ctx_prompt; capture final
    last-token logits and MLP pre-activations (last token) at capture_layers."""
    inj_hook = f"blocks.{layer}.hook_resid_post"
    full_act = torch.cat([prefix_act, last_act.unsqueeze(0)], dim=0).unsqueeze(0)

    def injection_hook(activation, hook):
        activation[:, :, :] = full_act
        return activation

    gates = {}

    def make_gate_hook(l):
        def gate_hook(activation, hook):
            gates[l] = activation[0, -1, :].detach()
            return activation
        return gate_hook

    fwd_hooks = [(inj_hook, injection_hook)]
    fwd_hooks += [(f"blocks.{l}.mlp.hook_pre", make_gate_hook(l)) for l in capture_layers]
    with torch.no_grad():
        logits = model.run_with_hooks(ctx_prompt, fwd_hooks=fwd_hooks)
    return logits[0, -1, :], gates


def topk_str(model, log_probs, k=8):
    p = log_probs.exp()
    tp, ti = torch.topk(p, k)
    return ' '.join(f"{model.to_string(i.item())!r}:{v.item():.3f}" for v, i in zip(tp, ti))


def run_pair(model, prompt_a, prompt_b, layer, measure_layer, n_steps, thresholds, target_id):
    print(f"\n=== {prompt_a!r} -> {prompt_b!r} ===")
    prefix_a = get_prefix_act(model, prompt_a, layer)     # shared context
    last_A = get_last_act(model, prompt_a, layer)
    last_B = get_last_act(model, prompt_b, layer)
    angle = torch.acos(torch.clamp(
        torch.dot(last_A / last_A.norm(), last_B / last_B.norm()), -1.0, 1.0)).item()
    print(f"  last-token angle: {np.degrees(angle):.2f} deg, ||A||={last_A.norm():.2f}, ||B||={last_B.norm():.2f}")

    capture_layers = list(range(layer + 1, measure_layer + 1))
    d_mlp = model.cfg.d_mlp

    logits_A, gates_A = forward_capture(model, prompt_a, prefix_a, last_A, layer, capture_layers)
    logits_B, _ = forward_capture(model, prompt_a, prefix_a, last_B, layer, capture_layers)
    lpA = F.log_softmax(logits_A.float(), dim=-1)
    print(f"  endpoint A (t=0): {topk_str(model, lpA)}")
    print(f"  endpoint B (t=1, transplanted into A ctx): {topk_str(model, F.log_softmax(logits_B.float(), dim=-1))}")

    t_values = np.linspace(0, 1, n_steps)
    rel_distances = np.zeros(n_steps)        # kept for reference (may be degenerate)
    kl_from_A = np.zeros(n_steps)
    p_target = np.zeros(n_steps)

    n_cap = len(capture_layers)
    perstep = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}
    cumulative = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}
    ref_bin = {tau: {l: (gates_A[l] > tau) for l in capture_layers} for tau in thresholds}
    prev_bin = {tau: {l: ref_bin[tau][l].clone() for l in capture_layers} for tau in thresholds}

    pA = lpA.exp()
    for i, t in enumerate(t_values):
        interp = slerp_with_linear_magnitude(last_A, last_B, float(t))
        logits, gates = forward_capture(model, prompt_a, prefix_a, interp, layer, capture_layers)
        lp = F.log_softmax(logits.float(), dim=-1)
        dA = (logits - logits_A).norm().item()
        dB = (logits - logits_B).norm().item()
        rel_distances[i] = dA / (dA + dB + 1e-10)
        kl_from_A[i] = (pA * (lpA - lp)).sum().item()
        p_target[i] = lp[target_id].exp().item()
        for tau in thresholds:
            for j, l in enumerate(capture_layers):
                cur = gates[l] > tau
                cumulative[tau][i, j] = (cur != ref_bin[tau][l]).float().mean().item()
                perstep[tau][i, j] = (cur != prev_bin[tau][l]).float().mean().item()
                prev_bin[tau][l] = cur
        if (i + 1) % 50 == 0:
            print(f"  step {i+1}/{n_steps}: t={t:.3f} P(target)={p_target[i]:.3f} "
                  f"KL_from_A={kl_from_A[i]:.3f} flips/step(t0)={perstep[0.0][i].sum()*d_mlp:.0f}")

    return {
        'prompt_a': prompt_a, 'prompt_b': prompt_b, 'layer': layer,
        'measure_layer': measure_layer, 'capture_layers': capture_layers, 'd_mlp': d_mlp,
        't_values': t_values, 'rel_distances': rel_distances,
        'kl_from_A': kl_from_A, 'p_target': p_target,
        'thresholds': thresholds, 'perstep': perstep, 'cumulative': cumulative,
        'last_token_angle_deg': np.degrees(angle), 'target_id': target_id,
    }


def plot_line(res, model_name, target_str, save_path):
    """Per-step flip count (left) overlaid with P(target) and KL-from-A (right)."""
    taus = res['thresholds']
    t = res['t_values']
    d_mlp = res['d_mlp']
    fig, axes = plt.subplots(1, len(taus), figsize=(8 * len(taus), 5), squeeze=False)
    for k, tau in enumerate(taus):
        ax = axes[0][k]
        perstep_count = res['perstep'][tau].sum(axis=1) * d_mlp
        ax.plot(t, perstep_count, color='C0', lw=2)
        ax.set_xlabel('Interpolation t')
        ax.set_ylabel(f'Per-step flips (count, {len(res["capture_layers"])} layers)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(perstep_count.max() * 1.1, 1))
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(t, res['p_target'], color='C2', lw=2, label=f'P({target_str!r})')
        ax2.plot(t, res['kl_from_A'] / max(res['kl_from_A'].max(), 1e-9), color='C3', lw=1.5,
                 ls='--', label='KL from A (norm.)')
        ax2.set_ylabel(f'P(target) / normalized KL-from-A', color='C2')
        ax2.tick_params(axis='y', labelcolor='C2')
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(loc='center right', fontsize=8)
        ax.set_title(f"tau={tau}")
    fig.suptitle(f"{model_name}  L{res['layer']}->{res['measure_layer']}  "
                 f"{res['prompt_a']!r} -> {res['prompt_b']!r}", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def plot_heatmap(res, tau, target_str, save_path):
    t = res['t_values']; layers = res['capture_layers']; d_mlp = res['d_mlp']
    counts = res['perstep'][tau] * d_mlp
    top_marg = counts.sum(axis=1); right_marg = counts.sum(axis=0)
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    im = ax_main.imshow(counts.T, aspect='auto', origin='lower', cmap='magma',
                        extent=[t[0], t[-1], layers[0] - 0.5, layers[-1] + 0.5])
    ax_main.set_xlabel('Interpolation t'); ax_main.set_ylabel('Layer (MLP)')
    ax_main.set_yticks(layers)
    ax_top.plot(t, top_marg, color='C0', lw=1.8)
    ax_top.set_ylabel('flips/step', fontsize=9, color='C0')
    ax_top.tick_params(axis='y', labelcolor='C0', labelsize=8); ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, alpha=0.3)
    ax_topr = ax_top.twinx()
    ax_topr.plot(t, res['p_target'], color='C2', lw=1.5)
    ax_topr.set_ylabel(f'P({target_str!r})', fontsize=9, color='C2')
    ax_topr.tick_params(axis='y', labelcolor='C2', labelsize=8); ax_topr.set_ylim(-0.02, 1.02)
    ax_right.plot(right_marg, layers, color='C0', lw=1.8)
    ax_right.fill_betweenx(layers, 0, right_marg, color='C0', alpha=0.25)
    ax_right.set_xlabel('total flips', fontsize=9); ax_right.tick_params(axis='y', labelleft=False)
    fig.colorbar(im, ax=ax_right, fraction=0.15, pad=0.35).set_label('flips per step', fontsize=9)
    ax_top.set_title(f"{res['prompt_a']!r} -> {res['prompt_b']!r}  (tau={tau})", fontsize=10)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-0.6b')
    p.add_argument('--layer', type=int, default=14)
    p.add_argument('--measure_layer', type=int, default=None)
    p.add_argument('--prompt_a', required=True)
    p.add_argument('--prompt_b', required=True)
    p.add_argument('--target', default=' Paris', help='token whose prob to track along the path')
    p.add_argument('--tag', required=True, help='filename tag, e.g. capital_to_louvre')
    p.add_argument('--n_steps', type=int, default=300)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.1])
    p.add_argument('--save_dir', default='results/interpolation_flips')
    args = p.parse_args()

    model, device = load_model(args.model)
    measure_layer = args.measure_layer if args.measure_layer is not None else model.cfg.n_layers - 1
    target_id = model.to_single_token(args.target)
    print(f"Inject L={args.layer}, measure up to L={measure_layer}, target={args.target!r} (id {target_id})")

    res = run_pair(model, args.prompt_a, args.prompt_b, args.layer, measure_layer,
                   args.n_steps, args.thresholds, target_id)
    res['model'] = args.model; res['target'] = args.target
    save_dir = Path(args.save_dir)
    model_short = args.model.replace('/', '_')
    stem = f"crossflips_{args.tag}_{model_short}_L{args.layer}"
    torch.save(res, save_dir / f"{stem}.pt")
    print(f"  saved data: {save_dir / f'{stem}.pt'}")
    plot_line(res, args.model, args.target, str(save_dir / f"{stem}.png"))
    plot_heatmap(res, 0.0, args.target, str(save_dir / f"{stem}_heatmap.png"))
    print("\nDone.")


if __name__ == '__main__':
    main()
