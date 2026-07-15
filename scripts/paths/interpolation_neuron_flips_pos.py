#!/usr/bin/env python3
"""
Position-aware interpolation neuron-flip experiment.

For two same-length prompts that differ at one (or a few) positions -- e.g.
"The capital of France is" vs "The capital of Germany is" -- the distinction
lives at the differing token, NOT the last token. Interpolating only the last
token does nothing (the held-fixed prefix keeps driving the answer). So here we
interpolate the layer-L residual at *every position where the two prompts'
residuals differ*, slerp each independently, and read the last-token prediction.

Because we interpolate all differing positions, the endpoints are the true
prompts: t=0 -> prompt A's real prediction, t=1 -> prompt B's. So the
relative-distance interpolation curve is meaningful, and we can ask whether the
prediction transitions A->B sharply (boundary) or smoothly, and whether neuron
flips (at the last token, across downstream MLP layers) peak at a boundary.

Usage:
    python scripts/paths/interpolation_neuron_flips_pos.py --model qwen3-0.6b \
        --layer 14 --n_steps 300 \
        --prompt_a "The capital of France is" --prompt_b "The capital of Germany is" \
        --targets " Paris" " Berlin" --tag france_germany_pos
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


def full_act(model, prompt, layer):
    _, cache = model.run_with_cache(prompt)
    return cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)   # [seq, d_model]


def slerp_lm(v0, v1, t):
    n0, n1 = v0.norm(), v1.norm()
    u0, u1 = v0 / n0, v1 / n1
    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    th = torch.acos(dot)
    if th.abs() < 1e-6:
        d = u0
    else:
        st = torch.sin(th)
        d = (torch.sin((1 - t) * th) / st) * u0 + (torch.sin(t * th) / st) * u1
    return d * ((1 - t) * n0 + t * n1)


def forward_capture(model, ctx_prompt, inj_act, layer, capture_layers):
    """Inject full residual `inj_act` ([seq,d]) at `layer`; capture last-token
    logits and last-token MLP pre-activations at capture_layers."""
    inj_hook = f"blocks.{layer}.hook_resid_post"
    full = inj_act.unsqueeze(0)

    def injection_hook(activation, hook):
        activation[:, :, :] = full
        return activation

    gates = {}

    def make_gate_hook(l):
        def gh(activation, hook):
            gates[l] = activation[0, -1, :].detach()
            return activation
        return gh

    fwd_hooks = [(inj_hook, injection_hook)]
    fwd_hooks += [(f"blocks.{l}.mlp.hook_pre", make_gate_hook(l)) for l in capture_layers]
    with torch.no_grad():
        logits = model.run_with_hooks(ctx_prompt, fwd_hooks=fwd_hooks)
    return logits[0, -1, :], gates


def topk_str(model, lp, k=8):
    p = lp.exp(); tp, ti = torch.topk(p, k)
    return ' '.join(f"{model.to_string(i.item())!r}:{v.item():.3f}" for v, i in zip(tp, ti))


def run(model, prompt_a, prompt_b, layer, measure_layer, n_steps, thresholds, target_ids, target_strs):
    print(f"\n=== {prompt_a!r} -> {prompt_b!r} ===")
    A = full_act(model, prompt_a, layer)
    B = full_act(model, prompt_b, layer)
    if A.shape != B.shape:
        raise ValueError(f"Prompts must be same token length: {A.shape} vs {B.shape}")
    seq = A.shape[0]
    # positions where the layer-L residuals differ
    pos_diff = (A - B).norm(dim=-1)
    diff_positions = [i for i in range(seq) if pos_diff[i] > 0.05 * A[i].norm()]
    toks_a = model.to_tokens(prompt_a)[0]
    print(f"  seq_len={seq}, interpolating positions {diff_positions} "
          f"(tokens { [model.to_string(toks_a[i].item()) for i in diff_positions] })")

    capture_layers = list(range(layer + 1, measure_layer + 1))
    d_mlp = model.cfg.d_mlp

    logits_A, gates_A = forward_capture(model, prompt_a, A, layer, capture_layers)
    logits_B, _ = forward_capture(model, prompt_a, B, layer, capture_layers)
    lpA = F.log_softmax(logits_A.float(), dim=-1)
    print(f"  endpoint A (t=0): {topk_str(model, lpA)}")
    print(f"  endpoint B (t=1): {topk_str(model, F.log_softmax(logits_B.float(), dim=-1))}")

    t_values = np.linspace(0, 1, n_steps)
    rel_distances = np.zeros(n_steps)
    kl_from_A = np.zeros(n_steps)
    p_targets = {s: np.zeros(n_steps) for s in target_strs}

    n_cap = len(capture_layers)
    perstep = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}
    cumulative = {tau: np.zeros((n_steps, n_cap)) for tau in thresholds}
    ref_bin = {tau: {l: (gates_A[l] > tau) for l in capture_layers} for tau in thresholds}
    prev_bin = {tau: {l: ref_bin[tau][l].clone() for l in capture_layers} for tau in thresholds}

    pA = lpA.exp()
    for i, t in enumerate(t_values):
        inj = A.clone()
        for pos in diff_positions:
            inj[pos] = slerp_lm(A[pos], B[pos], float(t))
        logits, gates = forward_capture(model, prompt_a, inj, layer, capture_layers)
        lp = F.log_softmax(logits.float(), dim=-1)
        dA = (logits - logits_A).norm().item()
        dB = (logits - logits_B).norm().item()
        rel_distances[i] = dA / (dA + dB + 1e-10)
        kl_from_A[i] = (pA * (lpA - lp)).sum().item()
        for s, tid in zip(target_strs, target_ids):
            p_targets[s][i] = lp[tid].exp().item()
        for tau in thresholds:
            for j, l in enumerate(capture_layers):
                cur = gates[l] > tau
                cumulative[tau][i, j] = (cur != ref_bin[tau][l]).float().mean().item()
                perstep[tau][i, j] = (cur != prev_bin[tau][l]).float().mean().item()
                prev_bin[tau][l] = cur
        if (i + 1) % 50 == 0:
            ps = ' '.join(f"P({s.strip()})={p_targets[s][i]:.2f}" for s in target_strs)
            print(f"  step {i+1}/{n_steps}: t={t:.3f} {ps} relD={rel_distances[i]:.3f} "
                  f"flips/step={perstep[0.0][i].sum()*d_mlp:.0f}")

    return {
        'prompt_a': prompt_a, 'prompt_b': prompt_b, 'layer': layer,
        'measure_layer': measure_layer, 'capture_layers': capture_layers, 'd_mlp': d_mlp,
        'diff_positions': diff_positions,
        't_values': t_values, 'rel_distances': rel_distances, 'kl_from_A': kl_from_A,
        'p_targets': p_targets, 'target_strs': target_strs,
        'thresholds': thresholds, 'perstep': perstep, 'cumulative': cumulative,
    }


def plot_all(res, model_name, save_stem):
    t = res['t_values']; d_mlp = res['d_mlp']

    # 1) classic interpolation curve
    plt.figure(figsize=(7, 5))
    plt.plot(t, res['rel_distances'], 'b-', lw=2, label='Observed')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Linear (d=t)')
    plt.xlabel('Interpolation t'); plt.ylabel('Relative distance d')
    plt.title(f"{res['prompt_a']!r} -> {res['prompt_b']!r}\n{model_name} L{res['layer']}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
    plt.tight_layout(); plt.savefig(f"{save_stem}_interp.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {save_stem}_interp.png")

    # 2) flips + target probs
    fig, axes = plt.subplots(1, len(res['thresholds']), figsize=(8 * len(res['thresholds']), 5), squeeze=False)
    for k, tau in enumerate(res['thresholds']):
        ax = axes[0][k]
        pc = res['perstep'][tau].sum(axis=1) * d_mlp
        ax.plot(t, pc, color='C0', lw=2)
        ax.set_xlabel('Interpolation t')
        ax.set_ylabel(f'Per-step flips (count, {len(res["capture_layers"])} layers)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0'); ax.set_ylim(0, max(pc.max() * 1.1, 1))
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        for s in res['target_strs']:
            ax2.plot(t, res['p_targets'][s], lw=2, label=f'P({s.strip()})')
        ax2.plot(t, res['rel_distances'], color='gray', lw=1.2, ls=':', label='rel-dist')
        ax2.set_ylabel('P(target) / rel-dist'); ax2.set_ylim(-0.02, 1.02)
        ax2.legend(loc='center right', fontsize=8)
        ax.set_title(f"tau={tau}")
    fig.suptitle(f"{model_name} L{res['layer']}->{res['measure_layer']}  "
                 f"{res['prompt_a']!r} -> {res['prompt_b']!r}", y=1.02)
    plt.tight_layout(); plt.savefig(f"{save_stem}_flips.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {save_stem}_flips.png")

    # 3) heatmap (tau=0)
    layers = res['capture_layers']
    counts = res['perstep'][0.0] * d_mlp
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    axm = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)
    im = axm.imshow(counts.T, aspect='auto', origin='lower', cmap='magma',
                    extent=[t[0], t[-1], layers[0] - 0.5, layers[-1] + 0.5])
    axm.set_xlabel('Interpolation t'); axm.set_ylabel('Layer (MLP)'); axm.set_yticks(layers)
    axt.plot(t, counts.sum(axis=1), color='C0', lw=1.8); axt.set_ylabel('flips/step', fontsize=9, color='C0')
    axt.tick_params(axis='x', labelbottom=False); axt.grid(True, alpha=0.3)
    axtr = axt.twinx()
    for s in res['target_strs']:
        axtr.plot(t, res['p_targets'][s], lw=1.4, label=f'P({s.strip()})')
    axtr.set_ylim(-0.02, 1.02); axtr.legend(fontsize=7, loc='center right')
    axr.plot(counts.sum(axis=0), layers, color='C0', lw=1.8)
    axr.fill_betweenx(layers, 0, counts.sum(axis=0), color='C0', alpha=0.25)
    axr.set_xlabel('total flips', fontsize=9); axr.tick_params(axis='y', labelleft=False)
    fig.colorbar(im, ax=axr, fraction=0.15, pad=0.35).set_label('flips per step', fontsize=9)
    axt.set_title(f"{res['prompt_a']!r} -> {res['prompt_b']!r}  (tau=0)", fontsize=10)
    plt.savefig(f"{save_stem}_heatmap.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {save_stem}_heatmap.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-0.6b')
    p.add_argument('--layer', type=int, default=14)
    p.add_argument('--measure_layer', type=int, default=None)
    p.add_argument('--prompt_a', required=True)
    p.add_argument('--prompt_b', required=True)
    p.add_argument('--targets', nargs='+', default=[' Paris', ' Berlin'])
    p.add_argument('--tag', required=True)
    p.add_argument('--n_steps', type=int, default=300)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.1])
    p.add_argument('--save_dir', default='results/interpolation_flips')
    args = p.parse_args()

    model, _ = load_model(args.model)
    measure_layer = args.measure_layer if args.measure_layer is not None else model.cfg.n_layers - 1
    target_ids = [model.to_single_token(s) for s in args.targets]
    print(f"Inject L={args.layer}, measure up to L={measure_layer}, targets={args.targets}")

    res = run(model, args.prompt_a, args.prompt_b, args.layer, measure_layer,
              args.n_steps, args.thresholds, target_ids, args.targets)
    res['model'] = args.model
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    stem = str(save_dir / f"posflips_{args.tag}_{args.model.replace('/', '_')}_L{args.layer}")
    torch.save(res, f"{stem}.pt"); print(f"  saved data: {stem}.pt")
    plot_all(res, args.model, stem)
    print("\nDone.")


if __name__ == '__main__':
    main()
