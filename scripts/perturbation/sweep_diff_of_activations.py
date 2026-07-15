#!/usr/bin/env python3
"""
Difference-of-activations perturbation sweep.

Idea (Francisco): take two prompts that share a prefix but differ in the last
token -- "The house was big" (A) and "The house was in" (B) -- and read off their
last-token residual-stream activations at an early layer L (default L2 of gpt2-xl).
The difference  d = a_B - a_A  is a concrete, interpretable direction in activation
space (it carries the model from the "big" completion toward the "in" completion).

We then perturb STARTING from a_A, stepping along d, and measure the *shape* of the
downstream response: L2 distance of the second-to-last-layer residual stream between
the perturbed and unperturbed forward passes, as a function of step size.

    a_perturbed = a_A + m * d_hat          (d_hat = d / ||d||)
    response(m) = ||resid_measure(a_perturbed) - resid_measure(a_A)||_2

The step size is parameterised as  m = alpha * ||d||,  so alpha=1 lands exactly on
a_B. We sweep alpha over [0, alpha_max].

Conditions (all use the SAME additive step sizes m, so they are compared at equal
input-space perturbation magnitude):
  - diff:   the single difference direction d_hat
  - random: n_dirs isotropic unit directions (median + IQR band)
  - contrastive (optional, --contrastive_pkl): named diff-of-means directions, only
    meaningful if computed at the SAME layer L. (gpt2-xl contrastive dirs we have are
    at L24, so this is off by default.)

Usage:
    python scripts/perturbation/sweep_diff_of_activations.py --model gpt2-xl --layer 2
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head, _get_blocks,
)


def get_prompt_activation(model, tokenizer, prompt, layer, device):
    """Return (context, activation) at `layer` for a text prompt.

    context:    [seq_len-1, d_model] prefix residual stream (last-token excluded)
    activation: [d_model]            last-token residual stream
    Uses hidden_states[layer+1] = residual stream AFTER block `layer`.
    """
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    hidden = out.hidden_states[layer + 1][0].float().cpu()  # [seq_len, d_model]
    return hidden[:-1], hidden[-1]


def run_additive_sweep(model, context, a_base, dirs, mags, layer, device,
                       measure_layer):
    """Additive perturbation sweep for one base activation over a set of directions.

    a_perturbed = a_base + m * dir   (dir unit-norm)

    Returns:
        l2_measure: [n_mags, n_dirs]   L2 at measure_layer (primary metric)
        kl_final:   [n_mags, n_dirs]   KL(ref || perturbed) on final logits
    """
    n_mags, n_dirs = len(mags), dirs.shape[0]
    l2_measure = np.zeros((n_mags, n_dirs))
    kl_final = np.zeros((n_mags, n_dirs))

    dtype = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=dtype)   # [1, seq-1, d]
    a = a_base.to(device=device, dtype=dtype)                   # [d]
    D = dirs.to(device=device, dtype=dtype)                     # [n_dirs, d]

    with torch.no_grad():
        ref_pt = a.unsqueeze(0).unsqueeze(0)                    # [1,1,d]
        ref_measure = forward_from_layer_to_layer(model, ctx, ref_pt, layer, measure_layer)
        ref_final = forward_from_layer_to_final(model, ctx, ref_pt, layer)
        ref_lp = F.log_softmax(_get_lm_head(model)(
            _get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)).float(), dim=-1)
        ref_p = ref_lp.exp()

        for mi, m in enumerate(mags):
            perturbed = (a.unsqueeze(0) + m * D).unsqueeze(1)   # [n_dirs, 1, d]
            pert_measure = forward_from_layer_to_layer(model, ctx, perturbed, layer, measure_layer)
            l2_measure[mi] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()

            pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)
            pert_lp = F.log_softmax(_get_lm_head(model)(
                _get_final_norm(model)(pert_final.unsqueeze(1)).squeeze(1)).float(), dim=-1)
            kl_final[mi] = (ref_p * (ref_lp - pert_lp)).sum(dim=-1).cpu().numpy()

    return l2_measure, kl_final


def load_contrastive_directions(path, layer, keys=None):
    """Load named diff-of-means directions from a pkl; warn if layer mismatches.

    keys: optional list of feature names to keep (default: all).
    """
    d = pickle.load(open(path, 'rb'))
    if d.get('layer') != layer:
        print(f"  WARNING: {path} is layer {d.get('layer')}, perturbing at layer {layer} "
              f"-- contrastive comparison is NOT layer-matched.")
    vecs, names = [], []
    for name, v in d['features'].items():
        if keys is not None and name not in keys:
            continue
        v = torch.as_tensor(v).float()
        vecs.append(v / v.norm())
        names.append(name)
    if not vecs:
        raise ValueError(f"No contrastive features matched keys={keys} in {path}")
    return torch.stack(vecs), names


def main():
    p = argparse.ArgumentParser(description='Difference-of-activations perturbation sweep')
    p.add_argument('--model', type=str, default='gpt2-xl')
    p.add_argument('--layer', type=int, default=2, help='injection layer (0-indexed)')
    p.add_argument('--measure_layer', type=int, default=-2,
                   help='layer to read L2 response at (default -2 = 2nd-to-last)')
    p.add_argument('--prompt_a', type=str, default='The house was big',
                   help='base prompt (sweep starts here)')
    p.add_argument('--prompt_b', type=str, default='The house was in',
                   help='contrast prompt (difference direction points toward this)')
    p.add_argument('--n_dirs', type=int, default=20, help='random directions')
    p.add_argument('--alpha_max', type=float, default=4.0,
                   help='max step size in units of ||a_B - a_A|| (alpha=1 reaches a_B)')
    p.add_argument('--n_steps', type=int, default=80)
    p.add_argument('--contrastive_pkl', type=str, default=None,
                   help='optional pkl of named diff-of-means dirs (must be same layer)')
    p.add_argument('--contrastive_keys', type=str, default=None,
                   help='comma-separated subset of feature names to keep (default: all)')
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.results_dir is None:
        args.results_dir = f'results/diff_of_activations/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    print("Config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 60)

    print(f"\nLoading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer >= 0 else n_layers + args.measure_layer
    print(f"device={device}, d_model={d_model}, n_layers={n_layers}, measure_layer={measure_layer}")

    # Activations for both prompts at the injection layer.
    ctx_a, a_A = get_prompt_activation(model, tokenizer, args.prompt_a, args.layer, device)
    ctx_b, a_B = get_prompt_activation(model, tokenizer, args.prompt_b, args.layer, device)
    print(f"\n'{args.prompt_a}' tokens: {tokenizer(args.prompt_a).input_ids}")
    print(f"'{args.prompt_b}' tokens: {tokenizer(args.prompt_b).input_ids}")

    d_vec = a_B - a_A
    d_norm = d_vec.norm().item()
    a_norm = a_A.norm().item()
    d_hat = d_vec / d_norm
    cos_d_a = F.cosine_similarity(d_hat, a_A / a_norm, dim=0).item()
    print(f"||a_A||={a_norm:.2f}  ||d||={d_norm:.2f}  cos(d_hat, a_A)={cos_d_a:.3f}")

    # Step sizes shared across all conditions: m = alpha * ||d||.
    alphas = np.linspace(0, args.alpha_max, args.n_steps + 1)[1:]  # skip 0
    mags = alphas * d_norm

    # Random unit directions (apples-to-apples: raw isotropic, not orthogonalised).
    rand_dirs = F.normalize(torch.randn(args.n_dirs, d_model), dim=-1)

    print(f"\n=== diff direction sweep ({args.n_steps} steps) ===")
    t0 = time.time()
    diff_l2, diff_kl = run_additive_sweep(
        model, ctx_a, a_A, d_hat.unsqueeze(0), mags, args.layer, device, measure_layer)
    print(f"  diff done ({time.time()-t0:.1f}s)")

    print(f"=== random directions sweep (n_dirs={args.n_dirs}) ===")
    t0 = time.time()
    rand_l2, rand_kl = run_additive_sweep(
        model, ctx_a, a_A, rand_dirs, mags, args.layer, device, measure_layer)
    print(f"  random done ({time.time()-t0:.1f}s)")

    results = {
        'args': vars(args), 'alphas': alphas, 'mags': mags,
        'd_norm': d_norm, 'a_norm': a_norm, 'cos_d_a': cos_d_a,
        'measure_layer': measure_layer, 'n_layers': n_layers,
        'diff': {'l2': diff_l2, 'kl': diff_kl},
        'random': {'l2': rand_l2, 'kl': rand_kl},
    }

    # Optional layer-matched contrastive directions.
    contrastive = None
    if args.contrastive_pkl:
        print(f"=== contrastive sweep ({args.contrastive_pkl}) ===")
        keys = args.contrastive_keys.split(',') if args.contrastive_keys else None
        cdirs, cnames = load_contrastive_directions(args.contrastive_pkl, args.layer, keys)
        c_l2, c_kl = run_additive_sweep(
            model, ctx_a, a_A, cdirs, mags, args.layer, device, measure_layer)
        contrastive = {'l2': c_l2, 'kl': c_kl, 'names': cnames}
        results['contrastive'] = contrastive

    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- Plot: L2 @ measure_layer and KL @ final vs alpha ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, name in [(axes[0], 'l2', f'L2 @ layer {measure_layer}'),
                          (axes[1], 'kl', 'KL @ final')]:
        ax.plot(alphas, results['diff'][key][:, 0], color='#EE6677', lw=2.5,
                label='diff (a_B - a_A)', zorder=3)
        rand = results['random'][key]                       # [n_steps, n_dirs]
        med = np.median(rand, axis=1)
        q25, q75 = np.percentile(rand, 25, axis=1), np.percentile(rand, 75, axis=1)
        ax.plot(alphas, med, color='#4477AA', lw=2, label='random (median)')
        ax.fill_between(alphas, q25, q75, color='#4477AA', alpha=0.18)
        if contrastive is not None:
            for j, nm in enumerate(contrastive['names']):
                ax.plot(alphas, contrastive[key][:, j], lw=1.2, alpha=0.7, label=f'contr:{nm}')
        ax.axvline(1.0, color='gray', ls=':', lw=1.5, label='alpha=1 (reaches a_B)')
        ax.set_xlabel('step size alpha  (m = alpha * ||a_B - a_A||)')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        # twin axis: absolute perturbation magnitude
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = [a for a in [0, 1, 2, 3, 4] if a <= args.alpha_max]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f'{a*d_norm:.0f}' for a in ticks])
        ax2.set_xlabel('perturbation magnitude ||m * d_hat||')

    fig.suptitle(f"{args.model} L{args.layer}: diff-of-activations vs random  "
                 f"('{args.prompt_a}' -> '{args.prompt_b}')  "
                 f"||d||={d_norm:.1f}, ||a||={a_norm:.1f}, cos={cos_d_a:.2f}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {path}")
    print(f"Saved data: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
