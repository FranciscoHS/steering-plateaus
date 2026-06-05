#!/usr/bin/env python3
"""
Learn a superellipse direction dictionary.

Discover unit directions in activation space whose downstream response geometry
is super-additive in the superellipse sense: the model is *less* responsive
along the 50/50 mixture of two directions than along either direction alone
(the p>2 regime of R(v) = F(sum_i w_i |d_i . v|^n)).

We optimize directions on the constant-norm sphere to minimize, per activation,
    L_a = R_mix / (R_1 + R_2)
where R_i is the downstream L2 response to a norm-matched spherical perturbation
along direction d_i (measured at a chosen layer, default 2nd-to-last). A soft
penalty lambda * (d_1 . d_2)^2 encourages approximate orthogonality.

Stage 1 learns the first pair (d1, d2) jointly. Stage 2+ grows the dictionary:
each new direction d_k is optimized against the frozen existing directions,
minimizing the average pair-loss L(d_k, d_j) over j < k.

Usage:
    python scripts/superellipse/learn_superellipse_dict.py \
        --model qwen3-1.7b --layer 2 --n_act 10 --n_dict 2 --angles 30
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
    load_model, forward_from_layer_to_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length


def extract_context_and_activation(model, token_ids_list, layer, device, batch_size=64):
    """Final-token activation + prefix context at `layer` (HF output_hidden_states)."""
    all_contexts = []
    all_activations = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
    return torch.cat(all_contexts, 0), torch.cat(all_activations, 0)


def tangent_dirs(a, d1, d2):
    """Orthogonalize d1, d2 against a, normalize, and form their bisector.

    Returns [3, d] = (d1_perp, d2_perp, mix), all unit and orthogonal to a.
    Differentiable w.r.t. d1, d2.
    """
    a_hat = a / a.norm()
    d1p = d1 - (d1 @ a_hat) * a_hat
    d1p = d1p / (d1p.norm() + 1e-10)
    d2p = d2 - (d2 @ a_hat) * a_hat
    d2p = d2p / (d2p.norm() + 1e-10)
    mix = d1p + d2p
    mix = mix / (mix.norm() + 1e-10)
    return torch.stack([d1p, d2p, mix])  # [3, d]


def pair_responses(model, a, ctx, f_ref, d1, d2, angle_rad, layer, measure_layer):
    """Downstream L2 responses (R1, R2, R_mix) for one activation at one angle.

    a: [d], ctx: [1, seq-1, d], f_ref: [d] (precomputed unperturbed response).
    Differentiable w.r.t. d1, d2.
    """
    # a, d1, d2, f_ref are float32; do the geometry in float32 and only cast
    # to the model dtype at the forward boundary (handles bf16 models).
    model_dtype = next(model.parameters()).dtype
    R = a.norm()
    td = tangent_dirs(a, d1, d2)  # [3, d], unit, orthogonal to a
    perturbed = np.cos(angle_rad) * a.unsqueeze(0) + np.sin(angle_rad) * R * td  # [3, d]
    point = perturbed.unsqueeze(1).to(model_dtype)  # [3, 1, d]
    f_pert = forward_from_layer_to_layer(model, ctx, point, layer, measure_layer).float()  # [3, d]
    responses = (f_pert - f_ref.unsqueeze(0)).norm(dim=-1)  # [3]
    return responses[0], responses[1], responses[2]


def pair_loss(model, acts, ctxs, f_refs, d1, d2, angles_rad, layer, measure_layer):
    """Mean over activations & angles of L_a = R_mix / (R1 + R2).

    Also returns mean R1, R2, R_mix (detached) for logging.
    """
    n_act = len(acts)
    ratio_sum = 0.0
    r1_sum = r2_sum = rmix_sum = 0.0
    count = 0
    for i in range(n_act):
        for angle_rad in angles_rad:
            r1, r2, rmix = pair_responses(
                model, acts[i], ctxs[i], f_refs[i], d1, d2,
                angle_rad, layer, measure_layer)
            ratio_sum = ratio_sum + rmix / (r1 + r2 + 1e-8)
            r1_sum += r1.item(); r2_sum += r2.item(); rmix_sum += rmix.item()
            count += 1
    loss = ratio_sum / count
    return loss, (r1_sum / count, r2_sum / count, rmix_sum / count)


def optimize(model, acts, ctxs, f_refs, fixed_dirs, n_new, angles_rad,
             layer, measure_layer, device, d_model, n_steps, lr, lam, seed,
             log_prefix=""):
    """Optimize new direction(s) on the sphere.

    fixed_dirs: list of frozen unit tensors (existing dictionary).
    n_new: 1 (grow) or 2 (initial pair). When 2, both are optimized jointly
        and paired against each other; when 1, the new direction is paired
        against every fixed direction.

    Returns (new_dirs [n_new, d], loss_history list).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    new = [torch.randn(d_model, generator=g, device=device) for _ in range(n_new)]
    new = [(v / v.norm()).requires_grad_(True) for v in new]
    optimizer = torch.optim.Adam(new, lr=lr)

    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()

        if n_new == 2:
            # Initial pair: optimize d1, d2 against each other.
            d1, d2 = new
            resp_loss, (mr1, mr2, mrmix) = pair_loss(
                model, acts, ctxs, f_refs, d1, d2, angles_rad, layer, measure_layer)
            d1u = d1 / (d1.norm() + 1e-10)
            d2u = d2 / (d2.norm() + 1e-10)
            orth = (d1u @ d2u) ** 2
        else:
            # Grow: pair the single new direction against each fixed direction.
            d_new = new[0]
            resp_terms = []
            mr1 = mr2 = mrmix = 0.0
            for d_fixed in fixed_dirs:
                rl, (a1, a2, amix) = pair_loss(
                    model, acts, ctxs, f_refs, d_new, d_fixed,
                    angles_rad, layer, measure_layer)
                resp_terms.append(rl)
                mr1 += a1; mr2 += a2; mrmix += amix
            resp_loss = torch.stack(resp_terms).mean()
            n_fix = len(fixed_dirs)
            mr1 /= n_fix; mr2 /= n_fix; mrmix /= n_fix
            d_newu = d_new / (d_new.norm() + 1e-10)
            orth = torch.stack([
                (d_newu @ (df / (df.norm() + 1e-10))) ** 2 for df in fixed_dirs
            ]).mean()

        loss = resp_loss + lam * orth
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for v in new:
                v.data.div_(v.data.norm())

        loss_history.append({
            'step': step, 'loss': loss.item(), 'resp_loss': resp_loss.item(),
            'orth': orth.item(), 'R1': mr1, 'R2': mr2, 'Rmix': mrmix,
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  orth={orth.item():.4f}  "
                  f"R1={mr1:.3f} R2={mr2:.3f} Rmix={mrmix:.3f}")

    new_dirs = torch.stack([v.detach() for v in new])
    return new_dirs, loss_history


def eval_dictionary(model, acts, ctxs, f_refs, directions, angles_rad,
                    layer, measure_layer):
    """Per-activation diagnostics for every pair in the dictionary.

    Returns dict with: per-direction mean single-response, per-pair mean L,
    and (for the first pair) the full per-activation R1/R2/Rmix arrays.
    """
    n_dir = directions.shape[0]
    n_act = len(acts)

    # Per-direction single-direction response (mean over acts & angles).
    single_R = np.zeros(n_dir)
    for k in range(n_dir):
        s = 0.0; c = 0
        for i in range(n_act):
            for angle_rad in angles_rad:
                # pair with itself, take R1 only
                with torch.no_grad():
                    r1, _, _ = pair_responses(
                        model, acts[i], ctxs[i], f_refs[i],
                        directions[k], directions[k], angle_rad, layer, measure_layer)
                s += r1.item(); c += 1
        single_R[k] = s / c

    # Per-pair mean L, plus full arrays for the first pair.
    pair_L = np.full((n_dir, n_dir), np.nan)
    first_pair = None
    for i_d in range(n_dir):
        for j_d in range(i_d + 1, n_dir):
            r1s, r2s, rmixs, Las = [], [], [], []
            for i in range(n_act):
                for angle_rad in angles_rad:
                    with torch.no_grad():
                        r1, r2, rmix = pair_responses(
                            model, acts[i], ctxs[i], f_refs[i],
                            directions[i_d], directions[j_d],
                            angle_rad, layer, measure_layer)
                    r1s.append(r1.item()); r2s.append(r2.item())
                    rmixs.append(rmix.item())
                    Las.append(rmix.item() / (r1.item() + r2.item() + 1e-8))
            pair_L[i_d, j_d] = float(np.mean(Las))
            if i_d == 0 and j_d == 1:
                first_pair = {
                    'R1': np.array(r1s), 'R2': np.array(r2s),
                    'Rmix': np.array(rmixs), 'La': np.array(Las),
                }

    gram = (directions @ directions.T).cpu().numpy()
    return {'single_R': single_R, 'pair_L': pair_L,
            'first_pair': first_pair, 'gram': gram}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2,
                        help='Perturbation layer (where directions are applied)')
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer to measure L2 response (default: 2nd-to-last)')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_act', type=int, default=10)
    parser.add_argument('--n_dict', type=int, default=2,
                        help='Target dictionary size (>=2). 2 = first pair only.')
    parser.add_argument('--angles', type=float, nargs='+', default=[30.0],
                        help='Perturbation angle(s) in degrees')
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lam', type=float, default=1.0,
                        help='Weight of the (d_i . d_j)^2 orthogonality penalty')
    parser.add_argument('--data_source', type=str, default='fineweb')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data sampling + global RNG')
    parser.add_argument('--init_seed', type=int, default=None,
                        help='Seed for random direction init only (default: --seed). '
                             'Set differently to vary init while keeping the same anchors.')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_seed = args.init_seed if args.init_seed is not None else args.seed

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/superellipse_dict/'
                            f'{args.model}_L{args.layer}_a{angle_tag}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    model_dtype = dtype_map[args.dtype] if args.dtype else None
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model, dtype=model_dtype)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"perturb_layer: {args.layer}, measure_layer: {measure_layer}")
    print(f"Config: n_act={args.n_act}, n_dict={args.n_dict}, "
          f"angles={args.angles} deg, n_steps={args.n_steps}, lr={args.lr}, "
          f"lam={args.lam}")

    # Data + activations
    print(f"Loading {args.n_act} {args.data_source} sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, token_ids_list, args.layer, device)
    model_dtype_actual = next(model.parameters()).dtype
    # Keep activations/directions in float32 (stable optimization); cast to the
    # model dtype only at forward boundaries.
    acts = [all_activations[i].to(device=device, dtype=torch.float32)
            for i in range(args.n_act)]
    ctxs = [all_contexts[i].unsqueeze(0).to(device=device, dtype=model_dtype_actual)
            for i in range(args.n_act)]
    print(f"Mean activation norm: {torch.stack(acts).norm(dim=-1).mean().item():.2f}")

    # Precompute unperturbed downstream response f(a) for each activation.
    f_refs = []
    with torch.no_grad():
        for i in range(args.n_act):
            point = acts[i].reshape(1, 1, d_model).to(model_dtype_actual)
            f_ref = forward_from_layer_to_layer(
                model, ctxs[i], point, args.layer, measure_layer).squeeze(0).float()
            f_refs.append(f_ref)

    # ---- Stage 1: first pair ----
    print(f"\n=== Stage 1: learning first pair (d1, d2) ===")
    t0 = time.time()
    pair, hist1 = optimize(
        model, acts, ctxs, f_refs, [], 2, angles_rad,
        args.layer, measure_layer, device, d_model,
        args.n_steps, args.lr, args.lam, init_seed)
    print(f"Stage 1 done in {time.time()-t0:.1f}s")
    directions = pair  # [2, d]
    loss_histories = {'pair_d1d2': hist1}

    ckpt_path = os.path.join(args.results_dir, 'checkpoint.pkl')
    torch.save({'directions': directions.cpu(), 'args': vars(args)}, ckpt_path)

    # ---- Stage 2+: grow dictionary ----
    for k in range(2, args.n_dict):
        print(f"\n=== Stage {k}: learning d{k+1} against {k} frozen directions ===")
        t0 = time.time()
        fixed_dirs = [directions[j] for j in range(directions.shape[0])]
        new_dir, hist_k = optimize(
            model, acts, ctxs, f_refs, fixed_dirs, 1, angles_rad,
            args.layer, measure_layer, device, d_model,
            args.n_steps, args.lr, args.lam, init_seed + k,
            log_prefix=f"d{k+1} ")
        print(f"Stage {k} done in {time.time()-t0:.1f}s")
        directions = torch.cat([directions, new_dir], dim=0)
        loss_histories[f'd{k+1}'] = hist_k
        torch.save({'directions': directions.cpu(), 'args': vars(args)}, ckpt_path)

    # ---- Diagnostics ----
    print(f"\n=== Evaluating dictionary ({directions.shape[0]} directions) ===")
    diag = eval_dictionary(model, acts, ctxs, f_refs, directions,
                           angles_rad, args.layer, measure_layer)
    print(f"Per-direction single-direction response: {np.round(diag['single_R'], 3)}")
    print(f"Pairwise L (lower = more superellipse-like, 0.5 = no interference):")
    for i_d in range(directions.shape[0]):
        for j_d in range(i_d + 1, directions.shape[0]):
            print(f"  L(d{i_d+1}, d{j_d+1}) = {diag['pair_L'][i_d, j_d]:.4f}")

    # ---- Save ----
    results = {
        'args': vars(args),
        'directions': directions.cpu(),
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_histories': loss_histories,
        'single_R': diag['single_R'],
        'pair_L': diag['pair_L'],
        'first_pair': diag['first_pair'],
        'gram': diag['gram'],
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Loss curves
    ax = axes[0, 0]
    for name, hist in loss_histories.items():
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['loss'] for h in hist], label=f'{name} (total)')
        ax.plot(steps, [h['resp_loss'] for h in hist], '--', alpha=0.6,
                label=f'{name} (ratio)')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_title('Optimization')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Histogram of per-activation L_a for the first pair
    ax = axes[0, 1]
    fp = diag['first_pair']
    if fp is not None:
        ax.hist(fp['La'], bins=20, alpha=0.7, color='C0')
        ax.axvline(0.5, color='gray', ls=':', label='0.5')
        ax.axvline(fp['La'].mean(), color='red', ls='--',
                   label=f"mean={fp['La'].mean():.3f}")
    ax.set_xlabel('L_a = R_mix/(R1+R2)')
    ax.set_title('Per-activation loss (first pair)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Bar of mean R1/R2/Rmix for first pair
    ax = axes[1, 0]
    if fp is not None:
        means = [fp['R1'].mean(), fp['R2'].mean(), fp['Rmix'].mean()]
        ax.bar(['R1', 'R2', 'Rmix'], means, color=['C0', 'C1', 'C2'])
        ax.axhline((means[0] + means[1]) / 2, color='gray', ls=':',
                   label='(R1+R2)/2')
    ax.set_ylabel('mean L2 response'); ax.set_title('Mean responses (first pair)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Gram matrix heatmap
    ax = axes[1, 1]
    im = ax.imshow(diag['gram'], vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title('Gram matrix (d_i . d_j)')
    ax.set_xticks(range(directions.shape[0]))
    ax.set_yticks(range(directions.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, '
                 f'{directions.shape[0]} dirs, alpha={args.angles} deg', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}")
    print(f"Saved plot: {plot_path}")


if __name__ == '__main__':
    main()
