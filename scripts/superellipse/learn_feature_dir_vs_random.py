#!/usr/bin/env python3
"""
Find feature directions by optimizing against a fixed random scaffold.

Instead of jointly learning a pair (or growing a dictionary against previously
learned directions, as in learn_superellipse_dict.py), we fix a scaffold of
R random unit directions and optimize a single candidate *feature* direction
d_f against them, minimizing the superellipse ratio averaged over the scaffold:

    L(d_f) = mean_r mean_a mean_alpha   r_mix / (r_r + r_f)

where, per activation a, random direction d_r, and angle alpha:
    r_f   = downstream L2 response to a spherical perturbation along d_f
    r_r   = response along the random direction d_r
    r_mix = response along the equal-mix bisector of d_f and d_r.

L < 0.5 is the superellipse / p>2 (super-additive interference) regime; the
hope is that pushing d_f to interfere maximally against a *random* basis pulls
it toward genuine feature directions of the model.

We "gather" several feature directions by running the single-direction
optimization n_feat times (different init seeds) against the *same* fixed
scaffold, with a push-apart penalty lambda * (d_f . d_prev)^2 against every
previously found feature direction so they stay distinct.

This reuses the geometry/response helpers from learn_superellipse_dict.py;
tangent_dirs(a, d_f, d_r) already returns exactly (r_f, r_r, r_mix).

Usage:
    python scripts/superellipse/learn_feature_dir_vs_random.py \
        --model qwen3-1.7b --layer 2 --n_act 10 --n_rand 10 --n_feat 5 --angles 30
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

# Reuse the geometry + response helpers verbatim.
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation, pair_loss, pair_responses,
)


def random_unit_dirs(n, d_model, device, seed):
    """n fixed random unit directions in R^d (seeded, reproducible)."""
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(n, d_model, generator=g, device=device)
    return v / v.norm(dim=-1, keepdim=True)


def optimize_feature_dir(model, acts, ctxs, f_refs, rand_dirs, prev_feats,
                         angles_rad, layer, measure_layer, device, d_model,
                         n_steps, lr, feat_orth_lam, rand_orth_lam, seed,
                         log_prefix=""):
    """Optimize one feature direction d_f against the fixed random scaffold.

    rand_dirs:  [R, d] fixed random unit directions (the scaffold).
    prev_feats: list of already-found unit feature directions (frozen); the
        new d_f is penalized for overlap with each of them.

    Loss = mean_r pair_loss(d_f, d_r)
           + feat_orth_lam * mean_prev (d_f . d_prev)^2
           + rand_orth_lam * mean_r    (d_f . d_r)^2

    Returns (d_f [d], loss_history list).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d_f = torch.randn(d_model, generator=g, device=device)
    d_f = (d_f / d_f.norm()).requires_grad_(True)
    optimizer = torch.optim.Adam([d_f], lr=lr)

    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()

        # Superellipse ratio against each random scaffold direction.
        resp_terms = []
        mr_f = mr_r = mr_mix = 0.0
        for d_r in rand_dirs:
            rl, (a_f, a_r, a_mix) = pair_loss(
                model, acts, ctxs, f_refs, d_f, d_r,
                angles_rad, layer, measure_layer)
            resp_terms.append(rl)
            mr_f += a_f; mr_r += a_r; mr_mix += a_mix
        n_rand = len(rand_dirs)
        resp_loss = torch.stack(resp_terms).mean()
        mr_f /= n_rand; mr_r /= n_rand; mr_mix /= n_rand

        d_fu = d_f / (d_f.norm() + 1e-10)

        # Push-apart penalty against previously found feature directions.
        if prev_feats:
            feat_overlap = torch.stack([
                (d_fu @ (p / (p.norm() + 1e-10))) ** 2 for p in prev_feats
            ]).mean()
        else:
            feat_overlap = torch.zeros((), device=device)

        # Optional penalty against collapsing onto a random scaffold direction.
        if rand_orth_lam != 0.0:
            rand_overlap = torch.stack([
                (d_fu @ d_r) ** 2 for d_r in rand_dirs
            ]).mean()
        else:
            rand_overlap = torch.zeros((), device=device)

        loss = resp_loss + feat_orth_lam * feat_overlap + rand_orth_lam * rand_overlap
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_f.data.div_(d_f.data.norm())

        loss_history.append({
            'step': step, 'loss': loss.item(), 'resp_loss': resp_loss.item(),
            'feat_overlap': feat_overlap.item(), 'rand_overlap': rand_overlap.item(),
            'R_f': mr_f, 'R_r': mr_r, 'R_mix': mr_mix,
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  feat_ov={feat_overlap.item():.4f}  "
                  f"R_f={mr_f:.3f} R_r={mr_r:.3f} R_mix={mr_mix:.3f}")

    return d_f.detach(), loss_history


def eval_against_scaffold(model, acts, ctxs, f_refs, d_f, rand_dirs,
                          angles_rad, layer, measure_layer):
    """Mean superellipse ratio L(d_f) over a (possibly held-out) random set.

    Returns (mean_L, mean_R_f, mean_R_r, mean_R_mix).
    """
    Ls, R_fs, R_rs, R_mixs = [], [], [], []
    for d_r in rand_dirs:
        for i in range(len(acts)):
            for angle_rad in angles_rad:
                with torch.no_grad():
                    r_f, r_r, r_mix = pair_responses(
                        model, acts[i], ctxs[i], f_refs[i], d_f, d_r,
                        angle_rad, layer, measure_layer)
                Ls.append(r_mix.item() / (r_f.item() + r_r.item() + 1e-8))
                R_fs.append(r_f.item()); R_rs.append(r_r.item())
                R_mixs.append(r_mix.item())
    return (float(np.mean(Ls)), float(np.mean(R_fs)),
            float(np.mean(R_rs)), float(np.mean(R_mixs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2,
                        help='Perturbation layer (where directions are applied)')
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer to measure L2 response (default: 2nd-to-last)')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_act', type=int, default=10)
    parser.add_argument('--n_rand', type=int, default=10,
                        help='Size of the fixed random scaffold')
    parser.add_argument('--n_rand_test', type=int, default=10,
                        help='Size of the held-out random set (generalization check)')
    parser.add_argument('--n_feat', type=int, default=5,
                        help='Number of feature directions to gather')
    parser.add_argument('--angles', type=float, nargs='+', default=[30.0],
                        help='Perturbation angle(s) in degrees')
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--feat_orth_lam', type=float, default=1.0,
                        help='Push-apart penalty against previously found feature dirs')
    parser.add_argument('--rand_orth_lam', type=float, default=0.0,
                        help='Penalty against overlap with the random scaffold')
    parser.add_argument('--data_source', type=str, default='fineweb')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/feature_dir_vs_random/'
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
    print(f"Config: n_act={args.n_act}, n_rand={args.n_rand}, n_feat={args.n_feat}, "
          f"angles={args.angles} deg, n_steps={args.n_steps}, lr={args.lr}, "
          f"feat_orth_lam={args.feat_orth_lam}, rand_orth_lam={args.rand_orth_lam}")

    # Data + activations
    print(f"Loading {args.n_act} {args.data_source} sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, token_ids_list, args.layer, device)
    model_dtype_actual = next(model.parameters()).dtype
    # Keep activations/directions in float32; cast to model dtype at forward only.
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

    # Fixed random scaffold + held-out random set (disjoint seeds).
    rand_dirs = random_unit_dirs(args.n_rand, d_model, device, args.seed + 1000)
    rand_test = random_unit_dirs(args.n_rand_test, d_model, device, args.seed + 2000)
    rand_dirs_list = [rand_dirs[r] for r in range(args.n_rand)]

    # ---- Gather feature directions ----
    feat_dirs = []           # list of [d] tensors
    prev_feats = []          # frozen feature dirs for the push-apart penalty
    loss_histories = {}
    train_L, test_L = [], []
    train_resp, test_resp = [], []   # (R_f, R_r, R_mix) tuples

    for j in range(args.n_feat):
        print(f"\n=== Feature direction {j+1}/{args.n_feat} ===")
        t0 = time.time()
        d_f, hist = optimize_feature_dir(
            model, acts, ctxs, f_refs, rand_dirs_list, prev_feats,
            angles_rad, args.layer, measure_layer, device, d_model,
            args.n_steps, args.lr, args.feat_orth_lam, args.rand_orth_lam,
            args.seed + j, log_prefix=f"f{j+1} ")
        print(f"  done in {time.time()-t0:.1f}s")

        feat_dirs.append(d_f)
        prev_feats.append(d_f)
        loss_histories[f'feat{j+1}'] = hist

        # Train (scaffold) vs held-out generalization.
        Ltr, ftr, rtr, mtr = eval_against_scaffold(
            model, acts, ctxs, f_refs, d_f, rand_dirs_list,
            angles_rad, args.layer, measure_layer)
        rand_test_list = [rand_test[r] for r in range(args.n_rand_test)]
        Lte, fte, rte, mte = eval_against_scaffold(
            model, acts, ctxs, f_refs, d_f, rand_test_list,
            angles_rad, args.layer, measure_layer)
        train_L.append(Ltr); test_L.append(Lte)
        train_resp.append((ftr, rtr, mtr)); test_resp.append((fte, rte, mte))
        print(f"  L(train scaffold)={Ltr:.4f}  L(held-out)={Lte:.4f}")

        # Checkpoint after each found direction.
        torch.save({'feat_dirs': torch.stack(feat_dirs).cpu(),
                    'rand_dirs': rand_dirs.cpu(), 'args': vars(args)},
                   os.path.join(args.results_dir, 'checkpoint.pkl'))

    feat_dirs = torch.stack(feat_dirs)  # [F, d]

    # ---- Grams ----
    feat_gram = (feat_dirs @ feat_dirs.T).cpu().numpy()
    feat_rand_gram = (feat_dirs @ rand_dirs.T).cpu().numpy()

    print(f"\n=== Summary ({args.n_feat} feature dirs) ===")
    for j in range(args.n_feat):
        print(f"  feat{j+1}: L_train={train_L[j]:.4f}  L_test={test_L[j]:.4f}  "
              f"R_f={train_resp[j][0]:.3f}")
    print("Feature-dir Gram (off-diagonal = overlap):")
    for i in range(args.n_feat):
        for k in range(i + 1, args.n_feat):
            print(f"  d_f{i+1}.d_f{k+1} = {feat_gram[i, k]:+.3f}")

    # ---- Save ----
    results = {
        'args': vars(args),
        'feat_dirs': feat_dirs.cpu(),
        'rand_dirs': rand_dirs.cpu(),
        'rand_test': rand_test.cpu(),
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_histories': loss_histories,
        'train_L': np.array(train_L),
        'test_L': np.array(test_L),
        'train_resp': np.array(train_resp),   # [F, 3] = R_f, R_r, R_mix
        'test_resp': np.array(test_resp),
        'feat_gram': feat_gram,
        'feat_rand_gram': feat_rand_gram,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Loss curves (all runs)
    ax = axes[0, 0]
    for name, hist in loss_histories.items():
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['resp_loss'] for h in hist], label=name)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('ratio loss')
    ax.set_title('Optimization (R_mix/(R_r+R_f))')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Train vs held-out final L per feature dir
    ax = axes[0, 1]
    x = np.arange(args.n_feat)
    ax.bar(x - 0.2, train_L, width=0.4, label='train scaffold', color='C0')
    ax.bar(x + 0.2, test_L, width=0.4, label='held-out', color='C1')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f'f{j+1}' for j in range(args.n_feat)])
    ax.set_ylabel('mean L'); ax.set_title('Final L: train vs held-out')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Feature-dir Gram heatmap (clustering / distinctness)
    ax = axes[1, 0]
    im = ax.imshow(feat_gram, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title('Feature-dir Gram (d_fi . d_fj)')
    ax.set_xticks(range(args.n_feat)); ax.set_yticks(range(args.n_feat))
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Mean responses per feature dir (train scaffold)
    ax = axes[1, 1]
    tr = np.array(train_resp)  # [F, 3]
    ax.bar(x - 0.25, tr[:, 0], width=0.25, label='R_f', color='C0')
    ax.bar(x, tr[:, 1], width=0.25, label='R_r', color='C1')
    ax.bar(x + 0.25, tr[:, 2], width=0.25, label='R_mix', color='C2')
    ax.set_xticks(x); ax.set_xticklabels([f'f{j+1}' for j in range(args.n_feat)])
    ax.set_ylabel('mean L2 response'); ax.set_title('Mean responses (train scaffold)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, '
                 f'{args.n_feat} feature dirs vs {args.n_rand} random, '
                 f'alpha={args.angles} deg', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}")
    print(f"Saved plot: {plot_path}")


if __name__ == '__main__':
    main()
