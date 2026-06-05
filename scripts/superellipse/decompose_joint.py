#!/usr/bin/env python3
"""Joint K-feature decomposition of an activation (non-greedy variant).

Same plateau-breaking proxy as decompose_plateau_breaking.py, but optimizes
all K features and their coefficients jointly with a single loss:

    L({d_k}, {alpha_k}) = ||a - sum_k alpha_k d_k||^2 / ||a||^2     (reconstruction)
        + lam_proxy * (1/K) sum_k mean_c [R_mix(d_k)/(R_f+R_c)]     (plateau-breaking)
        + lam_scaffold * (1/K) sum_k mean_c (d_k . d_c)^2           (avoid scaffold)
        + lam_orth * (2/(K(K-1))) sum_{i<j} (d_i . d_j)^2           (mutual orthogonality)

Reuses batched_proxy_loss from the greedy script. Renormalizes each d_k to
unit norm after each Adam step; alphas are free scalars.

Usage:
    python scripts/superellipse/decompose_joint.py \\
        --model gemma-2-9b --layer 15 --K 5 --n_act 10 --angles 30 \\
        --target_prompt 'Paris is the capital of' --seed 42
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

from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold, DEFAULT_NAMES,
)
from scripts.superellipse.decompose_plateau_breaking import batched_proxy_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemma-2-9b')
    ap.add_argument('--layer', type=int, default=15)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_act', type=int, default=10)
    ap.add_argument('--scaffold_names', nargs='+', default=DEFAULT_NAMES)
    ap.add_argument('--scaffold_pkl', default=None)
    ap.add_argument('--angles', type=float, nargs='+', default=[30.0])
    ap.add_argument('--K', type=int, default=5,
                    help='number of features (fixed)')
    ap.add_argument('--n_steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--lam_proxy', type=float, default=1.0)
    ap.add_argument('--lam_scaffold', type=float, default=1.0)
    ap.add_argument('--lam_orth', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target_seed', type=int, default=999)
    ap.add_argument('--target_idx', type=int, default=0)
    ap.add_argument('--target_prompt', default=None)
    ap.add_argument('--batch_chunk_k', type=int, default=None)
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/decomp_joint/'
                            f'{args.model}_L{args.layer}_K{args.K}_a{angle_tag}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device} d_model={d_model} K={args.K} '
          f'perturb_layer={args.layer} measure_layer={measure_layer}',
          flush=True)
    print(f'Config: n_act={args.n_act} scaffold={args.scaffold_names} '
          f'angles={args.angles} lam_proxy={args.lam_proxy} '
          f'lam_scaffold={args.lam_scaffold} lam_orth={args.lam_orth}',
          flush=True)

    # Opt-set
    print(f'Loading opt-set ({args.n_act} FineWeb, seed={args.seed})', flush=True)
    tok_opt = load_fineweb_fixed_length(args.n_act, tokenizer,
                                        seq_len=args.seq_len, seed=args.seed)
    ctx_opt, act_opt = extract_context_and_activation(model, tok_opt,
                                                     args.layer, device)
    model_dtype = next(model.parameters()).dtype
    acts_stack = act_opt.to(device=device, dtype=torch.float32)
    ctx_stack = ctx_opt.to(device=device, dtype=model_dtype)
    print(f'  mean ||a_opt||={acts_stack.norm(dim=-1).mean().item():.2f}',
          flush=True)

    # Target
    if args.target_prompt is not None:
        print(f'Target: custom prompt {args.target_prompt!r}', flush=True)
        tgt_ids = tokenizer(args.target_prompt,
                            return_tensors='pt').input_ids[0]
        print(f'  tokens: {tgt_ids.tolist()} '
              f'({tokenizer.decode(tgt_ids)!r})', flush=True)
        ctx_tgt, act_tgt = extract_context_and_activation(
            model, [tgt_ids], args.layer, device)
    else:
        tok_tgt_batch = load_fineweb_fixed_length(
            args.target_idx + 1, tokenizer,
            seq_len=args.seq_len, seed=args.target_seed)
        ctx_tgt, act_tgt = extract_context_and_activation(
            model, [tok_tgt_batch[args.target_idx]], args.layer, device)
    a_target = act_tgt[0].to(device=device, dtype=torch.float32)
    a_target_norm = a_target.norm().item()
    print(f'  ||a_target||={a_target_norm:.2f}', flush=True)

    # f_refs for opt-set
    with torch.no_grad():
        pts = acts_stack.unsqueeze(1).to(model_dtype)
        f_refs_stack = forward_from_layer_to_layer(
            model, ctx_stack, pts, args.layer, measure_layer).float()

    # Scaffold
    if args.scaffold_pkl is not None:
        with open(args.scaffold_pkl, 'rb') as f:
            blob = pickle.load(f)
        scaffold = torch.stack([(blob['directions'][n].to(device).float() /
                                 blob['directions'][n].to(device).float().norm())
                                for n in args.scaffold_names])
    else:
        scaffold = load_or_build_scaffold(model, tokenizer, device,
                                          args.model, args.layer,
                                          args.scaffold_names)
    print(f'Scaffold shape: {tuple(scaffold.shape)}', flush=True)

    # Only D is optimized; alphas are implicit (closed-form least-squares
    # given current D), so they're always at the optimal scale for the
    # current subspace. This bypasses the alpha-scale optimization issue.
    g = torch.Generator(device=device).manual_seed(args.seed)
    D = torch.randn(args.K, d_model, generator=g, device=device,
                    dtype=torch.float32)
    D = D / D.norm(dim=-1, keepdim=True)
    D = D.requires_grad_(True)

    optimizer = torch.optim.Adam([D], lr=args.lr)

    K_pairs = args.K * (args.K - 1) // 2
    history = []
    print('\n=== Joint optimization ===', flush=True)
    t0 = time.time()
    n_evals = args.K * len(angles_rad)
    for step in range(args.n_steps):
        optimizer.zero_grad()

        # --- 1. Per-feature proxy: backward each call separately so we don't
        #        hold K * (perturb-points) worth of autograd state at once. ---
        proxy_acc = 0.0
        mR_f = mR_c = mR_mix = 0.0
        for k in range(args.K):
            for angle_rad in angles_rad:
                d_k = D[k] / (D[k].norm() + 1e-10)
                rl, (a_f, a_c, a_mix) = batched_proxy_loss(
                    model, acts_stack, ctx_stack, f_refs_stack, d_k,
                    scaffold, angle_rad, args.layer, measure_layer,
                    chunk_k=args.batch_chunk_k)
                (args.lam_proxy * rl / n_evals).backward()
                proxy_acc = proxy_acc + rl.detach().item()
                mR_f += a_f; mR_c += a_c; mR_mix += a_mix
        proxy_loss_val = proxy_acc / n_evals
        mR_f /= n_evals; mR_c /= n_evals; mR_mix /= n_evals

        # --- 2. Structural losses (cheap): one shared backward. ---
        # Implicit alphas: solve (D D^T) alpha = D a (normal equations).
        # Tiny ridge for numerical stability if D rows get near-collinear.
        D_unit = D / (D.norm(dim=-1, keepdim=True) + 1e-10)
        G = D_unit @ D_unit.T + 1e-6 * torch.eye(args.K, device=device)
        b = D_unit @ a_target
        alpha_t = torch.linalg.solve(G, b)
        recon = (alpha_t.unsqueeze(-1) * D_unit).sum(dim=0)
        rec_loss = ((a_target - recon) ** 2).sum() / (a_target_norm ** 2)
        scaffold_overlap = ((D_unit @ scaffold.T) ** 2).mean()
        if args.K > 1:
            gram = D_unit @ D_unit.T
            iu = torch.triu_indices(args.K, args.K, offset=1, device=device)
            orth_loss = (gram[iu[0], iu[1]] ** 2).mean()
        else:
            orth_loss = torch.zeros((), device=device)
        struct_loss = (rec_loss
                       + args.lam_scaffold * scaffold_overlap
                       + args.lam_orth * orth_loss)
        struct_loss.backward()

        # Total loss for logging only
        loss_val = (rec_loss.item() + args.lam_proxy * proxy_loss_val
                    + args.lam_scaffold * scaffold_overlap.item()
                    + args.lam_orth * orth_loss.item())

        optimizer.step()

        # Hard renormalize D rows post-step (matches greedy convention)
        with torch.no_grad():
            D.data /= (D.data.norm(dim=-1, keepdim=True) + 1e-10)

        alpha_np = alpha_t.detach().cpu().numpy()
        history.append({
            'step': step, 'loss': loss_val,
            'rec_loss': rec_loss.item(),
            'proxy_loss': proxy_loss_val,
            'scaffold_overlap': scaffold_overlap.item(),
            'orth_loss': orth_loss.item(),
            'R_f': mR_f, 'R_c': mR_c, 'R_mix': mR_mix,
            'alphas': alpha_np.copy(),
        })
        if step % 20 == 0 or step == args.n_steps - 1:
            print(f'  step {step:4d}  L={loss_val:.4f}  '
                  f'rec={rec_loss.item():.4f}  proxy={proxy_loss_val:.4f}  '
                  f'scaf={scaffold_overlap.item():.4f}  '
                  f'orth={orth_loss.item():.4f}  '
                  f'alphas=[{", ".join(f"{a:+.1f}" for a in alpha_np)}]',
                  flush=True)
    print(f'Done in {time.time()-t0:.1f}s', flush=True)

    # Final state: alphas are the closed-form least-squares solution given
    # the final D, which is what we trained against.
    with torch.no_grad():
        D_final = D / (D.norm(dim=-1, keepdim=True) + 1e-10)
        D_final_np = D_final.detach().cpu().numpy()
        D_np = D_final_np.astype(np.float64)
        a_np = a_target.detach().cpu().numpy().astype(np.float64)
        alpha_lsq, *_ = np.linalg.lstsq(D_np.T, a_np, rcond=None)
        alpha_final = alpha_lsq.copy()
        recon_lsq = (alpha_lsq[:, None] * D_final_np).sum(axis=0)
        res_frac_lsq = float(np.linalg.norm(a_np - recon_lsq) /
                             np.linalg.norm(a_np))
        res_frac_free = res_frac_lsq  # same thing now

    print(f'\nFinal ||r||/||a|| = {res_frac_lsq:.4f}', flush=True)
    print(f'free alphas:  [{", ".join(f"{a:+.3f}" for a in alpha_final)}]')
    print(f'lsq  alphas:  [{", ".join(f"{a:+.3f}" for a in alpha_lsq)}]')

    # Grams
    feat_gram = D_final_np @ D_final_np.T
    feat_scaf_gram = D_final_np @ scaffold.cpu().numpy().T
    scaf_gram = (scaffold @ scaffold.T).cpu().numpy()

    # Match greedy's results.pkl schema so existing comparison scripts work.
    results = {
        'args': vars(args),
        'features': torch.from_numpy(D_final_np),
        'alphas': alpha_final,
        'alphas_lsq': alpha_lsq,
        'residual_norm_frac': np.array([1.0, res_frac_free]),
        'residual_norm_frac_lsq': np.array([1.0, res_frac_lsq]),
        'a_target': a_target.cpu(),
        'a_target_norm': a_target_norm,
        'scaffold': scaffold.cpu(),
        'scaffold_names': args.scaffold_names,
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_histories': [history],
        'feat_gram': feat_gram,
        'feat_scaffold_gram': feat_scaf_gram,
        'scaffold_gram': scaf_gram,
        'K_used': args.K,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    steps = [h['step'] for h in history]

    ax = axes[0, 0]
    ax.plot(steps, [h['loss'] for h in history], label='total')
    ax.plot(steps, [h['rec_loss'] for h in history], label='rec')
    ax.plot(steps, [h['proxy_loss'] for h in history], label='proxy')
    ax.plot(steps, [h['scaffold_overlap'] for h in history], label='scaffold')
    ax.plot(steps, [h['orth_loss'] for h in history], label='orth')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.legend(fontsize=9)
    ax.set_title('Loss components'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    alphas_hist = np.stack([h['alphas'] for h in history])  # [steps, K]
    for k in range(args.K):
        ax.plot(steps, alphas_hist[:, k], label=f'α_{k+1}')
    ax.set_xlabel('step'); ax.set_ylabel('α_k')
    ax.set_title('alpha trajectories'); ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    im = ax.imshow(feat_gram, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    for i in range(args.K):
        for j in range(args.K):
            ax.text(j, i, f'{feat_gram[i,j]:+.2f}', ha='center', va='center',
                    fontsize=8, color=('white' if abs(feat_gram[i,j]) > 0.6
                                       else 'black'))
    ax.set_xticks(range(args.K)); ax.set_yticks(range(args.K))
    ax.set_xticklabels([f'd{k+1}' for k in range(args.K)])
    ax.set_yticklabels([f'd{k+1}' for k in range(args.K)])
    ax.set_title('d_i · d_j (mutual)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(feat_scaf_gram, vmin=-1, vmax=1, cmap='RdBu_r',
                   aspect='auto')
    ax.set_xticks(range(len(args.scaffold_names)))
    ax.set_xticklabels(args.scaffold_names, rotation=45, ha='right')
    ax.set_yticks(range(args.K))
    ax.set_yticklabels([f'd{k+1}' for k in range(args.K)])
    ax.set_title('d_k · d_scaffold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, K={args.K}, '
                 f'seed={args.seed}, '
                 f'final ||r||/||a||={res_frac_free:.3f} (lsq={res_frac_lsq:.3f})',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(args.results_dir, "results.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
