#!/usr/bin/env python3
"""
Learn a single free direction against contrastive (class-pair) directions, MLP.

Variant of learn_superellipse_dict_mlp.py: instead of jointly learning a free
pair (d1, d2), we FIX one leg to a contrastive direction and optimize a single
free direction d against it, using the same superellipse ratio objective
    L = R_mix / (R_free + R_contrastive)   (+ lambda * (d . d_contrastive)^2)
averaged over anchors, angles, AND a set of contrastive directions.

Contrastive directions are the transformer-style diff-of-means: for a class
pair (A, B), d_contrastive = normalize( mean(h1 | class A) - mean(h1 | class B) ),
computed at the injection layer h1. We randomly select n_contrastive of the 45
class pairs for a given run and average the objective over them — i.e. the
"grow against fixed dirs" branch of the superellipse optimizer with
fixed_dirs = the selected contrastive directions.

Usage:
    python learn_against_contrastive_mlp.py \
        --ckpt results/grok_mlp/checkpoints/ckpt_s1000000_final.pt \
        --n_act 10 --angles 30 --n_contrastive 10
"""
import argparse
import itertools
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the MLP superellipse machinery verbatim (same objective).
from scripts.grok_mlp.learn_superellipse_dict_mlp import (
    build_mlp, hidden_act_indices, forward_to, forward_between,
    pair_responses, pair_loss,
)


def compute_class_means_h1(mlp, inject_after, split, cap, data_root, device, dtype):
    """Mean h1 activation per digit class (true labels), over up to `cap` images."""
    ds = torchvision.datasets.MNIST(root=data_root, train=(split == 'train'),
        transform=torchvision.transforms.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=500, shuffle=False)
    width = mlp[inject_after - 1].out_features if hasattr(mlp[inject_after - 1], 'out_features') else None
    sums, counts = {}, {}
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            h1 = forward_to(mlp, x.to(device), inject_after)  # [B, width]
            for c in range(10):
                m = (y == c)
                if m.any():
                    h = h1[m.to(device)]
                    sums[c] = sums.get(c, 0.0) + h.sum(0)
                    counts[c] = counts.get(c, 0) + int(m.sum())
            seen += x.shape[0]
            if seen >= cap:
                break
    means = torch.stack([sums[c] / counts[c] for c in range(10)])  # [10, width]
    return means, {c: counts[c] for c in range(10)}, seen


def build_contrastive_dirs(class_means, n_select, seed):
    """All 45 class-pair diff-of-means directions; randomly pick n_select."""
    all_pairs = list(itertools.combinations(range(10), 2))  # 45 pairs
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_pairs), size=n_select, replace=False)
    pairs = [all_pairs[i] for i in sorted(idx)]
    dirs = []
    for (a, b) in pairs:
        d = class_means[a] - class_means[b]
        dirs.append(d / d.norm())
    return torch.stack(dirs), pairs


def optimize_one_free(mlp, acts, f_refs, contrastive_dirs, prev_free, angles_rad,
                      inject_after, measure_after, device, d_model, n_steps, lr,
                      lam, lam_prev, seed, log_prefix=""):
    """Optimize one free direction against the fixed contrastive set, with a
    soft-orthogonality penalty against previously-found free directions.

        loss = mean_k L(d, c_k)  +  lam * mean_k (d.c_k)^2
                                  +  lam_prev * mean_j (d.prev_free_j)^2
    where L(d, c_k) = mean over anchors & angles of R_mix/(R_free + R_contrast).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d = torch.randn(d_model, generator=g, device=device, dtype=acts[0].dtype)
    d = (d / d.norm()).requires_grad_(True)
    optimizer = torch.optim.Adam([d], lr=lr)

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        resp_terms, mr1, mr2, mrmix = [], 0.0, 0.0, 0.0
        for dc in contrastive_dirs:
            rl, (a1, a2, amix) = pair_loss(
                mlp, acts, f_refs, d, dc, angles_rad, inject_after, measure_after)
            resp_terms.append(rl); mr1 += a1; mr2 += a2; mrmix += amix
        resp_loss = torch.stack(resp_terms).mean()
        n_c = len(contrastive_dirs)
        mr1 /= n_c; mr2 /= n_c; mrmix /= n_c

        d_u = d / (d.norm() + 1e-10)
        orth_c = torch.stack([(d_u @ (dc / (dc.norm() + 1e-10))) ** 2
                              for dc in contrastive_dirs]).mean()
        if len(prev_free):
            orth_p = torch.stack([(d_u @ (pf / (pf.norm() + 1e-10))) ** 2
                                  for pf in prev_free]).mean()
        else:
            orth_p = torch.zeros((), device=device, dtype=d.dtype)

        loss = resp_loss + lam * orth_c + lam_prev * orth_p
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d.data.div_(d.data.norm())

        history.append({'step': step, 'loss': loss.item(),
                        'resp_loss': resp_loss.item(), 'orth_c': orth_c.item(),
                        'orth_prev': float(orth_p.item()),
                        'R1': mr1, 'R2': mr2, 'Rmix': mrmix})
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  orth_c={orth_c.item():.4f}  "
                  f"orth_prev={float(orth_p.item()):.4f}  "
                  f"R_free={mr1:.3f} R_contrast={mr2:.3f} R_mix={mrmix:.3f}", flush=True)
    return d.detach(), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--width', type=int, default=200)
    p.add_argument('--activation', type=str, default='GeLU')
    p.add_argument('--inject_hidden', type=int, default=1)
    p.add_argument('--measure_hidden', type=int, default=-1)
    p.add_argument('--n_act', type=int, default=10)
    p.add_argument('--n_contrastive', type=int, default=10,
                   help='how many of the 45 class pairs to use (averaged over)')
    p.add_argument('--n_free', type=int, default=15,
                   help='how many free directions to learn (grown one at a time)')
    p.add_argument('--angles', type=float, nargs='+', default=[30.0])
    p.add_argument('--n_steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--lam', type=float, default=1.0,
                   help='weight of (d.c_k)^2 orthogonality vs contrastive dirs')
    p.add_argument('--lam_prev', type=float, default=None,
                   help='weight of soft-orthogonality vs previously-found free '
                        'directions (default: same as --lam)')
    p.add_argument('--class_mean_cap', type=int, default=10000)
    p.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--contrastive_seed', type=int, default=0,
                   help='seed for selecting the n_contrastive class pairs')
    p.add_argument('--init_seed', type=int, default=None)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    lam_prev = args.lam_prev if args.lam_prev is not None else args.lam

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.results_dir = (f'results/grok_mlp/against_contrastive/'
                            f'{tag}_a{angle_tag}_nc{args.n_contrastive}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    # Model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    depth = ckpt.get('depth', args.depth)
    width = ckpt.get('width', args.width)
    mlp = build_mlp(depth, width, act=args.activation).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    for prm in mlp.parameters():
        prm.requires_grad_(False)

    hidden_idxs = hidden_act_indices(mlp)
    inject_after = hidden_idxs[args.inject_hidden - 1]
    measure_after = hidden_idxs[args.measure_hidden if args.measure_hidden < 0
                                else args.measure_hidden - 1]
    d_model = width
    print(f"depth={depth} width={width} | inject after module {inject_after}, "
          f"measure after module {measure_after}", flush=True)

    # Contrastive directions (diff-of-class-means in h1)
    class_means, counts, seen = compute_class_means_h1(
        mlp, inject_after, args.split, args.class_mean_cap, args.data_root, device, dtype)
    contrastive_dirs, pairs = build_contrastive_dirs(
        class_means, args.n_contrastive, args.contrastive_seed)
    contrastive_dirs = contrastive_dirs.to(device)
    print(f"Class means over {seen} {args.split} images; counts={counts}", flush=True)
    print(f"Selected {args.n_contrastive} class pairs: {pairs}", flush=True)
    # pairwise cosines among contrastive dirs (info)
    cg = (contrastive_dirs @ contrastive_dirs.T).cpu().numpy()
    print(f"Contrastive dir gram: mean|offdiag|="
          f"{np.abs(cg[np.triu_indices(len(pairs),1)]).mean():.3f}", flush=True)

    # Anchors (same selection convention as the dict script)
    ds = torchvision.datasets.MNIST(root=args.data_root, train=(args.split == 'train'),
        transform=torchvision.transforms.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.n_act, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    x, _ = next(iter(loader))
    with torch.no_grad():
        h1 = forward_to(mlp, x.to(device), inject_after)
    acts = [h1[i].clone() for i in range(args.n_act)]
    f_refs = []
    with torch.no_grad():
        for i in range(args.n_act):
            f_refs.append(forward_between(mlp, acts[i].unsqueeze(0),
                                          inject_after, measure_after).squeeze(0))
    print(f"Mean h1 norm: {torch.stack(acts).norm(dim=-1).mean().item():.3f}", flush=True)

    # Grow n_free directions one at a time against the fixed contrastive set,
    # with soft-orthogonality vs the previously-found free directions.
    fixed_dirs = [contrastive_dirs[k] for k in range(len(pairs))]
    free_list = []
    loss_histories = {}
    for f_idx in range(args.n_free):
        print(f"\n=== Free direction {f_idx+1}/{args.n_free} vs {len(pairs)} "
              f"contrastive dirs (+ orth vs {f_idx} prev free) ===", flush=True)
        new_dir, hist = optimize_one_free(
            mlp, acts, f_refs, fixed_dirs, free_list, angles_rad,
            inject_after, measure_after, device, d_model,
            args.n_steps, args.lr, args.lam, lam_prev, init_seed + f_idx,
            log_prefix=f"free{f_idx+1} ")
        free_list.append(new_dir)
        loss_histories[f'free{f_idx+1}'] = hist
    free_dirs = torch.stack(free_list)  # [n_free, d]
    free_gram = (free_dirs @ free_dirs.T).cpu().numpy()
    iu = np.triu_indices(args.n_free, 1)
    print(f"\nFree-dir gram: mean|offdiag|={np.abs(free_gram[iu]).mean():.3f}, "
          f"max|offdiag|={np.abs(free_gram[iu]).max():.3f}", flush=True)

    # Eval: per free dir, L(free, contrastive_k) and single responses.
    print(f"\n=== Evaluating ===", flush=True)
    eval_rows = []
    for f_idx in range(args.n_free):
        df = free_dirs[f_idx]
        Ls, R1s, R2s, Rmixs = [], [], [], []
        for k in range(len(pairs)):
            dc = contrastive_dirs[k]
            r1a, r2a, rma = [], [], []
            for i in range(args.n_act):
                for ar in angles_rad:
                    with torch.no_grad():
                        r1, r2, rmix = pair_responses(
                            mlp, acts[i], f_refs[i], df, dc, ar, inject_after, measure_after)
                    r1a.append(r1.item()); r2a.append(r2.item()); rma.append(rmix.item())
            r1m, r2m, rmm = np.mean(r1a), np.mean(r2a), np.mean(rma)
            Lk = rmm / (r1m + r2m + 1e-8)
            Ls.append(Lk); R1s.append(r1m); R2s.append(r2m); Rmixs.append(rmm)
            print(f"  free{f_idx+1} vs pair{pairs[k]}: L={Lk:.4f} "
                  f"R_free={r1m:.3f} R_contrast={r2m:.3f} R_mix={rmm:.3f}", flush=True)
        eval_rows.append({'L': np.array(Ls), 'R_free': np.array(R1s),
                          'R_contrast': np.array(R2s), 'R_mix': np.array(Rmixs)})
        print(f"  free{f_idx+1}: mean L over contrastive set = {np.mean(Ls):.4f}", flush=True)

    # Save. directions = [free dirs..., contrastive dirs...] so the canonical
    # fit/boundary scripts can fit p for pair (free_idx, n_free + k).
    directions = torch.cat([free_dirs, contrastive_dirs], dim=0).cpu()
    results = {
        'args': vars(args), 'inject_after': inject_after, 'measure_after': measure_after,
        'n_free': args.n_free, 'class_pairs': pairs, 'angles_deg': args.angles,
        'directions': directions, 'free_dirs': free_dirs.cpu(),
        'contrastive_dirs': contrastive_dirs.cpu(), 'class_means': class_means.cpu(),
        'loss_histories': loss_histories, 'eval': eval_rows,
        'free_gram': free_gram, 'lam_prev': lam_prev,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot: ratio-loss curves (all free dirs) + mean-L per free dir + free-dir gram
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ax = axes[0]
    for name, hist in loss_histories.items():
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['resp_loss'] for h in hist], alpha=0.7, lw=1)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('ratio loss')
    ax.set_title(f'Optimization (ratio, {args.n_free} free dirs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_L = np.array([er['L'].mean() for er in eval_rows])
    ax.bar(range(args.n_free), mean_L, color='C0')
    ax.axhline(0.5, color='gray', ls=':', label='0.5')
    ax.axhline(mean_L.mean(), color='red', ls='--', label=f"overall mean={mean_L.mean():.3f}")
    ax.set_xlabel('free direction index'); ax.set_ylabel('mean L over contrastive set')
    ax.set_title('Per-free-dir mean L'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    im = ax.imshow(free_gram, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title('Free-dir Gram (d_i . d_j)')
    ax.set_xticks(range(args.n_free)); ax.set_yticks(range(args.n_free))
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'MNIST MLP d{depth}w{width}: {args.n_free} free dirs vs contrastive '
                 f'class-pair dirs (angles={args.angles} deg, n_c={args.n_contrastive}, '
                 f'lam_prev={lam_prev})', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}", flush=True)
    print(f"Saved plot: {plot_path}", flush=True)


if __name__ == '__main__':
    main()
