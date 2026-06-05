#!/usr/bin/env python3
"""
Learn a superellipse direction dictionary on a trained MNIST MLP.

Port of scripts/superellipse/learn_superellipse_dict.py to a plain feed-forward
MLP (no sequence/context). Discover unit directions in the *first latent layer*
(h1) whose downstream response geometry is super-additive: the model is *less*
responsive along the 50/50 mixture of two directions than along either alone.

Per activation a (= h1) and angle theta, with d1, d2 orthogonalized against a:
    R_i   = || f(a_perturbed along d_i) - f(a) ||_2   (L2 at the last hidden layer)
    R_mix = || f(a_perturbed along normalize(d1_perp + d2_perp)) - f(a) ||_2
We minimize  L_a = R_mix / (R1 + R2), averaged over activations & angles, plus a
soft orthogonality penalty  lambda * (d1 . d2)^2.

Measurement is at the last hidden layer (still high-dim, before the final Linear
that projects down to logits). Injection is at h1 (first Linear+activation).

Stage 1 learns the first pair (d1, d2) jointly. Stage 2+ grows the dictionary:
each new direction is optimized against the frozen existing directions.

Usage:
    python learn_superellipse_dict_mlp.py \
        --ckpt results/grok_mlp/checkpoints/ckpt_s1000000_final.pt \
        --n_act 10 --n_dict 2 --angles 30
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACT = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid, 'GeLU': nn.GELU}


def build_mlp(depth, width, in_dims=(28, 28), n_classes=10, act='GeLU'):
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(int(np.prod(in_dims)), width))
            layers.append(ACT[act]())
        elif i == depth - 1:
            layers.append(nn.Linear(width, n_classes))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(ACT[act]())
    return nn.Sequential(*layers)


def hidden_act_indices(mlp):
    """Module indices of every hidden activation (post-GeLU) in the Sequential."""
    return [i for i, m in enumerate(mlp) if isinstance(m, tuple(ACT.values()))]


def forward_to(mlp, x, stop_after):
    """Run modules [0 .. stop_after] inclusive (grad-enabled)."""
    out = x
    for m in mlp[:stop_after + 1]:
        out = m(out)
    return out


def forward_between(mlp, h, start_after, stop_after):
    """Given output of module index start_after, run [start_after+1 .. stop_after]
    inclusive (grad-enabled — used inside the optimization)."""
    out = h
    for m in mlp[start_after + 1:stop_after + 1]:
        out = m(out)
    return out


def tangent_dirs(a, d1, d2):
    """Orthogonalize d1, d2 against a, normalize, and form their bisector.
    Returns [3, d] = (d1_perp, d2_perp, mix), unit and orthogonal to a."""
    a_hat = a / a.norm()
    d1p = d1 - (d1 @ a_hat) * a_hat
    d1p = d1p / (d1p.norm() + 1e-10)
    d2p = d2 - (d2 @ a_hat) * a_hat
    d2p = d2p / (d2p.norm() + 1e-10)
    mix = d1p + d2p
    mix = mix / (mix.norm() + 1e-10)
    return torch.stack([d1p, d2p, mix])


def pair_responses(mlp, a, f_ref, d1, d2, angle_rad, inject_after, measure_after):
    """Downstream L2 responses (R1, R2, R_mix) for one activation at one angle.
    a: [d] (= h1), f_ref: [d] (precomputed unperturbed last-hidden response)."""
    R = a.norm()
    td = tangent_dirs(a, d1, d2)  # [3, d]
    perturbed = np.cos(angle_rad) * a.unsqueeze(0) + np.sin(angle_rad) * R * td  # [3, d]
    f_pert = forward_between(mlp, perturbed, inject_after, measure_after)  # [3, d]
    responses = (f_pert - f_ref.unsqueeze(0)).norm(dim=-1)  # [3]
    return responses[0], responses[1], responses[2]


def pair_loss(mlp, acts, f_refs, d1, d2, angles_rad, inject_after, measure_after):
    """Mean over activations & angles of L_a = R_mix / (R1 + R2).
    Also returns mean R1, R2, R_mix (detached) for logging."""
    ratio_sum = 0.0
    r1_sum = r2_sum = rmix_sum = 0.0
    count = 0
    for i in range(len(acts)):
        for angle_rad in angles_rad:
            r1, r2, rmix = pair_responses(
                mlp, acts[i], f_refs[i], d1, d2, angle_rad, inject_after, measure_after)
            ratio_sum = ratio_sum + rmix / (r1 + r2 + 1e-8)
            r1_sum += r1.item(); r2_sum += r2.item(); rmix_sum += rmix.item()
            count += 1
    return ratio_sum / count, (r1_sum / count, r2_sum / count, rmix_sum / count)


def optimize(mlp, acts, f_refs, fixed_dirs, n_new, angles_rad,
             inject_after, measure_after, device, d_model, n_steps, lr, lam, seed,
             log_prefix=""):
    """Optimize new direction(s) on the sphere. n_new=2 (initial pair) or 1 (grow)."""
    g = torch.Generator(device=device).manual_seed(seed)
    new = [torch.randn(d_model, generator=g, device=device, dtype=acts[0].dtype)
           for _ in range(n_new)]
    new = [(v / v.norm()).requires_grad_(True) for v in new]
    optimizer = torch.optim.Adam(new, lr=lr)

    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        if n_new == 2:
            d1, d2 = new
            resp_loss, (mr1, mr2, mrmix) = pair_loss(
                mlp, acts, f_refs, d1, d2, angles_rad, inject_after, measure_after)
            d1u = d1 / (d1.norm() + 1e-10)
            d2u = d2 / (d2.norm() + 1e-10)
            orth = (d1u @ d2u) ** 2
        else:
            d_new = new[0]
            resp_terms = []
            mr1 = mr2 = mrmix = 0.0
            for d_fixed in fixed_dirs:
                rl, (a1, a2, amix) = pair_loss(
                    mlp, acts, f_refs, d_new, d_fixed, angles_rad,
                    inject_after, measure_after)
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

        loss_history.append({'step': step, 'loss': loss.item(),
                             'resp_loss': resp_loss.item(), 'orth': orth.item(),
                             'R1': mr1, 'R2': mr2, 'Rmix': mrmix})
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  orth={orth.item():.4f}  "
                  f"R1={mr1:.3f} R2={mr2:.3f} Rmix={mrmix:.3f}", flush=True)

    return torch.stack([v.detach() for v in new]), loss_history


def eval_dictionary(mlp, acts, f_refs, directions, angles_rad, inject_after, measure_after):
    n_dir = directions.shape[0]
    single_R = np.zeros(n_dir)
    for k in range(n_dir):
        s, c = 0.0, 0
        for i in range(len(acts)):
            for angle_rad in angles_rad:
                with torch.no_grad():
                    r1, _, _ = pair_responses(mlp, acts[i], f_refs[i],
                        directions[k], directions[k], angle_rad, inject_after, measure_after)
                s += r1.item(); c += 1
        single_R[k] = s / c

    pair_L = np.full((n_dir, n_dir), np.nan)
    first_pair = None
    for i_d in range(n_dir):
        for j_d in range(i_d + 1, n_dir):
            r1s, r2s, rmixs, Las = [], [], [], []
            for i in range(len(acts)):
                for angle_rad in angles_rad:
                    with torch.no_grad():
                        r1, r2, rmix = pair_responses(mlp, acts[i], f_refs[i],
                            directions[i_d], directions[j_d], angle_rad,
                            inject_after, measure_after)
                    r1s.append(r1.item()); r2s.append(r2.item()); rmixs.append(rmix.item())
                    Las.append(rmix.item() / (r1.item() + r2.item() + 1e-8))
            pair_L[i_d, j_d] = float(np.mean(Las))
            if i_d == 0 and j_d == 1:
                first_pair = {'R1': np.array(r1s), 'R2': np.array(r2s),
                              'Rmix': np.array(rmixs), 'La': np.array(Las)}
    gram = (directions @ directions.T).cpu().numpy()
    return {'single_R': single_R, 'pair_L': pair_L, 'first_pair': first_pair, 'gram': gram}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--width', type=int, default=200)
    p.add_argument('--activation', type=str, default='GeLU')
    p.add_argument('--inject_hidden', type=int, default=1,
                   help='1-based index of hidden layer to inject at (default 1 = first)')
    p.add_argument('--measure_hidden', type=int, default=-1,
                   help='1-based index of hidden layer to measure at (default -1 = last)')
    p.add_argument('--n_act', type=int, default=10)
    p.add_argument('--n_dict', type=int, default=2, help='target dictionary size (>=2)')
    p.add_argument('--angles', type=float, nargs='+', default=[30.0])
    p.add_argument('--n_steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init_seed', type=int, default=None)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_seed = args.init_seed if args.init_seed is not None else args.seed

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.results_dir = f'results/grok_mlp/superellipse_dict/{tag}_a{angle_tag}'
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
        prm.requires_grad_(False)  # freeze the network; we only optimize directions

    hidden_idxs = hidden_act_indices(mlp)
    inject_after = hidden_idxs[args.inject_hidden - 1]
    measure_after = hidden_idxs[args.measure_hidden if args.measure_hidden < 0
                                else args.measure_hidden - 1]
    d_model = width
    print(f"depth={depth} width={width} | inject after module {inject_after} "
          f"(hidden {args.inject_hidden}), measure after module {measure_after} "
          f"(hidden {args.measure_hidden}) | final module {len(mlp)-1}", flush=True)
    print(f"Config: n_act={args.n_act}, n_dict={args.n_dict}, angles={args.angles} deg, "
          f"n_steps={args.n_steps}, lr={args.lr}, lam={args.lam}", flush=True)

    # Data: extract h1 for n_act inputs
    ds = torchvision.datasets.MNIST(root=args.data_root, train=(args.split == 'train'),
        transform=torchvision.transforms.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.n_act, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    x, _ = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        h1 = forward_to(mlp, x, inject_after)  # [n_act, width]
    acts = [h1[i].clone() for i in range(args.n_act)]
    print(f"Mean h1 norm: {torch.stack(acts).norm(dim=-1).mean().item():.3f}", flush=True)

    # Precompute unperturbed downstream (last-hidden) response for each activation
    f_refs = []
    with torch.no_grad():
        for i in range(args.n_act):
            f_refs.append(forward_between(mlp, acts[i].unsqueeze(0),
                                          inject_after, measure_after).squeeze(0))

    # Stage 1: first pair
    print(f"\n=== Stage 1: learning first pair (d1, d2) ===", flush=True)
    pair, hist1 = optimize(mlp, acts, f_refs, [], 2, angles_rad,
        inject_after, measure_after, device, d_model,
        args.n_steps, args.lr, args.lam, init_seed)
    directions = pair
    loss_histories = {'pair_d1d2': hist1}
    ckpt_path = os.path.join(args.results_dir, 'checkpoint.pkl')
    torch.save({'directions': directions.cpu(), 'args': vars(args)}, ckpt_path)

    # Stage 2+: grow
    for k in range(2, args.n_dict):
        print(f"\n=== Stage {k}: learning d{k+1} against {k} frozen directions ===", flush=True)
        fixed_dirs = [directions[j] for j in range(directions.shape[0])]
        new_dir, hist_k = optimize(mlp, acts, f_refs, fixed_dirs, 1, angles_rad,
            inject_after, measure_after, device, d_model,
            args.n_steps, args.lr, args.lam, init_seed + k, log_prefix=f"d{k+1} ")
        directions = torch.cat([directions, new_dir], dim=0)
        loss_histories[f'd{k+1}'] = hist_k
        torch.save({'directions': directions.cpu(), 'args': vars(args)}, ckpt_path)

    # Diagnostics
    print(f"\n=== Evaluating dictionary ({directions.shape[0]} directions) ===", flush=True)
    diag = eval_dictionary(mlp, acts, f_refs, directions, angles_rad, inject_after, measure_after)
    print(f"Per-direction single-direction response: {np.round(diag['single_R'], 3)}", flush=True)
    print(f"Pairwise L (lower = more superellipse-like, 0.5 = no interference):", flush=True)
    for i_d in range(directions.shape[0]):
        for j_d in range(i_d + 1, directions.shape[0]):
            print(f"  L(d{i_d+1}, d{j_d+1}) = {diag['pair_L'][i_d, j_d]:.4f}", flush=True)

    results = {'args': vars(args), 'directions': directions.cpu(),
               'inject_after': inject_after, 'measure_after': measure_after,
               'angles_deg': args.angles, 'loss_histories': loss_histories,
               'single_R': diag['single_R'], 'pair_L': diag['pair_L'],
               'first_pair': diag['first_pair'], 'gram': diag['gram']}
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plots (same 2x2 summary as the transformer version)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for name, hist in loss_histories.items():
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['loss'] for h in hist], label=f'{name} (total)')
        ax.plot(steps, [h['resp_loss'] for h in hist], '--', alpha=0.6, label=f'{name} (ratio)')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_title('Optimization')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    fp = diag['first_pair']
    if fp is not None:
        ax.hist(fp['La'], bins=20, alpha=0.7, color='C0')
        ax.axvline(0.5, color='gray', ls=':', label='0.5')
        ax.axvline(fp['La'].mean(), color='red', ls='--', label=f"mean={fp['La'].mean():.3f}")
    ax.set_xlabel('L_a = R_mix/(R1+R2)'); ax.set_title('Per-activation loss (first pair)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if fp is not None:
        means = [fp['R1'].mean(), fp['R2'].mean(), fp['Rmix'].mean()]
        ax.bar(['R1', 'R2', 'Rmix'], means, color=['C0', 'C1', 'C2'])
        ax.axhline((means[0] + means[1]) / 2, color='gray', ls=':', label='(R1+R2)/2')
    ax.set_ylabel('mean L2 response'); ax.set_title('Mean responses (first pair)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    im = ax.imshow(diag['gram'], vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title('Gram matrix (d_i . d_j)')
    ax.set_xticks(range(directions.shape[0])); ax.set_yticks(range(directions.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'MNIST MLP d{depth}w{width}, h{args.inject_hidden}->h{args.measure_hidden}, '
                 f'{directions.shape[0]} dirs, angles={args.angles} deg', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}", flush=True)
    print(f"Saved plot: {plot_path}", flush=True)


if __name__ == '__main__':
    main()
