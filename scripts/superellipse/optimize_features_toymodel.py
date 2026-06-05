#!/usr/bin/env python3
"""Do we recover the true features by optimizing against random directions?

Ports the randcontrast/step-resample dictionary build
(scripts/superellipse/build_dictionary.py + decompose_plateau_breaking.
batched_proxy_loss) into the denoising toy model, where the ground-truth
features are exactly the canonical basis e_0..e_{d-1}.

Per atom d_k (unit vector in input space), against fresh random contrast dirs c
(resampled every gradient step), with a fixed two-hot anchor x_clean (S=(0,1)):

    R_f   = ||model(x_clean + a*d_p)   - model(x_clean)||      (a single dir)
    R_c   = ||model(x_clean + a*c_p)   - model(x_clean)||      (per contrast)
    R_mix = ||model(x_clean + a*mix)   - model(x_clean)||      mix=norm(d_p+c_p)
    L(d_k) = mean_{c, a} [ R_mix / (R_f + R_c) ]               (proxy)
           + lam_prev * mean_prev (d_k . d_prev)^2             (spread atoms)

d_p / c_p are orthogonalized against x_clean (single-vector, mirroring the
transformer proxy). Perturbations are Euclidean (flat input space), averaged
over a small set of magnitudes `a`.

Recovery metric: for a unit atom, cos(d_k, e_i) = |d_k[i]|, so
max_i |d_k[i]| ~ 1 means d_k is a pure canonical feature. We report the
distribution of max|coord|, the argmax feature per atom, and whether the K
atoms hit distinct off-features (i not in S).

Usage:
    python scripts/superellipse/optimize_features_toymodel.py \\
        --ckpt /workspace/denoising/runs/phase1_tanh3/model.pt \\
        --denoising_repo /workspace/denoising --device cuda \\
        --K 20 --n_steps 250
"""

import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.superellipse.fit_pairs_toymodel import load_toy_model


def toy_proxy_loss(model, x_clean, base, x_hat, d, contrast, alphas):
    """Euclidean plateau-breaking proxy mean_{c,a} R_mix/(R_f+R_c).

    x_clean: [din]   base=model(x_clean): [dout] (detached)   x_hat: x_clean unit
    d:       [din] (requires grad)        contrast: [K, din] (unit)
    alphas:  1D tensor of perturbation magnitudes
    Returns (loss, (R_f_mean, R_c_mean, R_mix_mean)).
    """
    K = contrast.shape[0]
    # orthogonalize against the anchor (single-vector, as in batched_proxy_loss)
    d_p = d - (d @ x_hat) * x_hat
    d_p = d_p / (d_p.norm() + 1e-10)                                # [din]
    c_p = contrast - (contrast @ x_hat).unsqueeze(-1) * x_hat       # [K, din]
    c_p = c_p / (c_p.norm(dim=-1, keepdim=True) + 1e-10)
    mix = d_p.unsqueeze(0) + c_p
    mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-10)            # [K, din]

    # dirs stacked: [d_p (1), c_p (K), mix (K)] = [2K+1, din]
    dirs = torch.cat([d_p.unsqueeze(0), c_p, mix], dim=0)          # [2K+1, din]
    na = alphas.numel()
    # points: [na, 2K+1, din]
    pts = x_clean.view(1, 1, -1) + alphas.view(na, 1, 1) * dirs.view(1, 2 * K + 1, -1)
    out = model(pts.reshape(na * (2 * K + 1), -1))
    resp = (out - base.unsqueeze(0)).norm(dim=-1).view(na, 2 * K + 1)
    R_f = resp[:, 0:1]                                              # [na, 1]
    R_c = resp[:, 1:1 + K]                                          # [na, K]
    R_mix = resp[:, 1 + K:]                                         # [na, K]
    ratio = R_mix / (R_f + R_c + 1e-8)                             # [na, K]
    loss = ratio.mean()
    return loss, (R_f.mean().item(), R_c.mean().item(), R_mix.mean().item())


def optimize_one_atom(model, x_clean, base, x_hat, prev_stack, alphas,
                      din, n_contrast, n_steps, lr, lam_prev, device,
                      seed, contrast_seed_base, log_every):
    g = torch.Generator(device=device).manual_seed(seed)
    d = torch.randn(din, generator=g, device=device)
    d = (d / d.norm()).requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)
    history = []
    for step in range(n_steps):
        gc = torch.Generator(device=device).manual_seed(contrast_seed_base + step)
        contrast = torch.randn(n_contrast, din, generator=gc, device=device)
        contrast = contrast / contrast.norm(dim=-1, keepdim=True)
        opt.zero_grad()
        proxy, (rf, rc, rmix) = toy_proxy_loss(model, x_clean, base, x_hat,
                                               d, contrast, alphas)
        du = d / (d.norm() + 1e-10)
        if prev_stack is not None and prev_stack.shape[0] > 0:
            prev_overlap = ((du.unsqueeze(0) * prev_stack).sum(-1) ** 2).mean()
        else:
            prev_overlap = torch.zeros((), device=device)
        loss = proxy + lam_prev * prev_overlap
        loss.backward()
        opt.step()
        with torch.no_grad():
            d.data.div_(d.data.norm())
        history.append({'step': step, 'loss': loss.item(),
                        'proxy': proxy.item(), 'prev_overlap': float(prev_overlap),
                        'R_f': rf, 'R_c': rc, 'R_mix': rmix})
        if log_every and (step % log_every == 0 or step == n_steps - 1):
            print(f'    step {step:4d}  L={loss.item():.4f}  proxy={proxy.item():.4f}  '
                  f'prev={float(prev_overlap):.4f}', flush=True)
    return d.detach(), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--K', type=int, default=20, help='number of atoms')
    ap.add_argument('--n_steps', type=int, default=250)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--lam_prev', type=float, default=1.0)
    ap.add_argument('--n_contrast', type=int, default=10)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                    help='Euclidean perturbation magnitudes (averaged over). '
                         'Validated regime where the proxy prefers features over '
                         'random; alpha>~2 weakens/inverts this (diagnose_toy_proxy.py)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contrast_seed_base', type=int, default=7_000_003)
    ap.add_argument('--log_every', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results/dictionary/toymodel_phase1_tanh3/optimize_features_randcontrast')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    din = cfg['d']
    sparsity = cfg['sparsity']
    S = tuple(range(sparsity))

    x_clean = torch.zeros(din, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0].detach()
    x_hat = x_clean / x_clean.norm()
    alphas = torch.tensor(args.alphas, dtype=torch.float32, device=device)

    print(f'device={device} d={din} H={cfg["hidden_size"]} act={cfg["activation"]} '
          f'S={S}', flush=True)
    print(f'K={args.K} n_steps={args.n_steps} lr={args.lr} lam_prev={args.lam_prev} '
          f'n_contrast={args.n_contrast} alphas={args.alphas}', flush=True)

    dictionary = torch.zeros(args.K, din, device=device)
    metadata = {'args': vars(args), 'config': cfg, 'S': S, 'atoms': []}
    for k in range(args.K):
        prev_stack = dictionary[:k] if k > 0 else None
        d_k, hist = optimize_one_atom(
            model, x_clean, base, x_hat, prev_stack, alphas, din,
            args.n_contrast, args.n_steps, args.lr, args.lam_prev, device,
            seed=args.seed * 1_000_003 + k,
            contrast_seed_base=args.seed * args.contrast_seed_base + k * 100_003,
            log_every=args.log_every)
        dictionary[k] = d_k
        # recovery for this atom
        coord = d_k.abs()
        max_coord = float(coord.max().item())
        argmax = int(coord.argmax().item())
        metadata['atoms'].append({
            'k': k, 'max_coord': max_coord, 'argmax_feature': argmax,
            'in_S': argmax in S, 'final_loss': hist[-1]['loss'],
            'final_proxy': hist[-1]['proxy'],
        })
        print(f'  atom {k:3d}: max|coord|={max_coord:.3f}  feature={argmax:4d}'
              f'{" (in S!)" if argmax in S else "":8s}  '
              f'proxy={hist[-1]["proxy"]:.4f}  L={hist[-1]["loss"]:.4f}', flush=True)

    # ---- aggregate recovery ----
    dn = dictionary / dictionary.norm(dim=-1, keepdim=True)
    max_coords = dn.abs().max(dim=-1).values.cpu().numpy()
    argmaxes = dn.abs().argmax(dim=-1).cpu().numpy()
    n_recovered = int((max_coords > 0.9).sum())
    distinct = len(set(argmaxes.tolist()))
    n_in_S = int(sum(a in S for a in argmaxes.tolist()))
    # random-baseline max|coord| for reference
    gb = torch.Generator(device=device).manual_seed(0)
    rb = torch.randn(2000, din, generator=gb, device=device)
    rb = rb / rb.norm(dim=-1, keepdim=True)
    rand_maxcoord = float(rb.abs().max(dim=-1).values.mean().item())

    print(f'\n=== Recovery summary (K={args.K}) ===')
    print(f'  max|coord|: median={np.median(max_coords):.3f}  '
          f'mean={np.mean(max_coords):.3f}  min={max_coords.min():.3f}  '
          f'max={max_coords.max():.3f}')
    print(f'  atoms with max|coord|>0.9 (pure feature): {n_recovered}/{args.K}')
    print(f'  distinct argmax features: {distinct}/{args.K}  '
          f'(in active set S: {n_in_S})')
    print(f'  random-vector baseline max|coord| (d={din}): {rand_maxcoord:.3f}')
    print(f'  argmax features: {sorted(argmaxes.tolist())}')

    metadata['recovery'] = {
        'max_coords': max_coords, 'argmaxes': argmaxes,
        'n_recovered_0p9': n_recovered, 'distinct': distinct, 'n_in_S': n_in_S,
        'rand_baseline_maxcoord': rand_maxcoord,
    }
    torch.save({'dictionary': dictionary.cpu(), 'K': args.K, 'args': vars(args)},
               os.path.join(args.out, 'atoms.pt'))
    with open(os.path.join(args.out, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # plot: recovery histogram + sorted bar
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(max_coords, bins=np.linspace(0, 1, 26), edgecolor='black', alpha=0.85)
    ax.axvline(rand_maxcoord, color='C3', ls='--', lw=2,
               label=f'random baseline={rand_maxcoord:.2f}')
    ax.axvline(0.9, color='gray', ls=':', lw=1, label='0.9 (pure feature)')
    ax.set_xlabel('max_i |cos(atom, e_i)|')
    ax.set_ylabel('# atoms')
    ax.set_title(f'feature recovery ({n_recovered}/{args.K} atoms > 0.9)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    order = np.argsort(max_coords)[::-1]
    ax.bar(range(args.K), max_coords[order], color='C0', alpha=0.85)
    ax.axhline(rand_maxcoord, color='C3', ls='--', lw=1.5, label='random baseline')
    ax.set_xlabel('atom (sorted)')
    ax.set_ylabel('max|coord|')
    ax.set_title(f'distinct features hit: {distinct}/{args.K}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f'toy denoiser ({cfg["activation"]}, d={din}, H={cfg["hidden_size"]})  '
                 f'optimize vs random contrast (step-resample)', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'recovery.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {os.path.join(args.out, "atoms.pt")}')
    print(f'Saved {os.path.join(args.out, "metadata.pkl")}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
