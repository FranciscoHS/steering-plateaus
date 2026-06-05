#!/usr/bin/env python3
"""
Spherical perturbation sweep on a trained MNIST MLP (grok-adversarial recipe).

Mirrors scripts/perturbation/run_perturbation_sweep_spherical.py, adapted to a
plain feed-forward MLP with no sequence/context dimension.

- Inject at h1 = output of the first Linear+activation ("first latent layer").
- Perturb spherically: a' = ||a|| * (cos(theta) * a_unit + sin(theta) * v_perp),
  v_perp a random unit vector orthogonal to a.
- Measure downstream effect at h_{depth-1} = output of the LAST hidden
  Linear+activation, i.e. still in the high-dim (width) space, just BEFORE the
  final Linear that projects down to logits. Metric: L2 distance perturbed vs
  unperturbed (also cosine distance; KL on the softmax of the final logits).
- Two conditions, as in the transformer version: real activations vs
  norm-matched random activations.

Usage:
    python perturb_mlp_spherical.py --ckpt results/grok_mlp/checkpoints/ckpt_sXXANXX.pt
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


def layer_indices(mlp):
    """Module index after the first hidden act (inject) and after the last
    hidden act (measure). Sequential layout for depth d:
      0 Flatten, 1 Linear, 2 Act, 3 Linear, 4 Act, ..., (2d-1) final Linear.
    h1 is output of module index 2; last hidden act is index (2d-2)."""
    act_idxs = [i for i, m in enumerate(mlp) if isinstance(m, tuple(ACT.values()))]
    inject_after = act_idxs[0]      # first activation
    measure_after = act_idxs[-1]    # last activation (before final Linear)
    return inject_after, measure_after


@torch.no_grad()
def forward_to(mlp, x, stop_after):
    """Run modules [0 .. stop_after] inclusive on input x; return that output."""
    out = x
    for m in mlp[:stop_after + 1]:
        out = m(out)
    return out


@torch.no_grad()
def forward_between(mlp, h, start_after, stop_after):
    """Given the output of module index `start_after`, run modules
    [start_after+1 .. stop_after] inclusive and return the result."""
    out = h
    for m in mlp[start_after + 1:stop_after + 1]:
        out = m(out)
    return out


def sample_orthogonal_directions(activations, n_dirs, generator):
    """Random unit vectors orthogonal to each activation (Gram-Schmidt)."""
    n_act, d = activations.shape
    v_perps = torch.zeros(n_act, n_dirs, d, dtype=activations.dtype)
    for i in range(n_act):
        a_unit = activations[i] / activations[i].norm()
        v = torch.randn(n_dirs, d, dtype=activations.dtype, generator=generator)
        v = v - (v @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        v_perps[i] = F.normalize(v, dim=-1)
    return v_perps


@torch.no_grad()
def run_sweep(mlp, activation, v_perps, angles_rad, inject_after, measure_after, device):
    """Spherical sweep for one activation. Returns (l2, cos, kl) each [n_angles, n_dirs]."""
    n_angles, n_dirs = len(angles_rad), v_perps.shape[0]
    l2 = np.zeros((n_angles, n_dirs))
    cos = np.zeros((n_angles, n_dirs))
    kl = np.zeros((n_angles, n_dirs))

    a = activation.to(device)
    R = a.norm()
    a_unit = a / R
    vp = v_perps.to(device)

    ref_measure = forward_between(mlp, a.unsqueeze(0), inject_after, measure_after)  # [1, w]
    ref_logits = forward_between(mlp, a.unsqueeze(0), inject_after, len(mlp) - 1)
    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)

    for k, theta in enumerate(angles_rad):
        perturbed = R * (np.cos(theta) * a_unit.unsqueeze(0) + np.sin(theta) * vp)  # [n_dirs, w]
        pert_measure = forward_between(mlp, perturbed, inject_after, measure_after)
        l2[k] = (pert_measure - ref_measure).float().norm(dim=-1).cpu().numpy()
        cos[k] = (1 - F.cosine_similarity(pert_measure.float(), ref_measure.float(), dim=-1)).cpu().numpy()

        pert_logits = forward_between(mlp, perturbed, inject_after, len(mlp) - 1)
        pert_logp = F.log_softmax(pert_logits.float(), dim=-1)
        ref_p = ref_logp.exp()
        kl[k] = (ref_p * (ref_logp - pert_logp)).sum(dim=-1).cpu().numpy()
    return l2, cos, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--width', type=int, default=200)
    p.add_argument('--activation', type=str, default='GeLU')
    p.add_argument('--n_act', type=int, default=100, help='number of real inputs')
    p.add_argument('--n_dirs', type=int, default=10, help='random directions per activation')
    p.add_argument('--n_angles', type=int, default=60)
    p.add_argument('--max_angle', type=float, default=60.0)
    p.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()

    if args.results_dir is None:
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.results_dir = f'results/grok_mlp/perturb_spherical/{tag}'
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    gen = torch.Generator().manual_seed(args.seed)
    np.random.seed(args.seed)

    # model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    depth = ckpt.get('depth', args.depth)
    width = ckpt.get('width', args.width)
    mlp = build_mlp(depth, width, act=args.activation).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    inject_after, measure_after = layer_indices(mlp)
    print(f"depth={depth} width={width} | inject after module {inject_after}, "
          f"measure after module {measure_after} (final module {len(mlp)-1})", flush=True)

    # data: extract h1 for n_act real inputs
    train = (args.split == 'train')
    ds = torchvision.datasets.MNIST(root=args.data_root, train=train,
        transform=torchvision.transforms.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.n_act, shuffle=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    x, _ = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        real_h1 = forward_to(mlp, x, inject_after).cpu()  # [n_act, width]
    act_norm_mean = real_h1.norm(dim=-1).mean().item()
    print(f"Mean h1 norm: {act_norm_mean:.3f}", flush=True)

    # norm-matched random comparison
    rand_h1 = torch.randn(args.n_act, width, dtype=dtype, generator=gen)
    rand_h1 = F.normalize(rand_h1, dim=-1) * act_norm_mean

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    real_vp = sample_orthogonal_directions(real_h1, args.n_dirs, gen)
    rand_vp = sample_orthogonal_directions(rand_h1, args.n_dirs, gen)

    def sweep_all(acts, vps, label):
        l2 = np.zeros((args.n_act, len(angles_deg), args.n_dirs))
        cos = np.zeros_like(l2)
        kl = np.zeros_like(l2)
        for i in range(args.n_act):
            l2[i], cos[i], kl[i] = run_sweep(mlp, acts[i], vps[i], angles_rad,
                                             inject_after, measure_after, device)
        print(f"  {label} done", flush=True)
        return {'l2': l2, 'cos': cos, 'kl': kl}

    print("Sweeping real...", flush=True)
    real = sweep_all(real_h1, real_vp, 'real')
    print("Sweeping random (norm-matched)...", flush=True)
    rand = sweep_all(rand_h1, rand_vp, 'random')

    results = {'args': vars(args), 'angles_deg': angles_deg, 'angles_rad': angles_rad,
               'act_norm_mean': act_norm_mean, 'depth': depth, 'width': width,
               'real': real, 'random': rand}
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # plot: median over dirs then over acts, IQR band
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for mi, (key, name) in enumerate([('l2', 'L2 distance @ last hidden'),
                                      ('cos', 'Cosine distance @ last hidden'),
                                      ('kl', 'KL divergence @ logits')]):
        ax = axes[mi]
        for cond, color, lab in [(real, 'blue', 'Real'), (rand, 'red', 'Random (norm-matched)')]:
            per_act = np.median(cond[key], axis=2)
            med = np.median(per_act, axis=0)
            q25 = np.percentile(per_act, 25, axis=0)
            q75 = np.percentile(per_act, 75, axis=0)
            ax.plot(angles_deg, med, color=color, label=lab, linewidth=2)
            ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.15)
        ax.set_xlabel('Perturbation angle (degrees)')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.max_angle)
    fig.suptitle(f'MNIST MLP d{depth}w{width} (spherical, inject h1, '
                 f'measure last hidden)\n{os.path.basename(args.ckpt)}', y=1.02)
    plt.tight_layout()
    path = os.path.join(args.results_dir, 'sweep_median.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}", flush=True)


if __name__ == '__main__':
    main()
