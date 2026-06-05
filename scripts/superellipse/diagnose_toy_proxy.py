#!/usr/bin/env python3
"""Does the random-contrast proxy actually prefer a true feature over a random
direction in the toy model? Compares mean_c R_mix/(R_f+R_c) for
  d = e_j (true off-feature)  vs  d = random unit  vs  d = (e_j+e_k)/sqrt2
across single perturbation magnitudes alpha, with random contrasts.
Lower proxy = preferred by the optimizer.
"""
import argparse, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scripts.superellipse.fit_pairs_toymodel import load_toy_model
from scripts.superellipse.optimize_features_toymodel import toy_proxy_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n_contrast', type=int, default=64)
    ap.add_argument('--n_trials', type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    din = cfg['d']; S = (0, 1)
    x_clean = torch.zeros(din, device=device); x_clean[0] = 1; x_clean[1] = 1
    base = model(x_clean.unsqueeze(0))[0].detach()
    x_hat = x_clean / x_clean.norm()
    rng = np.random.default_rng(0)

    alphas_list = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    print(f'{"alpha":>6} | {"feature e_j":>22} | {"random dir":>22} | {"two-feature":>22}')
    print(f'{"":>6} | {"median[IQR]":>22} | {"median[IQR]":>22} | {"median[IQR]":>22}')
    for a in alphas_list:
        alphas = torch.tensor([a], dtype=torch.float32, device=device)
        feat, rand, two = [], [], []
        for _ in range(args.n_trials):
            gc = torch.Generator(device=device).manual_seed(int(rng.integers(1 << 30)))
            c = torch.randn(args.n_contrast, din, generator=gc, device=device)
            c = c / c.norm(dim=-1, keepdim=True)
            j = int(rng.integers(2, din))
            d_feat = torch.zeros(din, device=device); d_feat[j] = 1.0
            k = int(rng.integers(2, din));
            while k == j: k = int(rng.integers(2, din))
            d_two = torch.zeros(din, device=device); d_two[j] = 1; d_two[k] = 1
            d_two = d_two / d_two.norm()
            dr = torch.randn(din, generator=torch.Generator(device=device).manual_seed(int(rng.integers(1<<30))), device=device)
            dr = dr / dr.norm()
            feat.append(toy_proxy_loss(model, x_clean, base, x_hat, d_feat, c, alphas)[0].item())
            rand.append(toy_proxy_loss(model, x_clean, base, x_hat, dr, c, alphas)[0].item())
            two.append(toy_proxy_loss(model, x_clean, base, x_hat, d_two, c, alphas)[0].item())
        def fmt(v):
            v = np.array(v); return f'{np.median(v):.3f}[{np.quantile(v,.25):.3f},{np.quantile(v,.75):.3f}]'
        print(f'{a:>6.2f} | {fmt(feat):>22} | {fmt(rand):>22} | {fmt(two):>22}')


if __name__ == '__main__':
    main()
