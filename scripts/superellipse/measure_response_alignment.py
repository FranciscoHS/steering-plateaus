#!/usr/bin/env python3
"""Are the optimized atoms less aligned than the features in the space that
controls p? p between two directions is set by cos(A, B), where A, B are their
cubic OUTPUT response images (A.B=0 -> p=6, A.B parallel -> p=3). Input-space
cosine is the wrong space (features are exactly orthogonal there). Here we
measure cos(Delta_out_u, Delta_out_v) at small alpha, plus the linear hidden
overlap cos(W_enc u, W_enc v), for feature pairs vs atom pairs.
"""
import argparse, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scripts.superellipse.fit_pairs_toymodel import load_toy_model


@torch.no_grad()
def response_images(model, x_clean, base, dirs, alpha):
    """Delta_out for each unit direction (rows of dirs), at magnitude alpha."""
    out = model(x_clean.unsqueeze(0) + alpha * dirs)
    return out - base.unsqueeze(0)                       # [n, d_out]


def pairwise_abscos(M, pairs):
    Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-12)
    return np.array([float(abs((Mn[i] * Mn[j]).sum().item())) for i, j in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atoms', default='results/dictionary/toymodel_phase1_tanh3/'
                    'optimize_features_randcontrast/atoms.pt')
    ap.add_argument('--ckpt', default='~/Research/denoising/runs/phase1_tanh3/model.pt')
    ap.add_argument('--denoising_repo', default='~/Research/denoising')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--n_feat_pairs', type=int, default=190)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_toy_model(args.ckpt, args.denoising_repo, device)
    d = cfg['d']; S = (0, 1)
    x_clean = torch.zeros(d, device=device); x_clean[0] = 1; x_clean[1] = 1
    base = model(x_clean.unsqueeze(0))[0].detach()
    x_hat = x_clean / x_clean.norm()
    W_enc = model.fc1.weight.detach()                    # [H, d]
    rng = np.random.default_rng(args.seed)

    # --- atoms (orthogonalized to x_clean, as in the atom x atom fit) ---
    A = torch.load(args.atoms.replace('~', __import__('os').path.expanduser('~')),
                   map_location=device)['dictionary'].to(device).float()
    A = A - (A @ x_hat).unsqueeze(-1) * x_hat
    A = A / A.norm(dim=-1, keepdim=True)
    K = A.shape[0]
    atom_pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]

    # --- features e_j (j not in S) ---
    inactive = [i for i in range(d) if i not in S]
    nfp = min(args.n_feat_pairs, len(atom_pairs))
    feat_pair_idx = set()
    while len(feat_pair_idx) < nfp:
        j, k = rng.choice(inactive, size=2, replace=False)
        feat_pair_idx.add((int(min(j, k)), int(max(j, k))))
    feat_pair_idx = sorted(feat_pair_idx)
    feats = sorted({i for pr in feat_pair_idx for i in pr})
    fpos = {f: n for n, f in enumerate(feats)}
    F = torch.zeros(len(feats), d, device=device)
    for n, f in enumerate(feats):
        F[n, f] = 1.0
    feat_pairs_local = [(fpos[j], fpos[k]) for j, k in feat_pair_idx]

    # response images
    dA = response_images(model, x_clean, base, A, args.alpha)
    dF = response_images(model, x_clean, base, F, args.alpha)
    # hidden linear overlaps
    hA = A @ W_enc.T
    hF = F @ W_enc.T

    out_cos_atom = pairwise_abscos(dA, atom_pairs)
    out_cos_feat = pairwise_abscos(dF, feat_pairs_local)
    hid_cos_atom = pairwise_abscos(hA, atom_pairs)
    hid_cos_feat = pairwise_abscos(hF, feat_pairs_local)

    def stat(name, v):
        print(f'  {name:34s} median={np.median(v):.4f}  mean={np.mean(v):.4f}  '
              f'IQR=({np.quantile(v,.25):.4f},{np.quantile(v,.75):.4f})  '
              f'max={v.max():.4f}  n={len(v)}')

    print(f'd={d} H={cfg["hidden_size"]} alpha={args.alpha}  '
          f'atom_pairs={len(atom_pairs)} feat_pairs={len(feat_pairs_local)}\n')
    print('|cos| of OUTPUT response images Delta_out  (controls p; 0->p6, 1->p3):')
    stat('atoms', out_cos_atom)
    stat('features', out_cos_feat)
    print('\n|cos| of hidden linear overlap W_enc d:')
    stat('atoms', hid_cos_atom)
    stat('features', hid_cos_feat)
    print('\n|cos| in raw input space (reference):')
    stat('atoms', pairwise_abscos(A, atom_pairs))
    stat('features', pairwise_abscos(F, feat_pairs_local))


if __name__ == '__main__':
    main()
