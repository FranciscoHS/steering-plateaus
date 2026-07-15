#!/usr/bin/env python3
"""(1) Bootstrap 95% CI on the per-pair median p. (2) Greedily keep a maximal
set of dictionary atoms whose pairwise |cosine| stays below a threshold, and
report how many remain (a diversity/near-orthogonality check)."""
import argparse, pickle
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_pkl', default=None)
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--cos_thresh', type=float, default=0.15)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_subset', default=None,
                    help='save the |cos|<cos_thresh greedy subset as a dict.pt here')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- median CI (optional) ---
    if args.fit_pkl:
        d = pickle.load(open(args.fit_pkl, 'rb'))
        p = np.asarray(d['p_fits'], float); p = p[np.isfinite(p)]
        rng = np.random.default_rng(args.seed)
        meds = np.median(rng.choice(p, size=(args.n_boot, len(p)), replace=True), axis=1)
        lo, hi = np.percentile(meds, [2.5, 97.5])
        print(f'per-pair median p = {np.median(p):.3f}  '
              f'95% CI (bootstrap n={args.n_boot}) [{lo:.3f}, {hi:.3f}]  (n_pairs={len(p)})')

    # --- greedy |cos| < thresh subset (build order) ---
    blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].float()
    n_done = blob.get('n_done', D.shape[0])
    D = D[:n_done]
    D = (D / (D.norm(dim=1, keepdim=True) + 1e-12)).to(dev)
    K = D.shape[0]
    kept_idx = [0]
    kept = D[0:1].clone()                      # [m, d]
    thr = args.cos_thresh
    for i in range(1, K):
        if (kept @ D[i]).abs().max().item() < thr:
            kept_idx.append(i)
            kept = torch.cat([kept, D[i:i+1]], 0)
    print(f'dict atoms = {K}; greedy |cos| < {thr} subset = {len(kept_idx)} atoms')
    if args.save_subset:
        torch.save({'dictionary': D[kept_idx].cpu(), 'n_done': len(kept_idx),
                    'src_idx': kept_idx, 'cos_thresh': thr,
                    'src_dict': args.dict_path}, args.save_subset)
        print(f'  saved {len(kept_idx)}-atom subset -> {args.save_subset}')
    # also report a few other thresholds for context
    for t in (0.1, 0.2, 0.3):
        kidx = [0]; kp = D[0:1].clone()
        for i in range(1, K):
            if (kp @ D[i]).abs().max().item() < t:
                kidx.append(i); kp = torch.cat([kp, D[i:i+1]], 0)
        print(f'  |cos| < {t}: {len(kidx)} atoms')


if __name__ == '__main__':
    main()
