#!/usr/bin/env python3
"""Prepare sweep_p_dict_to_sae.py inputs from two TopKSAE checkpoints.

sweep_p_dict_to_sae.py expects (dict blob with 'dictionary', HF-style sae ckpt
with 'W_dec' [d, d_sae], align pkl with found_max_cos/found_max_idx). This shim
builds those from our popt/baseline TopKSAE checkpoints + the alignment.pkl of
sae_vs_sae_alignment.py, restricted to ALIVE popt atoms whose best-match
baseline latent is also alive (dead atoms are shared-init clutter), subsampled
so the sweep's O(n^2) pair enumeration stays cheap.

Usage:
    python scripts/superellipse/prep_popt_dtosae_inputs.py \\
        --popt .../l10_lam1/sae_weights.pt --baseline .../l10_doubled_k32_2M/sae_weights.pt \\
        --align_pkl .../alignment.pkl --n_sub 400 --seed 0 --out_dir .../dtosae_inputs
"""
import argparse
import os
import pickle

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--popt', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--align_pkl', required=True,
                    help='alignment.pkl from sae_vs_sae_alignment.py (popt=A, baseline=B)')
    ap.add_argument('--n_sub', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    W_popt = torch.load(args.popt, map_location='cpu',
                        weights_only=False)['state_dict']['W_dec'].float()
    W_base = torch.load(args.baseline, map_location='cpu',
                        weights_only=False)['state_dict']['W_dec'].float()

    al = pickle.load(open(args.align_pkl, 'rb'))
    a2b_cos = np.asarray(al['a2b_cos'])
    a2b_idx = np.asarray(al['a2b_idx'])
    fires_a = np.asarray(al['fires_a'])
    fires_b = np.asarray(al['fires_b'])

    eligible = np.where((fires_a > 0) & (fires_b[a2b_idx] > 0))[0]
    print(f'eligible popt atoms (alive, alive match): {eligible.size}')
    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(eligible, size=min(args.n_sub, eligible.size),
                             replace=False))
    print(f'subsampled: {sub.size}  '
          f'median match cos: {np.median(a2b_cos[sub]):.3f}')

    torch.save({'dictionary': W_popt[torch.from_numpy(sub)],
                'orig_idx': sub}, os.path.join(args.out_dir, 'dict_popt_sub.pt'))
    torch.save({'W_dec': W_base.T.contiguous()},
               os.path.join(args.out_dir, 'sae_baseline_shim.pt'))
    with open(os.path.join(args.out_dir, 'align_sub.pkl'), 'wb') as f:
        pickle.dump({'found_max_cos': a2b_cos[sub],
                     'found_max_idx': a2b_idx[sub], 'orig_idx': sub}, f)
    print(f'Saved inputs to {args.out_dir}')


if __name__ == '__main__':
    main()
