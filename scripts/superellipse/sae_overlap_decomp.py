#!/usr/bin/env python3
"""Compare decomposition features against Gemma Scope SAE latents at L15.

Two views:
  (1) Forward: encode a_target through the SAE, take the top-K active latents,
      and compute cosine overlap of each latent's unit decoder direction with
      our K_d decomposition features d_1..d_K.
  (2) Reverse: for each d_k, encode d_k itself through the SAE (treating it
      as if it were a residual-stream vector) and report the top-K latents
      whose encoder direction it lights up. This says 'what does d_k look
      like through the SAE's eyes'.

Also reports the SAE's reconstruction of a_target and how much variance it
captures, for context.

SAE params are loaded from a Gemma Scope npz file (JumpReLU SAE):
    keys = W_enc (d_model, d_sae), b_enc (d_sae,),
           W_dec (d_sae, d_model), b_dec (d_model,),
           threshold (d_sae,)

Encode:
    pre = (x - b_dec) @ W_enc + b_enc
    acts = pre * (pre > threshold)

Default: gemma-scope-9b-pt-res layer_15 width_16k average_l0_131.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def load_sae(repo_id, sae_path, device, dtype):
    npz_path = hf_hub_download(repo_id, sae_path + '/params.npz')
    hparams_path = hf_hub_download(repo_id, sae_path + '/hparams.json')
    npz = np.load(npz_path)
    with open(hparams_path) as f:
        hparams = json.load(f)
    sae = {k: torch.from_numpy(npz[k]).to(device=device, dtype=dtype)
           for k in npz.files}
    return sae, hparams


def encode(sae, x):
    """JumpReLU encode. x: (..., d_model) -> acts: (..., d_sae)."""
    pre = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']
    return pre * (pre > sae['threshold'])


def decode(sae, acts):
    return acts @ sae['W_dec'] + sae['b_dec']


def topk_with_labels(values, k, indices=None):
    vals, idx = torch.topk(values, k)
    rows = []
    for v, i in zip(vals, idx):
        gi = int(indices[i].item()) if indices is not None else int(i.item())
        rows.append((gi, float(v.item())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True)
    ap.add_argument('--sae_repo', default='google/gemma-scope-9b-pt-res')
    ap.add_argument('--sae_path', default='layer_15/width_16k/average_l0_131')
    ap.add_argument('--topk_latents', type=int, default=30)
    ap.add_argument('--topk_overlap', type=int, default=10,
                    help='for each d_k, also list the top SAE latents it '
                         'activates (encoder view)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='float32',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    dtype = dtype_map[args.dtype]

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features'].to(dtype).to(args.device)         # (K, hidden)
    a_target = res['a_target'].to(dtype).to(args.device)      # (hidden,)
    K, hidden = feats.shape
    feats_unit = feats / feats.norm(dim=-1, keepdim=True)
    print(f'loaded {K} features, hidden={hidden}, '
          f'||a_target||={float(a_target.norm()):.3f}', flush=True)

    print(f'loading SAE {args.sae_repo}/{args.sae_path}', flush=True)
    sae, hparams = load_sae(args.sae_repo, args.sae_path, args.device, dtype)
    d_sae = sae['W_enc'].shape[1]
    print(f'  d_sae={d_sae}, threshold mean={sae["threshold"].mean():.4f}, '
          f'hparams={hparams}', flush=True)

    # Reconstruction
    acts = encode(sae, a_target)
    recon = decode(sae, acts)
    err = a_target - recon
    var_explained = 1.0 - (err.float().pow(2).sum() /
                           a_target.float().pow(2).sum())
    n_active = int((acts > 0).sum().item())
    print(f'  L0={n_active}, ||recon||={float(recon.norm()):.3f}, '
          f'||err||={float(err.norm()):.3f}, '
          f'var_explained={float(var_explained):.4f}', flush=True)

    # Forward view: top-K active latents on a_target, then overlap with d_k.
    top_rows = topk_with_labels(acts, args.topk_latents)
    top_idx = torch.tensor([r[0] for r in top_rows], device=args.device)
    top_dec = sae['W_dec'][top_idx]                                    # (k, h)
    top_dec_unit = top_dec / top_dec.norm(dim=-1, keepdim=True)
    # cosine: feats_unit (K, h) @ top_dec_unit.T (h, k) -> (K, k)
    overlap_fwd = (feats_unit.float() @ top_dec_unit.float().T).cpu()

    # Also the abs contribution of each latent to a_target along d_k:
    # contribution_to_d_k = alpha_f * (W_dec[f] . d_k_unit), where
    # alpha_f is the SAE activation magnitude on a_target.
    contribs = (acts[top_idx].unsqueeze(0).float() *
                (feats_unit.float() @ top_dec.float().T)).cpu()       # (K, k)

    # Reverse view: encode each d_k through the SAE (using unit d_k) and find
    # the top latents it lights up.
    rev_top = []
    for k_idx in range(K):
        d_unit = feats_unit[k_idx]
        acts_k = encode(sae, d_unit)
        rev_top.append(topk_with_labels(acts_k, args.topk_overlap))

    # Also compute encoder-direction cosine: how much does d_k align with
    # each latent's encoder column W_enc[:, f] (unit-normalized)?
    W_enc_unit = sae['W_enc'] / sae['W_enc'].norm(dim=0, keepdim=True)
    cos_enc = (feats_unit.float() @ W_enc_unit.float()).cpu()        # (K, d_sae)
    enc_top = []
    for k_idx in range(K):
        vals, idx = torch.topk(cos_enc[k_idx].abs(), args.topk_overlap)
        enc_top.append([(int(i), float(cos_enc[k_idx, i]))
                        for i in idx])

    # Write the report.
    report_path = os.path.join(args.out_dir, 'sae_overlap_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Gemma Scope SAE overlap for decomposition features\n')
        f.write(f'  results: {args.results_pkl}\n')
        f.write(f'  SAE: {args.sae_repo}/{args.sae_path}\n')
        f.write(f'  d_sae={d_sae}, hparams={hparams}\n')
        f.write(f'  ||a_target||={float(a_target.norm()):.3f}, '
                f'L0_active={n_active}, '
                f'var_explained={float(var_explained):.4f}\n')
        f.write('=' * 78 + '\n\n')

        f.write('### Top active latents on a_target (forward view)\n')
        f.write('    rank  latent_id   act_value  '
                'cos with d1..dK -> (and absolute contribution to a along d_k)\n\n')
        for r, (lid, val) in enumerate(top_rows):
            cos_str = '  '.join(f'd{k+1}:{overlap_fwd[k, r]:+.3f}'
                                for k in range(K))
            contrib_str = '  '.join(f'd{k+1}:{contribs[k, r]:+7.2f}'
                                    for k in range(K))
            f.write(f'  {r+1:>3d}  id={lid:6d}  act={val:8.3f}  cos[{cos_str}]'
                    f'   contrib[{contrib_str}]\n')
        f.write('\n' + '-' * 78 + '\n\n')

        f.write('### Reverse view: top SAE latents activated by each d_k\n')
        f.write('    (encode(d_k_unit) through SAE; latents with nonzero acts)\n\n')
        for k_idx in range(K):
            f.write(f'  d{k_idx+1}:\n')
            for lid, val in rev_top[k_idx]:
                f.write(f'      id={lid:6d}  act={val:+.4f}\n')
            f.write('\n')
        f.write('-' * 78 + '\n\n')

        f.write('### Encoder-direction cosine: cos(d_k, W_enc[:, f]_unit)\n')
        f.write('    (top by |cos|; sign retained)\n\n')
        for k_idx in range(K):
            f.write(f'  d{k_idx+1}:\n')
            for lid, val in enc_top[k_idx]:
                f.write(f'      id={lid:6d}  cos={val:+.4f}\n')
            f.write('\n')
    print(f'wrote {report_path}', flush=True)

    out = {
        'args': vars(args),
        'sae_repo': args.sae_repo,
        'sae_path': args.sae_path,
        'd_sae': d_sae,
        'sae_hparams': hparams,
        'a_target_var_explained': float(var_explained),
        'a_target_L0_active': n_active,
        'top_latents': top_rows,                # list of (lid, act)
        'top_latents_overlap_fwd': overlap_fwd.numpy(),     # (K, k)
        'top_latents_contribs': contribs.numpy(),           # (K, k)
        'reverse_top_latents': rev_top,
        'encoder_top_latents': enc_top,
        'feature_labels': [f'd{k+1}' for k in range(K)],
    }
    pkl_path = os.path.join(args.out_dir, 'sae_overlap.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
