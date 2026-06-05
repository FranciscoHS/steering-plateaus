#!/usr/bin/env python3
"""For each decomposition feature d_k, find the SAE latents in the full
dictionary that align most strongly with it. Two views per d_k:

  (a) Decoder-direction cosine: cos(d_k, W_dec[f] / ||W_dec[f]||).
      This says: 'if this latent fires, does it write a residual update
      pointing in the d_k direction?'

  (b) Encoder-direction cosine: cos(d_k, W_enc[:, f] / ||W_enc[:, f]||).
      'Does the latent's read-out direction align with d_k?'

Both are computed over the FULL d_sae latents (no thresholding, no
forward pass), so this is independent of the specific activation we
decomposed.

For each d_k and view, reports the top-K by |cos|, with sign retained,
and pulls auto-interp explanations from Neuronpedia where possible.
"""
import argparse
import json
import os
import pickle
import sys
import urllib.error
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def load_sae_arrays(repo_id, sae_path, dtype=torch.float32):
    npz_path = hf_hub_download(repo_id, sae_path + '/params.npz')
    npz = np.load(npz_path)
    return {k: torch.from_numpy(npz[k]).to(dtype) for k in npz.files}


def fetch_neuronpedia(model_id, source_id, feature_idx, timeout=10):
    url = f'https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{feature_idx}'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'steering-plateaus/sae-global'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'__error__': f'{type(e).__name__}: {e}'}


def get_np_short(data):
    if data is None:
        return None
    if isinstance(data, dict) and '__error__' in data:
        return None
    expls = data.get('explanations') or []
    if not expls:
        return None
    return (expls[0].get('description') or '').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True)
    ap.add_argument('--sae_repo', default='google/gemma-scope-9b-pt-res')
    ap.add_argument('--sae_path', default='layer_15/width_16k/average_l0_131')
    ap.add_argument('--np_model', default='gemma-2-9b')
    ap.add_argument('--np_source', default='15-gemmascope-res-16k')
    ap.add_argument('--topk', type=int, default=25,
                    help='top latents to report per d_k per view')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--no_neuronpedia', action='store_true',
                    help='skip Neuronpedia lookups (faster, offline)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features']
    if not isinstance(feats, torch.Tensor):
        feats = torch.from_numpy(feats)
    feats = feats.float()                                          # (K, h)
    K, hidden = feats.shape
    feats_unit = feats / feats.norm(dim=-1, keepdim=True)
    print(f'loaded {K} features, hidden={hidden}', flush=True)

    print(f'loading SAE {args.sae_repo}/{args.sae_path}', flush=True)
    sae = load_sae_arrays(args.sae_repo, args.sae_path)
    W_dec = sae['W_dec']                                        # (d_sae, h)
    W_enc = sae['W_enc']                                        # (h, d_sae)
    d_sae, _ = W_dec.shape
    print(f'  d_sae={d_sae}, ||W_dec[0]||={float(W_dec[0].norm()):.4f}',
          flush=True)

    W_dec_unit = W_dec / W_dec.norm(dim=-1, keepdim=True)          # (d_sae, h)
    W_enc_unit = W_enc / W_enc.norm(dim=0, keepdim=True)           # (h, d_sae)

    cos_dec = feats_unit @ W_dec_unit.T                            # (K, d_sae)
    cos_enc = feats_unit @ W_enc_unit                              # (K, d_sae)

    # Gather top-K latent IDs across all views so we can batch Neuronpedia.
    top_ids_per_view = {}
    for view_name, cos_mat in [('dec', cos_dec), ('enc', cos_enc)]:
        for k_idx in range(K):
            abs_cos = cos_mat[k_idx].abs()
            vals, idx = torch.topk(abs_cos, args.topk)
            top_ids_per_view[(view_name, k_idx)] = [
                (int(i.item()), float(cos_mat[k_idx, i].item()))
                for i in idx
            ]

    unique_ids = sorted({lid for entries in top_ids_per_view.values()
                         for (lid, _) in entries})
    print(f'  {len(unique_ids)} unique latents across all top-{args.topk} '
          f'lists', flush=True)

    np_cache = {}
    if not args.no_neuronpedia:
        print('fetching Neuronpedia auto-interp (this may take a bit)',
              flush=True)
        for i, lid in enumerate(unique_ids):
            np_cache[lid] = fetch_neuronpedia(args.np_model, args.np_source,
                                              lid)
            if (i + 1) % 25 == 0:
                print(f'  fetched {i+1}/{len(unique_ids)}', flush=True)

    # Build report.
    lines = []
    lines.append('Global SAE-latent alignment with decomposition features')
    lines.append(f'  results: {args.results_pkl}')
    lines.append(f'  SAE: {args.sae_repo}/{args.sae_path}  d_sae={d_sae}')
    lines.append(f'  Neuronpedia: {args.np_model}/{args.np_source}'
                 + ('  (skipped)' if args.no_neuronpedia else ''))
    lines.append(f'  top-{args.topk} per d_k per view, ranked by |cos|')
    lines.append('=' * 100)
    lines.append('')

    for view_name, view_pretty in [('dec', 'DECODER direction cos(d_k, W_dec[f]/||W_dec[f]||)'),
                                   ('enc', 'ENCODER direction cos(d_k, W_enc[:,f]/||W_enc[:,f]||)')]:
        lines.append(f'### {view_pretty}')
        lines.append('')
        for k_idx in range(K):
            lines.append(f'  d{k_idx+1}:')
            entries = top_ids_per_view[(view_name, k_idx)]
            for rank, (lid, cos_v) in enumerate(entries, start=1):
                short = ''
                if not args.no_neuronpedia:
                    short = get_np_short(np_cache.get(lid)) or '(no explanation)'
                lines.append(f'    {rank:>2d}. id={lid:6d}  cos={cos_v:+.4f}'
                             + (f'  | {short}' if short else ''))
            lines.append('')
        lines.append('-' * 100)
        lines.append('')

    report_path = os.path.join(args.out_dir, 'sae_global_alignment.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {report_path}', flush=True)

    out = {
        'args': vars(args),
        'cos_dec': cos_dec.numpy(),       # (K, d_sae)
        'cos_enc': cos_enc.numpy(),       # (K, d_sae)
        'top_per_view': top_ids_per_view,
        'neuronpedia': np_cache,
        'feature_labels': [f'd{k+1}' for k in range(K)],
    }
    pkl_path = os.path.join(args.out_dir, 'sae_global_alignment.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
