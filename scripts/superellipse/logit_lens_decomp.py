#!/usr/bin/env python3
"""Logit-lens / direct vocab effect for the K decomposition features.

For each unit direction d_k from a decomposition results.pkl, project through
the model's unembedding W_U (the lm_head weight matrix) and report the most
promoted and most suppressed tokens. Two variants are computed per direction:

  - raw:       logits = W_U @ d_k
  - post-norm: logits = W_U @ ln_f(d_k.unsqueeze(0)).squeeze(0)

The post-norm variant is the standard "logit lens" treatment (apply the
final RMSNorm before unembedding); the raw variant is the linear direct
effect ignoring norm. Both are reported. Magnitudes differ but rankings
are usually similar; show both so we can sanity-check.

a_target (the activation being decomposed) is included as a reference row
so we can see what tokens the original activation itself promotes.

Output: a single text report (top/bottom K per direction, both variants)
plus a pickle with the full per-direction logit vectors.

Run on a pod where gemma-2-9b is cached at HF_HOME=/workspace/hf_cache:

    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/logit_lens_decomp.py \
      --results_pkl results/decomp_plateau/gemma-2-9b_L15_a30_lam1.0/results.pkl \
      --model_name google/gemma-2-9b \
      --out_dir results/decomp_plateau/gemma-2-9b_L15_a30_lam1.0/logit_lens
"""
import argparse
import os
import pickle
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from src.model import _get_final_norm, _get_lm_head


def topk_tokens(logits, tokenizer, k=20):
    vals, idx = torch.topk(logits, k)
    return [(int(i), tokenizer.decode([int(i)]), float(v))
            for v, i in zip(vals, idx)]


def fmt_token_list(rows, indent='    '):
    lines = []
    for tid, tok, val in rows:
        lines.append(f'{indent}{val:+9.4f}  id={tid:6d}  {tok!r}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='decomp results.pkl with features (K, hidden) + a_target')
    ap.add_argument('--model_name', default='google/gemma-2-9b')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features'].float()           # (K, hidden)
    a_target = res['a_target'].float() if 'a_target' in res else None  # optional
    K, hidden = feats.shape
    print(f'loaded {K} features (hidden={hidden})', flush=True)
    if a_target is not None:
        print(f'||a_target|| = {a_target.norm():.3f}', flush=True)

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    print(f'loading {args.model_name} ({args.dtype}) on {args.device}',
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype_map[args.dtype],
        device_map=args.device)
    model.eval()

    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float()      # (vocab, hidden)
    print(f'W_U shape: {tuple(W_U.shape)}', flush=True)

    device = next(model.parameters()).device
    feats = feats.to(device)
    rows_raw, rows_norm = [], []

    # a_target as a reference row (use unit-norm a too so magnitudes are
    # comparable to feature directions). Skipped when no target is present
    # (e.g. a plain dictionary rather than a single-activation decomposition).
    if a_target is not None:
        a_target = a_target.to(device)
        a_unit = (a_target / a_target.norm()).unsqueeze(0)
        with torch.no_grad():
            ln_a = ln_f(a_unit.to(next(ln_f.parameters()).dtype)).float().squeeze(0)
        logits_a_raw = (W_U @ a_unit.squeeze(0).float()).cpu()
        logits_a_norm = (W_U @ ln_a).cpu()
        rows_raw.append(('a_target_hat', logits_a_raw))
        rows_norm.append(('a_target_hat', logits_a_norm))

    for k in range(K):
        d = feats[k]
        d_unit = d / d.norm()
        with torch.no_grad():
            ln_d = ln_f(d_unit.unsqueeze(0).to(next(ln_f.parameters()).dtype))
            ln_d = ln_d.float().squeeze(0)
        logits_raw = (W_U @ d_unit).cpu()
        logits_norm = (W_U @ ln_d).cpu()
        rows_raw.append((f'd{k+1}', logits_raw))
        rows_norm.append((f'd{k+1}', logits_norm))

    # Write the text report.
    report_path = os.path.join(args.out_dir, 'logit_lens_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Logit lens for decomposition features from {args.results_pkl}\n')
        f.write(f'Model: {args.model_name}\n')
        f.write(f'K = {K}, topk = {args.topk}\n')
        if a_target is not None:
            f.write(f'||a_target|| = {float(a_target.norm()):.4f}\n')
        f.write('=' * 78 + '\n\n')

        for variant, rows in [('RAW (W_U @ d)', rows_raw),
                              ('POST-NORM (W_U @ ln_f(d))', rows_norm)]:
            f.write(f'### {variant}\n\n')
            for label, lg in rows:
                f.write(f'  {label}:\n')
                f.write(f'    TOP {args.topk} promoted:\n')
                f.write(fmt_token_list(topk_tokens(lg, tokenizer, args.topk),
                                       indent='      ') + '\n')
                f.write(f'    TOP {args.topk} suppressed:\n')
                f.write(fmt_token_list(topk_tokens(-lg, tokenizer, args.topk),
                                       indent='      ') + '\n\n')
            f.write('-' * 78 + '\n\n')
    print(f'wrote {report_path}', flush=True)

    # Save the raw logits for downstream analysis. When a_target is present it
    # is the first row; otherwise the rows are exactly the K features.
    off = 1 if a_target is not None else 0
    out = {
        'args': vars(args),
        'model': args.model_name,
        'feature_labels': [f'd{k+1}' for k in range(K)],
        'feature_logits_raw': torch.stack([r[1] for r in rows_raw[off:]]),
        'feature_logits_norm': torch.stack([r[1] for r in rows_norm[off:]]),
        'vocab_size': W_U.shape[0],
    }
    if a_target is not None:
        out['a_target_unit_logits_raw'] = rows_raw[0][1]
        out['a_target_unit_logits_norm'] = rows_norm[0][1]
    pkl_path = os.path.join(args.out_dir, 'logit_lens.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
