#!/usr/bin/env python3
"""Identify what specific Gemma Scope SAE latents represent, two ways:

  (1) Logit lens of W_dec[f]: project the latent's decoder direction through
      the model's final RMSNorm + unembedding, report top-promoted /
      top-suppressed tokens. This is what the latent 'writes' to the
      residual stream when it fires.
  (2) Neuronpedia API: pull auto-interp explanation + top-activating
      examples (if exposed for this L0).

Default latents are those identified as strongly aligned with d1/d2 in the
sae_overlap report.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from src.model import _get_final_norm, _get_lm_head


def load_sae(repo_id, sae_path, device, dtype):
    npz_path = hf_hub_download(repo_id, sae_path + '/params.npz')
    npz = np.load(npz_path)
    sae = {k: torch.from_numpy(npz[k]).to(device=device, dtype=dtype)
           for k in npz.files}
    return sae


def topk_tokens(logits, tokenizer, k=20):
    vals, idx = torch.topk(logits, k)
    return [(int(i), tokenizer.decode([int(i)]), float(v))
            for v, i in zip(vals, idx)]


def fetch_neuronpedia(model_id, source_id, feature_idx, timeout=10):
    """Try the public Neuronpedia API. Returns dict or None on failure."""
    url = f'https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{feature_idx}'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'steering-plateaus/sae-identify'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError) as e:
        return {'__error__': f'{type(e).__name__}: {e}'}


def summarize_neuronpedia(np_data):
    """Pull a short summary out of the Neuronpedia feature JSON."""
    if np_data is None:
        return '  (Neuronpedia: no data)'
    if '__error__' in np_data:
        return f'  (Neuronpedia error: {np_data["__error__"]})'
    lines = []
    expls = np_data.get('explanations') or []
    for e in expls[:3]:
        desc = e.get('description', '').strip()
        author = e.get('authorId', '?')
        lines.append(f'  explanation [{author}]: {desc}')
    if not lines:
        lines.append('  (no explanations on Neuronpedia)')
    # Top activations
    top_acts = np_data.get('activations') or []
    if top_acts:
        lines.append('  top activating snippets:')
        for a in top_acts[:5]:
            tokens = a.get('tokens') or []
            max_idx = a.get('maxValueTokenIndex')
            text = ''
            if tokens:
                start = max(0, (max_idx or 0) - 6)
                end = min(len(tokens), (max_idx or 0) + 4)
                pieces = []
                for i, t in enumerate(tokens[start:end], start=start):
                    if i == max_idx:
                        pieces.append(f'[[{t}]]')
                    else:
                        pieces.append(t)
                text = ''.join(pieces)
            mv = a.get('maxValue', None)
            lines.append(f'    (max={mv}) ...{text}...')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae_repo', default='google/gemma-scope-9b-pt-res')
    ap.add_argument('--sae_path', default='layer_15/width_16k/average_l0_131')
    ap.add_argument('--np_model', default='gemma-2-9b',
                    help='Neuronpedia model id')
    ap.add_argument('--np_sources', nargs='+',
                    default=['15-gemmascope-res-16k',
                             '15-gemmascope-res-16k-l0-131'],
                    help='Neuronpedia source candidates to try')
    ap.add_argument('--model_name', default='google/gemma-2-9b')
    ap.add_argument('--latents', type=int, nargs='+',
                    default=[7633, 8036, 11583, 3003, 1074, 13655, 14151,
                             8315, 11789, 8467],
                    help='SAE latent indices to identify')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    dtype = dtype_map[args.dtype]

    print(f'loading SAE {args.sae_repo}/{args.sae_path}', flush=True)
    sae = load_sae(args.sae_repo, args.sae_path, args.device, torch.float32)
    W_dec = sae['W_dec']                           # (d_sae, d_model)
    d_sae, d_model = W_dec.shape
    print(f'  d_sae={d_sae}, d_model={d_model}', flush=True)

    print(f'loading {args.model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=args.device)
    model.eval()
    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float()           # (vocab, d_model)
    print(f'  W_U: {tuple(W_U.shape)}', flush=True)

    out_records = []
    report_lines = []
    report_lines.append(f'SAE latent identification')
    report_lines.append(f'  SAE: {args.sae_repo}/{args.sae_path}')
    report_lines.append(f'  Neuronpedia model: {args.np_model}, sources tried: {args.np_sources}')
    report_lines.append(f'  Latents: {args.latents}')
    report_lines.append('=' * 78)
    report_lines.append('')

    for lid in args.latents:
        d_dec = W_dec[lid].to(args.device)          # (d_model,) raw decoder
        d_dec_unit = d_dec / d_dec.norm()
        with torch.no_grad():
            ln_d = ln_f(d_dec_unit.unsqueeze(0).to(
                next(ln_f.parameters()).dtype)).float().squeeze(0)
        logits_raw = (W_U @ d_dec_unit.float()).cpu()
        logits_norm = (W_U @ ln_d).cpu()

        report_lines.append(f'### latent {lid}')
        report_lines.append(f'  ||W_dec[f]|| = {float(d_dec.norm()):.4f}')
        report_lines.append(f'  threshold    = {float(sae["threshold"][lid]):.4f}')
        report_lines.append('')

        # Try Neuronpedia for each source candidate
        np_summary = None
        np_data = None
        for src in args.np_sources:
            data = fetch_neuronpedia(args.np_model, src, lid)
            if data is not None and '__error__' not in data and (
                    data.get('explanations') or data.get('activations')):
                np_data = data
                np_summary = summarize_neuronpedia(data)
                report_lines.append(f'  Neuronpedia source={src}:')
                report_lines.append(np_summary)
                report_lines.append('')
                break
        if np_summary is None:
            report_lines.append('  Neuronpedia: no source matched (errors below)')
            for src in args.np_sources:
                data = fetch_neuronpedia(args.np_model, src, lid)
                err = data.get('__error__') if isinstance(data, dict) else 'no-data'
                report_lines.append(f'    {src}: {err}')
            report_lines.append('')

        # Logit lens (POST-NORM) — this is the standard direct vocab effect.
        report_lines.append(f'  Logit lens W_U @ ln_f(W_dec[{lid}]_unit) — top {args.topk} promoted:')
        for tid, tok, val in topk_tokens(logits_norm, tokenizer, args.topk):
            report_lines.append(f'    {val:+9.3f}  id={tid:6d}  {tok!r}')
        report_lines.append('')
        report_lines.append(f'  ... top {args.topk} suppressed:')
        for tid, tok, val in topk_tokens(-logits_norm, tokenizer, args.topk):
            report_lines.append(f'    {val:+9.3f}  id={tid:6d}  {tok!r}')
        report_lines.append('')
        report_lines.append('-' * 78)
        report_lines.append('')

        out_records.append({
            'latent_id': lid,
            'logits_raw': logits_raw,
            'logits_norm': logits_norm,
            'neuronpedia_raw': np_data,
        })

    report_path = os.path.join(args.out_dir, 'identify_latents_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'wrote {report_path}', flush=True)

    pkl_path = os.path.join(args.out_dir, 'identify_latents.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'records': out_records,
        }, f)
    print(f'wrote {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
