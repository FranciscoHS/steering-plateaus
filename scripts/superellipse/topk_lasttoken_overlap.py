#!/usr/bin/env python3
"""Collect last-token residual-stream activations at layer L across a FineWeb
corpus, then for each decomposition feature d_k report the prompts whose
last-token activation is most cosine-aligned with d_k (top + bottom).

Per-prompt activation: residual stream at layer L, last token only.
Metric: cosine similarity (norm-independent), matching top_token_analysis.py.
"""
import argparse
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, '.')
from src.data import load_fineweb_fixed_length
from src.model import load_model, _get_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_pkl', required=True,
                    help='decomposition results.pkl with features (K, h)')
    ap.add_argument('--n_seq', type=int, default=4000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--topk', type=int, default=30)
    ap.add_argument('--ctx_window', type=int, default=20,
                    help='tokens of context to show before the last token')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results_pkl, 'rb') as f:
        res = pickle.load(f)
    feats = res['features']
    if not isinstance(feats, torch.Tensor):
        feats = torch.from_numpy(feats)
    feats = feats.float()
    K, hidden = feats.shape
    feats_unit = feats / feats.norm(dim=-1, keepdim=True)
    model_name = res['args']['model']
    layer = res['args']['layer']
    print(f'{K} features, hidden={hidden}, model={model_name}, layer={layer}',
          flush=True)

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    print(f'loading {model_name} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(model_name,
                                          dtype=dtype_map[args.dtype])
    feats_unit = feats_unit.to(device)
    model.eval()

    # Hook to grab residual stream at layer L, last position only.
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['act'] = h[:, -1, :].detach().to(torch.float32).cpu()

    block = _get_blocks(model)[layer]
    handle = block.register_forward_hook(hook)

    # Load corpus.
    tokens_list = load_fineweb_fixed_length(args.n_seq, tokenizer,
                                            seq_len=args.seq_len,
                                            seed=args.seed)
    n_seq = len(tokens_list)
    print(f'collecting last-token L{layer} acts for {n_seq} sequences '
          f'(seq_len={args.seq_len}, bs={args.batch_size})', flush=True)

    all_acts = torch.empty(n_seq, hidden, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_seq, args.batch_size):
            batch = torch.stack(tokens_list[i:i + args.batch_size]).to(device)
            _ = model(batch)
            n = batch.shape[0]
            all_acts[i:i + n] = captured['act']
            if (i // args.batch_size) % 25 == 0:
                rate = (i + n) / max(1e-9, time.time() - t0)
                print(f'  {i+n}/{n_seq}  rate={rate:.1f}/s', flush=True)
    handle.remove()
    print(f'done in {time.time()-t0:.1f}s', flush=True)

    # Cosine similarity: (n_seq, h) @ (h, K) -> (n_seq, K)
    norms = all_acts.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    acts_unit = all_acts / norms
    cos = (acts_unit @ feats_unit.cpu().T)                         # (n_seq, K)
    norm_a = norms.squeeze(-1)
    raw_dot = (all_acts @ feats_unit.cpu().T)                      # signed dot

    # Decode tokens for the report.
    def context_str(seq_tokens, last_idx, win):
        start = max(0, last_idx - win)
        text = tokenizer.decode(seq_tokens[start:last_idx].tolist(),
                                skip_special_tokens=False)
        last_tok = tokenizer.decode([int(seq_tokens[last_idx])],
                                    skip_special_tokens=False)
        return text, last_tok

    # Build report.
    report_lines = []
    report_lines.append('Top last-token activations vs decomposition features')
    report_lines.append(f'  results_pkl: {args.results_pkl}')
    report_lines.append(f'  corpus: FineWeb-Edu n={n_seq} seq_len={args.seq_len} '
                        f'seed={args.seed}')
    report_lines.append(f'  model={model_name}  layer={layer}  '
                        f'metric=cosine(a_last_token, d_k)')
    report_lines.append(f'  norm stats: ||a|| mean={norm_a.mean():.2f}  '
                        f'std={norm_a.std():.2f}  max={norm_a.max():.2f}  '
                        f'min={norm_a.min():.2f}')
    report_lines.append('=' * 90)
    report_lines.append('')

    for k_idx in range(K):
        col = cos[:, k_idx]
        report_lines.append(f'### d{k_idx+1}  '
                            f'(corpus stats: mean={col.mean():.4f}  '
                            f'std={col.std():.4f}  '
                            f'min={col.min():.4f}  max={col.max():.4f})')
        for sign_label, sgn in [('TOP positive cos', +1.0),
                                ('TOP negative cos', -1.0)]:
            vals, idx = torch.topk(sgn * col, args.topk)
            report_lines.append(f'  {sign_label}:')
            for rank, (v, i) in enumerate(zip(vals, idx), start=1):
                seq = tokens_list[int(i)]
                last_idx = args.seq_len - 1
                ctx, last_tok = context_str(seq, last_idx, args.ctx_window)
                rdot = float(raw_dot[int(i), k_idx])
                na = float(norm_a[int(i)])
                # Replace newlines in ctx for cleaner reading.
                ctx_safe = ctx.replace('\n', '\\n')
                last_safe = last_tok.replace('\n', '\\n')
                cos_v = float(col[int(i)]) * (1 if sgn > 0 else 1)
                report_lines.append(
                    f'    {rank:>3d}.  cos={cos_v:+.4f}  dot={rdot:+8.2f}  '
                    f'||a||={na:7.2f}  ctx[...{ctx_safe}]<<{last_safe}>>')
            report_lines.append('')
        report_lines.append('-' * 90)
        report_lines.append('')

    report_path = os.path.join(args.out_dir, 'topk_lasttoken_overlap.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'wrote {report_path}', flush=True)

    pkl_path = os.path.join(args.out_dir, 'topk_lasttoken_overlap.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'cos': cos.numpy(),
            'raw_dot': raw_dot.numpy(),
            'norm_a': norm_a.numpy(),
            'tokens_list': [t.tolist() for t in tokens_list],
            'feature_labels': [f'd{k+1}' for k in range(K)],
        }, f)
    print(f'wrote {pkl_path}', flush=True)


if __name__ == '__main__':
    main()
