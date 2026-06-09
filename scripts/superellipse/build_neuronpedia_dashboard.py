#!/usr/bin/env python3
"""Neuronpedia-style static HTML dashboard for found dictionary directions.

For a selected subset of atoms d_k from a dictionary.pt, this builds a single
browsable HTML page that mimics a Neuronpedia feature dashboard:

  - Activation histogram (distribution of a . d_k over a FineWeb corpus, all
    token positions, pos-0/BOS skipped).
  - Top activating examples: the highest-scoring snippets, with EVERY token
    background-coloured by its projection score (orange = positive,
    blue = negative), so you can read what the direction fires on in context.
  - Logit effects: W_U @ ln_f(d_k), top promoted (green) and suppressed (red)
    vocab tokens — the direct vocab effect / logit lens of the direction.

Score = a . d_k with d_k unit. This is the same projection used by
dict_max_activating.py; here we keep the full per-token score grid for the
selected atoms so we can highlight whole snippets and draw histograms.

No server, no database — just one self-contained .html file you open in a
browser. Forward passes need the GPU, so run on a pod (like dict_max_activating):

    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/build_neuronpedia_dashboard.py \
      --dict_path results/dictionary/qwen3-1.7b_L14_K8000/dictionary.pt \
      --model qwen3-1.7b --layer 14 --atoms 0-49 \
      --n_seqs 2000 --seq_len 64 --n_examples 20 \
      --out results/dictionary/qwen3-1.7b_L14_K8000/dashboard
"""

import argparse
import html
import os
import sys

sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_lm_head, _get_final_norm
from src.data import load_fineweb_fixed_length


def parse_atoms(spec, K):
    """Parse '0-49' / '0,3,7' / '0-9,20,30-32' into a sorted list of ints."""
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < K)


def color_for_score(score, vmax_pos, vmax_neg):
    """Background colour for a token given its projection score.

    Positive -> orange (Neuronpedia-style), negative -> blue, both with alpha
    proportional to magnitude relative to the per-atom max. Returns a CSS rgba.
    """
    if score >= 0:
        a = min(1.0, score / vmax_pos) if vmax_pos > 0 else 0.0
        return f'rgba(255,140,0,{a:.3f})'
    a = min(1.0, (-score) / vmax_neg) if vmax_neg > 0 else 0.0
    return f'rgba(40,120,255,{a:.3f})'


def svg_histogram(scores, n_bins=50, width=440, height=120):
    """Inline SVG histogram of a 1-D score array (self-contained, no deps)."""
    if scores.size == 0:
        return '<svg></svg>'
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = np.histogram(scores, bins=n_bins, range=(lo, hi))
    cmax = max(1, int(counts.max()))
    bw = width / n_bins
    bars = []
    for i, c in enumerate(counts):
        bh = (c / cmax) * (height - 18)
        x = i * bw
        y = height - 18 - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-0.6:.1f}" '
                    f'height="{bh:.1f}" fill="#888"/>')
    # zero line
    if lo < 0 < hi:
        zx = (0 - lo) / (hi - lo) * width
        bars.append(f'<line x1="{zx:.1f}" y1="0" x2="{zx:.1f}" '
                    f'y2="{height-18}" stroke="#c33" stroke-width="1"/>')
    axis = (f'<text x="0" y="{height-4}" font-size="10" fill="#666">{lo:.2f}</text>'
            f'<text x="{width}" y="{height-4}" font-size="10" fill="#666" '
            f'text-anchor="end">{hi:.2f}</text>')
    return (f'<svg width="{width}" height="{height}" '
            f'style="background:#fafafa;border:1px solid #ddd">'
            + ''.join(bars) + axis + '</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--atoms', default='0-49',
                    help="which atoms to render, e.g. '0-49' or '0,3,7'")
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--n_examples', type=int, default=20,
                    help='top activating snippets per atom')
    ap.add_argument('--ctx_left', type=int, default=20)
    ap.add_argument('--ctx_right', type=int, default=8)
    ap.add_argument('--topk_logits', type=int, default=15)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    blob = torch.load(args.dict_path, map_location='cpu')
    D_full = blob['dictionary'].to(torch.float32)
    D_full = D_full / (D_full.norm(dim=-1, keepdim=True) + 1e-12)
    K_full, d_model = D_full.shape
    atoms = parse_atoms(args.atoms, K_full)
    print(f'Dictionary: {K_full} atoms, rendering {len(atoms)}: '
          f'{atoms[:5]}{"..." if len(atoms) > 5 else ""}', flush=True)
    D = D_full[atoms].to(device)                      # [Ksel, d]
    Ksel = D.shape[0]

    # --- Corpus + per-token score grid for the selected atoms -------------
    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)                   # [N, T] cpu
    N, T = tok_ids.shape

    print('Capturing activations + scoring', flush=True)
    seq_scores = torch.empty(N, T, Ksel, dtype=torch.float32)   # cpu
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
            sc = h.float() @ D.T                      # [b, T, Ksel]
        seq_scores[i:i + bt.shape[0]] = sc.cpu()
    print('  done.', flush=True)

    # Pre-decode tokens per sequence (individual decode keeps it readable).
    def decode_tok(tid):
        return tokenizer.decode([int(tid)])

    # --- Logit lens for the selected atoms --------------------------------
    print('Computing logit lens', flush=True)
    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float()             # [vocab, d]
    ln_dtype = next(ln_f.parameters()).dtype
    with torch.no_grad():
        ln_D = ln_f(D.to(ln_dtype)).float()           # [Ksel, d]
        logits = (ln_D @ W_U.T.to(device)).cpu()      # [Ksel, vocab]

    # --- Render -----------------------------------------------------------
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
         background:#f4f4f4;color:#222}
    .nav{position:fixed;top:0;left:0;width:140px;height:100%;overflow-y:auto;
         background:#222;color:#ddd;padding:8px;font-size:12px;box-sizing:border-box}
    .nav a{display:block;color:#9cf;text-decoration:none;padding:2px 0}
    .main{margin-left:156px;padding:16px;max-width:920px}
    .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;
          margin-bottom:24px}
    .card h2{margin:0 0 8px;font-size:18px}
    .stat{color:#666;font-size:13px;margin-bottom:10px}
    .cols{display:flex;gap:24px;flex-wrap:wrap}
    .logit-col{font-size:12px;font-family:ui-monospace,monospace}
    .logit-col .pos{color:#1a7f37}.logit-col .neg{color:#c0392b}
    .ex{font-family:ui-monospace,monospace;font-size:13px;line-height:1.9;
        margin:3px 0;white-space:pre-wrap;word-break:break-word}
    .ex .tgt{outline:1.5px solid #c33;border-radius:2px}
    .exscore{color:#999;font-size:11px;margin-right:6px}
    h3{font-size:13px;color:#444;margin:14px 0 4px}
    """
    nav = ['<div class="nav"><b>atoms</b><br>']
    for k in atoms:
        nav.append(f'<a href="#a{k}">atom {k}</a>')
    nav.append('</div>')

    body = ['<div class="main">',
            f'<h1>Dictionary dashboard — {html.escape(args.model)} '
            f'L{args.layer}</h1>',
            f'<p class="stat">{os.path.basename(os.path.dirname(args.dict_path) or args.dict_path)} '
            f'· {N} seqs × {T} tok · projection a·d_k (unit d_k)</p>']

    for j, k in enumerate(atoms):
        s_all = seq_scores[:, 1:, j].reshape(-1).numpy()   # skip pos0
        vmax_pos = float(np.percentile(s_all[s_all > 0], 99)) if (s_all > 0).any() else 1.0
        vmax_neg = float(np.percentile(-s_all[s_all < 0], 99)) if (s_all < 0).any() else 1.0
        gmax = float(s_all.max()) if s_all.size else 0.0
        gmean = float(s_all.mean()) if s_all.size else 0.0
        frac_pos = float((s_all > 0).mean()) if s_all.size else 0.0

        body.append(f'<div class="card" id="a{k}">')
        body.append(f'<h2>Atom {k}</h2>')
        body.append(f'<div class="stat">max={gmax:.2f} · mean={gmean:.3f} · '
                    f'{frac_pos*100:.1f}% tokens &gt; 0</div>')

        # histogram + logit columns side by side
        body.append('<div class="cols">')
        body.append('<div><h3>activation histogram</h3>'
                    + svg_histogram(s_all) + '</div>')

        lg = logits[j]
        top_pos = torch.topk(lg, args.topk_logits)
        top_neg = torch.topk(-lg, args.topk_logits)
        body.append('<div class="logit-col"><h3>logit effect (W_U·ln_f(d))</h3>')
        body.append('<table><tr><td style="padding-right:18px">'
                    '<b>promoted</b><br>')
        for v, idx in zip(top_pos.values, top_pos.indices):
            tok = html.escape(repr(decode_tok(idx)))
            body.append(f'<span class="pos">{float(v):+6.3f}</span> {tok}<br>')
        body.append('</td><td><b>suppressed</b><br>')
        for v, idx in zip(top_neg.values, top_neg.indices):
            tok = html.escape(repr(decode_tok(idx)))
            body.append(f'<span class="neg">{-float(v):+6.3f}</span> {tok}<br>')
        body.append('</td></tr></table></div></div>')

        # top activating examples
        body.append('<h3>top activating examples</h3>')
        pos_scores = seq_scores[:, 1:, j]                  # [N, T-1]
        seq_max, seq_argmax = pos_scores.max(dim=1)        # over positions
        order = torch.argsort(seq_max, descending=True)[:args.n_examples]
        for rank, si in enumerate(order.tolist()):
            peak_pos = int(seq_argmax[si]) + 1             # +1: pos0 skipped
            lo = max(1, peak_pos - args.ctx_left)
            hi = min(T, peak_pos + args.ctx_right + 1)
            spans = []
            for p in range(lo, hi):
                sc = float(seq_scores[si, p, j])
                tok = html.escape(decode_tok(tok_ids[si, p]))
                cls = ' class="tgt"' if p == peak_pos else ''
                bg = color_for_score(sc, vmax_pos, vmax_neg)
                spans.append(f'<span{cls} style="background:{bg}" '
                             f'title="{sc:.3f}">{tok}</span>')
            body.append(f'<div class="ex"><span class="exscore">'
                        f'{float(seq_max[si]):.2f}</span>{"".join(spans)}</div>')
        body.append('</div>')

    body.append('</div>')

    out_html = os.path.join(args.out, 'dashboard.html')
    with open(out_html, 'w') as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8">'
                f'<style>{css}</style></head><body>'
                + ''.join(nav) + ''.join(body) + '</body></html>')
    print(f'\nSaved {out_html}', flush=True)


if __name__ == '__main__':
    main()
