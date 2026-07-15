#!/usr/bin/env python3
"""Neuronpedia-style dashboard for SPECIFIC TopK-SAE latents, scored FAITHFULLY.

Unlike build_neuronpedia_dashboard.py (which scores tokens by projection onto a
decoder direction), this scores each token by the SAE's true latent activation
    z_L(x) = relu( W_enc[L] . (x - b_dec) + b_enc[L] )
(the pre-topk feature activation) — the actual encoder feature, which is what we
selected on. The logit-lens still uses the decoder row W_dec[L] (what the feature
writes). Reuses the render helpers from build_neuronpedia_dashboard.py.

Each --spec is `label:sae_ckpt:latent_idx`; the model runs once over the corpus
and every spec's latent is read off the shared residual, so multiple SAEs/latents
render as cards in one page.

Run on a pod (needs the model):
    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/build_sae_latent_dashboard.py \
      --specs baseline:results/trained_sae/l10_doubled_k32_2M/sae_weights.pt:11645,popt:results/trained_sae_popt/l10_lam1/sae_weights.pt:23249 \
      --model qwen3-1.7b --layer 10 --n_seqs 2000 --seq_len 64 \
      --out results/sae_dash/sports_latent
"""
import argparse
import html
import importlib.util
import os
import sys

sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, _get_lm_head, _get_final_norm
from src.data import load_fineweb_fixed_length
from src.autoencoders import load_sae

_nd_path = os.path.join(os.path.dirname(__file__), 'build_neuronpedia_dashboard.py')
_spec = importlib.util.spec_from_file_location('nd', _nd_path)
nd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd)
svg_histogram, color_for_score = nd.svg_histogram, nd.color_for_score


@torch.no_grad()
def latent_score(h, sp, mode, abar):
    """Per-token score for one latent. h [B,T,d] -> [B,T].

    mode='activation'   : true pre-topk latent activation relu(W_enc.(x-b_dec)+b).
    mode='cos_centered' : centered cosine of the activation with the (normalized)
                          encoder read-direction — norm-invariant, so high-norm
                          outlier tokens no longer dominate.
    """
    sae = sp['sae']
    if mode == 'activation':
        w = sae.encoder.weight[sp['latent']].float()
        b = float(sae.encoder.bias[sp['latent']])
        return torch.relu((h - sae.b_dec.float()) @ w + b)
    hc = h - abar
    hc = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return hc @ sp['wdir']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True, help='label:sae_ckpt:latent,...')
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=10)
    ap.add_argument('--score_mode', default='cos_centered',
                    choices=['activation', 'cos_centered'],
                    help='cos_centered removes norm outliers; activation = true latent value')
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--n_examples', type=int, default=25)
    ap.add_argument('--ctx_left', type=int, default=25)
    ap.add_argument('--ctx_right', type=int, default=8)
    ap.add_argument('--topk_logits', type=int, default=15)
    ap.add_argument('--n_top_tokens', type=int, default=25)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])

    specs = []
    sae_cache = {}
    for s in args.specs.split(','):
        label, path, lat = s.split(':')
        if path not in sae_cache:
            sae_cache[path] = load_sae(path, device)
        sae = sae_cache[path]
        wdir = torch.nn.functional.normalize(
            sae.encoder.weight[int(lat)].float(), dim=0).to(device)
        specs.append({'label': label, 'sae': sae, 'latent': int(lat),
                      'path': path, 'wdir': wdir})
    print(f'{len(specs)} latent card(s), score_mode={args.score_mode}', flush=True)

    tok_ids = torch.stack(load_fineweb_fixed_length(
        args.n_seqs, tokenizer, seq_len=args.seq_len, seed=args.seed))
    N, T = tok_ids.shape

    abar = None
    if args.score_mode == 'cos_centered':
        print('Computing corpus mean abar', flush=True)
        abar = nd.corpus_mean(model, tok_ids, args.layer, args.batch_size, device)

    scores = torch.zeros(len(specs), N, T, dtype=torch.float32)
    print(f'Scoring corpus ({args.score_mode})', flush=True)
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1].float()
        for j, sp in enumerate(specs):
            scores[j, i:i + bt.shape[0]] = latent_score(h, sp, args.score_mode, abar).cpu()

    # logit lens from decoder rows
    lm_head, ln_f = _get_lm_head(model), _get_final_norm(model)
    W_U = lm_head.weight.detach().float()
    ln_dtype = next(ln_f.parameters()).dtype
    dirs = torch.stack([torch.nn.functional.normalize(
        sp['sae'].W_dec[sp['latent']].float(), dim=0) for sp in specs]).to(device)
    with torch.no_grad():
        logits = (ln_f(dirs.to(ln_dtype)).float() @ W_U.T.to(device)).cpu()

    def dtok(tid):
        return tokenizer.decode([int(tid)])

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f4f4;color:#222}
    .main{padding:16px;max-width:960px;margin:auto}
    .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:24px}
    .card h2{margin:0 0 4px;font-size:18px}
    .stat{color:#666;font-size:13px;margin-bottom:10px}
    .cols{display:flex;gap:24px;flex-wrap:wrap}
    .logit-col{font-size:12px;font-family:ui-monospace,monospace}
    .logit-col .pos{color:#1a7f37}.logit-col .neg{color:#c0392b}
    .ex{font-family:ui-monospace,monospace;font-size:13px;line-height:1.9;margin:3px 0;
        white-space:pre-wrap;word-break:break-word}
    .ex .tgt{outline:1.5px solid #c33;border-radius:2px}
    .exscore{color:#999;font-size:11px;margin-right:6px}
    h3{font-size:13px;color:#444;margin:14px 0 4px}
    .toktop{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 8px}
    .tokchip{font-family:ui-monospace,monospace;font-size:12px;padding:2px 6px;border-radius:4px;
             border:1px solid #ccc;white-space:pre}
    .tokchip .cs{color:#555;font-size:10px;margin-left:4px}
    """
    body = ['<div class="main">',
            f'<h1>SAE latent dashboard — {html.escape(args.model)} L{args.layer} '
            f'(score={args.score_mode})</h1>',
            f'<p class="stat">{N} seqs × {T} tok · FineWeb</p>']

    for j, sp in enumerate(specs):
        s_all = scores[j, :, 1:].reshape(-1).numpy()
        vmax_pos = float(np.percentile(s_all[s_all > 0], 99)) if (s_all > 0).any() else 1.0
        vmax_neg = float(np.percentile(-s_all[s_all < 0], 99)) if (s_all < 0).any() else 1.0
        gmax, gmean = float(s_all.max()), float(s_all.mean())
        frac_pos = float((s_all > 0).mean())
        body.append('<div class="card">')
        body.append(f'<h2>{html.escape(sp["label"])} — latent {sp["latent"]}</h2>')
        body.append(f'<div class="stat">{html.escape(os.path.basename(sp["path"]))} · '
                    f'max={gmax:.3f} · mean={gmean:.4f} · {frac_pos*100:.1f}% tokens active</div>')

        # top activating token types
        flat_sc = scores[j, :, 1:].reshape(-1)
        flat_tok = tok_ids[:, 1:].reshape(-1)
        seen, toplist = set(), []
        for oi in torch.argsort(flat_sc, descending=True).tolist():
            tid = int(flat_tok[oi])
            if tid in seen:
                continue
            seen.add(tid); toplist.append((tid, float(flat_sc[oi])))
            if len(toplist) >= args.n_top_tokens:
                break
        body.append('<h3>top activating tokens</h3><div class="toktop">')
        for tid, sc in toplist:
            bg = color_for_score(sc, vmax_pos, vmax_neg)
            body.append(f'<span class="tokchip" style="background:{bg}" title="{sc:.3f}">'
                        f'{html.escape(dtok(tid))}<span class="cs">{sc:.2f}</span></span>')
        body.append('</div>')

        body.append('<div class="cols">')
        body.append('<div><h3>activation histogram</h3>' + svg_histogram(s_all) + '</div>')
        lg = logits[j]
        tp, tn = torch.topk(lg, args.topk_logits), torch.topk(-lg, args.topk_logits)
        body.append('<div class="logit-col"><h3>logit effect (W_U·ln_f(W_dec))</h3>'
                    '<table><tr><td style="padding-right:18px"><b>promoted</b><br>')
        for v, idx in zip(tp.values, tp.indices):
            body.append(f'<span class="pos">{float(v):+6.3f}</span> {html.escape(repr(dtok(idx)))}<br>')
        body.append('</td><td><b>suppressed</b><br>')
        for v, idx in zip(tn.values, tn.indices):
            body.append(f'<span class="neg">{-float(v):+6.3f}</span> {html.escape(repr(dtok(idx)))}<br>')
        body.append('</td></tr></table></div></div>')

        # top activating examples
        pos_scores = scores[j, :, 1:]
        seq_max, seq_argmax = pos_scores.max(dim=1)
        body.append('<h3>top activating examples</h3>')
        for si in torch.argsort(seq_max, descending=True)[:args.n_examples].tolist():
            peak = int(seq_argmax[si]) + 1
            lo, hi = max(1, peak - args.ctx_left), min(T, peak + args.ctx_right + 1)
            spans = []
            for p in range(lo, hi):
                sc = float(scores[j, si, p])
                cls = ' class="tgt"' if p == peak else ''
                bg = color_for_score(sc, vmax_pos, vmax_neg)
                spans.append(f'<span{cls} style="background:{bg}" title="{sc:.3f}">'
                             f'{html.escape(dtok(tok_ids[si, p]))}</span>')
            body.append(f'<div class="ex"><span class="exscore">{float(seq_max[si]):.3f}</span>'
                        f'{"".join(spans)}</div>')
        body.append('</div>')
    body.append('</div>')

    out_html = os.path.join(args.out, 'dashboard.html')
    with open(out_html, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8">'
                f'<style>{css}</style></head><body>' + ''.join(body) + '</body></html>')
    print(f'\nSaved {out_html}', flush=True)


if __name__ == '__main__':
    main()
