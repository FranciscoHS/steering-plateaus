#!/usr/bin/env python3
"""Neuronpedia-style static HTML dashboard for found dictionary directions.

For a selected set of directions, builds a single browsable HTML page that
mimics a Neuronpedia feature dashboard:

  - Activation histogram (distribution of the per-token score over a FineWeb
    corpus, all positions, pos-0/BOS skipped).
  - Activating examples: by default a random sample from the top
    --sample_top_frac (default 10%) of sequences ranked by peak score, with
    EVERY token background-coloured by its score (orange = positive, blue =
    negative). Set --sample_top_frac<=0 for the strict top-n behaviour.
  - Logit effects: W_U @ ln_f(d_k), top promoted (green) / suppressed (red).

Per-token SCORE depends on --score_mode:
  - dot          : a . d_k                 (raw projection; norm-sensitive)
  - cos_centered : (a - abar).d_k / ||a - abar||   (centered cosine; removes
                   the norm outliers and the shared-mean component — matches
                   rank_directions_featureness.py)

Which directions get rendered:
  - default: the atoms named by --atoms (e.g. '0-49').
  - if --ranking_pkl is given: a CONTRAST BAND built from the feature-ness
    ranking — the top-N by kurtosis, a band of median-kurtosis atoms, the
    bottom-N, and a few random unit directions as a hard null. Each card is
    labelled with its group + kurtosis so you can eyeball top-vs-baseline.

No server, no DB — one self-contained .html. Run on a pod (needs the model):

    HF_HOME=/workspace/hf_cache python -u \
      scripts/superellipse/build_neuronpedia_dashboard.py \
      --dict_path /tmp/dict_25k_snapshot.pt --model qwen3-1.7b --layer 14 \
      --score_mode cos_centered \
      --ranking_pkl .../featureness_25k/ranking.pkl \
      --n_top 50 --n_median 10 --n_bottom 10 --n_random 5 \
      --out .../dashboard_top50
"""

import argparse
import html
import os
import pickle
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
    """Token background: orange for positive, blue for negative, alpha ~ |score|."""
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
    counts, _ = np.histogram(scores, bins=n_bins, range=(lo, hi))
    cmax = max(1, int(counts.max()))
    bw = width / n_bins
    bars = []
    for i, c in enumerate(counts):
        bh = (c / cmax) * (height - 18)
        x = i * bw
        y = height - 18 - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-0.6:.1f}" '
                    f'height="{bh:.1f}" fill="#888"/>')
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


def corpus_mean(model, tok_ids, layer, batch_size, device):
    """Mean layer-L activation over all tokens (pos-0 skipped)."""
    N, T = tok_ids.shape
    s = None; n = 0
    for i in range(0, N, batch_size):
        bt = tok_ids[i:i + batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        h = h[:, 1:, :].float().reshape(-1, h.shape[-1])
        s = h.sum(0) if s is None else s + h.sum(0)
        n += h.shape[0]
    return (s / n)


def score_corpus(model, tok_ids, D_sel, layer, batch_size, device,
                 score_mode, abar):
    """Per-token scores [N, T, Ksel] for the selected directions."""
    N, T = tok_ids.shape
    Ksel = D_sel.shape[0]
    seq_scores = torch.empty(N, T, Ksel, dtype=torch.float32)
    for i in range(0, N, batch_size):
        bt = tok_ids[i:i + batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[layer + 1].float()
            if score_mode == 'cos_centered':
                hc = h - abar
                hc = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                sc = hc @ D_sel.T
            else:
                sc = h @ D_sel.T
        seq_scores[i:i + bt.shape[0]] = sc.cpu()
    return seq_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--score_mode', default='dot',
                    choices=['dot', 'cos_centered'])
    ap.add_argument('--atoms', default='0-49',
                    help="manual mode: which atoms, e.g. '0-49' or '0,3,7'")
    ap.add_argument('--ranking_pkl', default=None,
                    help='featureness ranking.pkl -> build a contrast band '
                         '(top/median/bottom/random) instead of --atoms')
    ap.add_argument('--n_top', type=int, default=50)
    ap.add_argument('--n_median', type=int, default=10)
    ap.add_argument('--n_bottom', type=int, default=10)
    ap.add_argument('--n_random', type=int, default=5)
    ap.add_argument('--random_seed', type=int, default=0)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--n_examples', type=int, default=20)
    ap.add_argument('--sample_top_frac', type=float, default=0.10,
                    help='draw the displayed examples by random sampling from '
                         'the top this-fraction of sequences (ranked by peak '
                         'score), instead of the strict top-n. <=0 restores '
                         'the old strict-top-n behaviour.')
    ap.add_argument('--sample_seed', type=int, default=0,
                    help='RNG seed for --sample_top_frac example sampling')
    ap.add_argument('--ctx_left', type=int, default=20)
    ap.add_argument('--ctx_right', type=int, default=8)
    ap.add_argument('--topk_logits', type=int, default=15)
    ap.add_argument('--n_top_tokens', type=int, default=20,
                    help='# of distinct top-activating token TYPES to list '
                         '(Neuronpedia-style "top activating tokens")')
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
    names_full = blob.get('names')   # e.g. contrastive dicts carry labels

    # --- Build the set of directions to render (entries) ------------------
    # entry: {group, atom (int|None), kurt (float|None), tail (float|None), dir}
    entries = []
    abar_from_rank = None
    if args.ranking_pkl:
        rk = pickle.load(open(args.ranking_pkl, 'rb'))
        order = np.asarray(rk['order']); kurt = rk['kurtosis']; tail = rk['tail_ratio']
        abar_from_rank = rk.get('abar')
        top = order[:args.n_top]
        mid = len(order) // 2
        h = args.n_median
        median = order[mid - h // 2: mid - h // 2 + h] if h > 0 else []
        bottom = order[-args.n_bottom:] if args.n_bottom > 0 else []
        for grp, idxs in [('top', top), ('median', median), ('bottom', bottom)]:
            for a in idxs:
                a = int(a)
                entries.append({'group': grp, 'atom': a, 'kurt': float(kurt[a]),
                                'tail': float(tail[a]), 'dir': D_full[a]})
        g = torch.Generator().manual_seed(args.random_seed)
        for r in range(args.n_random):
            v = torch.randn(d_model, generator=g)
            entries.append({'group': 'random', 'atom': None, 'kurt': None,
                            'tail': None, 'dir': v / v.norm()})
        print(f'Contrast band: {args.n_top} top / {args.n_median} median / '
              f'{args.n_bottom} bottom / {args.n_random} random '
              f'= {len(entries)} cards', flush=True)
    else:
        for a in parse_atoms(args.atoms, K_full):
            entries.append({'group': 'manual', 'atom': a, 'kurt': None,
                            'tail': None, 'dir': D_full[a],
                            'name': names_full[a] if names_full else None})
        print(f'Rendering {len(entries)} atoms (manual)', flush=True)

    D_sel = torch.stack([e['dir'] for e in entries]).to(device)
    Ksel = D_sel.shape[0]

    # --- Corpus + per-token scores ----------------------------------------
    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})', flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape

    abar = None
    if args.score_mode == 'cos_centered':
        if abar_from_rank is not None:
            abar = torch.as_tensor(abar_from_rank, dtype=torch.float32, device=device)
            print('Using abar from ranking.pkl (same corpus)', flush=True)
        else:
            print('Computing corpus mean abar', flush=True)
            abar = corpus_mean(model, tok_ids, args.layer, args.batch_size, device)

    print(f'Scoring corpus (score_mode={args.score_mode})', flush=True)
    seq_scores = score_corpus(model, tok_ids, D_sel, args.layer,
                              args.batch_size, device, args.score_mode, abar)
    print('  done.', flush=True)

    def decode_tok(tid):
        return tokenizer.decode([int(tid)])

    # --- Logit lens -------------------------------------------------------
    print('Computing logit lens', flush=True)
    lm_head = _get_lm_head(model)
    ln_f = _get_final_norm(model)
    W_U = lm_head.weight.detach().float()
    ln_dtype = next(ln_f.parameters()).dtype
    with torch.no_grad():
        ln_D = ln_f(D_sel.to(ln_dtype)).float()
        logits = (ln_D @ W_U.T.to(device)).cpu()

    # --- Render -----------------------------------------------------------
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
         background:#f4f4f4;color:#222}
    .nav{position:fixed;top:0;left:0;width:150px;height:100%;overflow-y:auto;
         background:#222;color:#ddd;padding:8px;font-size:12px;box-sizing:border-box}
    .nav a{display:block;color:#9cf;text-decoration:none;padding:1px 0}
    .nav .grp{color:#fc8;margin-top:8px;font-weight:bold}
    .main{margin-left:166px;padding:16px;max-width:920px}
    .sec{font-size:20px;margin:26px 0 8px;border-bottom:2px solid #ccc;padding-bottom:4px}
    .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;
          margin-bottom:24px}
    .card h2{margin:0 0 4px;font-size:17px}
    .badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;
           color:#fff;margin-left:8px;vertical-align:middle}
    .b-top{background:#1a7f37}.b-median{background:#b08900}.b-bottom{background:#c0392b}
    .b-random{background:#666}.b-manual{background:#345}
    .stat{color:#666;font-size:13px;margin-bottom:10px}
    .cols{display:flex;gap:24px;flex-wrap:wrap}
    .logit-col{font-size:12px;font-family:ui-monospace,monospace}
    .logit-col .pos{color:#1a7f37}.logit-col .neg{color:#c0392b}
    .ex{font-family:ui-monospace,monospace;font-size:13px;line-height:1.9;
        margin:3px 0;white-space:pre-wrap;word-break:break-word}
    .ex .tgt{outline:1.5px solid #c33;border-radius:2px}
    .exscore{color:#999;font-size:11px;margin-right:6px}
    h3{font-size:13px;color:#444;margin:14px 0 4px}
    .toktop{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 8px}
    .tokchip{font-family:ui-monospace,monospace;font-size:12px;padding:2px 6px;
             border-radius:4px;border:1px solid #ccc;white-space:pre}
    .tokchip .cs{color:#555;font-size:10px;margin-left:4px}
    """
    group_titles = {'top': f'TOP {args.n_top} by kurtosis',
                    'median': f'MEDIAN band (n={args.n_median})',
                    'bottom': f'BOTTOM {args.n_bottom} by kurtosis',
                    'random': f'RANDOM null (n={args.n_random})',
                    'manual': 'Atoms'}

    nav = ['<div class="nav"><b>cards</b>']
    body = ['<div class="main">',
            f'<h1>Dictionary dashboard — {html.escape(args.model)} L{args.layer}</h1>',
            f'<p class="stat">{html.escape(os.path.basename(args.dict_path))} · '
            f'{N} seqs × {T} tok · score_mode={args.score_mode}</p>']

    last_group = None
    for j, e in enumerate(entries):
        grp = e['group']
        anchor = f'c{j}'
        if grp != last_group:
            nav.append(f'<div class="grp">{group_titles.get(grp, grp)}</div>')
            body.append(f'<div class="sec" id="sec_{grp}">{group_titles.get(grp, grp)}</div>')
            last_group = grp
        label = (f'atom {e["atom"]}' if e['atom'] is not None
                 else f'random #{j}')
        if e.get('name'):
            label = f'{e["name"]} (atom {e["atom"]})'
        nav.append(f'<a href="#{anchor}">{label}</a>')

        s_all = seq_scores[:, 1:, j].reshape(-1).numpy()
        vmax_pos = float(np.percentile(s_all[s_all > 0], 99)) if (s_all > 0).any() else 1.0
        vmax_neg = float(np.percentile(-s_all[s_all < 0], 99)) if (s_all < 0).any() else 1.0
        gmax = float(s_all.max()) if s_all.size else 0.0
        gmean = float(s_all.mean()) if s_all.size else 0.0
        frac_pos = float((s_all > 0).mean()) if s_all.size else 0.0

        body.append(f'<div class="card" id="{anchor}">')
        kbadge = (f' <span class="badge b-{grp}">{grp}</span>')
        body.append(f'<h2>{label}{kbadge}</h2>')
        rk_txt = ''
        if e['kurt'] is not None:
            rk_txt = f'kurtosis={e["kurt"]:.2f} · tailR={e["tail"]:.2f} · '
        body.append(f'<div class="stat">{rk_txt}max={gmax:.3f} · mean={gmean:.4f} · '
                    f'{frac_pos*100:.1f}% tokens &gt; 0</div>')

        # top activating TOKEN TYPES (distinct tokens by their peak activation)
        flat_sc = seq_scores[:, 1:, j].reshape(-1)
        flat_tok = tok_ids[:, 1:].reshape(-1)
        order_tok = torch.argsort(flat_sc, descending=True)
        seen_tok = set(); toplist = []
        for oi in order_tok.tolist():
            tid = int(flat_tok[oi])
            if tid in seen_tok:
                continue
            seen_tok.add(tid); toplist.append((tid, float(flat_sc[oi])))
            if len(toplist) >= args.n_top_tokens:
                break
        body.append('<h3>top activating tokens</h3><div class="toktop">')
        for tid, sc in toplist:
            bg = color_for_score(sc, vmax_pos, vmax_neg)
            body.append(f'<span class="tokchip" style="background:{bg}" '
                        f'title="{sc:.3f}">{html.escape(decode_tok(tid))}'
                        f'<span class="cs">{sc:.2f}</span></span>')
        body.append('</div>')

        body.append('<div class="cols">')
        body.append('<div><h3>activation histogram</h3>' + svg_histogram(s_all) + '</div>')
        lg = logits[j]
        top_pos = torch.topk(lg, args.topk_logits)
        top_neg = torch.topk(-lg, args.topk_logits)
        body.append('<div class="logit-col"><h3>logit effect (W_U·ln_f(d))</h3>')
        body.append('<table><tr><td style="padding-right:18px"><b>promoted</b><br>')
        for v, idx in zip(top_pos.values, top_pos.indices):
            body.append(f'<span class="pos">{float(v):+6.3f}</span> '
                        f'{html.escape(repr(decode_tok(idx)))}<br>')
        body.append('</td><td><b>suppressed</b><br>')
        for v, idx in zip(top_neg.values, top_neg.indices):
            body.append(f'<span class="neg">{-float(v):+6.3f}</span> '
                        f'{html.escape(repr(decode_tok(idx)))}<br>')
        body.append('</td></tr></table></div></div>')

        pos_scores = seq_scores[:, 1:, j]
        seq_max, seq_argmax = pos_scores.max(dim=1)
        ranked = torch.argsort(seq_max, descending=True)
        if args.sample_top_frac > 0:
            pool_n = max(args.n_examples,
                         int(round(args.sample_top_frac * ranked.numel())))
            pool_n = min(pool_n, ranked.numel())
            pool = ranked[:pool_n]
            rng = np.random.default_rng(args.sample_seed + j)
            take = min(args.n_examples, pool_n)
            pick = rng.choice(pool_n, size=take, replace=False)
            # show the sampled examples high-to-low score within the sample
            order_ex = pool[np.sort(pick)]
            body.append(f'<h3>activating examples '
                        f'(random sample of {take} from top '
                        f'{args.sample_top_frac:.0%}, n={pool_n})</h3>')
        else:
            order_ex = ranked[:args.n_examples]
            body.append('<h3>top activating examples</h3>')
        for si in order_ex.tolist():
            peak_pos = int(seq_argmax[si]) + 1
            lo = max(1, peak_pos - args.ctx_left)
            hi = min(T, peak_pos + args.ctx_right + 1)
            spans = []
            for p in range(lo, hi):
                sc = float(seq_scores[si, p, j])
                cls = ' class="tgt"' if p == peak_pos else ''
                bg = color_for_score(sc, vmax_pos, vmax_neg)
                spans.append(f'<span{cls} style="background:{bg}" '
                             f'title="{sc:.3f}">{html.escape(decode_tok(tok_ids[si, p]))}</span>')
            body.append(f'<div class="ex"><span class="exscore">'
                        f'{float(seq_max[si]):.3f}</span>{"".join(spans)}</div>')
        body.append('</div>')

    nav.append('</div>')
    body.append('</div>')

    out_html = os.path.join(args.out, 'dashboard.html')
    with open(out_html, 'w') as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8">'
                f'<style>{css}</style></head><body>'
                + ''.join(nav) + ''.join(body) + '</body></html>')
    print(f'\nSaved {out_html}', flush=True)


if __name__ == '__main__':
    main()
