#!/usr/bin/env python3
"""Context-dependence test: token detectors vs context/semantic features.

For each direction we find its top activating positions over a FineWeb corpus
(centered-cosine score, as in rank_directions_featureness.py). Then for each
such (seq, pos) we DESTROY THE CONTEXT: replace tokens 0..pos-1 with random
tokens drawn from the corpus unigram pool, keep the original token at pos, and
truncate there (exact for causal attention). The per-direction statistic is

    drop = 1 - score_shuffled / score_orig

averaged over the top positions and --n_resample shuffles. A pure
current-token detector keeps firing (drop ~ 0); a direction that needs the
context loses its activation (drop ~ 1).

Directions tested:
  - top --n_top_kurt dictionary atoms by kurtosis (from --ranking_pkl)
  - --n_random_atoms random other atoms (null band)
  - every direction of each --control_dicts file (labeled, e.g. the
    contrastive scaffold/langcode dicts) — known-semantic positive controls

Run on a GPU pod:
    HF_HOME=/workspace/hf_cache python -u \\
        scripts/superellipse/context_dependence_test.py \\
        --dict_path .../dictionary.pt --ranking_pkl .../featureness/ranking.pkl \\
        --control_dicts results/scaffold_qwen_L14/dictionary.pt \\
                        results/langcode_qwen_L14/dictionary.pt \\
        --out .../featureness/context_dependence
"""
import argparse
import os
import pickle
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from src.data import load_fineweb_fixed_length


def corpus_mean(model, tok_ids, layer, batch_size, device):
    N, T = tok_ids.shape
    s = None; n = 0
    for i in range(0, N, batch_size):
        bt = tok_ids[i:i + batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        h = h[:, 1:, :].float().reshape(-1, h.shape[-1])
        s = h.sum(0) if s is None else s + h.sum(0)
        n += h.shape[0]
    return s / n


def cscore(h, D_sel, abar):
    """Centered-cosine scores. h [..., d] -> [..., Ndir]."""
    hc = h - abar
    hc = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return hc @ D_sel.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--ranking_pkl', required=True)
    ap.add_argument('--n_top_kurt', type=int, default=500)
    ap.add_argument('--n_random_atoms', type=int, default=100)
    ap.add_argument('--random_atom_seed', type=int, default=0)
    ap.add_argument('--control_dicts', nargs='*', default=[])
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_seqs', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--n_pos', type=int, default=10,
                    help='top activating positions per direction')
    ap.add_argument('--min_pos', type=int, default=8,
                    help='only consider peaks at pos >= this (need context '
                         'worth destroying)')
    ap.add_argument('--n_resample', type=int, default=3)
    ap.add_argument('--resample_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])

    # --- Directions: top-kurt atoms + random atoms + control dicts ---------
    blob = torch.load(args.dict_path, map_location='cpu')
    D_full = blob['dictionary'].float()
    D_full = D_full / (D_full.norm(dim=-1, keepdim=True) + 1e-12)
    rk = pickle.load(open(args.ranking_pkl, 'rb'))
    kurt = np.asarray(rk['kurtosis'])
    order = np.argsort(-kurt)
    top = [int(a) for a in order[:args.n_top_kurt]]
    g = torch.Generator().manual_seed(args.random_atom_seed)
    rest = np.setdiff1d(np.arange(D_full.shape[0]), np.array(top))
    rnd = rest[torch.randperm(len(rest), generator=g)[
        :args.n_random_atoms].numpy()]

    dirs, meta = [], []   # meta: (group, label, kurt_or_None)
    for a in top:
        dirs.append(D_full[a]); meta.append(('top_kurt', f'atom {a}',
                                             float(kurt[a])))
    for a in rnd:
        dirs.append(D_full[int(a)]); meta.append(('random_atom',
                                                  f'atom {int(a)}',
                                                  float(kurt[int(a)])))
    for path in args.control_dicts:
        cb = torch.load(path, map_location='cpu')
        C = cb['dictionary'].float()
        C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
        tag = os.path.basename(os.path.dirname(path))
        for n, v in zip(cb['names'], C):
            dirs.append(v); meta.append(('control', f'{n} [{tag}]', None))
    D_sel = torch.stack(dirs).to(device)
    Ndir = D_sel.shape[0]
    print(f'{Ndir} directions: {args.n_top_kurt} top-kurt + '
          f'{args.n_random_atoms} random atoms + '
          f'{sum(1 for m in meta if m[0] == "control")} controls', flush=True)

    # --- Corpus + original scores ------------------------------------------
    print(f'Loading {args.n_seqs} FineWeb seqs (seq_len={args.seq_len})',
          flush=True)
    tok_list = load_fineweb_fixed_length(args.n_seqs, tokenizer,
                                         seq_len=args.seq_len, seed=args.seed)
    tok_ids = torch.stack(tok_list)
    N, T = tok_ids.shape
    print('Computing corpus mean abar', flush=True)
    abar = corpus_mean(model, tok_ids, args.layer, args.batch_size, device)

    print('Scoring corpus (original contexts)', flush=True)
    seq_scores = torch.empty(N, T, Ndir, dtype=torch.float32)
    for i in range(0, N, args.batch_size):
        bt = tok_ids[i:i + args.batch_size].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[
                args.layer + 1].float()
            seq_scores[i:i + bt.shape[0]] = cscore(h, D_sel, abar).cpu()

    # --- Top positions per direction (|score|, pos >= min_pos) -------------
    flat = seq_scores[:, args.min_pos:, :].reshape(-1, Ndir)   # [N*(T-mp), Ndir]
    n_per_seq = T - args.min_pos
    top_idx = flat.abs().argsort(dim=0, descending=True)[:args.n_pos]  # [n_pos, Ndir]
    pos_of = {}   # (si, p) -> set of dir indices needing it
    per_dir_pos = [[] for _ in range(Ndir)]
    for j in range(Ndir):
        for fi in top_idx[:, j].tolist():
            si, p = fi // n_per_seq, fi % n_per_seq + args.min_pos
            per_dir_pos[j].append((si, p))
            pos_of.setdefault((si, p), set()).add(j)
    uniq = sorted(pos_of.keys())
    print(f'{len(uniq)} unique (seq,pos) to re-forward '
          f'(x{args.n_resample} resamples)', flush=True)

    # --- Shuffled-context forwards -----------------------------------------
    # Group by p so each batch is rectangular; random prefix tokens drawn from
    # the corpus unigram pool (pos-0 tokens excluded).
    pool = tok_ids[:, 1:].reshape(-1)
    shuf_scores = {u: [] for u in uniq}    # (si,p) -> list over resamples [Ndir]
    by_p = {}
    for (si, p) in uniq:
        by_p.setdefault(p, []).append(si)
    for r in range(args.n_resample):
        gr = torch.Generator().manual_seed(args.resample_seed + r)
        for p, sis in sorted(by_p.items()):
            seqs = []
            for si in sis:
                pref = pool[torch.randint(len(pool), (p,), generator=gr)]
                seqs.append(torch.cat([pref, tok_ids[si, p:p + 1]]))
            batch = torch.stack(seqs)                          # [n, p+1]
            for i in range(0, batch.shape[0], args.batch_size):
                bt = batch[i:i + args.batch_size].to(device)
                with torch.no_grad():
                    h = model(bt, output_hidden_states=True).hidden_states[
                        args.layer + 1][:, -1, :].float()
                    sc = cscore(h, D_sel, abar).cpu()          # [n_b, Ndir]
                for k in range(bt.shape[0]):
                    shuf_scores[(sis[i + k], p)].append(sc[k])
        print(f'  resample {r + 1}/{args.n_resample} done', flush=True)

    # --- Per-direction drop -------------------------------------------------
    results = []
    for j in range(Ndir):
        ratios, origs = [], []
        for (si, p) in per_dir_pos[j]:
            s0 = float(seq_scores[si, p, j])
            origs.append(s0)
            for sc in shuf_scores[(si, p)]:
                ratios.append(float(sc[j]) / s0)
        ratios = np.asarray(ratios)
        grp, lab, kv = meta[j]
        results.append({'group': grp, 'label': lab, 'kurt': kv,
                        'mean_drop': float(1 - ratios.mean()),
                        'median_drop': float(1 - np.median(ratios)),
                        'mean_orig': float(np.mean(np.abs(origs))),
                        'positions': per_dir_pos[j]})

    with open(os.path.join(args.out, 'context_dependence.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'results': results}, f)

    # --- Report -------------------------------------------------------------
    lines = ['# Context-dependence test (drop = 1 - shuf/orig; '
             '1 = needs context, 0 = pure token detector)', '']
    for grp in ['control', 'top_kurt', 'random_atom']:
        rs = [r for r in results if r['group'] == grp]
        rs.sort(key=lambda r: -r['mean_drop'])
        d = np.array([r['mean_drop'] for r in rs])
        lines += [f'## {grp}  (n={len(rs)})',
                  f'- drop: mean={d.mean():.3f}  median={np.median(d):.3f}  '
                  f'p10={np.quantile(d, .1):.3f}  p90={np.quantile(d, .9):.3f}']
        show = rs if grp == 'control' else rs[:40]
        for r in show:
            kt = f'  kurt={r["kurt"]:.2f}' if r['kurt'] is not None else ''
            lines.append(f'- {r["label"]:28s} drop={r["mean_drop"]:+.3f}  '
                         f'(med {r["median_drop"]:+.3f}, '
                         f'orig {r["mean_orig"]:.3f}){kt}')
        lines.append('')
    report = os.path.join(args.out, 'report.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]), flush=True)
    print(f'Saved {report}', flush=True)

    # --- Plot: kurtosis vs drop, colored by group ---------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for grp, col, mk in [('top_kurt', 'C0', 'o'), ('random_atom', 'C7', '.'),
                         ('control', 'C3', '*')]:
        rs = [r for r in results if r['group'] == grp]
        x = [r['kurt'] if r['kurt'] is not None else -0.5 for r in rs]
        y = [r['mean_drop'] for r in rs]
        ax.scatter(x, y, s=60 if grp == 'control' else 14, c=col, marker=mk,
                   alpha=0.45 if grp != 'control' else 0.95, label=grp)
    for r in results:
        if r['group'] == 'control':
            ax.annotate(r['label'].split(' [')[0],
                        (r['kurt'] if r['kurt'] is not None else -0.5,
                         r['mean_drop']), fontsize=6, alpha=0.8)
    ax.set_xlabel('excess kurtosis (controls at x=-0.5; unranked)')
    ax.set_ylabel('context drop  (1 - shuf/orig)')
    ax.set_title(f'Context dependence vs kurtosis ({args.model} '
                 f'L{args.layer})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plot = os.path.join(args.out, 'context_dependence.png')
    plt.savefig(plot, dpi=150, bbox_inches='tight')
    print(f'Saved {plot}', flush=True)


if __name__ == '__main__':
    main()
