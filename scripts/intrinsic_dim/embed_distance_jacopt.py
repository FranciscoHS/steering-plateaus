#!/usr/bin/env python3
"""Distance from activations to token embeddings: jac-opt vs real.

Question: do the jac-norm-minimized points sit differently with respect to the
token embedding vectors than real activations do?

For every activation x and every token embedding e we take cosine(x, e) and
||x - e||_2, then summarize the vocab-sized distribution per activation:
  cos_max, cos_top10 (mean of the 10 largest), cos_mean
  l2_min (nearest embedding), l2_median

Two centering variants, both run:
  own_mean          embeddings minus the embedding-matrix mean; each activation
                    population minus its own mean
  global_real_mean  embeddings and every activation population minus the same
                    vector, the mean of the original FineWeb real activations

Embeddings are the INPUT embeddings (model.get_input_embeddings()). The script
prints config.tie_word_embeddings so we know whether that also equals W_U.

Populations: jac-opt raw, jac-opt rescaled to the real mean norm (so an L2 gap
cannot be a pure norm effect), and every real source cached by
c2st_data_sources.py, plus the original FineWeb real set.

Outputs in --outdir (default results/intrinsic_dim/embed_distance/):
  results.pkl (per-activation stats), summary.txt, embed_distance.png
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JAC_DIR = 'results/intrinsic_dim'
ACTS_DIR = 'results/intrinsic_dim/c2st_data_sources/acts'
STATS = ['cos_max', 'cos_top10', 'cos_mean', 'l2_min', 'l2_median']


def embedding_stats(X, E, E_n, E_sq, device, chunk=128, topk=10):
    """Per-row summaries of cosine / L2 against every embedding row.

    X: (n, d) activations, E: (V, d) embeddings — both already centered.
    E_n / E_sq are the row-normalized embeddings and their squared norms,
    passed in so they are built once per centering variant.
    """
    out = {k: [] for k in STATS}
    out['top1_id'] = []
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].to(device)
        x_n = torch.nn.functional.normalize(x, dim=-1)
        cos = x_n @ E_n.T                          # (b, V)
        top = cos.topk(topk, dim=-1)
        out['cos_max'].append(top.values[:, 0].cpu())
        out['cos_top10'].append(top.values.mean(-1).cpu())
        out['cos_mean'].append(cos.mean(-1).cpu())
        out['top1_id'].append(top.indices[:, 0].cpu())
        d2 = (x ** 2).sum(-1, keepdim=True) + E_sq[None, :] - 2 * (x @ E.T)
        d = d2.clamp_min(0).sqrt()
        out['l2_min'].append(d.min(-1).values.cpu())
        out['l2_median'].append(d.median(-1).values.cpu())
        del cos, d2, d
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--dtype', default='float16')
    p.add_argument('--outdir', default='results/intrinsic_dim/embed_distance')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from transformers import AutoModelForCausalLM
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                 'float32': torch.float32}
    print(f'Loading {args.model} ...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto')
    model.eval()
    device = next(model.parameters()).device
    E = model.get_input_embeddings().weight.detach().float().cpu()
    tied = getattr(model.config, 'tie_word_embeddings', None)
    print(f'  W_E {tuple(E.shape)}  tie_word_embeddings={tied}  '
          f'mean||e||={E.norm(dim=-1).mean():.3f}', flush=True)
    del model
    torch.cuda.empty_cache()

    # Populations.
    filt = torch.load(os.path.join(JAC_DIR, 'jac_filtered_data.pt'),
                      map_location='cpu', weights_only=False)
    opt = torch.load(os.path.join(JAC_DIR, 'jac_optimized_data.pt'),
                     map_location='cpu', weights_only=False)
    X_opt = opt['X_opt'].float()
    X_fw_orig = filt['X_real'].float()
    mu_real = X_fw_orig.mean(0)
    real_mean_norm = X_fw_orig.norm(dim=-1).mean()

    pops = {'jacopt_raw': X_opt,
            'jacopt_resc': X_opt * (real_mean_norm / X_opt.norm(dim=-1).mean())}
    if os.path.isdir(ACTS_DIR):
        for f in sorted(os.listdir(ACTS_DIR)):
            if f.endswith('.pt'):
                pops[f[:-3]] = torch.load(os.path.join(ACTS_DIR, f),
                                          map_location='cpu',
                                          weights_only=False).float()
    pops['fineweb_orig'] = X_fw_orig

    print('\nPopulations:')
    for n, X in pops.items():
        print(f'  {n:14s} n={X.shape[0]:6d}  mean||x||={X.norm(dim=-1).mean():.2f}')

    # (activation centering, embedding centering) — embeddings are centered
    # once per variant, activations once per population.
    variants = {
        'own_mean': (lambda X: X - X.mean(0), E - E.mean(0)),
        'global_real_mean': (lambda X: X - mu_real, E - mu_real),
    }

    res = {'model': args.model, 'tie_word_embeddings': tied,
           'vocab': int(E.shape[0]), 'stats': {}, 'mean_norms': {}}
    for n, X in pops.items():
        res['mean_norms'][n] = float(X.norm(dim=-1).mean())

    for vname, (center_act, Ec) in variants.items():
        print(f'\n=== centering: {vname} ===', flush=True)
        res['stats'][vname] = {}
        Ec_dev = Ec.to(device)
        E_n = torch.nn.functional.normalize(Ec_dev, dim=-1)
        E_sq = (Ec_dev ** 2).sum(-1)
        for n, X in pops.items():
            Xc = center_act(X)
            s = embedding_stats(Xc, Ec_dev, E_n, E_sq, device, chunk=args.chunk)
            res['stats'][vname][n] = s
            print(f'  {n:14s} cos_max={np.median(s["cos_max"]):.4f}  '
                  f'cos_top10={np.median(s["cos_top10"]):.4f}  '
                  f'cos_mean={np.median(s["cos_mean"]):+.4f}  '
                  f'l2_min={np.median(s["l2_min"]):.3f}  '
                  f'l2_med={np.median(s["l2_median"]):.3f}   (medians)',
                  flush=True)
            with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
                pickle.dump(res, f)
        del Ec_dev, E_n, E_sq
        torch.cuda.empty_cache()

    out = os.path.join(args.outdir, 'results.pkl')
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    print(f'\nSaved {out}')

    # Histograms: rows = centering variant, cols = statistic.
    cols = ['cos_max', 'cos_top10', 'l2_min', 'l2_median']
    fig, axes = plt.subplots(len(variants), len(cols),
                             figsize=(4.2 * len(cols), 3.4 * len(variants)))
    axes = np.atleast_2d(axes)
    for r, vname in enumerate(variants):
        for c, stat in enumerate(cols):
            ax = axes[r, c]
            for n in pops:
                v = res['stats'][vname][n][stat]
                ax.hist(v, bins=50, histtype='step', density=True,
                        label=n, lw=1.4 if n.startswith('jacopt') else 1.0)
            ax.set_title(f'{vname} — {stat}', fontsize=9)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=6)
    plt.tight_layout()
    plot = os.path.join(args.outdir, 'embed_distance.png')
    plt.savefig(plot, dpi=150)
    print(f'Saved {plot}')

    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'{args.model}  vocab={E.shape[0]}  d={E.shape[1]}  '
                f'tie_word_embeddings={tied}\n')
        f.write(f'mean||e||={E.norm(dim=-1).mean():.3f}\n')
        f.write('medians over each population\n')
        for vname in variants:
            f.write(f'\ncentering: {vname}\n')
            f.write(f'{"pop":14s} {"mean|x|":>8s} ' +
                    ' '.join(f'{s:>10s}' for s in STATS) + '\n')
            for n in pops:
                s = res['stats'][vname][n]
                f.write(f'{n:14s} {res["mean_norms"][n]:8.2f} ' +
                        ' '.join(f'{np.median(s[k]):10.4f}' for k in STATS) + '\n')
    print(f'Saved {os.path.join(args.outdir, "summary.txt")}')


if __name__ == '__main__':
    main()
