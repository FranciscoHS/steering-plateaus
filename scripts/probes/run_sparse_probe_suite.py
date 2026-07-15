#!/usr/bin/env python3
"""Sparse-probing over a SUITE of binary concepts: found dicts vs the Qwen SAE.

Generalizes run_probe_dict_vs_sae.py from one sentiment task to many binary
concepts, so we get a distribution (higher n) instead of a single curve. Same
k-sparse machinery: at each k, compress the last-token activation to k atoms,
reconstruct, probe the reconstruction; report accuracy vs k, in-distribution
(held-out split) per concept, aggregated as mean ± sem across concepts.

Per Francisco: k <= 50, the L0_50 Qwen SAE (matched to that sparsity budget),
and ONLY the standard SAE (native encoder top-k) — no decoder-OMP variant.

Methods: raw (ceiling) · sae_native top-k · our dicts (OMP-k) · random (OMP-k).
Mean-as-free-bias on every OMP method (recon = abar + OMP_k(a - abar, D)), abar
= per-concept training mean.

Concept suite (config-driven, edit build_concepts): Bias-in-Bios top professions
(one-vs-rest), AG News topics (one-vs-rest), SST-2 sentiment.

    HF_HOME=/workspace/hf_cache python -u scripts/probes/run_sparse_probe_suite.py \
      --layer 14 --sae_ckpt /workspace/.../layer14.sae.pt \
      --dicts lam100:/...,lam1000:/...,lam10000:/... \
      --out /workspace/.../probe_suite
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from scripts.probes.run_probe_generalization import get_activations, reconstruct_topk
from scripts.probes.run_probe_dict_vs_sae import (
    omp_gpu, load_qwen_sae, load_dict, run_probe,
)


def _balance(texts, labels, n_per_class, rng):
    """Subsample to a balanced binary set of <=2*n_per_class examples."""
    texts = np.asarray(texts, dtype=object); labels = np.asarray(labels)
    out_t, out_y = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        idx = idx[:n_per_class]
        out_t.extend(texts[idx].tolist()); out_y.extend([c] * len(idx))
    order = rng.permutation(len(out_y))
    return [out_t[i] for i in order], np.array([out_y[i] for i in order])


def _split(texts, labels, frac_train, rng):
    n = len(labels); idx = rng.permutation(n); cut = int(frac_train * n)
    tr, te = idx[:cut], idx[cut:]
    return ([texts[i] for i in tr], labels[tr],
            [texts[i] for i in te], labels[te])


def build_concepts(n_per_class, seed, n_prof=8, max_chars=400):
    """Return list of {name, train_texts, train_y, test_texts, test_y}."""
    rng = np.random.RandomState(seed)
    concepts = []

    # --- Bias-in-Bios: top professions, one-vs-rest ----------------------
    try:
        bib = load_dataset('LabHC/bias_in_bios', split='train')
        text_key = 'hard_text' if 'hard_text' in bib.column_names else 'text'
        prof = np.array(bib['profession'])
        texts_all = bib[text_key]
        counts = np.bincount(prof)
        top = np.argsort(counts)[::-1][:n_prof]
        for p in top:
            y = (prof == p).astype(int)
            t, yy = _balance(texts_all, y, n_per_class, rng)
            t = [s[:max_chars] for s in t]
            tr_t, tr_y, te_t, te_y = _split(t, yy, 0.7, rng)
            concepts.append({'name': f'bib_prof{int(p)}', 'train_texts': tr_t,
                             'train_y': tr_y, 'test_texts': te_t, 'test_y': te_y})
        print(f'  bias_in_bios: {n_prof} profession concepts', flush=True)
    except Exception as e:
        print(f'  [skip] bias_in_bios: {e}', flush=True)

    # --- AG News: topics, one-vs-rest ------------------------------------
    try:
        ag = load_dataset('fancyzhx/ag_news', split='train')
        lab = np.array(ag['label']); texts_all = ag['text']
        names = {0: 'world', 1: 'sports', 2: 'business', 3: 'scitech'}
        for c, nm in names.items():
            y = (lab == c).astype(int)
            t, yy = _balance(texts_all, y, n_per_class, rng)
            t = [s[:max_chars] for s in t]
            tr_t, tr_y, te_t, te_y = _split(t, yy, 0.7, rng)
            concepts.append({'name': f'agnews_{nm}', 'train_texts': tr_t,
                             'train_y': tr_y, 'test_texts': te_t, 'test_y': te_y})
        print('  ag_news: 4 topic concepts', flush=True)
    except Exception as e:
        print(f'  [skip] ag_news: {e}', flush=True)

    # --- SST-2 sentiment -------------------------------------------------
    try:
        sst = load_dataset('stanfordnlp/sst2', split='train')
        y = np.array(sst['label']); texts_all = sst['sentence']
        t, yy = _balance(texts_all, y, n_per_class, rng)
        tr_t, tr_y, te_t, te_y = _split(t, yy, 0.7, rng)
        concepts.append({'name': 'sst2_sentiment', 'train_texts': tr_t,
                         'train_y': tr_y, 'test_texts': te_t, 'test_y': te_y})
        print('  sst2: 1 sentiment concept', flush=True)
    except Exception as e:
        print(f'  [skip] sst2: {e}', flush=True)

    return concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--sae_ckpt', default=None,
                    help='L0_50 Qwen SAE .pt; omit to skip the SAE baseline')
    ap.add_argument('--dicts', required=True, help='label:path,label:path,...')
    ap.add_argument('--n_random', type=int, default=None,
                    help='random-direction pool size (default max(k)*4)')
    ap.add_argument('--max_atoms', type=int, default=None,
                    help='cap each dict to its first N atoms (matched pool size)')
    ap.add_argument('--n_per_class', type=int, default=1500)
    ap.add_argument('--n_prof', type=int, default=8)
    ap.add_argument('--top_k_values', default='1,2,5,10,20,50')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--omp_batch', type=int, default=2000)
    args = ap.parse_args()

    k_values = [int(k) for k in args.top_k_values.split(',')]
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dict_specs = [(p.split(':', 1)[0].strip(), p.split(':', 1)[1].strip())
                  for p in args.dicts.split(',')]

    print('Building concept suite...', flush=True)
    concepts = build_concepts(args.n_per_class, args.seed, args.n_prof)
    print(f'{len(concepts)} concepts total', flush=True)

    # --- extract activations per concept (cache) -------------------------
    cache = os.path.join(args.out, 'suite_acts.pt')
    if os.path.exists(cache):
        print(f'Loading cached activations {cache}', flush=True)
        acts = torch.load(cache, weights_only=False)
    else:
        print(f'Loading model {args.model}', flush=True)
        model, tokenizer, device = load_model(args.model)
        acts = {}
        for ci in concepts:
            tr = get_activations(model, tokenizer, ci['train_texts'], args.layer, device, args.batch_size)
            te = get_activations(model, tokenizer, ci['test_texts'], args.layer, device, args.batch_size)
            acts[ci['name']] = {'train': tr, 'train_y': ci['train_y'],
                                'test': te, 'test_y': ci['test_y']}
            print(f'  acts {ci["name"]}: train{tuple(tr.shape)} test{tuple(te.shape)}', flush=True)
        torch.save(acts, cache)
        del model; torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- load SAE (optional) + dicts -------------------------------------
    sae = None
    sae_dec_dirs = None
    if args.sae_ckpt:
        print(f'Loading L0_50 SAE {args.sae_ckpt}', flush=True)
        sae = load_qwen_sae(args.sae_ckpt, device)
        # decoder as a plain dictionary for OMP-k; cap to max_atoms (random
        # subset) so it's matched in size to the maha/random/align dicts.
        sae_dec_dirs = sae.W_dec.detach().cpu().float()
        sae_dec_dirs = sae_dec_dirs / (sae_dec_dirs.norm(dim=-1, keepdim=True) + 1e-12)
        if args.max_atoms is not None and sae_dec_dirs.shape[0] > args.max_atoms:
            sub = np.random.RandomState(args.seed + 2000).choice(
                sae_dec_dirs.shape[0], args.max_atoms, replace=False)
            sae_dec_dirs = sae_dec_dirs[sub]
        print(f'  SAE decoder for sae_omp: {sae_dec_dirs.shape[0]} latents', flush=True)
    dicts = []
    for label, path in dict_specs:
        D = load_dict(path)
        if args.max_atoms is not None and D.shape[0] > args.max_atoms:
            D = D[:args.max_atoms]
        dicts.append((label, D))
        print(f'  dict {label}: {D.shape[0]} atoms', flush=True)
    n_rand = args.n_random if args.n_random is not None else max(k_values) * 4
    rng = np.random.RandomState(args.seed + 1000)
    rand_dirs = torch.from_numpy(rng.randn(n_rand,
                                           acts[concepts[0]['name']]['train'].shape[1]
                                           ).astype(np.float32))
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)
    print(f'  random pool: {n_rand} directions', flush=True)

    sae_keys = ['sae_native', 'sae_omp'] if sae is not None else []
    method_keys = ['raw'] + sae_keys + [f'dict:{l}' for l, _ in dicts] + ['random']
    # results[concept][method][k] = acc
    results = {'args': vars(args), 'k_values': k_values,
               'dict_labels': [l for l, _ in dicts], 'concepts': {}}

    for ci in concepts:
        name = ci['name']
        a = acts[name]
        tr_a, tr_y = a['train'], a['train_y']
        te_a, te_y = a['test'], a['test_y']
        abar = tr_a.mean(dim=0, keepdim=True)
        res = {m: {} for m in method_keys}

        # raw ceiling (k-independent)
        raw_acc = run_probe(tr_a.numpy(), tr_y, te_a.numpy(), te_y, args.seed)
        res['raw'] = raw_acc

        def omp_acc(D, k):
            trr = abar + omp_gpu(D, tr_a - abar, k, device, args.omp_batch)
            ter = abar + omp_gpu(D, te_a - abar, k, device, args.omp_batch)
            return run_probe(trr.numpy(), tr_y, ter.numpy(), te_y, args.seed)

        for k in k_values:
            if sae is not None:
                trr, _ = reconstruct_topk(sae, tr_a, k, device)
                ter, _ = reconstruct_topk(sae, te_a, k, device)
                res['sae_native'][k] = run_probe(trr.numpy(), tr_y, ter.numpy(), te_y, args.seed)
                res['sae_omp'][k] = omp_acc(sae_dec_dirs, k)
            for label, D in dicts:
                res[f'dict:{label}'][k] = omp_acc(D, k)
            res['random'][k] = omp_acc(rand_dirs, k)
        results['concepts'][name] = res
        sline = ' '.join(f'{l}:{res["dict:"+l][k_values[-1]]:.3f}' for l, _ in dicts)
        print(f'[{name}] raw={raw_acc:.3f} dicts@{k_values[-1]}=({sline})', flush=True)
        with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # --- aggregate: mean +/- sem across concepts -------------------------
    names = list(results['concepts'].keys()); nC = len(names)
    def agg(method, k=None):
        vals = [results['concepts'][n][method] if k is None
                else results['concepts'][n][method][k] for n in names]
        vals = np.array(vals, float)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(nC)

    fig, ax = plt.subplots(figsize=(7, 5))
    raw_m, raw_s = agg('raw')
    ax.axhline(raw_m, ls='--', color='black', alpha=0.6, label=f'raw ({raw_m:.3f})')
    ax.fill_between(k_values, raw_m - raw_s, raw_m + raw_s, color='black', alpha=0.07)
    plot_specs = []
    if sae is not None:
        plot_specs.append(('sae_native', 'Qwen SAE (native top-k)', 'o-', 'tab:blue'))
        plot_specs.append(('sae_omp', 'Qwen SAE decoder (OMP-k, 20k)', 'D-', 'tab:cyan'))
    dcolors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
    for i, (label, _) in enumerate(dicts):
        plot_specs.append((f'dict:{label}', f'dict {label}', 's-', dcolors[i % len(dcolors)]))
    plot_specs.append(('random', 'random', '^-', 'tab:gray'))
    for key, lbl, style, color in plot_specs:
        ms = [agg(key, k) for k in k_values]
        m = [x[0] for x in ms]; s = [x[1] for x in ms]
        ax.errorbar(k_values, m, yerr=s, fmt=style, color=color, capsize=3, label=lbl)
    ax.set_xlabel('k (atoms / latents)'); ax.set_ylabel('accuracy')
    ax.set_xscale('log'); ax.set_xticks(k_values); ax.set_xticklabels(k_values)
    ax.set_title(f'Sparse probing over {nC} concepts (mean ± sem)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'probe_suite.png')
    plt.savefig(plot_path, dpi=150)
    print(f'\nPlot:    {plot_path}\nResults: {os.path.join(args.out, "results.pkl")}')


if __name__ == '__main__':
    main()
