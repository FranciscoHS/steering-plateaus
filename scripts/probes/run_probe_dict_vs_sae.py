#!/usr/bin/env python3
"""Sparse-probing comparison: our found dictionaries vs the Qwen SAE.

Trains a sentiment probe (SST-2) and evaluates in-distribution (held-out SST-2)
and out-of-distribution (Amazon polarity) at a range of sparsity levels k.  For
each k we compress the layer-L last-token activation to k atoms, reconstruct,
and probe the reconstruction with logistic regression.  The question is *how
few atoms* carry generalizable sentiment — that's where a good dictionary earns
its keep (a dense linear probe can't separate these representations, since they
are all linear re-encodings of the same activation).

Methods compared (all at the same k grid, same probe):
  - raw            : probe on the raw activation (ceiling; horizontal line).
  - sae_native     : the REAL Qwen SAE — encode, keep top-k active latents by
                     magnitude, decode (+b_dec). Gets its learned encoder.
  - sae_omp        : the SAME Qwen SAE *decoder* treated as a plain dictionary,
                     reconstructed by OMP-k. Isolates atom quality (apples to
                     apples with our dicts, which have no encoder).
  - dict:<label>   : each of our found dictionaries, OMP-k.
  - pca / random   : OMP-k baselines (best linear subspace / a hard null).

Mean-as-free-bias: every OMP method reconstructs the *deviation* from the mean
activation abar and adds abar back (recon = abar + OMP_k(a - abar, D)). So the k
atoms are spent only on the deviation, not on re-deriving the mean — which the
SAE gets for free through its decoder bias b_dec. abar is the training-set mean
(fit on train, applied to all splits), so it stays a fixed bias like b_dec.

Run on a pod (needs the model + the Qwen SAE ckpt + dicts on the volume):

    HF_HOME=/workspace/hf_cache python -u scripts/probes/run_probe_dict_vs_sae.py \
      --model qwen3-1.7b --layer 14 \
      --sae_ckpt /workspace/hf_cache/.../layer14.sae.pt \
      --dicts lam100:/workspace/.../align1.0_k100_40k/dictionary.pt,\
lam1000:/workspace/.../lamprev1000_40k/dictionary.pt,\
lam10000:/workspace/.../lamprev10000_40k/dictionary.pt \
      --out /workspace/.../probe_dict_vs_sae
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from scripts.probes.run_probe_generalization import (
    get_activations, SimpleSAE, reconstruct_topk,
)


def run_probe(train_X, train_y, test_X, test_y, seed=42):
    # liblinear: coordinate descent, much faster than lbfgs for binary L2
    # logistic on this scale (few-k samples, 2048 dense features); same L2/C=1.0
    # optimum, so accuracy is unchanged vs the old max_iter=2000 lbfgs.
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    clf.fit(train_X, train_y)
    return accuracy_score(test_y, clf.predict(test_X))


def omp_gpu(dictionary, activations, k, device, batch=2000):
    """Batched greedy OMP-k reconstruction on GPU. Returns recon [N, d] (CPU).

    dictionary: [n_dirs, d] (assumed unit-norm rows), activations: [N, d].
    """
    D = dictionary.to(device).float()
    out = torch.empty_like(activations)
    for i in range(0, len(activations), batch):
        X = activations[i:i + batch].to(device).float()
        n = X.shape[0]
        selected = torch.zeros(n, k, dtype=torch.long, device=device)
        residual = X.clone()
        recon = torch.zeros_like(X)
        for step in range(k):
            corr = (residual @ D.T).abs()                  # [n, n_dirs]
            for s in range(step):
                corr.scatter_(1, selected[:, s:s + 1], 0.0)
            selected[:, step] = corr.argmax(dim=1)
            idx = selected[:, :step + 1]                   # [n, step+1]
            D_sel = D[idx]                                 # [n, step+1, d]
            gram = D_sel @ D_sel.transpose(1, 2)
            gram += 1e-6 * torch.eye(step + 1, device=device).unsqueeze(0)
            rhs = (D_sel @ X.unsqueeze(2)).squeeze(2)      # [n, step+1]
            c = torch.linalg.solve(gram, rhs)
            recon = (c.unsqueeze(2) * D_sel).sum(dim=1)    # [n, d]
            residual = X - recon
        out[i:i + n] = recon.cpu()
    return out


def load_dict(path):
    """Load a dictionary.pt and return unit-norm direction rows [K, d]."""
    blob = torch.load(path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].float()
    return D / (D.norm(dim=-1, keepdim=True) + 1e-12)


def load_qwen_sae(ckpt_path, device):
    """Build a SimpleSAE from a Qwen-Scope SAE checkpoint (W_enc/W_dec/b_*)."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    W_enc = sd['W_enc'].float()                 # [n_features, d_model]
    b_enc = sd['b_enc'].float()
    W_dec = sd['W_dec'].float()                 # [d_model, n_features]
    if W_dec.shape[0] != W_enc.shape[0]:
        W_dec = W_dec.T                         # -> [n_features, d_model]
    b_dec = sd['b_dec'].float()
    sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, threshold=0.0)
    return sae.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--sae_ckpt', default=None,
                    help='Qwen-Scope SAE .pt (W_enc/W_dec/b_enc/b_dec); omit to '
                         'skip the SAE baselines (sae_native/sae_omp)')
    ap.add_argument('--dicts', required=True,
                    help='comma-sep label:path pairs, e.g. lam100:/a.pt,lam1000:/b.pt')
    ap.add_argument('--n_train', type=int, default=5000)
    ap.add_argument('--n_id_test', type=int, default=2000)
    ap.add_argument('--n_ood_test', type=int, default=2000)
    ap.add_argument('--top_k_values', default='1,5,10,20,40,80,160')
    ap.add_argument('--include_pca', action='store_true', default=True)
    ap.add_argument('--include_random', action='store_true', default=True)
    ap.add_argument('--n_random', type=int, default=None,
                    help='size of the random-direction pool OMP selects from '
                         '(default max(k)*4). Set to match dict atom counts.')
    ap.add_argument('--max_atoms', type=int, default=None,
                    help='cap each loaded dict to its first N atoms (for a '
                         'matched-pool-size comparison across dicts).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--omp_batch', type=int, default=2000)
    ap.add_argument('--activations_cache', default=None)
    args = ap.parse_args()

    k_values = [int(k) for k in args.top_k_values.split(',')]
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dict_specs = []
    for part in args.dicts.split(','):
        label, path = part.split(':', 1)
        dict_specs.append((label.strip(), path.strip()))

    # --- Activations (cache to skip model reload) -------------------------
    cache_path = args.activations_cache or os.path.join(args.out, 'activations.pt')
    if os.path.exists(cache_path):
        print(f'Loading cached activations from {cache_path}', flush=True)
        c = torch.load(cache_path, weights_only=False)
        train_acts, train_labels = c['train_acts'], c['train_labels']
        id_acts, id_labels = c['id_test_acts'], c['id_test_labels']
        ood_acts, ood_labels = c['ood_acts'], c['ood_labels']
    else:
        print('Loading SST-2...', flush=True)
        sst2 = load_dataset('stanfordnlp/sst2')
        texts = sst2['train']['sentence']; labs = sst2['train']['label']
        n_total = min(args.n_train + args.n_id_test, len(texts))
        idx = np.random.permutation(n_total)
        tr, te = idx[:args.n_train], idx[args.n_train:args.n_train + args.n_id_test]
        train_texts = [texts[i] for i in tr]
        train_labels = np.array([labs[i] for i in tr])
        id_texts = [texts[i] for i in te]
        id_labels = np.array([labs[i] for i in te])

        print('Loading Amazon polarity (OOD)...', flush=True)
        amazon = load_dataset('fancyzhx/amazon_polarity', split='test')
        oi = np.random.choice(len(amazon), args.n_ood_test, replace=False)
        ood_texts = [amazon[int(i)]['content'] for i in oi]
        ood_labels = np.array([amazon[int(i)]['label'] for i in oi])

        print(f'Loading model {args.model}', flush=True)
        model, tokenizer, device = load_model(args.model)
        print('Extracting activations...', flush=True)
        train_acts = get_activations(model, tokenizer, train_texts, args.layer, device, args.batch_size)
        id_acts = get_activations(model, tokenizer, id_texts, args.layer, device, args.batch_size)
        ood_acts = get_activations(model, tokenizer, ood_texts, args.layer, device, args.batch_size)
        torch.save({'train_acts': train_acts, 'train_labels': train_labels,
                    'id_test_acts': id_acts, 'id_test_labels': id_labels,
                    'ood_acts': ood_acts, 'ood_labels': ood_labels}, cache_path)
        del model; torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d_model = train_acts.shape[1]
    print(f'train={len(train_acts)} id={len(id_acts)} ood={len(ood_acts)} d={d_model}', flush=True)

    # mean-as-free-bias: fixed bias fit on train, applied to all splits
    abar = train_acts.mean(dim=0, keepdim=True)

    def omp_probe(D, k):
        """OMP-k with mean bias, then probe; returns (id_acc, ood_acc)."""
        tr = abar + omp_gpu(D, train_acts - abar, k, device, args.omp_batch)
        idr = abar + omp_gpu(D, id_acts - abar, k, device, args.omp_batch)
        odr = abar + omp_gpu(D, ood_acts - abar, k, device, args.omp_batch)
        return (run_probe(tr.numpy(), train_labels, idr.numpy(), id_labels, args.seed),
                run_probe(tr.numpy(), train_labels, odr.numpy(), ood_labels, args.seed))

    results = {'args': vars(args), 'k_values': k_values, 'dict_labels': [l for l, _ in dict_specs]}

    # --- raw ceiling ------------------------------------------------------
    raw_id = run_probe(train_acts.numpy(), train_labels, id_acts.numpy(), id_labels, args.seed)
    raw_ood = run_probe(train_acts.numpy(), train_labels, ood_acts.numpy(), ood_labels, args.seed)
    results['raw'] = {'id_acc': raw_id, 'ood_acc': raw_ood}
    print(f'\nraw: ID {raw_id:.4f}  OOD {raw_ood:.4f}', flush=True)

    # --- load SAE + decoder dictionary (optional) -------------------------
    sae = None
    if args.sae_ckpt:
        print(f'\nLoading Qwen SAE {args.sae_ckpt}', flush=True)
        sae = load_qwen_sae(args.sae_ckpt, device)
        sae_dec_dirs = sae.W_dec.detach().cpu().float()           # [n_feat, d]
        sae_dec_dirs = sae_dec_dirs / (sae_dec_dirs.norm(dim=-1, keepdim=True) + 1e-12)
        print(f'  SAE features: {sae_dec_dirs.shape[0]}', flush=True)

    # --- load our dicts ---------------------------------------------------
    dicts = []
    for label, path in dict_specs:
        D = load_dict(path)
        if args.max_atoms is not None and D.shape[0] > args.max_atoms:
            D = D[:args.max_atoms]
        dicts.append((label, D))
        print(f'  dict {label}: {D.shape[0]} atoms', flush=True)

    # --- PCA / random dictionaries ---------------------------------------
    n_ref = max(d.shape[0] for _, d in dicts)
    if args.include_pca:
        _, S, Vt = torch.linalg.svd(train_acts - abar, full_matrices=False)
        pca_dirs = Vt[:max(k_values)].float()
    if args.include_random:
        n_rand = args.n_random if args.n_random is not None else max(k_values) * 4
        rng = np.random.RandomState(args.seed + 1000)
        rand_dirs = torch.from_numpy(rng.randn(n_rand, d_model).astype(np.float32))
        rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)
        print(f'  random pool: {n_rand} directions', flush=True)

    results['sae_native'] = {}; results['sae_omp'] = {}
    results['pca'] = {}; results['random'] = {}
    for label, _ in dicts:
        results[f'dict:{label}'] = {}

    # --- sweep k ----------------------------------------------------------
    for k in k_values:
        print(f'\n--- k={k} ---', flush=True)

        if sae is not None:
            # real SAE: native encoder top-k
            tr, _ = reconstruct_topk(sae, train_acts, k, device)
            idr, _ = reconstruct_topk(sae, id_acts, k, device)
            odr, _ = reconstruct_topk(sae, ood_acts, k, device)
            sn = (run_probe(tr.numpy(), train_labels, idr.numpy(), id_labels, args.seed),
                  run_probe(tr.numpy(), train_labels, odr.numpy(), ood_labels, args.seed))
            results['sae_native'][k] = {'id_acc': sn[0], 'ood_acc': sn[1]}
            print(f'  sae_native top-{k}: ID {sn[0]:.4f}  OOD {sn[1]:.4f}', flush=True)

            # SAE decoder as a plain dictionary, OMP-k (fair atom-vs-atom)
            so = omp_probe(sae_dec_dirs, k)
            results['sae_omp'][k] = {'id_acc': so[0], 'ood_acc': so[1]}
            print(f'  sae_omp    OMP-{k}: ID {so[0]:.4f}  OOD {so[1]:.4f}', flush=True)

        # our dicts
        for label, D in dicts:
            a = omp_probe(D, k)
            results[f'dict:{label}'][k] = {'id_acc': a[0], 'ood_acc': a[1]}
            print(f'  dict:{label} OMP-{k}: ID {a[0]:.4f}  OOD {a[1]:.4f}', flush=True)

        # baselines
        if args.include_pca:
            a = omp_probe(pca_dirs, k)
            results['pca'][k] = {'id_acc': a[0], 'ood_acc': a[1]}
            print(f'  pca        OMP-{k}: ID {a[0]:.4f}  OOD {a[1]:.4f}', flush=True)
        if args.include_random:
            a = omp_probe(rand_dirs, k)
            results['random'][k] = {'id_acc': a[0], 'ood_acc': a[1]}
            print(f'  random     OMP-{k}: ID {a[0]:.4f}  OOD {a[1]:.4f}', flush=True)

        with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # --- plot -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    series = [('sae_native', 'Qwen SAE (native top-k)', 'o-', 'tab:blue'),
              ('sae_omp', 'Qwen SAE decoder (OMP-k)', 'D-', 'tab:cyan')]
    dcolors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
    for i, (label, _) in enumerate(dicts):
        series.append((f'dict:{label}', f'dict {label} (OMP-k)', 's-', dcolors[i % len(dcolors)]))
    if args.include_pca:
        series.append(('pca', 'PCA (OMP-k)', 'v-', 'tab:green'))
    if args.include_random:
        series.append(('random', 'random (OMP-k)', '^-', 'tab:gray'))

    for ax, split, title in [(ax1, 'id_acc', 'In-distribution (SST-2)'),
                             (ax2, 'ood_acc', 'Out-of-distribution (Amazon)')]:
        for key, lbl, style, color in series:
            ks = sorted(results[key].keys())
            if ks:
                ax.plot(ks, [results[key][k][split] for k in ks], style, color=color, label=lbl)
        ax.axhline(results['raw'][split], ls='--', color='black', alpha=0.6, label='raw activations')
        ax.set_xlabel('k (atoms / latents)'); ax.set_ylabel('accuracy')
        ax.set_xscale('log'); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out, 'probe_dict_vs_sae.png')
    plt.savefig(plot_path, dpi=150)
    print(f'\nPlot:    {plot_path}')
    print(f'Results: {os.path.join(args.out, "results.pkl")}')


if __name__ == '__main__':
    main()
