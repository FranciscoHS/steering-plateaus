#!/usr/bin/env python3
"""
Compare probes trained on SAE vs MELBO vs random direction reconstructions
at different sparsity levels (using OMP for MELBO/random).

Train on SST-2, ID test on held-out SST-2, OOD test on Amazon reviews.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from scripts.probes.run_probe_generalization import (
    get_activations, load_sae_from_hf, reconstruct_topk, SimpleSAE
)


def reconstruct_omp(dictionary, activations, k):
    """Reconstruct activations using OMP with k nonzero coefficients.

    Args:
        dictionary: [n_dirs, d_model] unit direction vectors
        activations: [n_samples, d_model]
        k: number of nonzero coefficients
    Returns:
        reconstructed activations [n_samples, d_model]
    """
    D = dictionary.numpy().T  # sklearn wants [d_model, n_dirs]
    X = activations.numpy()

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    omp.fit(D, X.T)  # fit: D @ coefs ≈ X.T
    # omp.coef_ is [n_samples, n_dirs]
    recon = omp.coef_ @ dictionary.numpy()  # [n_samples, d_model]
    return torch.from_numpy(recon).float()


def reconstruct_omp_batch(dictionary, activations, k, batch_size=500):
    """OMP reconstruction in batches to manage memory."""
    D = dictionary.numpy().T  # [d_model, n_dirs]
    all_recon = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size].numpy()
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        omp.fit(D, batch.T)
        recon = omp.coef_ @ dictionary.numpy()
        all_recon.append(torch.from_numpy(recon).float())
    return torch.cat(all_recon, dim=0)


def run_probe(train_X, train_y, test_X, test_y, seed=42):
    """Train logistic regression and return accuracy."""
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(train_X, train_y)
    return accuracy_score(test_y, clf.predict(test_X))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--sae_repo', type=str, default='adamkarvonen/qwen3-1.7b-saes')
    parser.add_argument('--sae_trainer', type=str, default='trainer_0')
    parser.add_argument('--melbo_path', type=str, required=True,
                        help='Path to MELBO checkpoint pkl with "directions" key')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_id_test', type=int, default=2000)
    parser.add_argument('--n_ood_test', type=int, default=2000)
    parser.add_argument('--top_k_values', type=str, default='1,5,10,20,40,80,160')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--activations_cache', type=str, default=None,
                        help='Path to cached activations.pt (skip model loading)')
    args = parser.parse_args()

    k_values = [int(k) for k in args.top_k_values.split(',')]

    if args.results_dir is None:
        args.results_dir = f'results/probe_comparison/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load or extract activations ---
    if args.activations_cache and os.path.exists(args.activations_cache):
        print(f"Loading cached activations from {args.activations_cache}")
        cache = torch.load(args.activations_cache, weights_only=False)
        train_acts = cache['train_acts']
        train_labels = cache['train_labels']
        id_test_acts = cache['id_test_acts']
        id_test_labels = cache['id_test_labels']
        ood_acts = cache['ood_acts']
        ood_labels = cache['ood_labels']
    else:
        print("Loading SST-2...")
        sst2 = load_dataset('stanfordnlp/sst2')
        train_texts = sst2['train']['sentence']
        train_labels_all = sst2['train']['label']

        n_total = min(args.n_train + args.n_id_test, len(train_texts))
        idx = np.random.permutation(n_total)
        train_idx = idx[:args.n_train]
        test_idx = idx[args.n_train:args.n_train + args.n_id_test]

        train_texts_sel = [train_texts[i] for i in train_idx]
        train_labels = np.array([train_labels_all[i] for i in train_idx])
        id_test_texts = [train_texts[i] for i in test_idx]
        id_test_labels = np.array([train_labels_all[i] for i in test_idx])

        print("Loading Amazon polarity...")
        amazon = load_dataset('amazon_polarity', split='test')
        ood_idx = np.random.choice(len(amazon), args.n_ood_test, replace=False)
        ood_texts = [amazon[int(i)]['content'] for i in ood_idx]
        ood_labels = np.array([amazon[int(i)]['label'] for i in ood_idx])

        print(f"Train: {len(train_texts_sel)}, ID test: {len(id_test_texts)}, "
              f"OOD test: {len(ood_texts)}")

        print(f"\nLoading model: {args.model}")
        model, tokenizer, device = load_model(args.model)

        print("Extracting activations...")
        train_acts = get_activations(model, tokenizer, train_texts_sel,
                                    args.layer, device, args.batch_size)
        id_test_acts = get_activations(model, tokenizer, id_test_texts,
                                      args.layer, device, args.batch_size)
        ood_acts = get_activations(model, tokenizer, ood_texts,
                                   args.layer, device, args.batch_size)
        print(f"  Activation shape: {train_acts.shape}")

        cache_path = os.path.join(args.results_dir, 'activations.pt')
        torch.save({
            'train_acts': train_acts, 'train_labels': train_labels,
            'id_test_acts': id_test_acts, 'id_test_labels': id_test_labels,
            'ood_acts': ood_acts, 'ood_labels': ood_labels,
        }, cache_path)
        print(f"Activations cached to {cache_path}")

        del model
        torch.cuda.empty_cache()

    d_model = train_acts.shape[1]
    print(f"\nActivations: train={len(train_acts)}, id_test={len(id_test_acts)}, "
          f"ood_test={len(ood_acts)}, d_model={d_model}")

    # --- Load MELBO directions ---
    print(f"\nLoading MELBO directions from {args.melbo_path}")
    melbo_data = pickle.load(open(args.melbo_path, 'rb'))
    melbo_dirs_raw = melbo_data['directions']
    # Stack and normalize
    melbo_dirs = torch.stack([d.float().squeeze() for d in melbo_dirs_raw])
    melbo_dirs = melbo_dirs / melbo_dirs.norm(dim=-1, keepdim=True)
    n_melbo = len(melbo_dirs)
    print(f"  {n_melbo} MELBO directions, shape {melbo_dirs.shape}")

    # --- Generate random directions (same count as MELBO) ---
    print(f"\nGenerating {n_melbo} random directions...")
    rng = np.random.RandomState(args.seed + 1000)
    rand_dirs = torch.from_numpy(
        rng.randn(n_melbo, d_model).astype(np.float32))
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)

    # --- Baseline: raw activations ---
    print("\n=== Raw activations ===")
    raw_id = run_probe(train_acts.numpy(), train_labels,
                       id_test_acts.numpy(), id_test_labels, args.seed)
    raw_ood = run_probe(train_acts.numpy(), train_labels,
                        ood_acts.numpy(), ood_labels, args.seed)
    print(f"  ID: {raw_id:.4f}  OOD: {raw_ood:.4f}")

    results = {
        'args': vars(args), 'k_values': k_values, 'n_melbo': n_melbo,
        'raw': {'id_acc': raw_id, 'ood_acc': raw_ood},
        'sae': {}, 'sae_restricted': {}, 'melbo': {}, 'random': {}, 'pca': {},
    }

    # --- Load SAE ---
    print(f"\nLoading SAE: {args.sae_repo} / {args.sae_trainer}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae = load_sae_from_hf(args.sae_repo, args.layer, args.sae_trainer, device)

    # --- SAE default (trained threshold) ---
    print("\n=== SAE default threshold ===")
    train_recon, train_l0 = reconstruct_topk(sae, train_acts, None, device)
    id_recon, _ = reconstruct_topk(sae, id_test_acts, None, device)
    ood_recon, _ = reconstruct_topk(sae, ood_acts, None, device)
    sae_def_id = run_probe(train_recon.numpy(), train_labels,
                           id_recon.numpy(), id_test_labels, args.seed)
    sae_def_ood = run_probe(train_recon.numpy(), train_labels,
                            ood_recon.numpy(), ood_labels, args.seed)
    print(f"  Mean L0: {train_l0:.1f}  ID: {sae_def_id:.4f}  OOD: {sae_def_ood:.4f}")
    results['sae_default'] = {
        'id_acc': sae_def_id, 'ood_acc': sae_def_ood, 'mean_l0': train_l0}

    # --- Top-N SAE features by activity on training set ---
    print(f"\nSelecting top-{n_melbo} SAE features by training set activity...")
    all_feature_activity = []
    for i in range(0, len(train_acts), 256):
        batch = train_acts[i:i+256].to(device)
        with torch.no_grad():
            z = torch.relu(sae.encode_pre(batch))
        all_feature_activity.append(z.cpu())
    all_feature_activity = torch.cat(all_feature_activity, dim=0)
    # Rank by mean activation magnitude
    mean_activity = all_feature_activity.mean(dim=0)
    top_feature_idx = mean_activity.argsort(descending=True)[:n_melbo]
    # Extract decoder directions for these features
    sae_restricted_dirs = sae.W_dec[top_feature_idx].cpu().float()
    sae_restricted_dirs = sae_restricted_dirs / sae_restricted_dirs.norm(dim=-1, keepdim=True)
    print(f"  Selected {len(sae_restricted_dirs)} SAE directions")

    # --- PCA directions from training set ---
    print(f"\nComputing top-{n_melbo} PCA directions from training set...")
    train_centered = train_acts - train_acts.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(train_centered, full_matrices=False)
    pca_dirs = Vt[:n_melbo].float()  # [n_melbo, d_model], already unit norm
    print(f"  PCA directions shape: {pca_dirs.shape}")
    print(f"  Variance explained by top-{n_melbo}: "
          f"{(S[:n_melbo]**2).sum() / (S**2).sum():.4f}")

    # --- Sweep over k values ---
    for k in k_values:
        print(f"\n--- k={k} ---")

        # SAE top-k
        train_recon, _ = reconstruct_topk(sae, train_acts, k, device)
        id_recon, _ = reconstruct_topk(sae, id_test_acts, k, device)
        ood_recon, _ = reconstruct_topk(sae, ood_acts, k, device)
        sae_id = run_probe(train_recon.numpy(), train_labels,
                           id_recon.numpy(), id_test_labels, args.seed)
        sae_ood = run_probe(train_recon.numpy(), train_labels,
                            ood_recon.numpy(), ood_labels, args.seed)
        print(f"  SAE top-{k}:    ID: {sae_id:.4f}  OOD: {sae_ood:.4f}")
        results['sae'][k] = {'id_acc': sae_id, 'ood_acc': sae_ood}

        # MELBO OMP-k
        if k <= n_melbo:
            train_recon = reconstruct_omp_batch(melbo_dirs, train_acts, k)
            id_recon = reconstruct_omp_batch(melbo_dirs, id_test_acts, k)
            ood_recon = reconstruct_omp_batch(melbo_dirs, ood_acts, k)
            melbo_id = run_probe(train_recon.numpy(), train_labels,
                                id_recon.numpy(), id_test_labels, args.seed)
            melbo_ood = run_probe(train_recon.numpy(), train_labels,
                                 ood_recon.numpy(), ood_labels, args.seed)
            print(f"  MELBO OMP-{k}:  ID: {melbo_id:.4f}  OOD: {melbo_ood:.4f}")
            results['melbo'][k] = {'id_acc': melbo_id, 'ood_acc': melbo_ood}

        # SAE restricted OMP-k (top-N SAE features only)
        if k <= n_melbo:
            train_recon = reconstruct_omp_batch(sae_restricted_dirs, train_acts, k)
            id_recon = reconstruct_omp_batch(sae_restricted_dirs, id_test_acts, k)
            ood_recon = reconstruct_omp_batch(sae_restricted_dirs, ood_acts, k)
            sr_id = run_probe(train_recon.numpy(), train_labels,
                              id_recon.numpy(), id_test_labels, args.seed)
            sr_ood = run_probe(train_recon.numpy(), train_labels,
                               ood_recon.numpy(), ood_labels, args.seed)
            print(f"  SAE-{n_melbo} OMP-{k}: ID: {sr_id:.4f}  OOD: {sr_ood:.4f}")
            results['sae_restricted'][k] = {'id_acc': sr_id, 'ood_acc': sr_ood}

        # PCA OMP-k
        if k <= n_melbo:
            train_recon = reconstruct_omp_batch(pca_dirs, train_acts, k)
            id_recon = reconstruct_omp_batch(pca_dirs, id_test_acts, k)
            ood_recon = reconstruct_omp_batch(pca_dirs, ood_acts, k)
            pca_id = run_probe(train_recon.numpy(), train_labels,
                               id_recon.numpy(), id_test_labels, args.seed)
            pca_ood = run_probe(train_recon.numpy(), train_labels,
                                ood_recon.numpy(), ood_labels, args.seed)
            print(f"  PCA OMP-{k}:  ID: {pca_id:.4f}  OOD: {pca_ood:.4f}")
            results['pca'][k] = {'id_acc': pca_id, 'ood_acc': pca_ood}

        # Random OMP-k
        if k <= n_melbo:
            train_recon = reconstruct_omp_batch(rand_dirs, train_acts, k)
            id_recon = reconstruct_omp_batch(rand_dirs, id_test_acts, k)
            ood_recon = reconstruct_omp_batch(rand_dirs, ood_acts, k)
            rand_id = run_probe(train_recon.numpy(), train_labels,
                                id_recon.numpy(), id_test_labels, args.seed)
            rand_ood = run_probe(train_recon.numpy(), train_labels,
                                 ood_recon.numpy(), ood_labels, args.seed)
            print(f"  Random OMP-{k}: ID: {rand_id:.4f}  OOD: {rand_ood:.4f}")
            results['random'][k] = {'id_acc': rand_id, 'ood_acc': rand_ood}

        # Checkpoint after each k
        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split, title in [
        (ax1, 'id_acc', 'In-distribution (SST-2)'),
        (ax2, 'ood_acc', 'Out-of-distribution (Amazon)'),
    ]:
        sae_ks = sorted(results['sae'].keys())
        ax.plot(sae_ks, [results['sae'][k][split] for k in sae_ks],
                'o-', color='tab:blue', label=f'SAE top-k (16k pool)')

        sr_ks = sorted(results.get('sae_restricted', {}).keys())
        if sr_ks:
            ax.plot(sr_ks, [results['sae_restricted'][k][split] for k in sr_ks],
                    'D-', color='tab:cyan', label=f'SAE top-{n_melbo} OMP-k')

        pca_ks = sorted(results.get('pca', {}).keys())
        if pca_ks:
            ax.plot(pca_ks, [results['pca'][k][split] for k in pca_ks],
                    'v-', color='tab:green', label='PCA OMP-k')

        melbo_ks = sorted(results['melbo'].keys())
        if melbo_ks:
            ax.plot(melbo_ks, [results['melbo'][k][split] for k in melbo_ks],
                    's-', color='tab:red', label='MELBO OMP-k')

        rand_ks = sorted(results['random'].keys())
        if rand_ks:
            ax.plot(rand_ks, [results['random'][k][split] for k in rand_ks],
                    '^-', color='tab:gray', label='Random OMP-k')

        ax.axhline(results['raw'][split], ls='--', color='black',
                   alpha=0.6, label='Raw activations')

        ax.set_xlabel('k (number of directions/features)')
        ax.set_ylabel('Accuracy')
        ax.set_xscale('log')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'probe_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot: {plot_path}")
    print(f"Results: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
