#!/usr/bin/env python3
"""
Run MELBO OMP probe comparison only (reuses cached activations if available).
Skips SAE/PCA/random — those don't change with more MELBO directions.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from sklearn.metrics import accuracy_score

from src.model import load_model


def get_activations(model, tokenizer, texts, layer, device, batch_size=32):
    """Extract last-token activations at a given layer."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        # Last token per sequence
        seq_lens = tokens['attention_mask'].sum(dim=1) - 1
        hidden = outputs.hidden_states[layer + 1]
        acts = hidden[torch.arange(len(batch)), seq_lens].cpu().float()
        all_acts.append(acts)
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(texts)}...")
    return torch.cat(all_acts, dim=0)


def reconstruct_omp_gpu(dictionary, activations, k, device='cuda'):
    """Batch OMP on GPU. dictionary: [n_dirs, d], activations: [N, d]."""
    D = dictionary.to(device)  # [n_dirs, d]
    X = activations.to(device)  # [N, d]
    N, d = X.shape
    n_dirs = D.shape[0]

    # Precompute all projections: [N, n_dirs]
    projections = X @ D.T

    # Greedy OMP: pick top-k one at a time with residual updates
    selected = torch.zeros(N, k, dtype=torch.long, device=device)
    coeffs = torch.zeros(N, k, device=device)
    residuals = X.clone()

    for step in range(k):
        # Correlations of residuals with all directions
        corr = (residuals @ D.T).abs()  # [N, n_dirs]
        # Mask already selected
        for s in range(step):
            corr.scatter_(1, selected[:, s:s+1], 0.0)
        # Pick best
        selected[:, step] = corr.argmax(dim=1)

        # Solve least squares for selected directions so far
        # Gather selected directions: [N, step+1, d]
        idx = selected[:, :step+1]  # [N, step+1]
        D_sel = D[idx]  # [N, step+1, d]

        # Batch least squares: c = (G + eps*I)^-1 @ D_sel @ x
        # D_sel: [N, step+1, d], X: [N, d]
        gram = D_sel @ D_sel.transpose(1, 2)  # [N, step+1, step+1]
        gram += 1e-6 * torch.eye(step + 1, device=device).unsqueeze(0)
        rhs = (D_sel @ X.unsqueeze(2)).squeeze(2)  # [N, step+1]
        c = torch.linalg.solve(gram, rhs)  # [N, step+1]

        # Update residuals
        recon = (c.unsqueeze(2) * D_sel).sum(dim=1)  # [N, d]
        residuals = X - recon

    return recon.cpu()


def reconstruct_omp_batch(dictionary, activations, k, batch_size=500):
    """OMP reconstruction in batches (GPU)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_recon = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        recon = reconstruct_omp_gpu(dictionary, batch, k, device)
        all_recon.append(recon)
    return torch.cat(all_recon, dim=0)


def run_probe(train_X, train_y, test_X, test_y, seed=42):
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(train_X, train_y)
    return accuracy_score(test_y, clf.predict(test_X))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--melbo_path', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_id_test', type=int, default=2000)
    parser.add_argument('--n_ood_test', type=int, default=2000)
    parser.add_argument('--top_k_values', type=str, default='1,5,10,20,40,80,100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--activations_cache', type=str, default=None)
    args = parser.parse_args()

    k_values = [int(k) for k in args.top_k_values.split(',')]

    if args.results_dir is None:
        args.results_dir = f'results/probe_comparison/{args.model}_L{args.layer}_melbo_v3'
    os.makedirs(args.results_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load or extract activations ---
    cache_path = args.activations_cache or os.path.join(args.results_dir, 'activations.pt')
    if os.path.exists(cache_path):
        print(f"Loading cached activations from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
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

        print(f"Train: {len(train_texts_sel)}, ID test: {len(id_test_texts)}, OOD: {len(ood_texts)}")

        print(f"\nLoading model: {args.model}")
        model, tokenizer, device = load_model(args.model)

        print("Extracting activations...")
        train_acts = get_activations(model, tokenizer, train_texts_sel, args.layer, device, args.batch_size)
        id_test_acts = get_activations(model, tokenizer, id_test_texts, args.layer, device, args.batch_size)
        ood_acts = get_activations(model, tokenizer, ood_texts, args.layer, device, args.batch_size)
        print(f"  Activation shape: {train_acts.shape}")

        torch.save({
            'train_acts': train_acts, 'train_labels': train_labels,
            'id_test_acts': id_test_acts, 'id_test_labels': id_test_labels,
            'ood_acts': ood_acts, 'ood_labels': ood_labels,
        }, os.path.join(args.results_dir, 'activations.pt'))
        print(f"Activations cached")

        del model
        torch.cuda.empty_cache()

    d_model = train_acts.shape[1]
    print(f"\nActivations: train={len(train_acts)}, id_test={len(id_test_acts)}, ood={len(ood_acts)}, d={d_model}")

    # --- Baseline ---
    raw_id = run_probe(train_acts.numpy(), train_labels, id_test_acts.numpy(), id_test_labels, args.seed)
    raw_ood = run_probe(train_acts.numpy(), train_labels, ood_acts.numpy(), ood_labels, args.seed)
    print(f"Raw: ID={raw_id:.4f} OOD={raw_ood:.4f}")

    # --- Load MELBO directions ---
    print(f"\nLoading MELBO directions from {args.melbo_path}")
    melbo_data = pickle.load(open(args.melbo_path, 'rb'))
    melbo_dirs_raw = melbo_data['directions']
    melbo_dirs = torch.stack([d.float().squeeze() for d in melbo_dirs_raw])
    melbo_dirs = melbo_dirs / melbo_dirs.norm(dim=-1, keepdim=True)
    n_melbo = len(melbo_dirs)
    print(f"  {n_melbo} MELBO directions")

    # --- Sweep k ---
    results = {'raw': {'id_acc': raw_id, 'ood_acc': raw_ood}, 'melbo': {}, 'n_melbo': n_melbo}
    for k in k_values:
        if k > n_melbo:
            print(f"  k={k} > n_melbo={n_melbo}, skipping")
            continue
        print(f"\n  MELBO OMP-{k}...")
        train_recon = reconstruct_omp_batch(melbo_dirs, train_acts, k)
        id_recon = reconstruct_omp_batch(melbo_dirs, id_test_acts, k)
        ood_recon = reconstruct_omp_batch(melbo_dirs, ood_acts, k)
        melbo_id = run_probe(train_recon.numpy(), train_labels, id_recon.numpy(), id_test_labels, args.seed)
        melbo_ood = run_probe(train_recon.numpy(), train_labels, ood_recon.numpy(), ood_labels, args.seed)
        print(f"    ID={melbo_id:.4f} OOD={melbo_ood:.4f}")
        results['melbo'][k] = {'id_acc': melbo_id, 'ood_acc': melbo_ood}

        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    print("\nDone")
    for k in sorted(results['melbo'].keys()):
        r = results['melbo'][k]
        print(f"  k={k}: ID={r['id_acc']:.4f} OOD={r['ood_acc']:.4f}")


if __name__ == '__main__':
    main()
