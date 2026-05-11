#!/usr/bin/env python3
"""
Sentiment probing: raw activations vs SAE reconstruction vs deep AE reconstruction.

Trains linear probes (LogisticRegression) on Qwen3-1.7B layer 2 last-token activations.
Compares three conditions:
  1. Raw activations (baseline)
  2. TopK SAE reconstruction (67M, d_sae=16384, k=32)
  3. Deep autoencoder reconstruction (67M, FlexAutoencoder, k=32)

Datasets:
  - Train: SST-2 (movie reviews)
  - ID test: held-out SST-2
  - OOD test: Amazon polarity reviews

Usage:
    python -u scripts/probes/run_probe_sae_vs_ae.py \
        --sae_weights results/final_comparison/sae_k32_50k/sae_weights.pt \
        --ae_weights results/final_comparison/ae_mse_cos_k32_50k/final_mse+cos_model.pt \
        --results_dir results/probe_sae_vs_ae
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.model import load_model
from src.autoencoders import TopKSAE, FlexAutoencoder, load_sae, load_ae


# ── Activation extraction ───────────────────────────────────────────────

def get_activations(model, tokenizer, texts, layer, device, batch_size=32):
    """Extract final-token activations at specified layer."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        mask = inputs['attention_mask']
        last_pos = mask.sum(dim=1) - 1
        acts = hidden[torch.arange(hidden.size(0)), last_pos]
        all_acts.append(acts.cpu().float())
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i+batch_size, len(texts))}/{len(texts)}", end='\r')
    print()
    return torch.cat(all_acts, dim=0)


def reconstruct(model, activations, device, batch_size=512):
    """Reconstruct activations through a model (SAE or AE).

    Both TopKSAE and FlexAutoencoder return tuples from forward(); unpack
    so callers see only the reconstructed activations.
    """
    all_recon = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
            recon = out[0] if isinstance(out, tuple) else out
        all_recon.append(recon.cpu().float())
    return torch.cat(all_recon, dim=0)


def recon_metrics(original, reconstructed):
    """Compute reconstruction quality metrics."""
    cos_sim = F.cosine_similarity(original, reconstructed, dim=-1).mean().item()
    mse = F.mse_loss(reconstructed, original).item()
    rel_l2 = ((original - reconstructed).norm(dim=-1) / original.norm(dim=-1)).mean().item()
    return {'cos_sim': cos_sim, 'mse': mse, 'rel_l2': rel_l2}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--sae_weights', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_id_test', type=int, default=500)
    parser.add_argument('--n_ood_test', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results/probe_sae_vs_ae')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load datasets ────────────────────────────────────────────────────
    print("Loading SST-2...")
    sst2 = load_dataset('stanfordnlp/sst2')
    train_texts = sst2['train']['sentence']
    train_labels = sst2['train']['label']

    n_total = min(args.n_train + args.n_id_test, len(train_texts))
    idx = np.random.permutation(n_total)
    train_idx = idx[:args.n_train]
    test_idx = idx[args.n_train:args.n_train + args.n_id_test]

    train_texts_sel = [train_texts[i] for i in train_idx]
    train_labels_sel = np.array([train_labels[i] for i in train_idx])
    id_test_texts = [train_texts[i] for i in test_idx]
    id_test_labels = np.array([train_labels[i] for i in test_idx])

    print("Loading Amazon polarity...")
    amazon = load_dataset('amazon_polarity', split='test')
    ood_idx = np.random.choice(len(amazon), args.n_ood_test, replace=False)
    ood_texts = [amazon[int(i)]['content'] for i in ood_idx]
    ood_labels = np.array([amazon[int(i)]['label'] for i in ood_idx])

    print(f"Train: {len(train_texts_sel)}, ID test: {len(id_test_texts)}, "
          f"OOD test: {len(ood_texts)}")
    print(f"Label balance — train: {train_labels_sel.mean():.2f}, "
          f"ID test: {id_test_labels.mean():.2f}, OOD: {ood_labels.mean():.2f}")

    # ── Extract activations ──────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    print("Extracting train activations...")
    train_acts = get_activations(model, tokenizer, train_texts_sel,
                                args.layer, device, args.batch_size)
    print("Extracting ID test activations...")
    id_test_acts = get_activations(model, tokenizer, id_test_texts,
                                  args.layer, device, args.batch_size)
    print("Extracting OOD test activations...")
    ood_acts = get_activations(model, tokenizer, ood_texts,
                               args.layer, device, args.batch_size)
    print(f"Activation shape: {train_acts.shape}")

    # Cache activations
    torch.save({
        'train_acts': train_acts, 'train_labels': train_labels_sel,
        'id_test_acts': id_test_acts, 'id_test_labels': id_test_labels,
        'ood_acts': ood_acts, 'ood_labels': ood_labels,
    }, os.path.join(args.results_dir, 'activations.pt'))

    del model
    torch.cuda.empty_cache()

    # ── Load SAE and AE ──────────────────────────────────────────────────
    print(f"\nLoading SAE: {args.sae_weights}")
    sae = load_sae(args.sae_weights, device)

    print(f"Loading AE: {args.ae_weights}")
    ae = load_ae(args.ae_weights, device)

    # ── Reconstruct ──────────────────────────────────────────────────────
    print("\nReconstructing through SAE...")
    train_sae = reconstruct(sae, train_acts, device)
    id_sae = reconstruct(sae, id_test_acts, device)
    ood_sae = reconstruct(sae, ood_acts, device)

    print("Reconstructing through AE...")
    train_ae = reconstruct(ae, train_acts, device)
    id_ae = reconstruct(ae, id_test_acts, device)
    ood_ae = reconstruct(ae, ood_acts, device)

    # ── Reconstruction quality ───────────────────────────────────────────
    all_acts = torch.cat([train_acts, id_test_acts, ood_acts], dim=0)
    all_sae = torch.cat([train_sae, id_sae, ood_sae], dim=0)
    all_ae = torch.cat([train_ae, id_ae, ood_ae], dim=0)

    sae_metrics = recon_metrics(all_acts, all_sae)
    ae_metrics = recon_metrics(all_acts, all_ae)
    print(f"\nReconstruction quality:")
    print(f"  SAE: cos={sae_metrics['cos_sim']:.4f}, mse={sae_metrics['mse']:.6f}, "
          f"rel_L2={sae_metrics['rel_l2']:.4f}")
    print(f"  AE:  cos={ae_metrics['cos_sim']:.4f}, mse={ae_metrics['mse']:.6f}, "
          f"rel_L2={ae_metrics['rel_l2']:.4f}")

    # ── Train probes ─────────────────────────────────────────────────────
    results = {
        'args': vars(args),
        'sae_recon': sae_metrics,
        'ae_recon': ae_metrics,
    }

    conditions = [
        ('raw', train_acts, id_test_acts, ood_acts),
        ('sae', train_sae, id_sae, ood_sae),
        ('ae', train_ae, id_ae, ood_ae),
    ]

    print(f"\nProbe results:")
    for name, tr, id_t, ood_t in conditions:
        clf = LogisticRegression(max_iter=2000, random_state=args.seed)
        clf.fit(tr.numpy(), train_labels_sel)
        id_acc = accuracy_score(id_test_labels, clf.predict(id_t.numpy()))
        ood_acc = accuracy_score(ood_labels, clf.predict(ood_t.numpy()))
        print(f"  {name:4s}: ID={id_acc:.4f}, OOD={ood_acc:.4f}")
        results[name] = {'id_acc': id_acc, 'ood_acc': ood_acc}

    # ── Save ─────────────────────────────────────────────────────────────
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
