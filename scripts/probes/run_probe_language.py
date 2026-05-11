#!/usr/bin/env python3
"""
Language identity probing: raw activations vs SAE vs deep AE reconstruction.

Binary classification (English vs Mandarin) on Qwen3-1.7B layer 2 last-token
activations. Uses Wikipedia sentences as data source.

Usage:
    python -u scripts/probes/run_probe_language.py \
        --sae_weights results/final_comparison/sae_k32_50k/sae_weights.pt \
        --ae_weights results/final_comparison/ae_mse_cos_k32_50k/final_mse+cos_model.pt \
        --results_dir results/probe_language
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


# ── Helpers ─────────────────────────────────────────────────────────────

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
    """Reconstruct activations through an SAE or AE. Both return tuples."""
    all_recon = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
            recon = out[0] if isinstance(out, tuple) else out
        all_recon.append(recon.cpu().float())
    return torch.cat(all_recon, dim=0)


def recon_metrics(original, reconstructed):
    cos_sim = F.cosine_similarity(original, reconstructed, dim=-1).mean().item()
    mse = F.mse_loss(reconstructed, original).item()
    rel_l2 = ((original - reconstructed).norm(dim=-1) / original.norm(dim=-1)).mean().item()
    return {'cos_sim': cos_sim, 'mse': mse, 'rel_l2': rel_l2}


# ── Data loading ────────────────────────────────────────────────────────

def load_wikipedia_sentences(lang, n_samples, seed=42):
    """Load sentences from Wikipedia. Returns list of strings."""
    rng = np.random.RandomState(seed)

    wiki_config = {'en': '20231101.en', 'zh': '20231101.zh'}
    print(f"  Loading Wikipedia ({lang})...")
    ds = load_dataset('wikimedia/wikipedia', wiki_config[lang], split='train', streaming=True)

    # Collect sentences by splitting articles on newlines
    sentences = []
    for article in ds:
        text = article['text']
        for line in text.split('\n'):
            line = line.strip()
            # Filter: reasonable sentence length
            if len(line) < 20 or len(line) > 500:
                continue
            sentences.append(line)
            if len(sentences) >= n_samples * 3:
                break
        if len(sentences) >= n_samples * 3:
            break

    # Shuffle and take n_samples
    idx = rng.permutation(len(sentences))[:n_samples]
    return [sentences[i] for i in idx]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--sae_weights', type=str, required=True)
    parser.add_argument('--ae_weights', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=2000,
                        help='Training samples per language')
    parser.add_argument('--n_test', type=int, default=2500,
                        help='Test samples per language')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results/probe_language')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_total = args.n_train + args.n_test

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data...")
    en_texts = load_wikipedia_sentences('en', n_total, seed=args.seed)
    zh_texts = load_wikipedia_sentences('zh', n_total, seed=args.seed)

    # Split into train/test
    en_train, en_test = en_texts[:args.n_train], en_texts[args.n_train:args.n_train + args.n_test]
    zh_train, zh_test = zh_texts[:args.n_train], zh_texts[args.n_train:args.n_train + args.n_test]

    train_texts = en_train + zh_train
    train_labels = np.array([0] * len(en_train) + [1] * len(zh_train))
    test_texts = en_test + zh_test
    test_labels = np.array([0] * len(en_test) + [1] * len(zh_test))

    # Shuffle
    rng = np.random.RandomState(args.seed)
    train_perm = rng.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in train_perm]
    train_labels = train_labels[train_perm]
    test_perm = rng.permutation(len(test_texts))
    test_texts = [test_texts[i] for i in test_perm]
    test_labels = test_labels[test_perm]

    print(f"Train: {len(train_texts)} ({train_labels.mean():.2f} zh), "
          f"Test: {len(test_texts)} ({test_labels.mean():.2f} zh)")

    # ── Extract activations ──────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    print("Extracting train activations...")
    train_acts = get_activations(model, tokenizer, train_texts,
                                args.layer, device, args.batch_size)
    print("Extracting test activations...")
    test_acts = get_activations(model, tokenizer, test_texts,
                                args.layer, device, args.batch_size)
    print(f"Activation shape: {train_acts.shape}")

    # Cache
    torch.save({
        'train_acts': train_acts, 'train_labels': train_labels,
        'test_acts': test_acts, 'test_labels': test_labels,
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
    test_sae = reconstruct(sae, test_acts, device)

    print("Reconstructing through AE...")
    train_ae = reconstruct(ae, train_acts, device)
    test_ae = reconstruct(ae, test_acts, device)

    # ── Reconstruction quality ───────────────────────────────────────────
    all_acts = torch.cat([train_acts, test_acts], dim=0)
    all_sae = torch.cat([train_sae, test_sae], dim=0)
    all_ae = torch.cat([train_ae, test_ae], dim=0)

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
        ('raw', train_acts, test_acts),
        ('sae', train_sae, test_sae),
        ('ae', train_ae, test_ae),
    ]

    print(f"\nProbe results (en vs zh, layer {args.layer}):")
    for name, tr, te in conditions:
        clf = LogisticRegression(max_iter=2000, random_state=args.seed)
        clf.fit(tr.numpy(), train_labels)
        acc = accuracy_score(test_labels, clf.predict(te.numpy()))
        se = np.sqrt(acc * (1 - acc) / len(test_labels))
        print(f"  {name:4s}: acc={acc:.4f} (±{1.96*se:.4f} 95% CI)")
        results[name] = {'acc': acc, 'se': se}

    # ── Save ─────────────────────────────────────────────────────────────
    results_path = os.path.join(args.results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
