#!/usr/bin/env python3
"""
Probe generalization experiment: compare linear probes trained on raw activations
vs SAE reconstructions at different sparsity levels (top-K).

Train on SST-2 (movie reviews), test ID on held-out SST-2, test OOD on Amazon reviews.
"""

import sys, os, argparse, pickle
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model


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
        # hidden_states[0] = embeddings, [L+1] = after block L
        hidden = outputs.hidden_states[layer + 1]
        # Last non-padding token
        mask = inputs['attention_mask']
        last_pos = mask.sum(dim=1) - 1
        acts = hidden[torch.arange(hidden.size(0)), last_pos]
        all_acts.append(acts.cpu().float())
    return torch.cat(all_acts, dim=0)


class SimpleSAE(torch.nn.Module):
    """Minimal SAE for inference. Supports custom top-k sparsity."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold=0.0):
        super().__init__()
        # W_enc: [n_features, d_model], W_dec: [n_features, d_model]
        self.register_buffer('W_enc', W_enc)
        self.register_buffer('b_enc', b_enc)
        self.register_buffer('W_dec', W_dec)
        self.register_buffer('b_dec', b_dec)
        self.register_buffer('threshold', torch.tensor(threshold))

    def encode_pre(self, x):
        """Pre-activation features."""
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x):
        """Encode with JumpReLU threshold."""
        return torch.relu(self.encode_pre(x) - self.threshold)

    def decode(self, z):
        """Decode sparse features."""
        return z @ self.W_dec + self.b_dec


def load_sae_from_hf(repo_id, layer, trainer='trainer_0', device='cpu', dtype=torch.float32):
    """Load BatchTopK SAE checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    if 'qwen3-1.7b' in repo_id.lower():
        model_subdir = 'Qwen_Qwen3-1.7B'
    elif 'qwen3-8b' in repo_id.lower():
        model_subdir = 'Qwen_Qwen3-8B'
    else:
        raise ValueError(f"Unknown repo: {repo_id}")

    filename = f"saes_{model_subdir}_batch_top_k/resid_post_layer_{layer}/{trainer}/ae.pt"
    print(f"  Downloading: {filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)

    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    print(f"  Checkpoint keys: {list(state_dict.keys())}")

    # Handle different key formats
    if 'encoder.weight' in state_dict:
        W_enc = state_dict['encoder.weight'].to(dtype)  # [n_features, d_model]
        b_enc = state_dict['encoder.bias'].to(dtype)     # [n_features]
        W_dec = state_dict['decoder.weight'].to(dtype)   # might be [d_model, n_features]
        b_dec = state_dict.get('decoder.bias', state_dict.get('b_dec', torch.zeros(W_enc.shape[1]))).to(dtype)
    elif 'W_enc' in state_dict:
        W_enc = state_dict['W_enc'].to(dtype)
        b_enc = state_dict['b_enc'].to(dtype)
        W_dec = state_dict['W_dec'].to(dtype)
        b_dec = state_dict.get('b_dec', torch.zeros(W_enc.shape[1])).to(dtype)
    else:
        raise ValueError(f"Unknown state dict format. Keys: {list(state_dict.keys())}")

    # Ensure W_dec is [n_features, d_model]
    n_features = W_enc.shape[0]
    d_model = W_enc.shape[1]
    if W_dec.shape == (d_model, n_features):
        W_dec = W_dec.T

    threshold = state_dict.get('threshold', torch.tensor(0.0))
    if threshold.numel() == 1:
        threshold = threshold.item()
    else:
        threshold = 0.0

    print(f"  d_model={d_model}, n_features={n_features}, threshold={threshold:.4f}")

    sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, threshold)
    return sae.to(device)


def reconstruct_topk(sae, activations, k, device, batch_size=256):
    """Encode through SAE, keep top-k features by activation magnitude, decode.

    If k is None, uses the SAE's trained JumpReLU threshold (default sparsity).
    """
    all_recon = []
    all_l0 = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size].to(device)
        with torch.no_grad():
            if k is None:
                # Use SAE's trained threshold
                z = sae.encode(batch)
            else:
                # Use pre-activations with ReLU, then keep top-k
                z = torch.relu(sae.encode_pre(batch))
                topk_vals, topk_idx = z.topk(k, dim=-1)
                sparse = torch.zeros_like(z)
                sparse.scatter_(-1, topk_idx, topk_vals)
                z = sparse
            l0 = (z > 0).sum(dim=-1).float()
            all_l0.append(l0.cpu())
            recon = sae.decode(z)
        all_recon.append(recon.cpu().float())
    mean_l0 = torch.cat(all_l0).mean().item()
    return torch.cat(all_recon, dim=0), mean_l0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--sae_repo', type=str, default='adamkarvonen/qwen3-1.7b-saes')
    parser.add_argument('--sae_trainer', type=str, default='trainer_0',
                        help='trainer_0=L0_80_16k, trainer_1=L0_160_16k, '
                             'trainer_2=L0_80_65k, trainer_3=L0_160_65k')
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_id_test', type=int, default=500)
    parser.add_argument('--n_ood_test', type=int, default=500)
    parser.add_argument('--top_k_values', type=str, default='1,5,10,20,40,80,160')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    k_values = [int(k) for k in args.top_k_values.split(',')]

    if args.results_dir is None:
        args.results_dir = (f'results/probe_generalization/'
                           f'{args.model}_L{args.layer}_{args.sae_trainer}')
    os.makedirs(args.results_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load datasets ---
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

    # --- Extract activations ---
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

    # Cache activations
    torch.save({
        'train_acts': train_acts, 'train_labels': train_labels_sel,
        'id_test_acts': id_test_acts, 'id_test_labels': id_test_labels,
        'ood_acts': ood_acts, 'ood_labels': ood_labels,
    }, os.path.join(args.results_dir, 'activations.pt'))
    print("Activations cached.")

    del model
    torch.cuda.empty_cache()

    # --- Load SAE ---
    print(f"\nLoading SAE: {args.sae_repo} / {args.sae_trainer}")
    sae = load_sae_from_hf(args.sae_repo, args.layer, args.sae_trainer, device)

    # --- Baseline: raw activations ---
    print("\n=== Raw activations ===")
    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(train_acts.numpy(), train_labels_sel)
    raw_id = accuracy_score(id_test_labels, clf.predict(id_test_acts.numpy()))
    raw_ood = accuracy_score(ood_labels, clf.predict(ood_acts.numpy()))
    print(f"  ID: {raw_id:.4f}  OOD: {raw_ood:.4f}")

    results = {
        'args': vars(args), 'k_values': k_values,
        'raw': {'id_acc': raw_id, 'ood_acc': raw_ood},
    }

    # --- Full SAE reconstruction (no top-k constraint) ---
    print("\n=== Full SAE reconstruction ===")
    train_recon, train_l0 = reconstruct_topk(sae, train_acts, None, device)
    id_recon, _ = reconstruct_topk(sae, id_test_acts, None, device)
    ood_recon, _ = reconstruct_topk(sae, ood_acts, None, device)
    print(f"  Mean L0: {train_l0:.1f}")

    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(train_recon.numpy(), train_labels_sel)
    full_id = accuracy_score(id_test_labels, clf.predict(id_recon.numpy()))
    full_ood = accuracy_score(ood_labels, clf.predict(ood_recon.numpy()))
    print(f"  ID: {full_id:.4f}  OOD: {full_ood:.4f}")
    results['sae_full'] = {'id_acc': full_id, 'ood_acc': full_ood, 'mean_l0': train_l0}

    # --- Top-k sweep ---
    results['topk'] = {}
    for k in k_values:
        print(f"\n=== SAE top-{k} ===")
        train_recon, _ = reconstruct_topk(sae, train_acts, k, device)
        id_recon, _ = reconstruct_topk(sae, id_test_acts, k, device)
        ood_recon, _ = reconstruct_topk(sae, ood_acts, k, device)

        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(train_recon.numpy(), train_labels_sel)
        k_id = accuracy_score(id_test_labels, clf.predict(id_recon.numpy()))
        k_ood = accuracy_score(ood_labels, clf.predict(ood_recon.numpy()))
        print(f"  ID: {k_id:.4f}  OOD: {k_ood:.4f}")
        results['topk'][k] = {'id_acc': k_id, 'ood_acc': k_ood}

    # --- Save ---
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(results['topk'].keys())
    id_accs = [results['topk'][k]['id_acc'] for k in ks]
    ood_accs = [results['topk'][k]['ood_acc'] for k in ks]

    ax.plot(ks, id_accs, 'o-', label='SAE top-k (ID)', color='tab:blue')
    ax.plot(ks, ood_accs, 's-', label='SAE top-k (OOD)', color='tab:orange')

    ax.axhline(raw_id, ls='--', color='tab:blue', alpha=0.5, label='Raw (ID)')
    ax.axhline(raw_ood, ls='--', color='tab:orange', alpha=0.5, label='Raw (OOD)')
    l0 = results['sae_full']['mean_l0']
    ax.axhline(full_id, ls=':', color='tab:blue', alpha=0.5,
               label=f'SAE default L0={l0:.0f} (ID)')
    ax.axhline(full_ood, ls=':', color='tab:orange', alpha=0.5,
               label=f'SAE default L0={l0:.0f} (OOD)')

    ax.set_xlabel('Top-K features')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.set_title(f'Probe generalization: {args.model} layer {args.layer}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'probe_generalization.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot: {plot_path}")
    print(f"Results: {os.path.join(args.results_dir, 'results.pkl')}")


if __name__ == '__main__':
    main()
