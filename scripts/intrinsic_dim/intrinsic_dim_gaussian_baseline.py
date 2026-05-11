"""Compare TWO-NN intrinsic dimensionality: real activations vs cov-matched Gaussian.

Collects activations from Qwen3-1.7B layer 2, computes covariance, samples
Gaussian points with matching mean/cov, and runs TWO-NN on both.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import skdim


def collect_activations(model, tokenizer, device, texts, layer, seq_len):
    acts = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        tokens = tokenizer(
            batch, return_tensors='pt', padding='max_length',
            truncation=True, max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        acts.append(out.hidden_states[layer + 1][:, -1, :].float().cpu())
    return torch.cat(acts, dim=0).numpy()


def main():
    layer = 2
    n_samples = 5000
    seq_len = 10
    n_gauss_seeds = 10

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.float16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    print("Loading texts...")
    ds = load_dataset("HuggingFaceFW/FineWeb-Edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=42)
    texts = []
    for ex in ds:
        if len(ex["text"]) >= seq_len * 3:
            texts.append(ex["text"])
        if len(texts) >= n_samples:
            break
    print(f"Collected {len(texts)} texts")

    print("Extracting activations...")
    X_real = collect_activations(model, tokenizer, device, texts, layer, seq_len)
    print(f"Shape: {X_real.shape}")

    # Real activations
    est = skdim.id.TwoNN()
    est.fit(X_real)
    print(f"\nReal activations: ID = {est.dimension_:.2f}")

    # Covariance stats
    mean = X_real.mean(axis=0)
    cov = np.cov(X_real, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total
    rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"Covariance rank (95% var): {rank_95}, (99% var): {rank_99}")

    # Cov-matched Gaussian
    print(f"\nGaussian baseline ({n_gauss_seeds} seeds):")
    gauss_ids = []
    for s in range(n_gauss_seeds):
        X_g = np.random.RandomState(s).multivariate_normal(mean, cov, size=n_samples)
        e = skdim.id.TwoNN()
        e.fit(X_g)
        gauss_ids.append(e.dimension_)
        print(f"  seed {s}: ID = {e.dimension_:.2f}")
    arr = np.array(gauss_ids)
    print(f"Gaussian mean = {arr.mean():.2f}, std = {arr.std():.2f}")


if __name__ == "__main__":
    main()
