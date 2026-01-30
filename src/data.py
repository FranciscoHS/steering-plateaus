"""
Data loading and sampling utilities for activation plateau analysis.

Provides FineWeb loading and covariance-matched random activation sampling.
"""

import torch
import numpy as np
from datasets import load_dataset


def load_fineweb_samples(n, tokenizer, min_tokens=10, max_tokens=512, seed=42):
    """
    Load and tokenize samples from FineWeb.

    Args:
        n: Number of samples to return
        tokenizer: GPT-2 tokenizer
        min_tokens: Minimum token length (filter shorter texts)
        max_tokens: Maximum token length (truncate longer texts)
        seed: Random seed for shuffling

    Returns:
        texts: list of n strings
        token_counts: list of token counts per text
    """
    # Load FineWeb-Edu sample (smaller, cleaner subset)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    # Shuffle with seed
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    texts = []
    token_counts = []

    for sample in dataset:
        text = sample['text']

        # Tokenize to check length
        tokens = tokenizer(text, truncation=True, max_length=max_tokens)
        n_tokens = len(tokens['input_ids'])

        if n_tokens >= min_tokens:
            texts.append(text)
            token_counts.append(n_tokens)

        if len(texts) >= n:
            break

    print(f"Loaded {len(texts)} FineWeb samples")
    print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={np.mean(token_counts):.1f}")

    return texts, token_counts


def sample_random_activations(real_activations, n, seed=42):
    """
    Sample random activations from a Gaussian fitted to real activations.

    Fits a multivariate Gaussian with full covariance to the real activations,
    then samples n vectors from it. Requires len(real_activations) > hidden_dim
    for full-rank covariance.

    Args:
        real_activations: [N, hidden_dim] tensor of real activations (N > d)
        n: Number of random samples to generate
        seed: Random seed

    Returns:
        random_activations: [n, hidden_dim] tensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(real_activations, torch.Tensor):
        real_activations = real_activations.numpy()

    N, d = real_activations.shape

    if N <= d:
        raise ValueError(
            f"Need more samples than dimensions for full-rank covariance. "
            f"Got {N} samples, {d} dimensions."
        )

    # Fit Gaussian
    mean = np.mean(real_activations, axis=0)
    cov = np.cov(real_activations, rowvar=False)

    # Check rank
    rank = np.linalg.matrix_rank(cov)
    print(f"Covariance matrix: shape={cov.shape}, rank={rank}")

    if rank < d:
        print(f"Warning: covariance is rank-deficient ({rank} < {d})")

    # Sample from multivariate Gaussian
    random_np = np.random.multivariate_normal(mean, cov, size=n)
    random_activations = torch.tensor(random_np, dtype=torch.float32)

    print(f"Sampled {n} random activations from fitted Gaussian")

    return random_activations
