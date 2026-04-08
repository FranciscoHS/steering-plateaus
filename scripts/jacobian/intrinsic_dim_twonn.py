"""Estimate intrinsic dimensionality of LM activations using TWO-NN (Facco et al., 2017).

Samples last-token activations from random FineWeb sequences at a given layer,
then runs skdim.id.TwoNN to estimate the intrinsic dimension.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import skdim


def collect_activations(model, tokenizer, device, texts, layer, seq_len):
    """Tokenize texts, truncate/pad to seq_len, extract last-token activations at given layer."""
    acts = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        # hidden_states[0] = embeddings, hidden_states[k] = after layer k-1
        h = out.hidden_states[layer + 1]  # (batch, seq_len, d_model)
        acts.append(h[:, -1, :].float().cpu())
    return torch.cat(acts, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    load_dtype = dtype_map[args.dtype]

    print(f"Loading model {args.model} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=load_dtype, device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"Loading FineWeb (streaming)...")
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT', split='train', streaming=True)
    ds = ds.shuffle(seed=args.seed)

    # Collect texts, filtering out very short ones
    min_chars = args.seq_len * 3  # rough heuristic: need enough chars to fill seq_len tokens
    texts = []
    for example in ds:
        text = example['text']
        if len(text) >= min_chars:
            texts.append(text)
        if len(texts) >= args.n_samples:
            break
    print(f"Collected {len(texts)} texts")

    print(f"Extracting layer {args.layer} last-token activations...")
    X = collect_activations(model, tokenizer, device, texts, args.layer, args.seq_len)
    print(f"Activation matrix shape: {X.shape}")

    # Run TWO-NN
    X_np = X.numpy()
    print("Running TWO-NN...")
    est = skdim.id.TwoNN()
    est.fit(X_np)
    print(f"\nResult: intrinsic dimension = {est.dimension_:.2f}")
    print(f"  model: {args.model}")
    print(f"  layer: {args.layer}")
    print(f"  n_samples: {X.shape[0]}")
    print(f"  ambient_dim: {X.shape[1]}")


if __name__ == '__main__':
    main()
