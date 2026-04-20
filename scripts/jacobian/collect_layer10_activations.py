"""Collect layer-10 last-token activations at seq_len=10 and save as a tensor
for downstream local-ID analysis. Matches the setup used in intrinsic_dim_layer_sweep.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_texts(n_texts, min_chars, seed=42):
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT', split='train', streaming=True)
    ds = ds.shuffle(seed=seed)
    texts = []
    for ex in ds:
        if len(ex['text']) >= min_chars:
            texts.append(ex['text'])
        if len(texts) >= n_texts:
            break
    return texts


def collect(model, tokenizer, device, texts, seq_len, layer):
    out_chunks = []
    batch_size = min(128, max(32, 4096 // seq_len))
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors='pt',
            padding='max_length', truncation=True, max_length=seq_len,
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        h = out.hidden_states[layer + 1][:, -1, :].float().cpu()
        out_chunks.append(h)
        if (i // batch_size) % 20 == 0:
            print(f"  batch {i // batch_size}: collected {sum(c.shape[0] for c in out_chunks)}")
    return torch.cat(out_chunks, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--n_texts', type=int, default=50000)
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--out', type=str,
                        default='results/jacobian/l10_last_token_seq10.pt')
    args = parser.parse_args()

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device

    min_chars = args.seq_len * 10
    print(f"Loading {args.n_texts} texts (min {min_chars} chars)...")
    texts = load_texts(args.n_texts, min_chars)
    print(f"  got {len(texts)} texts")

    print(f"\nCollecting layer {args.layer} last-token activations...")
    acts = collect(model, tokenizer, device, texts, args.seq_len, args.layer)
    print(f"  shape={tuple(acts.shape)}, dtype={acts.dtype}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(acts, args.out)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
