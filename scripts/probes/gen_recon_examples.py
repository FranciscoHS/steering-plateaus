#!/usr/bin/env python3
"""Example generations: natural vs SAE-reconstructed vs AE-reconstructed.

For each prompt (first `seq_len` tokens of a wiki_fr text):
  - Run natural generation (no hook) — baseline
  - Run generation with SAE reconstruction swapped in at last prompt position, layer L
  - Run generation with AE reconstruction swapped in at last prompt position, layer L

The reconstruction is applied exactly once (step 0, last position); all subsequent
generation steps run the model unmodified, using KV cache from the modified hidden
state.

Greedy decoding for reproducibility.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_sources import SOURCE_LOADERS
from scripts.probes.run_probe_sae_vs_ae import load_sae, load_ae


def _get_blocks(model):
    return model.model.layers


@torch.no_grad()
def generate(model, prompt_ids, max_new, recon_fn, layer, eos_id):
    blocks = _get_blocks(model)
    generated = prompt_ids.clone()
    past = None

    for step in range(max_new):
        inp = generated if step == 0 else generated[:, -1:]
        handle = None
        if step == 0 and recon_fn is not None:
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                recon = recon_fn(hidden[:, -1, :].float()).to(hidden.dtype)
                hidden[:, -1, :] = recon
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            handle = blocks[layer].register_forward_hook(hook)

        out = model(inp, past_key_values=past, use_cache=True)
        if handle is not None:
            handle.remove()
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--max_new', type=int, default=40)
    parser.add_argument('--n_examples', type=int, default=5)
    parser.add_argument('--source', default='wiki_fr', choices=list(SOURCE_LOADERS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--sae_weights',
                        default='results/trained_sae/l10_doubled_k32_10M/sae_weights.pt')
    parser.add_argument('--ae_weights',
                        default='results/trained_autoencoder/l10_scaling/l10_10M_drop_model.pt')
    parser.add_argument('--outdir', default='results/recon_ood')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                 'float32': torch.float32}

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"Loading SAE: {args.sae_weights}")
    sae = load_sae(args.sae_weights, device).eval()
    print(f"Loading AE:  {args.ae_weights}")
    ae = load_ae(args.ae_weights, device).eval()

    def sae_fn(x):
        # TopKSAE.forward returns (recon, top_vals, top_idx); take recon.
        return sae(x.to(next(sae.parameters()).dtype))[0].float()

    def ae_fn(x):
        # FlexAutoencoder.forward returns (recon, z); take recon.
        return ae(x.to(next(ae.parameters()).dtype))[0].float()

    min_chars = args.seq_len * 10
    print(f"Loading {args.n_examples + 3} {args.source} texts...")
    texts = SOURCE_LOADERS[args.source](args.n_examples + 3, min_chars, args.seed)

    results = {
        'model': args.model, 'layer': args.layer, 'seq_len': args.seq_len,
        'max_new': args.max_new, 'source': args.source,
        'sae_weights': args.sae_weights, 'ae_weights': args.ae_weights,
        'examples': [],
    }

    eos = tokenizer.eos_token_id
    print(f"\n{'='*80}")
    for i, text in enumerate(texts[:args.n_examples]):
        ids = tokenizer(text, truncation=True, max_length=args.seq_len,
                        return_tensors='pt').input_ids.to(device)
        # Skip if shorter than seq_len (should be rare given min_chars filter)
        if ids.shape[1] < args.seq_len:
            print(f"[{i}] skipping: only {ids.shape[1]} tokens")
            continue

        prompt_txt = tokenizer.decode(ids[0])
        print(f"\n[{i}] PROMPT: {prompt_txt!r}")

        gen_nat = generate(model, ids, args.max_new, None, args.layer, eos)
        gen_sae = generate(model, ids, args.max_new, sae_fn, args.layer, eos)
        gen_ae = generate(model, ids, args.max_new, ae_fn, args.layer, eos)

        nat_cont = tokenizer.decode(gen_nat[0, args.seq_len:])
        sae_cont = tokenizer.decode(gen_sae[0, args.seq_len:])
        ae_cont = tokenizer.decode(gen_ae[0, args.seq_len:])

        print(f"    NATURAL:  {nat_cont!r}")
        print(f"    SAE:      {sae_cont!r}")
        print(f"    AE:       {ae_cont!r}")

        results['examples'].append({
            'prompt': prompt_txt,
            'natural': nat_cont,
            'sae': sae_cont,
            'ae': ae_cont,
        })

    outpath = os.path.join(args.outdir, f'gen_examples_{args.source}_layer{args.layer}.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
