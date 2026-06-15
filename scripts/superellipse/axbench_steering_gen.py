#!/usr/bin/env python3
"""AxBench-style steering generations: dictionary vs SAE selected directions.

For a random subset of concept500 concepts, steer Qwen3-1.7B (chat mode,
thinking off) with the detection-selected direction of each method
(`best_dict_idx` / `best_sae_idx` from axbench_detection.py results.csv),
sign-corrected via the concept's contrastive vector, by adding
factor * d to resid post LAYER at every position during generation.

Generates responses to a fixed set of generic instructions over a factor
sweep and appends everything to <out>/generations.jsonl (checkpointed per
batch; rerun skips completed (concept, method, factor) units).

Usage (calibration, on pod):
    HF_HOME=/workspace/hf_cache python -u scripts/superellipse/axbench_steering_gen.py \\
        --base_out results/axbench_detection/qwen3-1.7b_L14_2b-l20 \\
        --out results/axbench_steering/calib \\
        --n_concepts 3 --n_instructions 3 --factors 4 8 16 32 64 128
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch

from src.model import load_model, _get_blocks


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_out', required=True,
                    help='axbench_detection output dir (results.csv, '
                         'contrastive_dirs.pt, dictionary_snapshot.pt, '
                         'train_df.parquet)')
    ap.add_argument('--sae_ckpt',
                    default='results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_concepts', type=int, default=50)
    ap.add_argument('--concept_seed', type=int, default=0)
    ap.add_argument('--n_instructions', type=int, default=10)
    ap.add_argument('--instr_seed', type=int, default=1)
    ap.add_argument('--factors', type=float, nargs='+',
                    default=[4, 8, 16, 32, 64],
                    help='absolute steering magnitudes (ignored if '
                         '--factor_norm_fracs is given)')
    ap.add_argument('--factor_norm_fracs', type=float, nargs='+', default=None,
                    help='steering magnitudes as fractions of the mean '
                         'activation norm (from --act_norm_path)')
    ap.add_argument('--act_norm_path',
                    default='results/dictionary/qwen3-1.7b_L14_K8000/held_acts.pt',
                    help='held activations used to compute the mean norm')
    ap.add_argument('--methods', nargs='+', default=['dict', 'sae'])
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.8)
    ap.add_argument('--batch_pairs', type=int, default=4,
                    help='(concept, method) pairs per generation batch')
    ap.add_argument('--gen_seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    if args.factor_norm_fracs is not None:
        held = torch.load(args.act_norm_path, map_location='cpu')
        acts = held['activations'] if isinstance(held, dict) else held
        act_norm = acts.float().norm(dim=-1).mean().item()
        args.factors = [round(f * act_norm, 1) for f in args.factor_norm_fracs]
        print(f'mean act norm {act_norm:.2f} -> factors {args.factors} '
              f'(fracs {args.factor_norm_fracs})', flush=True)

    res = pd.read_csv(os.path.join(args.base_out, 'results.csv'))
    res = res.sort_values('concept_id').reset_index(drop=True)
    D_contrast = torch.load(os.path.join(args.base_out, 'contrastive_dirs.pt'),
                            map_location='cpu')

    blob = torch.load(os.path.join(args.base_out, 'dictionary_snapshot.pt'),
                      map_location='cpu')
    D_dict = blob['dictionary'].float()
    D_dict = D_dict / D_dict.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    d_model = D_dict.shape[1]
    sae = torch.load(args.sae_ckpt, map_location='cpu')
    D_sae = sae['W_dec'].float()
    if D_sae.shape[0] == d_model:
        D_sae = D_sae.T
    D_sae = D_sae / D_sae.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- concept subset ----------------------------------------------------
    rng = np.random.default_rng(args.concept_seed)
    chosen = sorted(rng.choice(res.concept_id.values, size=args.n_concepts,
                               replace=False).tolist())
    print(f'{len(chosen)} concepts: {chosen}', flush=True)

    # ---- per (concept, method) signed steering vector ----------------------
    vecs = {}   # (cid, method) -> unit vector (torch [d])
    meta_rows = []
    for cid in chosen:
        row = res[res.concept_id == cid].iloc[0]
        ci = int(res.index[res.concept_id == cid][0])
        dc = D_contrast[ci]
        for m in args.methods:
            d = D_dict[int(row.best_dict_idx)] if m == 'dict' \
                else D_sae[int(row.best_sae_idx)]
            sign = 1.0 if float(d @ dc) >= 0 else -1.0
            vecs[(cid, m)] = sign * d
            meta_rows.append({'concept_id': cid, 'concept': row.concept,
                              'method': m, 'sign': sign,
                              'idx': int(row.best_dict_idx if m == 'dict'
                                         else row.best_sae_idx)})

    # ---- instructions -------------------------------------------------------
    train_df = pd.read_parquet(os.path.join(args.base_out, 'train_df.parquet'))
    pool = train_df[(train_df.concept_id < 0)
                    & (train_df.concept_genre == 'text')].input.unique()
    rng_i = np.random.default_rng(args.instr_seed)
    instructions = rng_i.choice(pool, size=args.n_instructions,
                                replace=False).tolist()
    print('instructions:', flush=True)
    for s in instructions:
        print('  -', s[:100], flush=True)

    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'concepts': chosen, 'instructions': instructions,
                   'factors': args.factors, 'methods': args.methods,
                   'selection': meta_rows, 'args': vars(args)}, f, indent=2)

    # ---- resume support ------------------------------------------------------
    out_jsonl = os.path.join(args.out, 'generations.jsonl')
    done = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                done.add((r['concept_id'], r['method'], r['factor']))
        print(f'resuming: {len(done)} units already done', flush=True)

    # ---- model ---------------------------------------------------------------
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    blocks = _get_blocks(model)

    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': instr}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for instr in instructions]
    n_i = len(prompts)

    # steering hook: per-sequence vector added to resid post LAYER
    state = {'vecs': None}     # [B, d] on device, rows repeat per instruction

    def hook(module, inputs, output):
        if state['vecs'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + state['vecs'][:, None, :].to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    handle = blocks[args.layer].register_forward_hook(hook)

    units = [(cid, m, f) for cid in chosen for m in args.methods
             for f in args.factors]
    units = [u for u in units if u not in done]
    print(f'{len(units)} units to generate '
          f'({len(units) * n_i} sequences)', flush=True)

    torch.manual_seed(args.gen_seed)
    import time
    t0 = time.time()
    for bi in range(0, len(units), args.batch_pairs):
        batch_units = units[bi:bi + args.batch_pairs]
        texts = prompts * len(batch_units)
        vec_rows = torch.stack([
            f * vecs[(cid, m)]
            for (cid, m, f) in batch_units for _ in range(n_i)])
        enc = tokenizer(texts, return_tensors='pt', padding=True,
                        add_special_tokens=False)
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        state['vecs'] = vec_rows.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id)
        state['vecs'] = None
        gen = out[:, ids.shape[1]:]
        texts_out = tokenizer.batch_decode(gen, skip_special_tokens=True)
        with open(out_jsonl, 'a') as f:
            k = 0
            for (cid, m, fac) in batch_units:
                for ii in range(n_i):
                    f.write(json.dumps({
                        'concept_id': int(cid), 'method': m,
                        'factor': float(fac), 'instruction_idx': ii,
                        'generation': texts_out[k]}) + '\n')
                    k += 1
        nb = (len(units) + args.batch_pairs - 1) // args.batch_pairs
        print(f'  batch {bi // args.batch_pairs + 1}/{nb}  '
              f'({time.time() - t0:.0f}s)', flush=True)

    handle.remove()
    print(f'Saved {out_jsonl}', flush=True)


if __name__ == '__main__':
    main()
