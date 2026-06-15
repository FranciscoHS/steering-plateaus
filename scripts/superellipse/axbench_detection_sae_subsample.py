#!/usr/bin/env python3
"""Size-matched SAE baseline for the AxBench detection eval.

Randomly subsample the SAE dictionary to n_sub latents (default 12800 =
dictionary size at snapshot), redo the per-concept best-direction selection
on the cached train scores from axbench_detection.py, then run a test pass
on the newly selected latents and report per-concept test AUROC.

Reuses from --base_out: train_scores.pt, train_df.parquet, test_df.parquet,
mu.pt, dictionary_snapshot.pt (for K_dict offset).

Usage (on pod):
    HF_HOME=/workspace/hf_cache python -u scripts/superellipse/axbench_detection_sae_subsample.py \\
        --base_out results/axbench_detection/qwen3-1.7b_L14_2b-l20 \\
        --sae_ckpt results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt \\
        --n_sub 12800 --seed 0
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch

from src.model import load_model
from scripts.superellipse.axbench_detection import rank_auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_out', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--n_sub', type=int, default=12800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_tokens', type=int, default=512)
    args = ap.parse_args()

    out = os.path.join(args.base_out, f'sae_sub{args.n_sub}_seed{args.seed}')
    os.makedirs(out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_parquet(os.path.join(args.base_out, 'train_df.parquet'))
    test_df = pd.read_parquet(os.path.join(args.base_out, 'test_df.parquet'))
    neg_pool = train_df[train_df.concept_id < 0]
    concept_ids = sorted(train_df[train_df.concept_id >= 0].concept_id.unique())

    blob = torch.load(os.path.join(args.base_out, 'dictionary_snapshot.pt'),
                      map_location='cpu')
    K_dict = blob['dictionary'].shape[0]
    d_model = blob['dictionary'].shape[1]
    del blob

    sae = torch.load(args.sae_ckpt, map_location='cpu')
    D_sae = sae['W_dec'].float()
    if D_sae.shape[0] == d_model:
        D_sae = D_sae.T
    D_sae = D_sae / D_sae.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K_sae = D_sae.shape[0]

    g = torch.Generator().manual_seed(args.seed)
    sub_idx = torch.randperm(K_sae, generator=g)[:args.n_sub]
    print(f'Subsampled {args.n_sub}/{K_sae} SAE latents (seed {args.seed})',
          flush=True)

    print('Loading cached train scores', flush=True)
    ck = torch.load(os.path.join(args.base_out, 'train_scores.pt'),
                    map_location='cpu')
    tr_scores = ck['scores']                         # [N, K_dict + K_sae] fp16
    del ck

    # ---- selection on the subsampled latents ------------------------------
    sel = {'concept_id': [], 'concept': [], 'genre': [],
           'best_sub': [], 'train_auroc_sub': []}
    for ci, cid in enumerate(concept_ids):
        pos = train_df.index[(train_df.concept_id == cid)
                             & (train_df.category == 'positive')]
        genre = train_df.loc[pos, 'concept_genre'].mode().iloc[0]
        neg = neg_pool.index[neg_pool.concept_genre == genre]
        if len(neg) == 0:
            neg = neg_pool.index
        rows = np.concatenate([pos.values, neg.values])
        labels = torch.zeros(len(rows), dtype=torch.bool, device=device)
        labels[:len(pos)] = True
        S = tr_scores[rows][:, K_dict + sub_idx].float().to(device)
        auroc = rank_auroc(S, labels)                # [n_sub]
        b = int(auroc.argmax())
        sel['concept_id'].append(int(cid))
        sel['concept'].append(train_df.loc[pos[0], 'output_concept'])
        sel['genre'].append(genre)
        sel['best_sub'].append(int(sub_idx[b]))      # index into full SAE
        sel['train_auroc_sub'].append(float(auroc[b]))
        if (ci + 1) % 100 == 0:
            print(f'  {ci+1}/{len(concept_ids)} concepts', flush=True)
    del tr_scores
    with open(os.path.join(out, 'selection_sub.pkl'), 'wb') as f:
        pickle.dump({**sel, 'sub_idx': sub_idx.numpy()}, f)

    # ---- test pass on the selected latents --------------------------------
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    mu = torch.load(os.path.join(args.base_out, 'mu.pt'))

    cid2row = {cid: i for i, cid in enumerate(sel['concept_id'])}
    D_sel = torch.stack([D_sae[sel['best_sub'][i]]
                         for i in range(len(concept_ids))]).to(device)

    texts = test_df.output.tolist()
    cids = test_df.concept_id.values
    N = len(texts)
    te_scores = torch.empty(N, dtype=torch.float32)
    order = np.argsort([-len(t) for t in texts])
    mu_dev = mu.to(device)
    t0 = time.time()
    n_batches = (N + args.batch_size - 1) // args.batch_size
    print('Test pass', flush=True)
    with torch.no_grad():
        for bi, i in enumerate(range(0, N, args.batch_size)):
            idx = order[i:i + args.batch_size]
            enc = tokenizer([texts[j] for j in idx], return_tensors='pt',
                            padding=True, truncation=True,
                            max_length=args.max_tokens,
                            add_special_tokens=True)
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            h = model(ids, attention_mask=mask, output_hidden_states=True
                      ).hidden_states[args.layer + 1].float()
            valid = mask.bool()
            valid[:, 0] = False
            hc = h - mu_dev
            norms = hc.norm(dim=-1).clamp_min(1e-6)
            rows = torch.tensor([cid2row[c] for c in cids[idx]], device=device)
            Db = D_sel[rows]                         # [B, d]
            s = torch.einsum('btd,bd->bt', hc, Db).abs() / norms
            s = s.masked_fill(~valid, -1.0)
            te_scores[idx] = s.max(dim=1).values.cpu()
            if (bi + 1) % 200 == 0 or bi + 1 == n_batches:
                print(f'  [test] batch {bi+1}/{n_batches}  '
                      f'({time.time()-t0:.0f}s)', flush=True)
    torch.save({'scores': te_scores}, os.path.join(out, 'test_scores_sub.pt'))

    # ---- per-concept test AUROC -------------------------------------------
    rows_out = []
    for i, cid in enumerate(sel['concept_id']):
        m = test_df.concept_id == cid
        y = torch.tensor((test_df.category[m] == 'positive').values)
        S = te_scores[m.values].float().unsqueeze(1)
        aur = rank_auroc(S, y)
        rows_out.append({
            'concept_id': cid, 'concept': sel['concept'][i],
            'genre': sel['genre'][i],
            'best_sub_idx': sel['best_sub'][i],
            'train_auroc_sub': sel['train_auroc_sub'][i],
            'test_auroc_sae_sub': float(aur[0]),
        })
    res = pd.DataFrame(rows_out)
    res.to_csv(os.path.join(out, 'results_sub.csv'), index=False)

    col = res.test_auroc_sae_sub
    summary = (f'SAE random subsample {args.n_sub}/{K_sae} (seed {args.seed}), '
               f'{len(res)} concepts\n'
               f'test AUROC  mean {col.mean():.4f}  median {col.median():.4f}  '
               f'frac>0.9 {(col > 0.9).mean():.3f}  ==1.0 {(col == 1.0).sum()}')
    print(summary, flush=True)
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')


if __name__ == '__main__':
    main()
