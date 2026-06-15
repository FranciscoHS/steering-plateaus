#!/usr/bin/env python3
"""AxBench concept500 detection eval: our dictionary vs SAE vs contrastive.

For each of the 500 AxBench concepts (pyvene/axbench-concept500, default
subset 2b/l20 — the concept text is model-agnostic, we compute Qwen3-1.7B
layer-14 activations on it):

  1. Train set = 72 positives + 72 genre-matched shared negatives (144 ex).
  2. Score every direction d on every train example:
       s(example) = max over tokens of |cos(h - mu, d)|
     with h = resid post LAYER, mu = FineWeb corpus mean, pos 0 skipped.
  3. Select the direction with the best train AUROC (dictionary and SAE
     separately; SAE directions = unit-normalized decoder rows).
  4. Contrastive baseline: d_c = mean(pos example-mean acts) - mean(neg
     example-mean acts), unit-normalized (no selection needed).
  5. Evaluate all three on the concept's held-out test slice
     (~36 pos + 36 neg + ~2 hard neg; label positive=1, rest=0) -> AUROC.

Stages checkpoint to --out and are skipped on rerun if their output exists.

Usage (on pod):
    HF_HOME=/workspace/hf_cache python -u scripts/superellipse/axbench_detection.py \\
        --dict_path results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_align1.0_k100_40k/dictionary.pt \\
        --sae_ckpt results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt \\
        --out results/axbench_detection/qwen3-1.7b_L14_2b-l20
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
from src.data import load_fineweb_fixed_length


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--subset', default='2b/l20',
                    help='axbench-concept500 subset (source model/layer of '
                         'the concept list; text is model-agnostic)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_tokens', type=int, default=512)
    ap.add_argument('--dir_chunk', type=int, default=8192)
    ap.add_argument('--mu_seqs', type=int, default=2000)
    ap.add_argument('--mu_seqlen', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def load_axbench_split(subset, split):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('pyvene/axbench-concept500',
                           f'{subset}/{split}/data.parquet',
                           repo_type='dataset')
    return pd.read_parquet(path)


def compute_mu(model, tokenizer, layer, device, n_seqs, seq_len, seed):
    """Corpus mean over FineWeb token activations (pos 0 skipped)."""
    tok_list = load_fineweb_fixed_length(n_seqs, tokenizer, seq_len=seq_len,
                                         seed=seed)
    tok_ids = torch.stack(tok_list)
    total, count = None, 0
    for i in range(0, tok_ids.shape[0], 64):
        bt = tok_ids[i:i + 64].to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[layer + 1]
        h = h[:, 1:, :].float()                      # drop pos 0
        s = h.reshape(-1, h.shape[-1]).sum(0)
        total = s if total is None else total + s
        count += h.shape[0] * h.shape[1]
    return (total / count).cpu()                     # [d]


@torch.no_grad()
def score_pass(model, tokenizer, texts, layer, device, mu, D_chunks,
               batch_size, max_tokens, d_model, tag):
    """Forward `texts`; return ([N, K_all] fp16 max-|cos| scores,
    [N, d] fp32 example-mean activations). Length-sorted internally."""
    N = len(texts)
    K_all = sum(D.shape[0] for D in D_chunks)
    scores = torch.empty(N, K_all, dtype=torch.float16)
    meanacts = torch.empty(N, d_model, dtype=torch.float32)
    order = np.argsort([-len(t) for t in texts])     # long first: fail fast on OOM
    mu_dev = mu.to(device)
    t0 = time.time()
    n_batches = (N + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, N, batch_size)):
        idx = order[i:i + batch_size]
        enc = tokenizer([texts[j] for j in idx], return_tensors='pt',
                        padding=True, truncation=True, max_length=max_tokens,
                        add_special_tokens=True)
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        h = model(ids, attention_mask=mask,
                  output_hidden_states=True).hidden_states[layer + 1].float()
        valid = mask.bool()
        valid[:, 0] = False                          # pos 0 = attention sink
        # example-mean activation (uncentered; centering cancels in diff-of-means)
        vf = valid.unsqueeze(-1).float()
        meanacts[idx] = ((h * vf).sum(1) / vf.sum(1).clamp_min(1.0)).cpu()
        hc = h - mu_dev
        norms = hc.norm(dim=-1).clamp_min(1e-6)      # [B, T]
        outs = []
        for D in D_chunks:
            s = torch.einsum('btd,kd->btk', hc, D).abs() / norms.unsqueeze(-1)
            s = s.masked_fill(~valid.unsqueeze(-1), -1.0)
            outs.append(s.max(dim=1).values)         # [B, k]
        scores[idx] = torch.cat(outs, 1).half().cpu()
        if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
            el = time.time() - t0
            print(f'  [{tag}] batch {bi+1}/{n_batches}  '
                  f'({el:.0f}s, {el/(bi+1):.2f}s/batch)', flush=True)
    return scores, meanacts


def rank_auroc(scores, labels):
    """Vectorized Mann-Whitney AUROC. scores [n, D] fp32, labels [n] bool.
    No tie correction (scores are continuous; ties only from fp16 rounding)."""
    n = scores.shape[0]
    order = scores.argsort(dim=0)
    ranks = torch.empty_like(order)
    ar = torch.arange(n, device=scores.device).unsqueeze(1).expand_as(order)
    ranks.scatter_(0, order, ar)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    rank_sum_pos = ranks[labels].sum(0).double()
    return ((rank_sum_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).float()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- data ----------------------------------------------------------
    train_df = load_axbench_split(args.subset, 'train').reset_index(drop=True)
    test_df = load_axbench_split(args.subset, 'test').reset_index(drop=True)
    neg_pool = train_df[train_df.concept_id < 0]
    print(f'train {len(train_df)} rows, test {len(test_df)} rows, '
          f'{train_df.concept_id.max() + 1} concepts, '
          f'{len(neg_pool)} shared negatives', flush=True)
    print('shared negative genres:',
          neg_pool.concept_genre.value_counts().to_dict(), flush=True)
    concept_ids = sorted(train_df[train_df.concept_id >= 0].concept_id.unique())

    # ---- directions ------------------------------------------------------
    # snapshot the dictionary (the build may still be appending to it)
    snap_path = os.path.join(args.out, 'dictionary_snapshot.pt')
    if os.path.exists(snap_path):
        blob = torch.load(snap_path, map_location='cpu')
        print(f'Loaded existing dictionary snapshot ({snap_path})', flush=True)
    else:
        for attempt in range(3):
            try:
                blob = torch.load(args.dict_path, map_location='cpu')
                break
            except Exception as e:                   # mid-checkpoint write
                print(f'dict load failed ({e}), retrying', flush=True)
                time.sleep(10)
        torch.save(blob, snap_path)
    D_dict = blob['dictionary'].float()
    D_dict = D_dict / D_dict.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K_dict, d_model = D_dict.shape
    print(f'dictionary: {K_dict} atoms (n_done={blob.get("n_done")}, '
          f'K_dict={blob.get("K_dict")})', flush=True)

    sae = torch.load(args.sae_ckpt, map_location='cpu')
    D_sae = sae['W_dec'].float()
    if D_sae.shape[0] == d_model:                    # [d, d_sae] -> rows = latents
        D_sae = D_sae.T
    D_sae = D_sae / D_sae.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K_sae = D_sae.shape[0]
    print(f'SAE: {K_sae} latents', flush=True)

    D_all = torch.cat([D_dict, D_sae], 0)
    K_all = D_all.shape[0]
    D_chunks = [D_all[i:i + args.dir_chunk].to(device)
                for i in range(0, K_all, args.dir_chunk)]

    # ---- model + mu ------------------------------------------------------
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    mu_path = os.path.join(args.out, 'mu.pt')
    if os.path.exists(mu_path):
        mu = torch.load(mu_path)
    else:
        print('Computing FineWeb corpus mean', flush=True)
        mu = compute_mu(model, tokenizer, args.layer, device,
                        args.mu_seqs, args.mu_seqlen, args.seed)
        torch.save(mu, mu_path)
    print(f'mu norm: {mu.norm():.2f}', flush=True)

    # ---- stage 1: train pass --------------------------------------------
    tr_scores_path = os.path.join(args.out, 'train_scores.pt')
    if os.path.exists(tr_scores_path):
        ck = torch.load(tr_scores_path, map_location='cpu')
        tr_scores, tr_meanacts = ck['scores'], ck['meanacts']
        print('Loaded cached train scores', flush=True)
    else:
        print('Train pass', flush=True)
        tr_scores, tr_meanacts = score_pass(
            model, tokenizer, train_df.output.tolist(), args.layer, device,
            mu, D_chunks, args.batch_size, args.max_tokens, d_model, 'train')
        torch.save({'scores': tr_scores, 'meanacts': tr_meanacts},
                   tr_scores_path)
        train_df.to_parquet(os.path.join(args.out, 'train_df.parquet'))

    # ---- stage 2: per-concept selection + contrastive dirs ---------------
    sel_path = os.path.join(args.out, 'selection.pkl')
    contrast_path = os.path.join(args.out, 'contrastive_dirs.pt')
    if os.path.exists(sel_path) and os.path.exists(contrast_path):
        with open(sel_path, 'rb') as f:
            sel = pickle.load(f)
        D_contrast = torch.load(contrast_path)
        print('Loaded cached selection', flush=True)
    else:
        print('Selecting best directions per concept', flush=True)
        sel = {'concept_id': [], 'concept': [], 'genre': [],
               'best_dict': [], 'best_sae': [],
               'train_auroc_dict': [], 'train_auroc_sae': [],
               'auroc_train_all': np.zeros((len(concept_ids), K_all),
                                           dtype=np.float16)}
        contrast_dirs = torch.zeros(len(concept_ids), d_model)
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
            S = tr_scores[rows].float().to(device)
            auroc = rank_auroc(S, labels)            # [K_all]
            sel['auroc_train_all'][ci] = auroc.cpu().numpy().astype(np.float16)
            bd = int(auroc[:K_dict].argmax())
            bs = int(auroc[K_dict:].argmax())
            sel['concept_id'].append(int(cid))
            sel['concept'].append(train_df.loc[pos[0], 'output_concept'])
            sel['genre'].append(genre)
            sel['best_dict'].append(bd)
            sel['best_sae'].append(bs)
            sel['train_auroc_dict'].append(float(auroc[bd]))
            sel['train_auroc_sae'].append(float(auroc[K_dict + bs]))
            dc = tr_meanacts[pos.values].mean(0) - tr_meanacts[neg.values].mean(0)
            contrast_dirs[ci] = dc / dc.norm().clamp_min(1e-12)
            if (ci + 1) % 100 == 0:
                print(f'  {ci+1}/{len(concept_ids)} concepts', flush=True)
        with open(sel_path, 'wb') as f:
            pickle.dump(sel, f)
        torch.save(contrast_dirs, contrast_path)
        D_contrast = contrast_dirs

    # ---- stage 3: test pass on selected directions -----------------------
    # D_sel[c] = (dict-best, sae-best, contrastive) for concept c
    cid2row = {cid: i for i, cid in enumerate(sel['concept_id'])}
    D_sel = torch.zeros(len(concept_ids), 3, d_model)
    for i in range(len(concept_ids)):
        D_sel[i, 0] = D_dict[sel['best_dict'][i]]
        D_sel[i, 1] = D_sae[sel['best_sae'][i]]
        D_sel[i, 2] = D_contrast[i]
    D_sel_dev = D_sel.to(device)

    te_scores_path = os.path.join(args.out, 'test_scores.pt')
    if os.path.exists(te_scores_path):
        te_scores = torch.load(te_scores_path)['scores']
        print('Loaded cached test scores', flush=True)
    else:
        print('Test pass', flush=True)
        texts = test_df.output.tolist()
        cids = test_df.concept_id.values
        N = len(texts)
        te_scores = torch.empty(N, 3, dtype=torch.float32)
        order = np.argsort([-len(t) for t in texts])
        mu_dev = mu.to(device)
        t0 = time.time()
        n_batches = (N + args.batch_size - 1) // args.batch_size
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
                rows = torch.tensor([cid2row[c] for c in cids[idx]],
                                    device=device)
                Db = D_sel_dev[rows]                 # [B, 3, d]
                s = torch.einsum('btd,bkd->btk', hc, Db).abs() \
                    / norms.unsqueeze(-1)
                s = s.masked_fill(~valid.unsqueeze(-1), -1.0)
                te_scores[idx] = s.max(dim=1).values.cpu()
                if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
                    el = time.time() - t0
                    print(f'  [test] batch {bi+1}/{n_batches}  ({el:.0f}s)',
                          flush=True)
        torch.save({'scores': te_scores}, te_scores_path)
        test_df.to_parquet(os.path.join(args.out, 'test_df.parquet'))

    # ---- stage 4: per-concept test AUROC ---------------------------------
    rows_out = []
    for i, cid in enumerate(sel['concept_id']):
        m = test_df.concept_id == cid
        y = torch.tensor((test_df.category[m] == 'positive').values)
        S = te_scores[m.values].float()
        aur = rank_auroc(S, y)
        cats = test_df.category[m].value_counts()
        rows_out.append({
            'concept_id': cid, 'concept': sel['concept'][i],
            'genre': sel['genre'][i],
            'n_pos': int(cats.get('positive', 0)),
            'n_neg': int(cats.get('negative', 0)),
            'n_hardneg': int(cats.get('hard negative', 0)),
            'best_dict_idx': sel['best_dict'][i],
            'best_sae_idx': sel['best_sae'][i],
            'train_auroc_dict': sel['train_auroc_dict'][i],
            'train_auroc_sae': sel['train_auroc_sae'][i],
            'test_auroc_dict': float(aur[0]),
            'test_auroc_sae': float(aur[1]),
            'test_auroc_contrast': float(aur[2]),
        })
    res = pd.DataFrame(rows_out)
    res.to_csv(os.path.join(args.out, 'results.csv'), index=False)

    lines = [f'AxBench concept500 ({args.subset}) detection, '
             f'{args.model} L{args.layer}, {len(res)} concepts',
             f'dictionary: {K_dict} atoms ({args.dict_path})',
             f'SAE: {K_sae} latents ({args.sae_ckpt})']
    for col, name in [('test_auroc_dict', 'dictionary'),
                      ('test_auroc_sae', 'SAE'),
                      ('test_auroc_contrast', 'contrastive')]:
        lines.append(f'{name:12s} test AUROC  mean {res[col].mean():.4f}  '
                     f'median {res[col].median():.4f}  '
                     f'frac>0.9 {(res[col] > 0.9).mean():.3f}')
    summary = '\n'.join(lines)
    print(summary, flush=True)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f'Saved {os.path.join(args.out, "results.csv")}', flush=True)


if __name__ == '__main__':
    main()
