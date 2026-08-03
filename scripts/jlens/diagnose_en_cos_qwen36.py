#!/usr/bin/env python3
"""Diagnose why EN J-Lens pairs failed to form on Qwen3.6-27B L36:
compute the full pairwise cosine matrix of the 30 EN candidate J-Lens
vectors (and ZH for comparison) and report distribution stats.

Loads only the lm_head rows needed from the safetensors shard (no model
load). Writes results/jlens/qwen3.6-27b/en_cos_diagnostic.txt.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from scripts.jlens.build_jlens_directions_qwen36 import (
    EN_CANDIDATES, ZH_CANDIDATES, LAYER, single_token_id)

OUT = 'results/jlens/qwen3.6-27b/en_cos_diagnostic.txt'


def find_snapshot(hf_home):
    pats = glob.glob(os.path.join(
        hf_home, 'hub/models--Qwen--Qwen3.6-27B/snapshots/*'))
    assert pats, "model snapshot not found"
    return pats[0]


def load_rows(snap, ids):
    with open(os.path.join(snap, 'model.safetensors.index.json')) as f:
        index = json.load(f)['weight_map']
    shard = index['lm_head.weight']
    with safe_open(os.path.join(snap, shard), framework='pt') as f:
        W = f.get_slice('lm_head.weight')
        rows = torch.stack([W[i] for i in ids]).float()
    return rows


def stats(vecs, label, lines):
    V = vecs / vecs.norm(dim=-1, keepdim=True)
    C = (V @ V.T).numpy()
    iu = np.triu_indices(len(V), k=1)
    a = np.abs(C[iu])
    lines.append(f"{label}: n={len(V)}, pairwise |cos|: "
                 f"min={a.min():.3f} p25={np.percentile(a, 25):.3f} "
                 f"median={np.median(a):.3f} p75={np.percentile(a, 75):.3f} "
                 f"max={a.max():.3f}; frac<=0.2: {(a <= 0.2).mean():.3f}, "
                 f"frac<=0.3: {(a <= 0.3).mean():.3f}, "
                 f"frac<=0.4: {(a <= 0.4).mean():.3f}")
    # mean-vector component: how much do vectors share one direction?
    m = V.mean(0)
    lines.append(f"  ||mean_vec||={m.norm():.3f} "
                 f"(1.0 = identical, ~1/sqrt(n) = random)")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lens', default='/home/user/lens/qwen3.6-27b/jlens/'
                    'Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt')
    ap.add_argument('--hf_home', default=os.environ.get('HF_HOME',
                    '/home/user/hf_local'))
    args = ap.parse_args()

    lines = [f"EN/ZH J-Lens cosine diagnostic — qwen3.6-27b L{LAYER}"]
    snap = find_snapshot(args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(snap)
    lens = torch.load(args.lens, map_location='cpu', weights_only=False)
    J = lens['J'][LAYER].float()

    for cands, label in [(EN_CANDIDATES, 'EN'), (ZH_CANDIDATES, 'ZH')]:
        ids, names = [], []
        for s in cands:
            tid = single_token_id(tokenizer, s)
            if tid is not None:
                ids.append(tid)
                names.append(s)
        rows = load_rows(snap, ids)
        vecs = (J.T @ rows.T).T
        stats(vecs, label, lines)
        # raw unembedding cosines for comparison (is it J or W_U?)
        stats(rows, f"{label} (raw W_U rows, no J)", lines)

    txt = '\n'.join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(txt)
    print(txt)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
