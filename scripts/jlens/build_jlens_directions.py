#!/usr/bin/env python3
"""Build J-Lens direction pairs for the superellipse experiment (Qwen3-8B, L20).

The J-Lens vector for token t at layer l is J_l^T @ w_t, where J_l is the
pre-fitted average Jacobian (neuronpedia/jacobian-lens) and w_t the unembedding
row of t. Final-LN scaling is omitted: it only affects norms and the spherical
sweep uses directions only.

Categories (per plan): EN×EN and ZH×ZH, 10 pairs each. Candidate tokens are
verified to be single tokens in the Qwen3 tokenizer; pairs are formed greedily
from a shuffled list, requiring |cosine| <= COS_MAX between the two J-Lens
vectors and using each token at most once.

Outputs: directions pkl + text report.
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model

LAYER = 20
SEED = 123
COS_MAX = 0.2
N_PAIRS = 10

# Concrete single-token-candidate content words. EN with leading space.
EN_CANDIDATES = [
    ' basketball', ' music', ' doctor', ' river', ' winter', ' gold',
    ' horse', ' church', ' science', ' coffee', ' fire', ' ocean',
    ' king', ' moon', ' garden', ' iron', ' hospital', ' teacher',
    ' island', ' snow', ' bridge', ' engine', ' forest', ' dragon',
    ' wine', ' salt', ' piano', ' soldier', ' mountain', ' poetry',
]
# Multi-character Chinese words (no leading space), incl. Neel's
# interpretative meta-tokens as candidates.
ZH_CANDIDATES = [
    '篮球', '音乐', '医生', '河流', '冬天', '黄金',
    '教堂', '科学', '咖啡', '火焰', '海洋', '国王',
    '月亮', '花园', '医院', '老师', '岛屿', '桥梁',
    '发动机', '森林', '葡萄酒', '钢琴', '士兵', '山脉',
    '什么意思', '是什么意思', '这句话', '是何含义',
]


def single_token_id(tokenizer, s):
    ids = tokenizer(s, add_special_tokens=False)['input_ids']
    return ids[0] if len(ids) == 1 else None


def make_pairs(names, vecs, rng, report):
    """Max-cardinality matching over edges with |cos| <= COS_MAX, preferring
    low |cos| among maximum matchings. Falls back gracefully if <N_PAIRS."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = float(vecs[i] @ vecs[j])
            if abs(c) <= COS_MAX:
                G.add_edge(i, j, weight=1.0 - abs(c), cos=c)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matched = sorted(matching, key=lambda e: abs(G.edges[e]['cos']))
    pairs = []
    for i, j in matched[:N_PAIRS]:
        c = G.edges[i, j]['cos']
        pairs.append({'tokens': (names[i], names[j]),
                      'vecs': (vecs[i], vecs[j]),
                      'cosine': c})
        report.append(f"  pair {len(pairs)}: {names[i]!r} x {names[j]!r}"
                      f"  cos={c:+.3f}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lens', default='results/jlens/qwen3-8b/qwen3-8b/'
                        'jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt')
    parser.add_argument('--out_dir', default='results/jlens/qwen3-8b')
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    report = []

    print("Loading lens...")
    lens = torch.load(args.lens, map_location='cpu', weights_only=False)
    J = lens['J'][LAYER].float()          # [d_model, d_model]

    print("Loading Qwen3-8B...")
    model, tokenizer, device = load_model('qwen3-8b', dtype=torch.bfloat16)
    W_U = model.get_output_embeddings().weight.detach().float().cpu()  # [vocab, d]

    def build(cands, label):
        names, vecs = [], []
        report.append(f"\n{label} candidates:")
        for s in cands:
            tid = single_token_id(tokenizer, s)
            if tid is None:
                report.append(f"  SKIP (multi-token): {s!r}")
                continue
            v = J.T @ W_U[tid]
            v = (v / v.norm()).numpy()
            names.append(s)
            vecs.append(v)
            report.append(f"  ok: {s!r} (id {tid})")
        report.append(f"{label}: {len(names)} single-token candidates")
        report.append(f"{label} pairs:")
        pairs = make_pairs(names, np.stack(vecs), rng, report)
        if len(pairs) < N_PAIRS:
            report.append(f"  WARNING: only {len(pairs)} pairs formed")
        return pairs

    en_pairs = build(EN_CANDIDATES, 'EN')
    zh_pairs = build(ZH_CANDIDATES, 'ZH')

    os.makedirs(args.out_dir, exist_ok=True)
    out_pkl = os.path.join(args.out_dir, f'jlens_direction_pairs_L{LAYER}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'layer': LAYER, 'en_pairs': en_pairs,
                     'zh_pairs': zh_pairs, 'cos_max': COS_MAX,
                     'lens_path': args.lens}, f)

    report_txt = '\n'.join(report)
    report_path = os.path.join(args.out_dir, f'jlens_direction_pairs_L{LAYER}.txt')
    with open(report_path, 'w') as f:
        f.write(report_txt)
    print(report_txt)
    print(f"\nSaved: {out_pkl}\nReport: {report_path}")


if __name__ == '__main__':
    main()
