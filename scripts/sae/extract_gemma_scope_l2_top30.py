#!/usr/bin/env python3
"""Extract top-30 Gemma Scope SAE latent directions at L=2.

Loads Gemma-2-9B + Gemma Scope L2 res-16k SAE, encodes the same 30 FineWeb
final-position activations that fig4_robustness_v2.py uses, picks the top 30
latents by mean activation, and saves their decoder directions as
v2_directions_sae_gemma_L2.pkl in results/fig4_robustness/.

Output format matches what fig4_robustness_v2.py:load_external_directions()
expects.

Usage: python scripts/sae/extract_gemma_scope_l2_top30.py
"""
import os, sys, pickle, time, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model
from src.data import load_fineweb_fixed_length

OUT_DIR = 'results/fig4_robustness'
N_ACT = 30
SEQ_LEN = 5
SEED = 42
N_TOP = 30


def ts(): return time.strftime('%H:%M:%S')


def get_acts_at_layer(model, tokenizer, device, layer, target,
                      n=N_ACT, seq_len=SEQ_LEN):
    cache_dir = f'/workspace/fineweb_cache_{target}'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n, tokenizer, seq_len=seq_len, seed=SEED, cache_dir=cache_dir)
    acts = []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        # final-position activation at layer L (post-block hidden state)
        acts.append(outs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
    return torch.stack(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=2)
    ap.add_argument('--width', default='16k', choices=['16k', '65k', '131k'])
    ap.add_argument('--target', default='gemma')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR,
                            f'v2_directions_sae_{args.target}_L{args.layer}.pkl')
    if os.path.exists(out_path):
        print(f"[{ts()}] {out_path} exists, exiting")
        return

    print(f"[{ts()}] loading gemma-2-9b")
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)

    print(f"[{ts()}] extracting {N_ACT} FineWeb activations at L{args.layer}")
    acts = get_acts_at_layer(model, tokenizer, device, args.layer, args.target)
    print(f"  acts shape: {tuple(acts.shape)}, mean norm: {acts.norm(dim=-1).mean():.2f}")

    print(f"[{ts()}] loading Gemma Scope SAE: layer_{args.layer}/width_{args.width}/canonical")
    from sae_lens import SAE
    sae, cfg, _ = SAE.from_pretrained(
        release='gemma-scope-9b-pt-res-canonical',
        sae_id=f'layer_{args.layer}/width_{args.width}/canonical',
        device='cuda',
    )
    print(f"  SAE d_in={sae.cfg.d_in}  d_sae={sae.cfg.d_sae}")

    acts_dev = acts.to(device=device, dtype=sae.W_enc.dtype)
    with torch.no_grad():
        feats = sae.encode(acts_dev)   # (N, d_sae)
    print(f"  feats shape: {tuple(feats.shape)}")

    mean_act = feats.float().mean(dim=0)   # (d_sae,)
    nonzero = (feats > 0).float().mean(dim=0)
    print(f"  active features (>0 in any of {N_ACT}): "
          f"{(nonzero > 0).sum().item()}")

    # Pick top N_TOP by mean activation; require they activate on at least 3 contexts
    # so we don't pick a feature that fires once huge and is otherwise dead
    eligible = (feats > 0).sum(dim=0) >= 3
    scores = mean_act.clone()
    scores[~eligible] = -1.0
    top_vals, top_idx = torch.topk(scores, N_TOP)
    print(f"[{ts()}] top {N_TOP} latents by mean activation:")
    for k in range(N_TOP):
        i = int(top_idx[k].item())
        print(f"  rank {k:2d}  feat={i:6d}  mean_act={float(mean_act[i]):.4f}  "
              f"n_fire={int((feats[:, i] > 0).sum().item())}/{N_ACT}")

    # Decoder directions: rows of W_dec
    W_dec = sae.W_dec.detach().cpu().float()   # (d_sae, d_in)
    directions = []
    for k in range(N_TOP):
        i = int(top_idx[k].item())
        d = W_dec[i]                           # (d_in,)
        d = F.normalize(d, dim=0)              # unit-norm
        directions.append(d)

    out = {
        'directions': directions,
        'feature_indices': [int(i.item()) for i in top_idx],
        'mean_activations': [float(mean_act[int(i.item())]) for i in top_idx],
        'layer': args.layer,
        'width': args.width,
        'model': 'gemma-2-9b',
        'sae_release': 'gemma-scope-9b-pt-res-canonical',
        'sae_id': f'layer_{args.layer}/width_{args.width}/canonical',
        'n_act': N_ACT,
        'seed': SEED,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"[{ts()}] saved {N_TOP} SAE directions → {out_path}")


if __name__ == '__main__':
    main()
