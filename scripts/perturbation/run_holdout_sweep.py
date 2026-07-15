#!/usr/bin/env python3
"""Concept-matched holdout sweep (BATCHED): perturb each concept's OWN holdout
activations along contrastive (matched) / random / high-p directions.

All directions x all angles for one activation go through the upper layers in a few
chunked forward passes (not one-forward-per-angle), so the sweep is ~100x faster.

Output (per metric in {l2, kl}):
    angles_deg          : [A]
    concepts            : [N]  concept tag per activation
    <m>_random          : [N, n_random, A]
    <m>_highp           : [N, n_atoms, A]
    <m>_contrast_matched: [N, A]
"""
import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F

from src.model import (load_model, _get_blocks, _get_final_norm, _get_lm_head,
                       forward_from_layer_to_final, forward_from_layer_to_layer)


def unit_rows(T):
    T = T.float()
    return T / T.norm(dim=-1, keepdim=True)


@torch.no_grad()
def batched_sweep(model, context, a, dirs_raw, angles_rad, layer, device,
                  measure_layer=None, chunk=512):
    """Returns kl [K, A], l2 [K, A] for K directions over A angles, one activation.

    dirs_raw: [K, d] arbitrary (orthogonalized to a and unit-normalized here)."""
    dtype = next(model.parameters()).dtype
    A = len(angles_rad)
    d = a.shape[0]
    ctx = context.unsqueeze(0).to(device=device, dtype=dtype)        # [1, seq-1, d]
    a = a.to(device=device, dtype=torch.float32)
    R = a.norm(); a_hat = a / R
    dirs = dirs_raw.to(device=device, dtype=torch.float32)           # [K, d]
    # component orthogonal to a, unit
    vp = dirs - (dirs @ a_hat).unsqueeze(1) * a_hat.unsqueeze(0)
    vp = vp / (vp.norm(dim=1, keepdim=True) + 1e-10)                  # [K, d]
    K = vp.shape[0]

    cos = torch.tensor(np.cos(angles_rad), device=device, dtype=torch.float32)
    sin = torch.tensor(np.sin(angles_rad), device=device, dtype=torch.float32)
    # perturbed points: [K, A, d] = R*(cos_t a_hat + sin_t vp_k)
    pts = R * (cos.view(1, A, 1) * a_hat.view(1, 1, d) + sin.view(1, A, 1) * vp.view(K, 1, d))
    flat = pts.reshape(K * A, 1, d)

    # reference (unperturbed)
    ref_final = forward_from_layer_to_final(model, ctx, a.view(1, 1, d).to(dtype), layer)
    ref_measure = (ref_final if measure_layer is None
                   else forward_from_layer_to_layer(model, ctx, a.view(1, 1, d).to(dtype), layer, measure_layer)).float()
    ref_logits = _get_lm_head(model)(_get_final_norm(model)(ref_final.unsqueeze(1)).squeeze(1)).float()
    ref_logp = F.log_softmax(ref_logits, dim=-1)      # [1, V]
    ref_p = ref_logp.exp()

    l2 = torch.empty(K * A, device=device)
    kl = torch.empty(K * A, device=device)
    for s in range(0, K * A, chunk):
        pc = flat[s:s + chunk].to(dtype)
        pf = forward_from_layer_to_final(model, ctx, pc, layer)
        meas = (pf if measure_layer is None
                else forward_from_layer_to_layer(model, ctx, pc, layer, measure_layer)).float()
        l2[s:s + chunk] = (meas - ref_measure).norm(dim=-1)
        logp = F.log_softmax(_get_lm_head(model)(_get_final_norm(model)(pf.unsqueeze(1)).squeeze(1)).float(), dim=-1)
        kl[s:s + chunk] = (ref_p * (ref_logp - logp)).sum(-1)
    return kl.view(K, A).cpu().numpy(), l2.view(K, A).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--prompts_pkl', required=True)
    ap.add_argument('--scaffold_pkl', required=True)
    ap.add_argument('--highp_pkl', default=None,
                    help='soft_p dictionary.pt; omit to run contrastive+random only')
    ap.add_argument('--n_per_concept', type=int, default=12)
    ap.add_argument('--n_random_dirs', type=int, default=10)
    ap.add_argument('--max_angle', type=float, default=90.0)
    ap.add_argument('--n_angles', type=int, default=60)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--dtype', default=None, help='e.g. bfloat16 (needed for big models)')
    ap.add_argument('--results_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.results_dir is None:
        args.results_dir = f'results/holdout_sweep/{args.model}_L{args.layer}'
    os.makedirs(args.results_dir, exist_ok=True)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    A = len(angles_deg)

    print(f"Loading model: {args.model}", flush=True)
    dtype = getattr(torch, args.dtype) if args.dtype else None
    model, tokenizer, device = load_model(args.model, dtype=dtype)
    if args.measure_layer is not None and args.measure_layer < 0:
        args.measure_layer = len(_get_blocks(model)) + args.measure_layer

    scaf = torch.load(args.scaffold_pkl, map_location='cpu', weights_only=False)
    cdir = {n: unit_rows(scaf['dictionary'][i]) for i, n in enumerate(scaf['names'])}
    if args.highp_pkl:
        D = unit_rows(torch.load(args.highp_pkl, map_location='cpu', weights_only=False)['dictionary'])
    else:
        D = torch.zeros(0, scaf['dictionary'].shape[1])  # no high-p family
    n_atoms = D.shape[0]
    prompts = pickle.load(open(args.prompts_pkl, 'rb'))
    concepts = [c for c in scaf['names'] if c in prompts and c in cdir]
    print(f"{len(concepts)} concepts, {n_atoms} high-p atoms, {args.n_random_dirs} random", flush=True)

    # extract matched holdout activations (last token)
    acts, ctxs, tags = [], [], []
    for c in concepts:
        for p in prompts[c][:args.n_per_concept]:
            toks = tokenizer.encode(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(toks, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1]
            ctxs.append(h[0, :-1, :].float().cpu()); acts.append(h[0, -1, :].float().cpu()); tags.append(c)
    N = len(acts)
    print(f"Holdout activations: N={N}, mean norm {torch.stack(acts).norm(dim=-1).mean():.2f}", flush=True)

    L2 = {'random': np.zeros((N, args.n_random_dirs, A)), 'highp': np.zeros((N, n_atoms, A)),
          'contrast_matched': np.zeros((N, A))}
    KL = {'random': np.zeros((N, args.n_random_dirs, A)), 'highp': np.zeros((N, n_atoms, A)),
          'contrast_matched': np.zeros((N, A))}
    nr = args.n_random_dirs

    def save(done):
        with open(os.path.join(args.results_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump({'args': vars(args), 'angles_deg': angles_deg, 'concepts': tags,
                         'l2': L2, 'kl': KL, 'done': done}, f)

    t0 = time.time()
    for i in range(N):
        a, ctx, c = acts[i], ctxs[i], tags[i]
        rand = torch.randn(nr, a.shape[0])                       # orthogonalized inside batched_sweep
        dirs = torch.cat([rand, D, cdir[c].unsqueeze(0)], dim=0)  # [nr+n_atoms+1, d]
        kl, l2 = batched_sweep(model, ctx, a, dirs, angles_rad, args.layer, device,
                               measure_layer=args.measure_layer, chunk=args.chunk)
        KL['random'][i], L2['random'][i] = kl[:nr], l2[:nr]
        KL['highp'][i], L2['highp'][i] = kl[nr:nr + n_atoms], l2[nr:nr + n_atoms]
        KL['contrast_matched'][i], L2['contrast_matched'][i] = kl[-1], l2[-1]
        if (i + 1) % 20 == 0 or i == N - 1:
            save(i + 1)
            print(f"  act {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.results_dir, 'holdout_sweep.pkl'), 'wb') as f:
        pickle.dump({'args': vars(args), 'angles_deg': angles_deg, 'concepts': tags,
                     'l2': L2, 'kl': KL}, f)
    print(f"saved: {os.path.join(args.results_dir, 'holdout_sweep.pkl')}", flush=True)


if __name__ == '__main__':
    main()
