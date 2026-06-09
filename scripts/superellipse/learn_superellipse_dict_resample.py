#!/usr/bin/env python3
"""
Learn a superellipse direction dictionary WITH per-step activation resampling.

Same objective and joint-pair / greedy-grow structure as
learn_superellipse_dict.py, but instead of a fixed set of n_act activations held
constant across the whole optimization, each gradient step draws a fresh batch
of `--n_act_per_step` activations from a large FineWeb pool (default 8000). This
mirrors the `--resample_act step` behaviour of build_dictionary.py (the builder
of dictionary_snap20k.pt), applied to the JOINT-PAIR objective.

Objective (per gradient step, unchanged from learn_superellipse_dict.py):
    L = mean_{a in batch, alpha} [ R_mix / (R_1 + R_2) ]  +  lam * (d_i . d_j)^2
where R_1, R_2, R_mix are downstream L2 responses to norm-matched spherical
perturbations along d_1, d_2 and their bisector, measured at `measure_layer`.
Stage 1 learns the first pair (d1, d2) jointly; Stage 2+ grows each new
direction against the frozen existing ones with the orthogonality (diversity)
penalty.

The unperturbed reference response f_ref for a pooled activation is exactly the
measure-layer last-token hidden state from the same forward used to extract the
perturb-layer activation, so the pool costs a single forward pass per sequence.

Usage:
    python scripts/superellipse/learn_superellipse_dict_resample.py \
        --model qwen3-1.7b --layer 14 --angles 30 \
        --n_dict 10 --n_steps 150 --n_act_per_step 10 \
        --act_pool_size 8000 --lam 1.0 \
        --out results/dictionary/qwen3-1.7b_L14_jointpair_resample_d10
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

# Reuse the exact response geometry + loss from the non-resampling script.
from scripts.superellipse.learn_superellipse_dict import (
    pair_loss, eval_dictionary,
)


def extract_pool(model, token_ids_list, layer, measure_layer, device,
                 batch_size=64):
    """Return (contexts, activations, f_refs) for a pool of sequences.

    activations = perturb-layer (`layer`) last-token hidden state.
    contexts    = perturb-layer prefix (all but last token).
    f_refs      = measure-layer last-token hidden state (the unperturbed
                  downstream response, taken from the SAME forward pass).
    """
    all_ctx, all_act, all_fref = [], [], []
    for i in range(0, len(token_ids_list), batch_size):
        batch = torch.stack(token_ids_list[i:i + batch_size]).to(device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
            h_perturb = out.hidden_states[layer + 1]
            h_measure = out.hidden_states[measure_layer + 1]
        all_ctx.append(h_perturb[:, :-1, :].float().cpu())
        all_act.append(h_perturb[:, -1, :].float().cpu())
        all_fref.append(h_measure[:, -1, :].float().cpu())
    return (torch.cat(all_ctx, 0), torch.cat(all_act, 0), torch.cat(all_fref, 0))


def optimize_resample(model, pool_acts, pool_ctxs, pool_frefs, fixed_dirs,
                      n_new, angles_rad, layer, measure_layer, device, d_model,
                      n_steps, lr, lam, n_act_per_step, init_seed, sample_seed,
                      log_prefix=""):
    """Like learn_superellipse_dict.optimize but resamples n_act_per_step
    activations from the pool at every gradient step."""
    g_init = torch.Generator(device=device).manual_seed(init_seed)
    new = [torch.randn(d_model, generator=g_init, device=device)
           for _ in range(n_new)]
    new = [(v / v.norm()).requires_grad_(True) for v in new]
    optimizer = torch.optim.Adam(new, lr=lr)

    sampler = np.random.default_rng(sample_seed)
    n_pool = len(pool_acts)
    loss_history = []
    for step in range(n_steps):
        idx = sampler.integers(0, n_pool, size=n_act_per_step)
        acts = [pool_acts[i] for i in idx]
        ctxs = [pool_ctxs[i] for i in idx]
        f_refs = [pool_frefs[i] for i in idx]

        optimizer.zero_grad()
        if n_new == 2:
            d1, d2 = new
            resp_loss, (mr1, mr2, mrmix) = pair_loss(
                model, acts, ctxs, f_refs, d1, d2, angles_rad,
                layer, measure_layer)
            d1u = d1 / (d1.norm() + 1e-10)
            d2u = d2 / (d2.norm() + 1e-10)
            orth = (d1u @ d2u) ** 2
        else:
            d_new = new[0]
            resp_terms = []
            mr1 = mr2 = mrmix = 0.0
            for d_fixed in fixed_dirs:
                rl, (a1, a2, amix) = pair_loss(
                    model, acts, ctxs, f_refs, d_new, d_fixed,
                    angles_rad, layer, measure_layer)
                resp_terms.append(rl)
                mr1 += a1; mr2 += a2; mrmix += amix
            resp_loss = torch.stack(resp_terms).mean()
            n_fix = len(fixed_dirs)
            mr1 /= n_fix; mr2 /= n_fix; mrmix /= n_fix
            d_newu = d_new / (d_new.norm() + 1e-10)
            orth = torch.stack([
                (d_newu @ (df / (df.norm() + 1e-10))) ** 2 for df in fixed_dirs
            ]).mean()

        loss = resp_loss + lam * orth
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for v in new:
                v.data.div_(v.data.norm())

        loss_history.append({
            'step': step, 'loss': loss.item(), 'resp_loss': resp_loss.item(),
            'orth': orth.item(), 'R1': mr1, 'R2': mr2, 'Rmix': mrmix,
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  orth={orth.item():.4f}  "
                  f"R1={mr1:.3f} R2={mr2:.3f} Rmix={mrmix:.3f}", flush=True)

    new_dirs = torch.stack([v.detach() for v in new])
    return new_dirs, loss_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--n_dict', type=int, default=10)
    ap.add_argument('--angles', type=float, nargs='+', default=[30.0])
    ap.add_argument('--n_steps', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--lam', type=float, default=1.0,
                    help='diversity / orthogonality (d_i . d_j)^2 penalty weight')
    ap.add_argument('--act_pool_size', type=int, default=8000)
    ap.add_argument('--n_act_per_step', type=int, default=10,
                    help='fresh activations resampled each gradient step')
    ap.add_argument('--act_pool_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=42,
                    help='seed for direction init + per-step resampling stream')
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f"Loading model: {args.model} ({args.dtype})", flush=True)
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    model_dtype = next(model.parameters()).dtype
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"perturb_layer: {args.layer}, measure_layer: {measure_layer}",
          flush=True)
    print(f"Config: n_dict={args.n_dict}, angles={args.angles} deg, "
          f"n_steps={args.n_steps}, lr={args.lr}, lam={args.lam}, "
          f"act_pool={args.act_pool_size}, n_act_per_step={args.n_act_per_step}",
          flush=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    # Activation pool (fresh-per-step source).
    print(f"Building activation pool ({args.act_pool_size} FineWeb seqs, "
          f"seq_len={args.seq_len}, seed={args.act_pool_seed})", flush=True)
    tok_pool = load_fineweb_fixed_length(args.act_pool_size, tokenizer,
                                         seq_len=args.seq_len,
                                         seed=args.act_pool_seed)
    ctx_pool, act_pool, fref_pool = extract_pool(
        model, tok_pool, args.layer, measure_layer, device)
    print(f"Mean activation norm: "
          f"{act_pool.norm(dim=-1).mean().item():.2f}", flush=True)

    # Move pool to device once. acts/frefs float32 for stable geometry; ctxs in
    # model dtype (used only at forward boundaries).
    pool_acts = [act_pool[i].to(device=device, dtype=torch.float32)
                 for i in range(act_pool.shape[0])]
    pool_frefs = [fref_pool[i].to(device=device, dtype=torch.float32)
                  for i in range(fref_pool.shape[0])]
    pool_ctxs = [ctx_pool[i].unsqueeze(0).to(device=device, dtype=model_dtype)
                 for i in range(ctx_pool.shape[0])]

    # ---- Stage 1: first pair (joint) ----
    print(f"\n=== Stage 1: learning first pair (d1, d2) jointly ===", flush=True)
    t0 = time.time()
    directions, hist1 = optimize_resample(
        model, pool_acts, pool_ctxs, pool_frefs, [], 2, angles_rad,
        args.layer, measure_layer, device, d_model, args.n_steps, args.lr,
        args.lam, args.n_act_per_step, init_seed=args.seed,
        sample_seed=args.seed + 1000)
    print(f"Stage 1 done in {time.time()-t0:.1f}s", flush=True)
    loss_histories = {'pair_d1d2': hist1}

    ckpt_path = os.path.join(args.out, 'checkpoint.pkl')
    torch.save({'directions': directions.cpu(), 'args': vars(args)}, ckpt_path)

    # ---- Stage 2+: grow with diversity penalty ----
    for k in range(2, args.n_dict):
        print(f"\n=== Stage {k}: learning d{k+1} against {k} frozen "
              f"directions ===", flush=True)
        t0 = time.time()
        fixed_dirs = [directions[j] for j in range(directions.shape[0])]
        new_dir, hist_k = optimize_resample(
            model, pool_acts, pool_ctxs, pool_frefs, fixed_dirs, 1, angles_rad,
            args.layer, measure_layer, device, d_model, args.n_steps, args.lr,
            args.lam, args.n_act_per_step, init_seed=args.seed + k,
            sample_seed=args.seed + 1000 + k, log_prefix=f"d{k+1} ")
        print(f"Stage {k} done in {time.time()-t0:.1f}s", flush=True)
        directions = torch.cat([directions, new_dir], dim=0)
        loss_histories[f'd{k+1}'] = hist_k
        torch.save({'directions': directions.cpu(), 'args': vars(args)},
                   ckpt_path)

    # ---- Diagnostics on a fixed held-out batch (just for logging) ----
    print(f"\n=== Evaluating dictionary ({directions.shape[0]} directions) ===",
          flush=True)
    eval_n = min(20, len(pool_acts))
    diag = eval_dictionary(model, pool_acts[:eval_n], pool_ctxs[:eval_n],
                           pool_frefs[:eval_n], directions, angles_rad,
                           args.layer, measure_layer)
    print(f"Per-direction single-direction response: "
          f"{np.round(diag['single_R'], 3)}", flush=True)
    print("Pairwise L (lower = more superellipse-like, 0.5 = no interference):",
          flush=True)
    for i_d in range(directions.shape[0]):
        for j_d in range(i_d + 1, directions.shape[0]):
            print(f"  L(d{i_d+1}, d{j_d+1}) = {diag['pair_L'][i_d, j_d]:.4f}",
                  flush=True)

    # ---- Save: results + dictionary blob for fit_pairs_dictionary.py ----
    results = {
        'args': vars(args),
        'directions': directions.cpu(),
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_histories': loss_histories,
        'single_R': diag['single_R'],
        'pair_L': diag['pair_L'],
        'gram': diag['gram'],
    }
    with open(os.path.join(args.out, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    # fit_pairs_dictionary.py expects blob['dictionary'] = [N, d].
    torch.save({'dictionary': directions.cpu(), 'args': vars(args)},
               os.path.join(args.out, 'dictionary_resample.pt'))

    # Loss curves.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in loss_histories.items():
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['loss'] for h in hist], label=f'{name} (total)')
        ax.plot(steps, [h['resp_loss'] for h in hist], '--', alpha=0.5)
    ax.axhline(0.5, color='gray', ls=':', label='L=0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.set_title(f'{args.model} L{args.layer}->{measure_layer} joint-pair '
                 f'resample (n_act/step={args.n_act_per_step})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'loss_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\nSaved {os.path.join(args.out, 'results.pkl')}", flush=True)
    print(f"Saved {os.path.join(args.out, 'dictionary_resample.pt')}",
          flush=True)


if __name__ == '__main__':
    main()
