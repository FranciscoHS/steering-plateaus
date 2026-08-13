#!/usr/bin/env python3
"""1D spherical perturbation: real activation vs norm-matched random activation,
both perturbed along random directions.

Mirrors eval_plateaus_1d.py (which is known to fit gemma-2-9b on a 4090) but
swaps the second condition: instead of contrastive directions, the comparison
is "random activations of the same norm, perturbed by the same random
directions through the same model forward". This shows whether real
activations sit on a special low-curvature manifold relative to off-manifold
points.

Output:
  - results.pkl with L2(angle) per (direction, anchor) for real + random_act
  - sweep.png with median+IQR for each
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks, forward_from_layer_to_layer
from src.data import load_fineweb_fixed_length


def ctx_acts_from_token_lists(model, device, layer, token_lists):
    contexts, activations = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def fineweb_ctx_acts(model, tokenizer, device, layer, n_act, seq_len, seed):
    cache_dir = '/workspace/fineweb_cache'
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n_act, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    return ctx_acts_from_token_lists(model, device, layer, token_lists)


def random_token_ctx_acts(model, tokenizer, device, layer, n_act, seq_len,
                          seed):
    """Anchors from random token-id sequences: uniform over the base vocab
    (special/added tokens excluded), same seq_len as the FineWeb condition.
    Each anchor keeps its own context, mirroring the real-activation setup."""
    g = torch.Generator().manual_seed(seed)
    vocab = tokenizer.vocab_size
    token_lists = [torch.randint(0, vocab, (seq_len,), generator=g)
                   for _ in range(n_act)]
    return ctx_acts_from_token_lists(model, device, layer, token_lists)


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device, batch=64):
    """Spherical L2 sweep at one anchor along direction d. Returns L2[n_steps]."""
    R = a_dev.norm(); a_unit = a_dev / R
    d_o = F.normalize(d_dev - (d_dev @ a_unit) * a_unit, dim=0)
    n_steps = len(angles_rad)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    pts32 = R * (c[:, None] * a_unit[None, :] + s[:, None] * d_o[None, :])
    pts = pts32.to(model_dtype)
    ref = forward_from_layer_to_layer(
        model, ctx_d, a_dev[None, None].to(model_dtype),
        layer, measure_layer).float()
    out = np.zeros(n_steps, dtype=np.float32)
    for st in range(0, n_steps, batch):
        ed = min(n_steps, st + batch)
        pf = forward_from_layer_to_layer(
            model, ctx_d, pts[st:ed].unsqueeze(1),
            layer, measure_layer).float()
        out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--measure_layer', type=int, default=None)
    ap.add_argument('--n_act', type=int, default=30)
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_steps', type=int, default=31)
    ap.add_argument('--n_dirs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rand_seed', type=int, default=0)
    ap.add_argument('--dict_path', default=None,
                    help='if given, add synthetic anchors = equal sum of a '
                         'subset of dict atoms, rescaled to the mean act norm')
    ap.add_argument('--n_synth_dirs', type=int, default=None,
                    help='# of dict atoms to combine per synthetic anchor '
                         '(default: all). A random subset is drawn per combo.')
    ap.add_argument('--n_synth_combos', type=int, default=1,
                    help='# of independent random subsets to build + sweep')
    ap.add_argument('--synth_seed', type=int, default=0,
                    help='seed for choosing the synthetic-anchor subsets')
    ap.add_argument('--synth_mode', choices=['equal_sum', 'sae_coef'],
                    default='equal_sum',
                    help="how to build the synthetic anchor. 'equal_sum': sum "
                         "of n_synth_dirs random atoms, rescaled to mean ||a||. "
                         "'sae_coef': decompose each real activation with a TopK "
                         "SAE, then rebuild b_dec + sum_i m_i*d_assigned(i) using "
                         "our atoms with the SAE per-latent contribution "
                         "magnitudes m_i (realistic heavy-tailed spectrum).")
    ap.add_argument('--sae_ckpt', default=None,
                    help='TopK SAE checkpoint (W_enc,b_enc,W_dec,b_dec) for '
                         '--synth_mode sae_coef')
    ap.add_argument('--extra_anchors_pt', action='append', default=None,
                    help='.pt file with extra synthetic anchors to sweep as '
                         'an additional condition (e.g. '
                         'jac_optimized_data.pt); the first n_act rows are '
                         'used, paired with the same contexts and random '
                         'directions. Repeat the flag for several sets.')
    ap.add_argument('--extra_anchors_key', action='append', default=None,
                    help='key of the anchor tensor inside the .pt (ignored '
                         'if the file holds a raw tensor); one per '
                         '--extra_anchors_pt (default X_opt)')
    ap.add_argument('--extra_label', action='append', default=None,
                    help='legend label, one per --extra_anchors_pt')
    ap.add_argument('--random_tokens', action='store_true',
                    help='add a condition: anchors from random token-id '
                         'sequences (uniform over vocab, same seq_len), each '
                         'swept with its own context')
    ap.add_argument('--rand_tok_seed', type=int, default=1234,
                    help='seed for sampling the random token sequences')
    ap.add_argument('--sae_K', type=int, default=50,
                    help='# of top SAE latents per activation (the SAE L0)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    d_model = model._plateau_info['d_model']
    model_dtype = next(model.parameters()).dtype
    print(f'Device: {device} d_model={d_model} L{args.layer}->{measure_layer}',
          flush=True)

    print(f'Extracting {args.n_act} FineWeb anchors '
          f'(seq_len={args.seq_len}, seed={args.seed})', flush=True)
    contexts, activations = fineweb_ctx_acts(model, tokenizer, device,
                                              args.layer, args.n_act,
                                              args.seq_len, args.seed)
    a_norms = activations.norm(dim=-1)
    print(f'  mean ||a||={a_norms.mean().item():.2f}  '
          f'std={a_norms.std().item():.2f}', flush=True)

    # Optional anchors from random token sequences (with their own contexts)
    rt_contexts, rt_acts = None, None
    if args.random_tokens:
        print(f'Extracting {args.n_act} random-token anchors '
              f'(seq_len={args.seq_len}, seed={args.rand_tok_seed})',
              flush=True)
        rt_contexts, rt_acts = random_token_ctx_acts(
            model, tokenizer, device, args.layer, args.n_act, args.seq_len,
            args.rand_tok_seed)
        rt_norms = rt_acts.norm(dim=-1)
        print(f'  mean ||a||={rt_norms.mean().item():.2f}  '
              f'std={rt_norms.std().item():.2f}', flush=True)

    # Norm-matched random "activations": same per-anchor norm, random unit dir
    g_act = torch.Generator(device=device).manual_seed(args.rand_seed)
    rand_acts = torch.randn(args.n_act, d_model, generator=g_act, device=device,
                            dtype=torch.float32)
    rand_acts = rand_acts / rand_acts.norm(dim=-1, keepdim=True)
    rand_acts = rand_acts * a_norms.to(device).unsqueeze(-1)
    rand_acts = rand_acts.cpu()

    # Random unit directions (n_dirs of them, shared across all anchors)
    g_d = torch.Generator(device=device).manual_seed(args.rand_seed + 1)
    rand_dirs = torch.randn(args.n_dirs, d_model, generator=g_d, device=device,
                            dtype=torch.float32)
    rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)
    rand_dirs = rand_dirs.cpu()

    # Optional synthetic "activations". synth_all: [n_combos, n_act, d] — one
    # synthetic last-token vector per (combo, anchor). For equal_sum the same
    # vector is broadcast across anchors; for sae_coef it is built per-anchor
    # from that anchor's own SAE decomposition.
    synth_all = None
    synth_meta = None
    if args.dict_path is not None:
        blob = torch.load(args.dict_path, map_location='cpu')
        D = blob['dictionary'].to(torch.float32)
        D = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        K = D.shape[0]
        n_combos = args.n_synth_combos
        synth_all = torch.zeros(n_combos, args.n_act, d_model, dtype=torch.float32)
        synth_meta = []

        if args.synth_mode == 'equal_sum':
            n_pick = min(args.n_synth_dirs if args.n_synth_dirs is not None
                         else K, K)
            rng = np.random.default_rng(args.synth_seed)
            for c in range(n_combos):
                idx = (np.arange(K) if n_pick >= K
                       else rng.choice(K, size=n_pick, replace=False))
                v = D[idx].sum(0)
                sa = (v / v.norm()) * a_norms.mean()
                synth_all[c, :] = sa                          # broadcast anchors
                synth_meta.append({'mode': 'equal_sum',
                                   'atoms': sorted(int(j) for j in idx)})
                cos_to_real = float(F.cosine_similarity(
                    sa[None], activations, dim=-1).abs().mean())
                print(f'Synthetic combo {c}: {n_pick} of {K} atoms summed, '
                      f'||v||={sa.norm():.2f}, mean|cos| to real={cos_to_real:.3f}',
                      flush=True)
        else:  # sae_coef
            sae = torch.load(args.sae_ckpt, map_location='cpu')
            W_enc = sae['W_enc'].float(); b_enc = sae['b_enc'].float()
            W_dec = sae['W_dec'].float(); b_dec = sae['b_dec'].float()  # [d,d_sae]
            dec_norm = W_dec.norm(dim=0)                       # [d_sae]
            # per-anchor SAE decomposition (top-K), shared across combos.
            pre = activations.float() @ W_enc.T + b_enc        # [n_act, d_sae]
            topv, topi = pre.topk(args.sae_K, dim=-1)          # [n_act, K]
            # effective contribution magnitude m_i = z_i * ||W_dec[:,i]||
            mags = topv * dec_norm[topi]                       # [n_act, K]
            for c in range(n_combos):
                rng = np.random.default_rng(args.synth_seed + c)
                for i in range(args.n_act):
                    pick = rng.choice(K, size=args.sae_K,
                                      replace=(K < args.sae_K))
                    contrib = (mags[i].unsqueeze(-1) * D[pick]).sum(0)  # [d]
                    synth_all[c, i] = b_dec + contrib
                synth_meta.append({'mode': 'sae_coef', 'sae_K': args.sae_K})
            cos_to_real = float(F.cosine_similarity(
                synth_all[0], activations, dim=-1).abs().mean())
            sn = synth_all[0].norm(dim=-1).mean()
            print(f'sae_coef: top-{args.sae_K} decomp (mean recon FVU unchecked), '
                  f'synth mean||v||={sn:.1f} vs mean||a||={a_norms.mean():.1f}, '
                  f'mean|cos| to real={cos_to_real:.3f}', flush=True)

    # Optional extra anchor sets from tensor files (e.g. Jacobian-optimized
    # points). NOTE: paired with THIS run's contexts, which need not be the
    # contexts the points were built/optimized with.
    extra_sets = []          # list of (label, [n_act, d] tensor)
    if args.extra_anchors_pt:
        keys = args.extra_anchors_key or []
        labels = args.extra_label or []
        for j, pt in enumerate(args.extra_anchors_pt):
            key = keys[j] if j < len(keys) else 'X_opt'
            label = (labels[j] if j < len(labels)
                     else os.path.basename(pt).rsplit('.', 1)[0])
            eb = torch.load(pt, map_location='cpu', weights_only=False)
            acts = (eb[key] if isinstance(eb, dict) else eb).float()
            if acts.shape[0] < args.n_act:
                raise SystemExit(f'extra anchors {pt}: only {acts.shape[0]} '
                                 f'rows for n_act={args.n_act}')
            acts = acts[:args.n_act]
            extra_sets.append((label, acts))
            print(f'Extra anchors ({label}): {tuple(acts.shape)} '
                  f'mean||a||={acts.norm(dim=-1).mean():.2f}', flush=True)

    angles_deg = np.linspace(0.0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    print(f'Sweep angles 0..{args.max_angle}° in {args.n_steps} steps  '
          f'n_dirs={args.n_dirs}', flush=True)

    out_real = np.zeros((args.n_act, args.n_dirs, args.n_steps),
                        dtype=np.float32)
    out_rand = np.zeros((args.n_act, args.n_dirs, args.n_steps),
                        dtype=np.float32)
    n_combos = synth_all.shape[0] if synth_all is not None else 0
    out_synth = (np.zeros((n_combos, args.n_act, args.n_dirs, args.n_steps),
                          dtype=np.float32) if n_combos else None)
    out_extras = [np.zeros((args.n_act, args.n_dirs, args.n_steps),
                           dtype=np.float32) for _ in extra_sets]
    out_randtok = (np.zeros((args.n_act, args.n_dirs, args.n_steps),
                            dtype=np.float32) if args.random_tokens else None)
    synth_dev = (synth_all.to(device=device, dtype=torch.float32)
                 if synth_all is not None else None)

    t0 = time.time()
    for i in range(args.n_act):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a_real = activations[i].to(device=device, dtype=torch.float32)
        a_rand = rand_acts[i].to(device=device, dtype=torch.float32)
        for r in range(args.n_dirs):
            d = rand_dirs[r].to(device=device, dtype=torch.float32)
            out_real[i, r] = sweep_1d(model, ctx, a_real, d, args.layer,
                                      measure_layer, angles_rad,
                                      model_dtype, device)
            out_rand[i, r] = sweep_1d(model, ctx, a_rand, d, args.layer,
                                      measure_layer, angles_rad,
                                      model_dtype, device)
            for c in range(n_combos):
                out_synth[c, i, r] = sweep_1d(model, ctx, synth_dev[c, i], d,
                                              args.layer, measure_layer,
                                              angles_rad, model_dtype, device)
            for j, (_, acts) in enumerate(extra_sets):
                a_extra = acts[i].to(device=device, dtype=torch.float32)
                out_extras[j][i, r] = sweep_1d(model, ctx, a_extra, d,
                                               args.layer, measure_layer,
                                               angles_rad, model_dtype,
                                               device)
            if out_randtok is not None:
                ctx_rt = rt_contexts[i].to(device=device, dtype=model_dtype)
                a_rt = rt_acts[i].to(device=device, dtype=torch.float32)
                out_randtok[i, r] = sweep_1d(model, ctx_rt, a_rt, d,
                                             args.layer, measure_layer,
                                             angles_rad, model_dtype, device)
        if (i + 1) % 5 == 0 or i == args.n_act - 1:
            print(f'  anchor {i+1}/{args.n_act}  elapsed={time.time()-t0:.1f}s',
                  flush=True)

    out_pkl = os.path.join(args.out, 'results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'angles_deg': angles_deg,
            'measure_layer': measure_layer,
            'real_l2': out_real,    # [n_act, n_dirs, n_steps]
            'rand_l2': out_rand,
            'synth_l2': out_synth,  # [n_combos, n_act, n_dirs, n_steps] or None
            'synth_meta': synth_meta,
            'extra_l2': out_extras,  # list of [n_act, n_dirs, n_steps]
            'extra_labels': [lab for lab, _ in extra_sets],
            'randtok_l2': out_randtok,  # [n_act, n_dirs, n_steps] or None
            'randtok_norms': (rt_acts.norm(dim=-1).numpy()
                              if rt_acts is not None else None),
            'a_norms': a_norms.numpy(),
        }, f)

    real_flat = out_real.reshape(-1, args.n_steps)
    rand_flat = out_rand.reshape(-1, args.n_steps)
    series = [(real_flat, 'C0', 'real activation'),
              (rand_flat, 'C1', 'norm-matched random activation')]
    extra_colors = ['C2', 'C4', 'C5', 'C6', 'C7']
    for j, (lab, _) in enumerate(extra_sets):
        series.append((out_extras[j].reshape(-1, args.n_steps),
                       extra_colors[j % len(extra_colors)], lab))
    if out_randtok is not None:
        series.append((out_randtok.reshape(-1, args.n_steps), 'C3',
                       'random-token activation'))
    if out_synth is not None:
        tag = (f'{args.n_synth_dirs or "all"} atoms'
               if args.synth_mode == 'equal_sum'
               else f'SAE-coef top-{args.sae_K}')
        for c in range(n_combos):
            series.append((out_synth[c].reshape(-1, args.n_steps),
                           plt.cm.autumn(c / max(n_combos - 1, 1)),
                           f'synthetic ({tag})' if n_combos == 1
                           else f'synth {c} ({tag})'))
    fig, ax = plt.subplots(figsize=(8, 5))
    for arr, col, lab in series:
        m = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        ax.plot(angles_deg, m, '-', color=col, label=lab)
        ax.fill_between(angles_deg, q25, q75, alpha=0.2, color=col)
    ax.set_xlabel('perturbation angle (deg)')
    ax.set_ylabel(r'L2 at L$_{measure}$  ($\|f(a_\theta)-f(a)\|$)')
    ax.set_title(f'{args.model}  L{args.layer}->{measure_layer}  '
                 f'(seq_len={args.seq_len}, n_act={args.n_act}×n_dirs={args.n_dirs})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(args.out, 'sweep.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    # Console summary: median L2 per condition at sample angles.
    print('\n=== median L2 at sample angles ===')
    cols = [('real', real_flat), ('rand', rand_flat)]
    for j, (lab, _) in enumerate(extra_sets):
        cols.append((lab[:9], out_extras[j].reshape(-1, args.n_steps)))
    if out_randtok is not None:
        cols.append(('randtok', out_randtok.reshape(-1, args.n_steps)))
    if out_synth is not None:
        for c in range(n_combos):
            cols.append((f'synth{c}', out_synth[c].reshape(-1, args.n_steps)))
    print('  angle  ' + '  '.join(f'{n:>9}' for n, _ in cols))
    for ang in [10, 20, 30, 45, 60]:
        if ang <= args.max_angle:
            j = int(np.argmin(np.abs(angles_deg - ang)))
            vals = '  '.join(f'{np.median(a[:, j]):9.2f}' for _, a in cols)
            print(f'  {angles_deg[j]:5.1f}  {vals}')

    print(f'\nSaved {out_pkl}')
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
