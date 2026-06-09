#!/usr/bin/env python3
"""Build a large dictionary of plateau-breaking unit directions via repeated
single-direction optimization with soft orthogonality.

For each new direction k = 1..K_dict:
  - sample 1 activation from a FineWeb pool (fresh per direction),
  - sample SAMPLE_SCAFFOLD of the available semantic scaffold directions,
  - optimize d_k via Adam with the same plateau-breaking proxy as
    learn_feature_dir_vs_contrastive.py, minus any reconstruction term:

      L(d_k) = mean_c [R_mix / (R_f + R_c)]         # plateau-breaking proxy
             + lam_scaffold * mean_c (d_k . d_c)^2  # avoid scaffold
             + lam_prev     * mean_prev (d_k . d_prev)^2  # avoid all prior atoms

Defaults match the dictionary build plan
(`Plans/dictionary_build_8k_directions.md`):

    Qwen3-1.7B at layer 14, K_dict=8000, n_steps=150, n_act=1, 10 of 17
    semantic scaffold dirs sampled per direction, angle=30deg.

Checkpoints incrementally to results/dictionary/<model>_L<layer>_K<K>/.
"""

import argparse
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import torch

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation,
)
from scripts.superellipse.learn_feature_dir_vs_contrastive import (
    load_or_build_scaffold,
)
from scripts.superellipse.decompose_plateau_breaking import (
    batched_proxy_loss, batched_ratio_multiangle,
)
from scripts.superellipse.soft_p_proxy import soft_p_estimate, diag_rho_estimate


# Semantic-only names (drop language and code — both anchored on English).
SEMANTIC_NAMES = [
    'Gender', 'Sentiment', 'Age', 'Era', 'Tense', 'TensePresent',
    'Number', 'Person', 'Status', 'Honesty', 'HonestyShort',
    'Refusal', 'Formal', 'Literary', 'Wealth', 'Health', 'Certainty',
]


def optimize_one_atom(model, acts_stack, ctx_stack, f_refs_stack,
                      scaffold, prev_stack, angles_rad,
                      layer, measure_layer, device, d_model,
                      n_steps, lr, lam_scaffold, lam_prev,
                      chunk_k, seed, log_every, contrast_fn=None,
                      objective='ratio', soft_p_cfg=None, act_fn=None):
    """Optimize one dictionary atom d_k.

    scaffold:    [K_scaf, d]  fixed contrast dirs for this atom (used when
                 contrast_fn is None).
    contrast_fn: optional callable(step) -> [K, d] returning fresh contrast
                 dirs each gradient step (per-step resampling). Overrides
                 `scaffold` for the proxy/overlap when provided.
    act_fn:      optional callable(step) -> (acts, ctx, f_refs) returning a
                 fresh activation each gradient step (per-step anchor resampling,
                 anti-overfitting). Overrides the passed-in stacks when provided.
    prev_stack:  [N_prev, d] or None  previously found atoms (unit).
    objective:   'ratio'  -> minimize mean R_mix/(R_f+R_c) (original proxy).
                 'soft_p' -> maximize the differentiable soft-intercept p_hat.
    soft_p_cfg:  dict with keys phis_rad, radii_rad, p_cap, n_newton when
                 objective == 'soft_p'.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d = torch.randn(d_model, generator=g, device=device)
    d = (d / d.norm()).requires_grad_(True)
    optimizer = torch.optim.Adam([d], lr=lr)
    history = []

    for step in range(n_steps):
        if act_fn is not None:
            acts_stack, ctx_stack, f_refs_stack = act_fn(step)
        contrast = contrast_fn(step) if contrast_fn is not None else scaffold
        optimizer.zero_grad()

        p_hat = None
        if objective == 'soft_p':
            p_hat = soft_p_estimate(
                model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
                layer, measure_layer,
                soft_p_cfg['phis_rad'], soft_p_cfg['radii_rad'],
                p_cap=soft_p_cfg['p_cap'], n_newton=soft_p_cfg['n_newton'],
                chunk=soft_p_cfg.get('chunk', 512))
            resp_loss = -p_hat                      # maximize p
        elif objective == 'diag_rho':
            rho_hat, rdiag = diag_rho_estimate(
                model, acts_stack, ctx_stack, f_refs_stack, d, contrast,
                layer, measure_layer,
                soft_p_cfg['phis_rad'], soft_p_cfg['radii_rad'],
                tau_frac=soft_p_cfg['tau_frac'],
                chunk=soft_p_cfg.get('chunk', 512), return_diag=True)
            resp_loss = -torch.log(rho_hat)         # maximize rho (monotone in p)
            p_hat = rdiag['p_monitor'].mean()       # closed-form p, for logging
        else:
            # All push magnitudes in one batched forward (latency-bound ->
            # ~free vs the per-angle loop). Equivalent to averaging the ratio
            # over angles.
            resp_loss, _ = batched_ratio_multiangle(
                model, acts_stack, ctx_stack, f_refs_stack, d,
                contrast, angles_rad, layer, measure_layer, chunk_k=chunk_k)

        du = d / (d.norm() + 1e-10)
        scaffold_overlap = ((du.unsqueeze(0) * contrast).sum(-1) ** 2).mean()
        if prev_stack is not None and prev_stack.shape[0] > 0:
            prev_overlap = ((du.unsqueeze(0) * prev_stack).sum(-1) ** 2).mean()
        else:
            prev_overlap = torch.zeros((), device=device)

        loss = (resp_loss
                + lam_scaffold * scaffold_overlap
                + lam_prev * prev_overlap)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d.data.div_(d.data.norm())

        history.append({
            'step': step, 'loss': loss.item(),
            'resp_loss': resp_loss.item(),
            'p_hat': p_hat.item() if p_hat is not None else float('nan'),
            'scaffold_overlap': scaffold_overlap.item(),
            'prev_overlap': prev_overlap.item(),
        })
        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            extra = (f'p_hat={p_hat.item():.3f}' if p_hat is not None
                     else f'ratio={resp_loss.item():.4f}')
            print(f'    step {step:4d}  L={loss.item():.4f}  {extra}  '
                  f'scaf={scaffold_overlap.item():.4f}  '
                  f'prev={prev_overlap.item():.4f}', flush=True)

    return d.detach(), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--measure_layer', type=int, default=None,
                    help='default: n_layers - 2')
    ap.add_argument('--seq_len', type=int, default=10)
    ap.add_argument('--K_dict', type=int, default=8000,
                    help='total dictionary size')
    ap.add_argument('--act_pool_size', type=int, default=8000,
                    help='size of FineWeb activation pool to sample from')
    ap.add_argument('--contrast_source', choices=['scaffold', 'random'],
                    default='random',
                    help='contrast dirs d_c: sampled semantic scaffold, or '
                         'fresh random unit dirs per atom')
    ap.add_argument('--n_contrast', type=int, default=10,
                    help='# of contrast dirs per atom (replaces '
                         '--scaffold_sample_size)')
    ap.add_argument('--contrast_seed_base', type=int, default=7_000_003,
                    help='per-atom RNG offset for random contrast dirs')
    ap.add_argument('--contrast_resample', choices=['atom', 'step'],
                    default='atom',
                    help='random contrast: sample once per atom, or fresh '
                         'every gradient step (anti-overfitting). '
                         'Only applies to --contrast_source random.')
    ap.add_argument('--resample_act', choices=['direction', 'step'],
                    default='direction',
                    help="activation/anchor: one fixed per direction "
                         "(default), or fresh from the pool every gradient "
                         "step (anti-overfitting to a single activation)")
    ap.add_argument('--n_act_per_step', type=int, default=1,
                    help='--resample_act step: # activations averaged per '
                         'gradient step (minibatch; cuts gradient variance)')
    ap.add_argument('--scaffold_sample_size', type=int, default=None,
                    help='deprecated alias for --n_contrast')
    ap.add_argument('--angles', type=float, nargs='+', default=[30.0])
    ap.add_argument('--objective', choices=['ratio', 'soft_p', 'diag_rho'],
                    default='ratio',
                    help="'ratio' (original R_mix/(R_f+R_c) proxy), 'soft_p' "
                         "(maximize Gauss-Newton p_hat), or 'diag_rho' "
                         "(maximize the smooth diagonal intercept rho; "
                         "simplest, monotone in p, cleanest gradient)")
    ap.add_argument('--tau_frac', type=float, default=0.15,
                    help='diag_rho: smoothing scale of the soft intercept '
                         '(fraction of the threshold level)')
    ap.add_argument('--n_radii', type=int, default=6,
                    help='soft_p: # geodesic radii per ray (incl. r=0)')
    ap.add_argument('--n_rays', type=int, default=5,
                    help='soft_p: # rays in phi [0,90] incl. d- and c-axis')
    ap.add_argument('--max_angle', type=float, default=60.0,
                    help='soft_p: max geodesic radius (deg)')
    ap.add_argument('--p_cap', type=float, default=10.0,
                    help='soft_p: cap on fitted p (bounds the p->inf corner)')
    ap.add_argument('--n_newton', type=int, default=20,
                    help='soft_p: Gauss-Newton iterations for the p fit')
    ap.add_argument('--n_steps', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--lam_scaffold', type=float, default=None,
                    help='penalty on (d_k . d_c)^2; default 0 for random '
                         'contrast, 1.0 for scaffold')
    ap.add_argument('--lam_prev', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42,
                    help='seed for fineweb pool + per-atom RNG offset')
    ap.add_argument('--act_pool_seed', type=int, default=42,
                    help='seed for the activation pool')
    ap.add_argument('--ckpt_every', type=int, default=100)
    ap.add_argument('--log_every_step', type=int, default=0,
                    help='print per-step inner-loop log every N steps; 0 = silent')
    ap.add_argument('--save_history', action='store_true',
                    help='dump full per-step optimization history per direction '
                         'to histories.pkl (for diagnosing trajectories)')
    ap.add_argument('--batch_chunk_k', type=int, default=None)
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--dtype', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--resume', action='store_true',
                    help='resume from existing checkpoint if present')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve deprecated alias + contrast-source-dependent defaults.
    if args.scaffold_sample_size is not None:
        args.n_contrast = args.scaffold_sample_size
    if args.lam_scaffold is None:
        args.lam_scaffold = 0.0 if args.contrast_source == 'random' else 1.0

    if args.results_dir is None:
        suffix = '_randcontrast' if args.contrast_source == 'random' else ''
        if args.contrast_source == 'random' and args.contrast_resample == 'step':
            suffix += '_stepresample'
        if args.resample_act == 'step':
            suffix += '_actresample'
        if args.objective == 'soft_p':
            suffix += '_softp'
        elif args.objective == 'diag_rho':
            suffix += '_diagrho'
        args.results_dir = (f'results/dictionary/'
                            f'{args.model}_L{args.layer}_K{args.K_dict}{suffix}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    # Ray/radius grids for the eval-frame objectives (built once, shared).
    soft_p_cfg = None
    if args.objective in ('soft_p', 'diag_rho'):
        from scripts.superellipse.soft_p_proxy import default_phis_radii
        phis_rad, radii_rad = default_phis_radii(
            args.n_rays, args.n_radii, args.max_angle)
        soft_p_cfg = {'phis_rad': phis_rad, 'radii_rad': radii_rad,
                      'p_cap': args.p_cap, 'n_newton': args.n_newton,
                      'tau_frac': args.tau_frac}
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f'Device: {device} d_model={d_model} n_layers={n_layers} '
          f'perturb_layer={args.layer} measure_layer={measure_layer}',
          flush=True)
    print(f'Config: K_dict={args.K_dict} act_pool={args.act_pool_size} '
          f'objective={args.objective} '
          f'contrast_source={args.contrast_source} n_contrast={args.n_contrast} '
          f'contrast_resample={args.contrast_resample} '
          f'resample_act={args.resample_act} '
          f'n_act_per_step={args.n_act_per_step} '
          f'n_steps={args.n_steps} lr={args.lr} '
          f'lam_scaffold={args.lam_scaffold} lam_prev={args.lam_prev}',
          flush=True)
    if args.objective in ('soft_p', 'diag_rho'):
        print(f'  {args.objective}: n_rays={args.n_rays} n_radii={args.n_radii} '
              f'max_angle={args.max_angle} p_cap={args.p_cap} '
              f'n_newton={args.n_newton} tau_frac={args.tau_frac}', flush=True)

    # Scaffold (all 17 semantic dirs) — only needed for scaffold contrast.
    if args.contrast_source == 'scaffold':
        scaffold_full = load_or_build_scaffold(model, tokenizer, device,
                                               args.model, args.layer,
                                               SEMANTIC_NAMES)
        print(f'Scaffold (full): shape={tuple(scaffold_full.shape)}', flush=True)
    else:
        scaffold_full = None
        print('Contrast: fresh random unit dirs per atom '
              f'(n_contrast={args.n_contrast}, seed_base='
              f'{args.contrast_seed_base})', flush=True)

    # Activation pool: act_pool_size FineWeb sequences at seq_len.
    print(f'Building activation pool ({args.act_pool_size} FineWeb '
          f'seqs, seed={args.act_pool_seed})', flush=True)
    tok_pool = load_fineweb_fixed_length(args.act_pool_size, tokenizer,
                                         seq_len=args.seq_len,
                                         seed=args.act_pool_seed)
    ctx_all, act_all = extract_context_and_activation(model, tok_pool,
                                                     args.layer, device)
    model_dtype = next(model.parameters()).dtype
    act_all = act_all.to(device=device, dtype=torch.float32)         # [N, d]
    ctx_all = ctx_all.to(device=device, dtype=model_dtype)           # [N, T-1, d]
    print(f'  mean ||a||={act_all.norm(dim=-1).mean().item():.2f}',
          flush=True)

    # Precompute f_refs for the pool (one batched forward).
    print('Precomputing f_refs for the activation pool', flush=True)
    f_refs_all = torch.empty(act_all.shape, dtype=torch.float32, device=device)
    chunk = 256
    with torch.no_grad():
        for i in range(0, act_all.shape[0], chunk):
            j = min(act_all.shape[0], i + chunk)
            pts = act_all[i:j].unsqueeze(1).to(model_dtype)            # [b, 1, d]
            ctx_b = ctx_all[i:j]
            f_refs_all[i:j] = forward_from_layer_to_layer(
                model, ctx_b, pts, args.layer, measure_layer).float()
    print('  done.', flush=True)

    # Dictionary buffer + optional resume.
    ckpt_path = os.path.join(args.results_dir, 'dictionary.pt')
    meta_path = os.path.join(args.results_dir, 'metadata.pkl')
    start_k = 0
    dictionary = torch.zeros(args.K_dict, d_model, device=device,
                             dtype=torch.float32)
    metadata = {'args': vars(args), 'atoms': []}
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device)
        n_have = blob['n_done']
        dictionary[:n_have] = blob['dictionary'].to(device=device,
                                                    dtype=torch.float32)
        start_k = n_have
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
        print(f'[resume] starting from k={start_k}', flush=True)

    # Per-atom RNG: numpy generator seeded per atom so sampling is
    # deterministic and resumable.
    def atom_rng(k):
        return np.random.default_rng(args.seed * 1_000_003 + k)

    # Per-atom random contrast dirs: independent torch generator so the
    # contrast sample doesn't shift the activation/scaffold sampling stream.
    def random_contrast(k):
        g = torch.Generator(device=device).manual_seed(
            args.seed * args.contrast_seed_base + k)
        dc = torch.randn(args.n_contrast, d_model, generator=g, device=device)
        return dc / dc.norm(dim=-1, keepdim=True)

    # Per-step variant: fresh dirs each gradient step, deterministic in
    # (atom k, step) so the build stays resumable.
    def random_contrast_step(k, step):
        g = torch.Generator(device=device).manual_seed(
            (args.seed * args.contrast_seed_base + k) * 100_003 + step)
        dc = torch.randn(args.n_contrast, d_model, generator=g, device=device)
        return (dc / dc.norm(dim=-1, keepdim=True)).to(torch.float32)

    # Per-step activation resampling: fresh anchor from the pool each step,
    # deterministic in (atom k, step) so the build stays resumable.
    def act_resample_step(k, step):
        g = np.random.default_rng((args.seed * 2_000_003 + k) * 100_003 + step)
        m = min(args.n_act_per_step, act_all.shape[0])
        idxs = torch.as_tensor(
            g.choice(act_all.shape[0], size=m, replace=False), device=device)
        return act_all[idxs], ctx_all[idxs], f_refs_all[idxs]

    # Per-step scaffold contrast resampling: fresh subset of the semantic
    # scaffold each gradient step, deterministic in (atom k, step). Enables
    # --contrast_source scaffold --contrast_resample step.
    def scaffold_contrast_step(k, step):
        g = np.random.default_rng((args.seed * 3_000_003 + k) * 100_003 + step)
        m = min(args.n_contrast, scaffold_full.shape[0])
        idxs = g.choice(scaffold_full.shape[0], size=m, replace=False)
        return scaffold_full[torch.as_tensor(idxs, device=device)].to(torch.float32)

    print('\n=== Building dictionary ===', flush=True)
    t_start = time.time()
    all_histories = []          # full per-step history (only if --save_history)
    for k in range(start_k, args.K_dict):
        rng = atom_rng(k)
        # Sample 1 activation (fixed per direction; overridden below if
        # --resample_act step, which draws a fresh anchor each gradient step).
        idx = int(rng.integers(0, act_all.shape[0]))
        acts_stack = act_all[idx:idx + 1]
        ctx_stack = ctx_all[idx:idx + 1]
        f_refs_stack = f_refs_all[idx:idx + 1]
        act_fn = None
        if args.resample_act == 'step':
            act_fn = lambda step, _k=k: act_resample_step(_k, step)
            idx = None                                   # no single anchor
        # Contrast dirs d_c: scaffold subset or fresh random unit dirs.
        contrast_fn = None
        if args.contrast_source == 'scaffold':
            if args.contrast_resample == 'step':
                contrast_fn = lambda step, _k=k: scaffold_contrast_step(_k, step)
                scaf = scaffold_contrast_step(k, 0)        # placeholder; unused
                scaf_idx = 'step-resampled'
            else:
                scaf_idx = rng.choice(scaffold_full.shape[0],
                                      size=args.n_contrast, replace=False)
                scaf = scaffold_full[torch.tensor(scaf_idx, device=device)]
                scaf_idx = scaf_idx.tolist()
        elif args.contrast_resample == 'step':
            scaf = random_contrast_step(k, 0)          # placeholder, unused
            contrast_fn = lambda step, _k=k: random_contrast_step(_k, step)
            scaf_idx = None
        else:
            scaf = random_contrast(k).to(torch.float32)
            scaf_idx = None
        # Previous atoms (slice — already on device)
        prev_stack = dictionary[:k] if k > 0 else None

        d_k, hist = optimize_one_atom(
            model, acts_stack, ctx_stack, f_refs_stack, scaf, prev_stack,
            angles_rad, args.layer, measure_layer, device, d_model,
            args.n_steps, args.lr, args.lam_scaffold, args.lam_prev,
            args.batch_chunk_k, args.seed * 1_000_003 + k,
            args.log_every_step, contrast_fn=contrast_fn,
            objective=args.objective, soft_p_cfg=soft_p_cfg, act_fn=act_fn)
        dictionary[k] = d_k
        metadata['atoms'].append({
            'k': k, 'act_idx': idx, 'scaf_idx': scaf_idx,
            'final_loss': hist[-1]['loss'],
            'final_resp': hist[-1]['resp_loss'],
            'final_p_hat': hist[-1]['p_hat'],
            'final_scaffold_overlap': hist[-1]['scaffold_overlap'],
            'final_prev_overlap': hist[-1]['prev_overlap'],
        })
        if args.save_history:
            all_histories.append({'k': k, 'history': hist})
            with open(os.path.join(args.results_dir, 'histories.pkl'), 'wb') as f:
                pickle.dump(all_histories, f)

        if (k + 1) % args.ckpt_every == 0 or k == args.K_dict - 1:
            torch.save({
                'dictionary': dictionary[:k + 1].cpu(),
                'n_done': k + 1, 'K_dict': args.K_dict,
                'args': vars(args),
            }, ckpt_path)
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            elapsed = time.time() - t_start
            done = k + 1 - start_k
            rate = elapsed / max(done, 1)
            eta = rate * (args.K_dict - (k + 1))
            print(f'[ckpt {k+1:5d}/{args.K_dict}]  '
                  f'last loss={hist[-1]["loss"]:.4f}  '
                  f'rate={rate:.2f}s/atom  ETA={eta/3600:.2f}h',
                  flush=True)

    elapsed = time.time() - t_start
    print(f'\nDONE: built {args.K_dict - start_k} atoms in '
          f'{elapsed/3600:.2f}h ({elapsed / max(args.K_dict-start_k,1):.2f}s/atom)',
          flush=True)


if __name__ == '__main__':
    main()
