#!/usr/bin/env python3
"""Generic separating-normal-shift replication (any supported model).

Generalization of gpt2xl_shift_replication.py (kept unchanged) to run the
same pipeline on other models/families:

  1. extract   real L2 activations + contexts (FineWeb, seq_len 10)
  2. gauss     mean+cov matched Gaussian samples
  3. optimize  jac-norm optimization of Gaussian points 0:200
               (min ||J||_F + lam*||x-x0||^2, Adam lr=0.1, lam=1e-3,
               200 steps, 5 Hutchinson probes, batched/chunked)
  4. analyze   logistic plane -> w_hat, shift c; gauss+c*w_hat anchors;
               ||J||_F for real / gauss / shifted / mirrored / jac-opt

Unlike the gpt2-xl script, the stage-4 ||J||_F measurement uses batched
AUTOGRAD Hutchinson (not finite differences): finite differences with
eps=1e-3 are numerically meaningless under bfloat16, which the large
models here need to fit on a 24GB card. Within-run comparisons are
unaffected; absolute values are not comparable to the finite-difference
numbers of the gpt2-xl/qwen runs.

Stages checkpoint to --out and are skipped/resumed when outputs exist.

Outputs in results/intrinsic_dim/<model>_shift/ (default):
  real_data.pt, gauss_data.pt, jac_optimized.pt (+ _ckpt.pt),
  shift_anchors.pt, jac_norms.pkl, jac_hist.png
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, "/workspace/steering-plateaus")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model
from intrinsic_dim_jac_filtered import forward_batch_layer_to_final

SEQ_LEN = 10
N_REAL = 5000
N_GAUSS = 1000
N_OPT = 200
N_STEPS = 200
LR = 0.1
LAM = 1e-3
N_PROBES_OPT = 5
N_PROBES_EVAL = 10
CKPT_EVERY = 20


def stage_extract(model, tokenizer, device, layer, out_dir):
    out = os.path.join(out_dir, 'real_data.pt')
    if os.path.exists(out):
        print(f'[extract] found {out}, skipping', flush=True)
        return torch.load(out, map_location='cpu', weights_only=False)
    print('[extract] streaming FineWeb...', flush=True)
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True)
    ds = ds.shuffle(seed=42)
    acts, ctxs = [], []
    n_kept = 0
    batch_texts = []
    for ex in ds:
        if len(ex['text']) < SEQ_LEN * 3:
            continue
        batch_texts.append(ex['text'])
        if len(batch_texts) == 32:
            tokens = tokenizer(batch_texts, return_tensors='pt',
                               padding='max_length', truncation=True,
                               max_length=SEQ_LEN).to(device)
            # keep only rows fully covered by real tokens (padding would
            # corrupt the last-token activation)
            full = tokens['attention_mask'].sum(-1) == SEQ_LEN
            with torch.no_grad():
                o = model(**tokens, output_hidden_states=True)
            h = o.hidden_states[layer + 1][full]
            acts.append(h[:, -1, :].float().cpu())
            ctxs.append(h[:, :-1, :].float().cpu())
            n_kept += int(full.sum())
            batch_texts = []
            if n_kept >= N_REAL:
                break
    X_real = torch.cat(acts)[:N_REAL]
    contexts = torch.cat(ctxs)[:N_REAL]
    blob = {'X_real': X_real, 'contexts': contexts}
    torch.save(blob, out)
    print(f'[extract] X_real {tuple(X_real.shape)}  '
          f'mean||a||={X_real.norm(dim=-1).mean():.2f}  saved {out}',
          flush=True)
    return blob


def stage_gauss(X_real, out_dir):
    out = os.path.join(out_dir, 'gauss_data.pt')
    if os.path.exists(out):
        print(f'[gauss] found {out}, skipping', flush=True)
        return torch.load(out, map_location='cpu', weights_only=False)
    Xf = X_real.numpy()
    mean = Xf.mean(axis=0)
    cov = np.cov(Xf, rowvar=False)
    X_gauss = torch.from_numpy(
        np.random.RandomState(0).multivariate_normal(
            mean, cov, size=N_GAUSS)).float()
    blob = {'X_gauss': X_gauss}
    torch.save(blob, out)
    print(f'[gauss] X_gauss {tuple(X_gauss.shape)}  '
          f'mean||a||={X_gauss.norm(dim=-1).mean():.2f}  saved {out}',
          flush=True)
    return blob


def optimize_batched(model, points, contexts, layer, device, chunk, out_dir):
    """Same objective/semantics as intrinsic_dim_jac_optimized.optimize_points,
    batched in `chunk`-sized groups with gradient accumulation."""
    ckpt_path = os.path.join(out_dir, 'jac_optimized_ckpt.pt')
    dtype = next(model.parameters()).dtype
    N, d = points.shape
    x0 = points.clone().to(device)
    start_step = 0
    jac_hist, dist_hist = [], []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        x = ck['x'].to(device).requires_grad_(True)
        opt_state = ck['opt_state']
        start_step = ck['step']
        jac_hist, dist_hist = ck['jac_hist'], ck['dist_hist']
        print(f'[optimize] resuming from step {start_step}', flush=True)
    else:
        x = points.clone().to(device).requires_grad_(True)
        opt_state = None
    optimizer = torch.optim.Adam([x], lr=LR)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    ctx_dev = contexts.to(device, dtype)

    for step in range(start_step, N_STEPS):
        optimizer.zero_grad()
        tot_jac, tot_dist = 0.0, 0.0
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            xc = x[st:ed]
            hidden = torch.cat([ctx_dev[st:ed],
                                xc.unsqueeze(1).to(dtype)], dim=1)
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.MATH):
                f = forward_batch_layer_to_final(model, hidden, layer)
            frob_sq = torch.zeros(ed - st, device=device)
            for _ in range(N_PROBES_OPT):
                u = torch.randn(ed - st, f.shape[-1], device=device,
                                dtype=f.dtype)
                s = (f * u).sum()
                gc = torch.autograd.grad(s, xc, create_graph=True,
                                         retain_graph=True)[0]
                frob_sq = frob_sq + (gc.float() ** 2).sum(dim=-1)
            frob_sq = frob_sq / N_PROBES_OPT
            jac_loss = torch.sqrt(frob_sq + 1e-8)
            dist_loss = LAM * ((xc - x0[st:ed]) ** 2).sum(dim=-1)
            loss = (jac_loss + dist_loss).sum() / N
            loss.backward()
            tot_jac += jac_loss.sum().item()
            tot_dist += dist_loss.sum().item()
        optimizer.step()
        jac_hist.append(tot_jac / N)
        dist_hist.append(tot_dist / N)
        if step % 5 == 0:
            move = (x.detach() - x0).norm(dim=-1).mean().item()
            print(f'  step {step}: jac={tot_jac/N:.1f} '
                  f'dist_loss={tot_dist/N:.2f} avg_move={move:.2f}',
                  flush=True)
        if (step + 1) % CKPT_EVERY == 0:
            torch.save({'x': x.detach().cpu(),
                        'opt_state': optimizer.state_dict(),
                        'step': step + 1, 'jac_hist': jac_hist,
                        'dist_hist': dist_hist}, ckpt_path)
            print(f'  [checkpoint at step {step + 1}]', flush=True)
    return x.detach().float().cpu(), {'jac': jac_hist, 'dist': dist_hist}


def stage_optimize(model, device, layer, gauss, real, chunk, out_dir):
    out = os.path.join(out_dir, 'jac_optimized.pt')
    if os.path.exists(out):
        print(f'[optimize] found {out}, skipping', flush=True)
        return torch.load(out, map_location='cpu', weights_only=False)
    gstart = gauss['X_gauss'][:N_OPT]
    ctx = real['contexts'][:N_OPT]
    print(f'[optimize] {N_OPT} points, {N_STEPS} steps, lr={LR}, lam={LAM}, '
          f'chunk={chunk}', flush=True)
    X_opt, history = optimize_batched(model, gstart, ctx, layer, device,
                                      chunk, out_dir)
    blob = {'X_opt': X_opt, 'X_gauss_subset': gstart, 'history': history,
            'n_steps': N_STEPS, 'lr': LR, 'lam': LAM,
            'n_probes': N_PROBES_OPT}
    torch.save(blob, out)
    print(f'[optimize] saved {out}', flush=True)
    return blob


def jac_norms_autograd(model, contexts, points, layer, device, chunk=32):
    """Batched autograd Hutchinson ||J||_F (first-order only, no
    create_graph) — numerically safe under bfloat16, unlike finite
    differences."""
    dtype = next(model.parameters()).dtype
    N = points.shape[0]
    out = torch.zeros(N)
    for st in range(0, N, chunk):
        ed = min(N, st + chunk)
        ctx = contexts[st:ed].to(device, dtype)
        x = points[st:ed].to(device, dtype).requires_grad_(True)
        hidden = torch.cat([ctx, x.unsqueeze(1)], dim=1)
        with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH):
            f = forward_batch_layer_to_final(model, hidden, layer)
        ns = torch.zeros(ed - st, device=device)
        for k in range(N_PROBES_EVAL):
            u = torch.randn(ed - st, f.shape[-1], device=device,
                            dtype=f.dtype)
            g = torch.autograd.grad((f * u).sum(), x,
                                    retain_graph=(k < N_PROBES_EVAL - 1))[0]
            ns = ns + (g.float() ** 2).sum(dim=-1)
        out[st:ed] = (ns / N_PROBES_EVAL).sqrt().cpu()
    return out


def stage_analyze(model, device, layer, real, gauss, optb, chunk, out_dir,
                  model_name):
    out_pt = os.path.join(out_dir, 'shift_anchors.pt')
    out_pkl = os.path.join(out_dir, 'jac_norms.pkl')
    if os.path.exists(out_pkl):
        print(f'[analyze] found {out_pkl}, skipping', flush=True)
        return
    X_gstart = optb['X_gauss_subset'].float()
    X_opt = optb['X_opt'].float()
    X = np.concatenate([X_gstart.numpy(), X_opt.numpy()])
    y = np.concatenate([np.zeros(N_OPT), np.ones(N_OPT)])
    clf = LogisticRegression(C=0.1, max_iter=5000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    w = torch.from_numpy(clf.coef_[0]).float()
    b = float(clf.intercept_[0])
    w_hat = w / w.norm()
    c = float(((X_opt - X_gstart) @ w_hat).mean())
    print(f'[analyze] logistic train acc={acc:.3f}  c={c:.2f}  '
          f'(mean total displacement '
          f'{(X_opt - X_gstart).norm(dim=-1).mean():.2f})', flush=True)

    G = gauss['X_gauss'][N_OPT:2 * N_OPT]
    G_shift = G + c * w_hat
    signed = (G @ w) + b
    G_mirror = G - 2 * (signed / (w.norm() ** 2)).unsqueeze(-1) * w
    torch.save({'X_shift': G_shift, 'X_fresh': G, 'X_mirror': G_mirror,
                'w': w, 'b': b, 'c': c}, out_pt)
    print(f'[analyze] saved {out_pt}', flush=True)

    contexts = real['contexts'][:N_OPT]
    conditions = {
        'real': real['X_real'][:N_OPT],
        'gauss fresh': G,
        'gauss + c*w': G_shift,
        'gauss mirrored': G_mirror,
        'jac-opt': X_opt,
    }
    jac = {}
    for name, pts in conditions.items():
        print(f'[analyze] ||J||_F: {name}', flush=True)
        jac[name] = jac_norms_autograd(model, contexts, pts, layer, device,
                                       chunk=max(chunk, 8))
    print(f'\n=== {model_name} ||J||_F summary (autograd Hutchinson) ===',
          flush=True)
    for name, v in jac.items():
        print(f'  {name:16s} mean={v.mean():.1f}  median={v.median():.1f}  '
              f'mean||a||={conditions[name].norm(dim=-1).mean():.1f}',
              flush=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump({'jac': {k: v.numpy() for k, v in jac.items()},
                     'w': w.numpy(), 'b': b, 'c': c, 'train_acc': acc,
                     'estimator': 'autograd_hutchinson',
                     'norms': {k: v.norm(dim=-1).numpy()
                               for k, v in conditions.items()}}, f)

    colors = {'real': 'gray', 'gauss fresh': '#4477aa',
              'gauss + c*w': '#ee7733', 'gauss mirrored': '#228833',
              'jac-opt': '#aa3377'}
    hi = max(float(v.max()) for v in jac.values())
    bins = np.linspace(0, hi * 1.02, 50)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, v in jac.items():
        ax.hist(v.numpy(), bins=bins, density=True, histtype='step',
                lw=1.8, color=colors[name], label=name)
    ax.set_xlabel(r'$\|J\|_F$ (autograd Hutchinson, L{}$\to$final)'
                  .format(layer))
    ax.set_ylabel('density')
    ax.set_title(f'{model_name} L{layer}: Jacobian norm after linear maps '
                 f'along the separating normal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'jac_hist.png'), dpi=150,
                bbox_inches='tight')
    print(f'[analyze] saved {out_pkl} and jac_hist.png', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layer', type=int, default=2)
    ap.add_argument('--dtype', default='float32',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--chunk', type=int, default=10,
                    help='points per double-backprop chunk (memory bound)')
    ap.add_argument('--out', default=None,
                    help='default results/intrinsic_dim/<model>_shift')
    args = ap.parse_args()
    out_dir = args.out or f'results/intrinsic_dim/{args.model}_shift'
    os.makedirs(out_dir, exist_ok=True)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f'Loading {args.model} ({args.dtype})...', flush=True)
    model, tokenizer, device = load_model(args.model,
                                          dtype=dtype_map[args.dtype])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    real = stage_extract(model, tokenizer, device, args.layer, out_dir)
    gauss = stage_gauss(real['X_real'], out_dir)
    optb = stage_optimize(model, device, args.layer, gauss, real,
                          args.chunk, out_dir)
    stage_analyze(model, device, args.layer, real, gauss, optb, args.chunk,
                  out_dir, args.model)
    print('\nPipeline done.', flush=True)


if __name__ == '__main__':
    main()
