#!/usr/bin/env python3
"""Replicate the separating-normal-shift finding on gpt2-xl (perturb L2).

Pipeline (each stage checkpoints to OUT_DIR and is skipped if its output
already exists, so the script is resumable after spot interruptions):

  1. extract   5000 real L2 activations + contexts (FineWeb, seq_len 10)
  2. gauss     mean+cov matched Gaussian samples (n=1000, seed 0)
  3. optimize  jac-norm optimization of Gaussian points 0:200
               (same objective/hyperparams as intrinsic_dim_jac_optimized.py:
               min ||J||_F + lam*||x-x0||^2, Adam lr=0.1, lam=1e-3,
               200 steps, 5 Hutchinson probes — but batched/chunked instead
               of per-point, for speed)
  4. analyze   logistic plane (gauss-start vs jac-opt) -> w_hat, shift c;
               build gauss+c*w_hat from fresh rows 200:400; measure
               finite-difference ||J||_F for real / gauss / shifted /
               mirrored / jac-opt

After this, run scripts/superellipse/eval_real_vs_random_acts.py on
gpt2-xl L2 with the saved anchors as extra conditions.

Outputs in results/intrinsic_dim/gpt2xl_shift/:
  real_data.pt, gauss_data.pt, jac_optimized.pt (+ _ckpt.pt during runs),
  shift_anchors.pt, jac_norms.pkl, jac_hist.png
"""

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

from src.model import load_model, forward_from_layer_to_final
from intrinsic_dim_jac_filtered import (estimate_jac_norms_batched,
                                        forward_batch_layer_to_final)

OUT_DIR = 'results/intrinsic_dim/gpt2xl_shift'
MODEL = 'gpt2-xl'
SOURCE_LAYER = 2
SEQ_LEN = 10
N_REAL = 5000
N_GAUSS = 1000
N_OPT = 200          # points to optimize (gauss rows 0:200)
N_STEPS = 200
LR = 0.1
LAM = 1e-3
N_PROBES_OPT = 5     # Hutchinson probes during optimization
N_PROBES_EVAL = 10   # probes for the final finite-difference measurement
CHUNK = 10           # points per double-backprop chunk (memory bound:
                     # ~6 forward-graphs per chunk; 50 OOMed a 24GB 4090)
CKPT_EVERY = 20


def stage_extract(model, tokenizer, device):
    out = os.path.join(OUT_DIR, 'real_data.pt')
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
            # keep only rows with a full SEQ_LEN of real tokens (no padding —
            # GPT-2 pads on the right, which would corrupt the last token)
            full = tokens['attention_mask'].sum(-1) == SEQ_LEN
            with torch.no_grad():
                o = model(**tokens, output_hidden_states=True)
            h = o.hidden_states[SOURCE_LAYER + 1][full]
            acts.append(h[:, -1, :].cpu())
            ctxs.append(h[:, :-1, :].cpu())
            n_kept += int(full.sum())
            batch_texts = []
            if n_kept >= N_REAL:
                break
    X_real = torch.cat(acts)[:N_REAL].float()
    contexts = torch.cat(ctxs)[:N_REAL].float()
    blob = {'X_real': X_real, 'contexts': contexts}
    torch.save(blob, out)
    print(f'[extract] X_real {tuple(X_real.shape)}  '
          f'mean||a||={X_real.norm(dim=-1).mean():.2f}  saved {out}',
          flush=True)
    return blob


def stage_gauss(X_real):
    out = os.path.join(OUT_DIR, 'gauss_data.pt')
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


def optimize_batched(model, points, contexts, device):
    """Batched version of intrinsic_dim_jac_optimized.optimize_points.

    Same objective and semantics (per-point Hutchinson sqrt-Frobenius loss,
    mean over points, one Adam step per iteration); points are processed in
    CHUNK-sized groups with gradient accumulation instead of one at a time.
    Resumes from the checkpoint if present.
    """
    ckpt_path = os.path.join(OUT_DIR, 'jac_optimized_ckpt.pt')
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
    ctx_dev = contexts.to(device)

    for step in range(start_step, N_STEPS):
        optimizer.zero_grad()
        tot_jac, tot_dist = 0.0, 0.0
        for st in range(0, N, CHUNK):
            ed = min(N, st + CHUNK)
            xc = x[st:ed]
            hidden = torch.cat([ctx_dev[st:ed], xc.unsqueeze(1)], dim=1)
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.MATH):
                f = forward_batch_layer_to_final(model, hidden, SOURCE_LAYER)
            frob_sq = torch.zeros(ed - st, device=device)
            for _ in range(N_PROBES_OPT):
                u = torch.randn(ed - st, d, device=device)
                s = (f * u).sum()
                gc = torch.autograd.grad(s, xc, create_graph=True,
                                         retain_graph=True)[0]
                frob_sq = frob_sq + (gc ** 2).sum(dim=-1)
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
    return x.detach().cpu(), {'jac': jac_hist, 'dist': dist_hist}


def stage_optimize(model, device, gauss, real):
    out = os.path.join(OUT_DIR, 'jac_optimized.pt')
    if os.path.exists(out):
        print(f'[optimize] found {out}, skipping', flush=True)
        return torch.load(out, map_location='cpu', weights_only=False)
    gstart = gauss['X_gauss'][:N_OPT]
    ctx = real['contexts'][:N_OPT]
    print(f'[optimize] {N_OPT} points, {N_STEPS} steps, lr={LR}, lam={LAM}, '
          f'chunk={CHUNK}', flush=True)
    X_opt, history = optimize_batched(model, gstart, ctx, device)
    blob = {'X_opt': X_opt, 'X_gauss_subset': gstart, 'history': history,
            'n_steps': N_STEPS, 'lr': LR, 'lam': LAM,
            'n_probes': N_PROBES_OPT}
    torch.save(blob, out)
    print(f'[optimize] saved {out}', flush=True)
    return blob


def stage_analyze(model, device, real, gauss, optb):
    out_pt = os.path.join(OUT_DIR, 'shift_anchors.pt')
    out_pkl = os.path.join(OUT_DIR, 'jac_norms.pkl')
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
        jac[name] = estimate_jac_norms_batched(
            model, contexts, pts, SOURCE_LAYER, n_probes=N_PROBES_EVAL)
    print('\n=== gpt2-xl ||J||_F summary ===', flush=True)
    for name, v in jac.items():
        print(f'  {name:16s} mean={v.mean():.1f}  median={v.median():.1f}  '
              f'mean||a||={conditions[name].norm(dim=-1).mean():.1f}',
              flush=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump({'jac': {k: v.numpy() for k, v in jac.items()},
                     'w': w.numpy(), 'b': b, 'c': c, 'train_acc': acc,
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
    ax.set_xlabel(r'$\|J\|_F$ (Hutchinson, L2$\to$final)')
    ax.set_ylabel('density')
    ax.set_title('gpt2-xl L2: Jacobian norm after linear maps along the '
                 'separating normal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'jac_hist.png'), dpi=150,
                bbox_inches='tight')
    print(f'[analyze] saved {out_pkl} and jac_hist.png', flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading {MODEL} (float32)...', flush=True)
    model, tokenizer, device = load_model(MODEL, dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    real = stage_extract(model, tokenizer, device)
    gauss = stage_gauss(real['X_real'])
    optb = stage_optimize(model, device, gauss, real)
    stage_analyze(model, device, real, gauss, optb)
    print('\nPipeline done.', flush=True)


if __name__ == '__main__':
    main()
