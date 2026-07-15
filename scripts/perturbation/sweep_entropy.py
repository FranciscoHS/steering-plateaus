#!/usr/bin/env python3
"""Entropy-vs-angle companion to run_feature_vs_random_sweep.py.

Same spherical perturbation (rotate the layer-L residual toward a direction by
angle theta), but records the Shannon entropy (nats) of the model's next-token
distribution softmax(final logits) at the last position — averaged over n_act
activations — as a function of angle. Compares supplied directions (atoms,
contrastive) against random orthogonal directions. Angle 0 = unperturbed.
"""
import sys, os, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_final, _get_final_norm, _get_lm_head
from src.data import load_fineweb_fixed_length
from scripts.perturbation.run_feature_vs_random_sweep import project_orthogonal


def entropy_curve(model, context, activation, v_perp, angles_rad, layer, device, ln, head):
    md = next(model.parameters()).dtype
    ctx = context.unsqueeze(0).to(device=device, dtype=md)
    a = activation.to(device=device, dtype=md)
    R = a.norm(); a_unit = a / R
    vp = v_perp.to(device=device, dtype=md)
    ent = np.zeros(len(angles_rad))
    for i, theta in enumerate(angles_rad):
        perturbed = (R * (np.cos(theta) * a_unit + np.sin(theta) * vp)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pf = forward_from_layer_to_final(model, ctx, perturbed, layer)
            logits = head(ln(pf.unsqueeze(1)).squeeze(1))
            lp = F.log_softmax(logits.float(), dim=-1)
            ent[i] = -(lp.exp() * lp).sum(-1).item()
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--n_act', type=int, default=50)
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_angles', type=int, default=40)
    ap.add_argument('--direction_pkls', nargs='+', required=True)
    ap.add_argument('--dir_labels', nargs='+', default=None)
    ap.add_argument('--n_random_dirs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)   # include 0
    angles_rad = np.deg2rad(angles_deg)

    model, tokenizer, device = load_model(args.model)
    ln, head = _get_final_norm(model), _get_lm_head(model)

    # named directions
    named = {}
    for i, p in enumerate(args.direction_pkls):
        v = pickle.load(open(p, 'rb'))['directions'][0].float()
        v = v / v.norm()
        label = (args.dir_labels[i] if args.dir_labels and i < len(args.dir_labels)
                 else os.path.splitext(os.path.basename(p))[0])
        named[label] = v

    # activations (same protocol as the sweep)
    print(f'Extracting {args.n_act} layer-{args.layer} activations', flush=True)
    toks = load_fineweb_fixed_length(args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    contexts, acts = [], []
    for i in range(0, len(toks), 4):
        bt = torch.stack(toks[i:i + 4]).to(device)
        with torch.no_grad():
            h = model(bt, output_hidden_states=True).hidden_states[args.layer + 1]
        contexts.append(h[:, :-1, :].float().cpu()); acts.append(h[:, -1, :].float().cpu())
    contexts = torch.cat(contexts, 0); activations = torch.cat(acts, 0)

    nA = len(angles_deg)
    ent = {'random': np.zeros((args.n_act, args.n_random_dirs, nA))}
    for name in named:
        ent[name] = np.zeros((args.n_act, nA))

    import time; t0 = time.time()
    for i in range(args.n_act):
        a_unit = activations[i] / activations[i].norm()
        rv = torch.randn(args.n_random_dirs, activations.shape[1])
        rv = rv - (rv @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        rv = F.normalize(rv, dim=-1)
        for d in range(args.n_random_dirs):
            ent['random'][i, d] = entropy_curve(model, contexts[i], activations[i], rv[d],
                                                 angles_rad, args.layer, device, ln, head)
        for name, vec in named.items():
            vp = project_orthogonal(vec, activations[i])
            ent[name][i] = entropy_curve(model, contexts[i], activations[i], vp,
                                         angles_rad, args.layer, device, ln, head)
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{args.n_act} ({el:.0f}s, ~{el/(i+1)*(args.n_act-i-1):.0f}s left)', flush=True)

    pickle.dump({'args': vars(args), 'angles_deg': angles_deg, 'entropy': ent},
                open(os.path.join(args.out, 'entropy_sweep.pkl'), 'wb'))

    # plot: median +/- IQR
    fig, ax = plt.subplots(figsize=(8, 5))
    rflat = ent['random'].reshape(-1, nA)
    ax.fill_between(angles_deg, np.percentile(rflat, 25, 0), np.percentile(rflat, 75, 0),
                    color='gray', alpha=0.25, label='random (IQR)')
    ax.plot(angles_deg, np.median(rflat, 0), color='gray', ls=':', lw=1.5)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(named))))
    for k, name in enumerate(named):
        ls = '--' if name.startswith('contrastive') else '-'
        lw = 2.4 if ls == '--' else 1.6
        ax.plot(angles_deg, np.median(ent[name], 0), ls=ls, lw=lw, color=cmap[k], label=name)
    ax.set_xlabel('angle (deg)'); ax.set_ylabel('next-token entropy (nats)')
    ax.set_title(f'{args.model} L{args.layer}: output entropy vs perturbation angle')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out, 'entropy_sweep.png')
    plt.savefig(out_png, dpi=150)
    print(f'\nPlot:    {out_png}\nResults: {os.path.join(args.out, "entropy_sweep.pkl")}')


if __name__ == '__main__':
    main()
