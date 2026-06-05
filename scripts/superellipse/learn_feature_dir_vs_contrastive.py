#!/usr/bin/env python3
"""
Find one feature direction by optimizing against a fixed contrastive scaffold.

Same proxy as learn_feature_dir_vs_random.py, but the K auxiliary directions
are diverse contrastive (diff-of-means) directions extracted from the
fsec-anonymous registry at the same layer the optimization runs at.

Per activation a, contrastive direction d_c, and angle alpha:
    r_f   = downstream L2 response to a spherical perturbation along d_f
    r_c   = response along d_c
    r_mix = response along the equal-mix bisector of d_f and d_c.

Loss:
    L(d_f) = mean_c mean_a mean_alpha   r_mix / (r_c + r_f)
             + contrast_orth_lam * mean_c (d_f . d_c)^2

L < 0.5 is the superellipse / p>2 (super-additive) regime; the hope is that
pushing d_f to interfere maximally against the contrastive scaffold pulls it
toward a real feature direction distinct from the scaffold.

Usage:
    python scripts/superellipse/learn_feature_dir_vs_contrastive.py \\
        --model gemma-2-9b --layer 15 --n_act 10 --angles 30
"""

import argparse
import contextlib
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, forward_from_layer_to_layer, _get_blocks
from src.data import load_fineweb_fixed_length

# Reuse the geometry + response helpers verbatim.
from scripts.superellipse.learn_superellipse_dict import (
    extract_context_and_activation, pair_loss, pair_responses,
)


def _find_fsec_path():
    """Resolve fsec-anonymous repo path: env var > ~/Research/... > /workspace/..."""
    env = os.environ.get('FSEC_PATH')
    if env and os.path.isdir(env):
        return env
    for p in [os.path.expanduser('~/Research/fsec-anonymous'),
              '/workspace/fsec-anonymous']:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Could not find fsec-anonymous repo. Set $FSEC_PATH or "
        "place it at ~/Research/fsec-anonymous or /workspace/fsec-anonymous.")


FSEC_PATH = _find_fsec_path()

DEFAULT_NAMES = ['Chinese', 'Python', 'Gender', 'Age', 'Sentiment', 'Tense',
                 'Number', 'Formal', 'Refusal', 'Honesty']

# Map src.model identifiers to fsec-anonymous registry target keys.
MODEL_TO_FSEC_TARGET = {
    'gemma-2-9b': 'gemma',
    'qwen3-1.7b': 'qwen',
    'llama-3.1-8b': 'llama',
    'mistral-7b-v0.3': 'mistral',
    'aya-expanse-8b': 'aya',
    'yi-1.5-9b': 'yi',
}


@contextlib.contextmanager
def _chdir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def load_or_build_scaffold(model, tokenizer, device, model_name, layer, names):
    """Load contrastive scaffold from fsec-anonymous, building the cache if needed.

    Calls fsec-anonymous scripts.lib.directions.extract_all, which handles
    on-disk caching at <fsec>/results/directions/dirs_<target>_L<layer>.pkl.
    Only the requested names are guaranteed to be present (extract_all extends
    an existing cache with missing names rather than rebuilding from scratch).

    Returns scaffold tensor [K, d] (rows in the order of `names`).
    """
    target = MODEL_TO_FSEC_TARGET.get(model_name)
    if target is None:
        raise ValueError(
            f"No fsec-anonymous target mapping for model {model_name!r}. "
            f"Add it to MODEL_TO_FSEC_TARGET.")

    # fsec-anonymous's registry uses `import data.*_prompts`, which collides
    # with steering-plateaus' own top-level data/ if our path is searched
    # first. Insert fsec at sys.path[0] so its data/ wins.
    if FSEC_PATH not in sys.path:
        sys.path.insert(0, FSEC_PATH)

    # `scripts` and `data` are already cached in sys.modules from
    # steering-plateaus' own packages, so a naive `from scripts.lib import ...`
    # resolves to our repo (which has no scripts/lib). Pop the cached entries
    # around the fsec import, then restore steering-plateaus' versions.
    cached = {k: sys.modules.pop(k) for k in list(sys.modules)
              if k == 'scripts' or k.startswith('scripts.')
              or k == 'data' or k.startswith('data.')}
    try:
        # OUT_DIR in fsec's directions module is module-level and relative;
        # chdir before first import so both the makedirs and the pkl path
        # resolve under the fsec-anonymous repo, not under our CWD.
        with _chdir(FSEC_PATH):
            from scripts.lib import directions as fsec_directions  # noqa: E402
            print(f"[scaffold] target={target} layer={layer} names={names}")
            blob = fsec_directions.extract_all(
                model, tokenizer, device, target=target,
                layer=layer, names=names)
    finally:
        # Drop fsec's modules and restore steering-plateaus' originals.
        for k in list(sys.modules):
            if (k == 'scripts' or k.startswith('scripts.')
                    or k == 'data' or k.startswith('data.')):
                del sys.modules[k]
        sys.modules.update(cached)

    dirs = []
    for n in names:
        if n not in blob['directions']:
            raise KeyError(f"Direction {n!r} not in fsec cache after extract_all")
        v = blob['directions'][n].to(device).float()
        v = v / v.norm()
        dirs.append(v)
    return torch.stack(dirs)  # [K, d]


def optimize_feature_dir(model, acts, ctxs, f_refs, scaffold,
                         angles_rad, layer, measure_layer, device, d_model,
                         n_steps, lr, contrast_orth_lam, seed,
                         log_prefix=""):
    """Optimize one feature direction d_f against the contrastive scaffold.

    scaffold: [K, d] frozen unit contrastive directions.

    Loss = mean_c pair_loss(d_f, d_c) + contrast_orth_lam * mean_c (d_f . d_c)^2
    """
    g = torch.Generator(device=device).manual_seed(seed)
    d_f = torch.randn(d_model, generator=g, device=device)
    d_f = (d_f / d_f.norm()).requires_grad_(True)
    optimizer = torch.optim.Adam([d_f], lr=lr)

    K = scaffold.shape[0]
    scaffold_list = [scaffold[k] for k in range(K)]

    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()

        # Superellipse ratio against each scaffold direction.
        resp_terms = []
        mr_f = mr_c = mr_mix = 0.0
        for d_c in scaffold_list:
            rl, (a_f, a_c, a_mix) = pair_loss(
                model, acts, ctxs, f_refs, d_f, d_c,
                angles_rad, layer, measure_layer)
            resp_terms.append(rl)
            mr_f += a_f; mr_c += a_c; mr_mix += a_mix
        resp_loss = torch.stack(resp_terms).mean()
        mr_f /= K; mr_c /= K; mr_mix /= K

        # Penalty: push d_f away from each scaffold direction.
        d_fu = d_f / (d_f.norm() + 1e-10)
        contrast_overlap = torch.stack([
            (d_fu @ d_c) ** 2 for d_c in scaffold_list
        ]).mean()

        loss = resp_loss + contrast_orth_lam * contrast_overlap
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            d_f.data.div_(d_f.data.norm())

        loss_history.append({
            'step': step, 'loss': loss.item(), 'resp_loss': resp_loss.item(),
            'contrast_overlap': contrast_overlap.item(),
            'R_f': mr_f, 'R_c': mr_c, 'R_mix': mr_mix,
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {log_prefix}step {step:4d}  L={loss.item():.4f}  "
                  f"ratio={resp_loss.item():.4f}  "
                  f"ov={contrast_overlap.item():.4f}  "
                  f"R_f={mr_f:.3f} R_c={mr_c:.3f} R_mix={mr_mix:.3f}")

    return d_f.detach(), loss_history


def eval_per_scaffold(model, acts, ctxs, f_refs, d_f, scaffold,
                      angles_rad, layer, measure_layer):
    """Per-scaffold mean r_f / r_c / r_mix / L across activations and angles.

    Returns array [K, 4] with columns (R_f, R_c, R_mix, L).
    """
    K = scaffold.shape[0]
    out = np.zeros((K, 4))
    for k in range(K):
        r_fs, r_cs, r_mixs, Ls = [], [], [], []
        for i in range(len(acts)):
            for angle_rad in angles_rad:
                with torch.no_grad():
                    r_f, r_c, r_mix = pair_responses(
                        model, acts[i], ctxs[i], f_refs[i],
                        d_f, scaffold[k], angle_rad, layer, measure_layer)
                r_fs.append(r_f.item()); r_cs.append(r_c.item())
                r_mixs.append(r_mix.item())
                Ls.append(r_mix.item() / (r_f.item() + r_c.item() + 1e-8))
        out[k] = [np.mean(r_fs), np.mean(r_cs), np.mean(r_mixs), np.mean(Ls)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemma-2-9b')
    parser.add_argument('--layer', type=int, default=15,
                        help='Perturbation + scaffold extraction layer')
    parser.add_argument('--measure_layer', type=int, default=None,
                        help='Layer to measure L2 response (default: 2nd-to-last)')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_act', type=int, default=10)
    parser.add_argument('--scaffold_names', type=str, nargs='+',
                        default=DEFAULT_NAMES,
                        help='Contrastive direction names (fsec-anonymous registry)')
    parser.add_argument('--scaffold_pkl', type=str, default=None,
                        help='Explicit scaffold pkl path (skips fsec build).'
                             ' Must contain {"directions": {name: tensor}, ...}.')
    parser.add_argument('--angles', type=float, nargs='+', default=[30.0],
                        help='Perturbation angle(s) in degrees')
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--contrast_orth_lam', type=float, default=1.0,
                        help='Penalty on (d_f . d_c)^2 against scaffold dirs')
    parser.add_argument('--data_source', type=str, default='fineweb')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Model dtype (bfloat16 default for 9B models)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    angle_tag = '-'.join(f'{a:g}' for a in args.angles)
    if args.results_dir is None:
        args.results_dir = (f'results/feature_dir_vs_contrastive/'
                            f'{args.model}_L{args.layer}_a{angle_tag}')
    os.makedirs(args.results_dir, exist_ok=True)

    angles_rad = [np.deg2rad(a) for a in args.angles]

    # Load model
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}
    print(f"Loading model: {args.model} ({args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=dtype_map[args.dtype])
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = (args.measure_layer if args.measure_layer is not None
                     else n_layers - 2)
    print(f"Device: {device}, d_model: {d_model}, n_layers: {n_layers}, "
          f"perturb_layer: {args.layer}, measure_layer: {measure_layer}")
    print(f"Config: n_act={args.n_act}, scaffold={args.scaffold_names}, "
          f"angles={args.angles} deg, n_steps={args.n_steps}, lr={args.lr}, "
          f"contrast_orth_lam={args.contrast_orth_lam}")

    # Data + activations
    print(f"Loading {args.n_act} {args.data_source} sequences (seq_len={args.seq_len})...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, token_ids_list, args.layer, device)
    model_dtype_actual = next(model.parameters()).dtype
    acts = [all_activations[i].to(device=device, dtype=torch.float32)
            for i in range(args.n_act)]
    ctxs = [all_contexts[i].unsqueeze(0).to(device=device, dtype=model_dtype_actual)
            for i in range(args.n_act)]
    print(f"Mean activation norm: {torch.stack(acts).norm(dim=-1).mean().item():.2f}")

    # Precompute unperturbed downstream response f(a) for each activation.
    f_refs = []
    with torch.no_grad():
        for i in range(args.n_act):
            point = acts[i].reshape(1, 1, d_model).to(model_dtype_actual)
            f_ref = forward_from_layer_to_layer(
                model, ctxs[i], point, args.layer, measure_layer).squeeze(0).float()
            f_refs.append(f_ref)

    # Contrastive scaffold
    if args.scaffold_pkl is not None:
        print(f"Loading scaffold from explicit pkl: {args.scaffold_pkl}")
        with open(args.scaffold_pkl, 'rb') as f:
            blob = pickle.load(f)
        dirs = []
        for n in args.scaffold_names:
            v = blob['directions'][n].to(device).float()
            dirs.append(v / v.norm())
        scaffold = torch.stack(dirs)
    else:
        scaffold = load_or_build_scaffold(
            model, tokenizer, device, args.model, args.layer,
            args.scaffold_names)
    print(f"Scaffold shape: {tuple(scaffold.shape)}")
    pre_gram = (scaffold @ scaffold.T).cpu().numpy()
    print(f"Scaffold pairwise |cos| (off-diag) mean: "
          f"{np.abs(pre_gram - np.eye(pre_gram.shape[0])).mean():.3f}")

    # ---- Optimize d_f ----
    print(f"\n=== Optimizing d_f against {len(args.scaffold_names)} contrastive dirs ===")
    t0 = time.time()
    d_f, hist = optimize_feature_dir(
        model, acts, ctxs, f_refs, scaffold,
        angles_rad, args.layer, measure_layer, device, d_model,
        args.n_steps, args.lr, args.contrast_orth_lam, args.seed,
        log_prefix="d_f ")
    print(f"  done in {time.time()-t0:.1f}s")

    # Checkpoint d_f before diagnostics (cheap insurance).
    torch.save({'d_f': d_f.cpu(), 'scaffold': scaffold.cpu(),
                'scaffold_names': args.scaffold_names, 'args': vars(args)},
               os.path.join(args.results_dir, 'checkpoint.pkl'))

    # Per-scaffold diagnostics
    print(f"\n=== Per-scaffold-direction diagnostics ===")
    per_scaffold = eval_per_scaffold(
        model, acts, ctxs, f_refs, d_f, scaffold,
        angles_rad, args.layer, measure_layer)
    overlaps = (d_f.unsqueeze(0) @ scaffold.T).squeeze(0).cpu().numpy()
    for k, name in enumerate(args.scaffold_names):
        r_f, r_c, r_mix, L = per_scaffold[k]
        print(f"  {name:12s}  L={L:.4f}  R_f={r_f:.3f}  R_c={r_c:.3f}  "
              f"R_mix={r_mix:.3f}  (d_f.d_c)={overlaps[k]:+.3f}")

    # ---- Save ----
    results = {
        'args': vars(args),
        'd_f': d_f.cpu(),
        'scaffold': scaffold.cpu(),
        'scaffold_names': args.scaffold_names,
        'measure_layer': measure_layer,
        'angles_deg': args.angles,
        'loss_history': hist,
        'per_scaffold': per_scaffold,        # [K, 4] = R_f, R_c, R_mix, L
        'overlaps': overlaps,                # [K] = d_f . d_c
        'scaffold_gram': pre_gram,
    }
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    K = len(args.scaffold_names)
    x = np.arange(K)

    # Loss curves
    ax = axes[0, 0]
    steps = [h['step'] for h in hist]
    ax.plot(steps, [h['loss'] for h in hist], label='total')
    ax.plot(steps, [h['resp_loss'] for h in hist], '--', alpha=0.7, label='ratio')
    ax.plot(steps, [h['contrast_overlap'] for h in hist], ':', alpha=0.7,
            label='mean (d_f.d_c)^2')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='0.5 (no interference)')
    ax.set_xlabel('step'); ax.set_ylabel('value'); ax.set_title('Optimization')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Per-scaffold L
    ax = axes[0, 1]
    ax.bar(x, per_scaffold[:, 3], color='C0')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(args.scaffold_names, rotation=45, ha='right')
    ax.set_ylabel('mean L'); ax.set_title('Per-scaffold L = R_mix/(R_f+R_c)')
    ax.grid(True, alpha=0.3)

    # Per-scaffold response bars
    ax = axes[1, 0]
    ax.bar(x - 0.25, per_scaffold[:, 0], width=0.25, label='R_f', color='C0')
    ax.bar(x,        per_scaffold[:, 1], width=0.25, label='R_c', color='C1')
    ax.bar(x + 0.25, per_scaffold[:, 2], width=0.25, label='R_mix', color='C2')
    ax.set_xticks(x); ax.set_xticklabels(args.scaffold_names, rotation=45, ha='right')
    ax.set_ylabel('mean L2 response'); ax.set_title('Per-scaffold mean responses')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Overlaps |d_f . d_c|
    ax = axes[1, 1]
    colors = ['C3' if v < 0 else 'C0' for v in overlaps]
    ax.bar(x, overlaps, color=colors)
    ax.axhline(0, color='gray', ls=':', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(args.scaffold_names, rotation=45, ha='right')
    ax.set_ylabel('d_f . d_c'); ax.set_title('Final overlap with each scaffold dir')
    ax.set_ylim(-1, 1); ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.model} L{args.layer}->{measure_layer}, '
                 f'1 feature dir vs {K} contrastive, '
                 f'alpha={args.angles} deg', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved results: {os.path.join(args.results_dir, 'results.pkl')}")
    print(f"Saved plot: {plot_path}")


if __name__ == '__main__':
    main()
