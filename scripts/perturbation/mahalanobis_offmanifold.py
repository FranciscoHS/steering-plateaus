#!/usr/bin/env python3
"""How far off-manifold do different direction families push activations?

Rough proxy: at a matched perturbation (geodesic rotation by angle theta toward
a unit direction d), measure the Mahalanobis distance of the rotated activation
to the natural activation distribution at the same layer. A direction that points
along low-variance axes of Sigma racks up large Mahalanobis distance -> "more off
manifold". Since the rotation is norm-preserving, every family moves each anchor
by the *same* chord length; the only thing that differs across families is which
subspace d points into, so the comparison isolates covariance geometry.

Three direction families:
  - contrastive : difference-of-means language directions
  - optimized   : superellipse dictionary atoms (response term on, alignment off)
  - random      : isotropic Gaussian unit vectors

Measured at the perturbation layer (L14) using a Ledoit-Wolf shrinkage estimate
of Sigma. No model forward passes needed.
"""
import os, sys, json, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
from sklearn.covariance import LedoitWolf


def unit(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def load_natural(cache_path):
    meta = json.load(open(cache_path + '.meta.json'))
    n, d = meta['shape']
    X = np.array(np.memmap(cache_path, dtype=meta['dtype'], mode='r', shape=(n, d)),
                 dtype=np.float64)
    return X


def load_families(dict_path, contrastive_path, sae_path, n_random, d, seed=0,
                  dict_align_path=None, extra_dicts=None):
    fam = {}
    # optimized: dictionary atoms (response term on, alignment off)
    blob = torch.load(dict_path, map_location='cpu', weights_only=False)
    D = blob['dictionary'].float().numpy().astype(np.float64)
    fam['optimized'] = unit(D)
    # optimized + align: response term on AND corpus-presence alignment term on.
    # These dicts have ~40k atoms; sample n_random to match the other families.
    if dict_align_path:
        ba = torch.load(dict_align_path, map_location='cpu', weights_only=False)
        Da = ba['dictionary'].float().numpy().astype(np.float64)
        rng_a = np.random.default_rng(seed + 1)
        sel = rng_a.choice(Da.shape[0], size=min(n_random, Da.shape[0]), replace=False)
        fam['optimized + align'] = unit(Da[sel])
    # contrastive: language difference-of-means
    c = pickle.load(open(contrastive_path, 'rb'))
    C = np.stack([c['features'][k].float().numpy() for k in c['languages']]).astype(np.float64)
    fam['contrastive'] = unit(C)
    # random: isotropic Gaussian unit vectors, matched count to optimized
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_random, d))
    fam['random'] = unit(R)
    # SAE latents: random decoder columns (W_dec is [d, n_latents])
    if sae_path:
        sae = torch.load(sae_path, map_location='cpu', weights_only=True)
        W_dec = sae['W_dec'].float().numpy().astype(np.float64)  # (d, n_lat)
        if W_dec.shape[0] != d:
            W_dec = W_dec.T
        lat_idx = rng.choice(W_dec.shape[1], size=min(n_random, W_dec.shape[1]), replace=False)
        fam['SAE latent'] = unit(W_dec[:, lat_idx].T)
    # arbitrary extra dictionaries: list of (label, path); sample n_random atoms
    for label, path in (extra_dicts or []):
        bx = torch.load(path, map_location='cpu', weights_only=False)
        Dx = bx['dictionary'].float().numpy().astype(np.float64)
        n_done = bx.get('n_done', Dx.shape[0])
        Dx = Dx[:n_done]
        rng_x = np.random.default_rng(seed + 7)
        sx = rng_x.choice(Dx.shape[0], size=min(n_random, Dx.shape[0]), replace=False)
        fam[label] = unit(Dx[sx])
    return fam


def geodesic_rotate(a, dvec, theta):
    """Rotate each anchor a[i] (n,d) toward unit direction dvec (d,) by angle theta
    (radians), norm-preserving, in the plane spanned by a-hat and d. Returns (n,d)."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    ahat = a / np.maximum(norms, 1e-12)
    # component of d orthogonal to each ahat
    proj = (ahat @ dvec)[:, None]                # (n,1)
    e = dvec[None, :] - proj * ahat              # (n,d)
    e = unit(e, axis=1)
    rot = np.cos(theta) * ahat + np.sin(theta) * e
    return rot * norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/mahalanobis_offmanifold/qwen3-1.7b_L14_fineweb_n10000_seq5.npy')
    ap.add_argument('--dict_path', default='results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_resample/dictionary.pt')
    ap.add_argument('--contrastive_path', default='results/allpairs_contrastive/qwen3-1.7b/directions.pkl')
    ap.add_argument('--sae_path', default='results/dictionary/qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt')
    ap.add_argument('--dict_align_path', default=None,
                    help='dictionary built with response AND alignment terms; '
                         'sampled to n_random atoms -> "optimized + align" family')
    ap.add_argument('--extra_dicts', default=None,
                    help='comma-separated label:path entries; each adds a family '
                         'of n_random sampled atoms (e.g. "maha rel1:/path/dict.pt")')
    ap.add_argument('--n_anchors', type=int, default=512)
    ap.add_argument('--n_random', type=int, default=100)
    ap.add_argument('--max_angle', type=float, default=60.0)
    ap.add_argument('--n_angles', type=int, default=13)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--families', nargs='+', default=None,
                    help='subset of families to compute (default: all). '
                         'Σ and anchors are deterministic given --seed, so a '
                         'subset is mergeable with a prior run via --merge_pkl.')
    ap.add_argument('--merge_pkl', default=None,
                    help='prior maha_sweep.pkl whose per_family entries are '
                         'merged in (not recomputed) before plotting.')
    ap.add_argument('--out_dir', default='results/mahalanobis_offmanifold')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X = load_natural(args.cache)
    n, d = X.shape
    print(f'natural acts: {X.shape}')

    # natural distribution
    mu = X.mean(0)
    print('fitting Ledoit-Wolf Sigma...')
    lw = LedoitWolf().fit(X - mu)
    Sigma = lw.covariance_
    print(f'  shrinkage = {lw.shrinkage_:.4f}')
    # whitening Sigma^{-1/2} via eigendecomp
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-12)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T   # (d,d)

    def maha(pts):
        z = (pts - mu) @ W
        return np.linalg.norm(z, axis=-1)

    extra = None
    if args.extra_dicts:
        extra = [tuple(e.split(':', 1)) for e in args.extra_dicts.split(',')]
    fam = load_families(args.dict_path, args.contrastive_path, args.sae_path,
                        args.n_random, d, args.seed, dict_align_path=args.dict_align_path,
                        extra_dicts=extra)
    if args.families:
        fam = {k: v for k, v in fam.items() if k in args.families}
    for k, v in fam.items():
        print(f'  {k}: {v.shape[0]} directions')

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=min(args.n_anchors, n), replace=False)
    A = X[idx]                                     # (n_anchors, d)
    base_dM = maha(A)                              # natural Mahalanobis of anchors
    print(f'baseline anchor Mahalanobis: mean {base_dM.mean():.2f} (sqrt(d)={np.sqrt(d):.1f})')

    angles = np.linspace(0, np.radians(args.max_angle), args.n_angles)
    results = {'angles_deg': np.degrees(angles), 'baseline_dM': base_dM,
               'd_model': d, 'shrinkage': float(lw.shrinkage_), 'per_family': {}}

    for fname, dirs in fam.items():
        # dM[angle, direction, anchor]; average over +/- sign
        dM = np.zeros((len(angles), dirs.shape[0], A.shape[0]))
        for di, dvec in enumerate(dirs):
            for ti, th in enumerate(angles):
                rp = geodesic_rotate(A, dvec, th)
                rm = geodesic_rotate(A, -dvec, th)
                dM[ti, di] = 0.5 * (maha(rp) + maha(rm))
        # aggregate over directions x anchors
        flat = dM.reshape(len(angles), -1)
        results['per_family'][fname] = {
            'mean': flat.mean(1), 'std': flat.std(1),
            'p25': np.percentile(flat, 25, axis=1),
            'p50': np.percentile(flat, 50, axis=1),
            'p75': np.percentile(flat, 75, axis=1),
            'delta_mean': flat.mean(1) - base_dM.mean(),
            'n_dirs': dirs.shape[0],
        }
        print(f'{fname}: dM at 0deg={flat[0].mean():.2f}, at max={flat[-1].mean():.2f}')

    if args.merge_pkl:
        prior = pickle.load(open(args.merge_pkl, 'rb'))
        for k, v in prior['per_family'].items():
            results['per_family'].setdefault(k, v)
        print(f"merged families from {args.merge_pkl}; now: {list(results['per_family'])}")

    # stable family order for plotting; append any extras not in the base order
    base_order = ['contrastive', 'optimized', 'optimized + align', 'random', 'SAE latent']
    pf = results['per_family']
    order = [k for k in base_order if k in pf] + [k for k in pf if k not in base_order]
    results['per_family'] = {k: pf[k] for k in order}

    out_pkl = os.path.join(args.out_dir, 'maha_sweep.pkl')
    pickle.dump(results, open(out_pkl, 'wb'))
    print(f'saved {out_pkl}')

    # plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    deg = results['angles_deg']
    colors = {'contrastive': 'C0', 'optimized': 'C1', 'optimized + align': 'C4',
              'random': 'C2', 'SAE latent': 'C3'}
    extra_cycle = ['C5', 'C6', 'C7', 'C8', 'C9']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ei = 0
    for fname, r in results['per_family'].items():
        if fname in colors:
            c = colors[fname]
        else:
            c = extra_cycle[ei % len(extra_cycle)]; ei += 1
        ax.plot(deg, r['p50'], label=f"{fname} (n={r['n_dirs']})", color=c)
        ax.fill_between(deg, r['p25'], r['p75'], alpha=0.18, color=c)
    ax.axhline(base_dM.mean(), ls='--', c='k', lw=0.8, label='natural baseline')
    ax.set_xlabel('rotation angle (deg)'); ax.set_ylabel('Mahalanobis distance')
    ax.set_title('Qwen3-1.7B L14: off-manifold-ness by direction family\n(median, IQR band)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, 'maha_sweep.png')
    fig.savefig(out_png, dpi=130)
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()
