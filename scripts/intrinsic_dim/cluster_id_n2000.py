"""Local ID per k-means cluster, with each cluster subsampled to fixed N=2000
before running TwoNN. Matches the N-scale of the earlier layer sweep."""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skdim
from sklearn.cluster import MiniBatchKMeans


def run_twonn(X):
    X = X + np.random.randn(*X.shape) * 1e-7 * X.std()
    return skdim.id.TwoNN().fit(X).dimension_


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_path', default='results/jacobian/l10_last_token_seq10.pt')
    p.add_argument('--n_clusters_list', type=int, nargs='+',
                   default=[5, 10, 15, 20, 25, 30, 40])
    p.add_argument('--n_samples', type=int, default=50000)
    p.add_argument('--subsample_n', type=int, default=2000)
    p.add_argument('--n_runs_per_cluster', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='results/jacobian/cluster_id_l10_n2000')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    X = torch.load(args.cache_path, weights_only=True).float().numpy()
    if args.n_samples < len(X):
        X = X[torch.randperm(len(X))[:args.n_samples].numpy()]
    print(f"X: {X.shape}")

    # Global ID at N=subsample_n
    global_runs = []
    for r in range(10):
        rng = np.random.RandomState(r)
        global_runs.append(run_twonn(X[rng.choice(len(X), args.subsample_n, replace=False)]))
    global_id = float(np.mean(global_runs))
    global_sem = float(np.std(global_runs, ddof=1) / np.sqrt(len(global_runs)))
    print(f"Global ID at N={args.subsample_n} ({len(global_runs)} runs): "
          f"{global_id:.2f} +/- {global_sem:.2f}")

    all_sizes, all_ids = [], []
    by_k = {}
    for n_clusters in sorted(args.n_clusters_list):
        print(f"\nk={n_clusters} (avg expected size {len(X)//n_clusters})...")
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed,
                             batch_size=min(10000, len(X)))
        labels = km.fit_predict(X)
        print(f"  clustering: {time.time()-t0:.1f}s")

        cluster_ids, cluster_sizes = [], []
        for c in range(n_clusters):
            mask = labels == c
            n_in = int(mask.sum())
            if n_in < args.subsample_n:
                continue
            Xc = X[mask]
            runs = []
            for r in range(args.n_runs_per_cluster):
                rng = np.random.RandomState(1000 + r)
                runs.append(run_twonn(Xc[rng.choice(n_in, args.subsample_n, replace=False)]))
            cid = float(np.mean(runs))
            cluster_ids.append(cid)
            cluster_sizes.append(n_in)
            all_sizes.append(n_in)
            all_ids.append(cid)
        cluster_ids = np.array(cluster_ids); cluster_sizes = np.array(cluster_sizes)
        if len(cluster_ids) > 0:
            print(f"  {len(cluster_ids)} clusters >= {args.subsample_n} pts; "
                  f"ID mean={cluster_ids.mean():.2f}, range=[{cluster_ids.min():.2f}, {cluster_ids.max():.2f}]")
        by_k[n_clusters] = {'cluster_ids': cluster_ids.tolist(),
                            'cluster_sizes': cluster_sizes.tolist()}

    all_sizes = np.array(all_sizes); all_ids = np.array(all_ids)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(all_sizes, all_ids, alpha=0.5, s=25, color='steelblue')
    ax.axhline(global_id, color='red', linestyle='--', linewidth=1.5,
               label=f'Global ID (N={args.subsample_n}) = {global_id:.1f} ± {global_sem:.1f}')
    if len(all_sizes) >= 10:
        bins = np.logspace(np.log10(all_sizes.min()), np.log10(all_sizes.max()), 10)
        c, m, q25, q75 = [], [], [], []
        for i in range(len(bins)-1):
            mask = (all_sizes >= bins[i]) & (all_sizes < bins[i+1])
            if mask.sum() >= 3:
                c.append(np.sqrt(bins[i]*bins[i+1]))
                m.append(np.median(all_ids[mask]))
                q25.append(np.percentile(all_ids[mask], 25))
                q75.append(np.percentile(all_ids[mask], 75))
        if c:
            ax.plot(c, m, 'o-', color='darkblue', linewidth=2, markersize=6, label='Binned median')
            ax.fill_between(c, q25, q75, color='darkblue', alpha=0.15, label='IQR')
    ax.set_xscale('log')
    ax.set_xlabel('Cluster size (number of points)')
    ax.set_ylabel(f'TWO-NN intrinsic dim (at N={args.subsample_n} per cluster)')
    ax.set_title(f'Local ID (N={args.subsample_n} subsample) vs cluster size (Qwen3-1.7B layer 10)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'id_vs_cluster_size_n2000.png'), dpi=150, bbox_inches='tight')

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({'global_id_n2000': global_id, 'global_sem': global_sem,
                   'subsample_n': args.subsample_n,
                   'all_sizes': all_sizes.tolist(), 'all_ids': all_ids.tolist(),
                   'by_k': {str(k): v for k, v in by_k.items()}}, f, indent=2)
    print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()
