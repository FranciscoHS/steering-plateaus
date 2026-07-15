#!/usr/bin/env python3
"""For each binary concept, find the atom OMP picks most often at k=1, for the
two best k=1 dictionaries (maha10 and the SAE decoder subset used in the suite),
then emit concept-ordered mini-dictionaries of those directions for the
neuronpedia dashboard (interpretability check).

k=1 OMP selects, per sample, argmax_j |(a - abar) . d_j|  (unit atoms, abar =
per-concept train mean — matches run_sparse_probe_suite's omp_acc). The
"latent used for a concept" = the atom selected by the most samples (mode),
pooling that concept's train+test examples; we also report how concentrated
the selection is (fraction of samples on the modal atom).
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, '.')
import numpy as np
import torch
from scripts.probes.run_probe_dict_vs_sae import load_dict, load_qwen_sae


def modal_atom(X, D, device):
    """X [N,d] (already centered), D [K,d] unit rows. Returns (modal_idx, frac, counts)."""
    Xt = torch.from_numpy(X).to(device).float()
    Dt = D.to(device).float()
    sel = (Xt @ Dt.T).abs().argmax(dim=1).cpu().numpy()   # [N]
    counts = np.bincount(sel, minlength=D.shape[0])
    idx = int(counts.argmax())
    return idx, counts[idx] / len(sel), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True, help='suite_acts.pt')
    ap.add_argument('--maha_dict', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--max_atoms', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    acts = torch.load(args.acts, weights_only=False)
    names = list(acts.keys())
    print(f'{len(names)} concepts: {names}', flush=True)

    Dm = load_dict(args.maha_dict)                                  # [20000, d] unit
    print(f'maha dict: {tuple(Dm.shape)}', flush=True)

    sae = load_qwen_sae(args.sae_ckpt, device)
    Ds = sae.W_dec.detach().cpu().float()
    Ds = Ds / (Ds.norm(dim=-1, keepdim=True) + 1e-12)               # [32768, d] unit
    # SAME random subset as the suite: RandomState(seed+2000).choice(.., max_atoms)
    sub = np.random.RandomState(args.seed + 2000).choice(Ds.shape[0], args.max_atoms, replace=False)
    Ds_sub = Ds[sub]                                                # [20000, d]
    print(f'SAE decoder subset: {tuple(Ds_sub.shape)}', flush=True)

    summary = []
    maha_rows, sae_rows = [], []
    for name in names:
        a = acts[name]
        tr, te = a['train'].numpy(), a['test'].numpy()
        abar = tr.mean(0, keepdims=True)
        X = np.concatenate([tr, te], 0) - abar
        mi, mf, _ = modal_atom(X, Dm, device)
        si_sub, sf, _ = modal_atom(X, Ds_sub, device)
        si_orig = int(sub[si_sub])
        summary.append({'concept': name,
                        'maha10_atom': mi, 'maha10_frac': float(mf),
                        'sae_latent': si_orig, 'sae_subset_idx': int(si_sub),
                        'sae_frac': float(sf)})
        maha_rows.append(Dm[mi])
        sae_rows.append(Ds[si_orig])
        print(f'  {name:18s} maha10 atom {mi:6d} ({mf:.2f})   '
              f'sae latent {si_orig:6d} ({sf:.2f})', flush=True)

    # concept-ordered mini-dicts (row i = concept i) for the dashboard
    torch.save({'dictionary': torch.stack(maha_rows),
                'concept_names': names,
                'src_atom_idx': [s['maha10_atom'] for s in summary]},
               os.path.join(args.out_dir, 'maha10_concept_dict.pt'))
    torch.save({'dictionary': torch.stack(sae_rows),
                'concept_names': names,
                'src_latent_idx': [s['sae_latent'] for s in summary]},
               os.path.join(args.out_dir, 'sae_concept_dict.pt'))
    with open(os.path.join(args.out_dir, 'k1_atoms.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nsaved mini-dicts + k1_atoms.json to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
