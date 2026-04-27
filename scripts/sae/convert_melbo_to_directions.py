#!/usr/bin/env python3
"""Convert MELBO output to v2_directions_melbo_*.pkl format expected by
fig4_robustness_v2.py:phase_I.

Reads results/melbo_diverse_capped/<model>_target<kl>/checkpoint.pkl and
re-saves the directions list under
results/fig4_robustness/v2_directions_melbo_<target>_L<layer>.pkl.

Usage:
  python scripts/sae/convert_melbo_to_directions.py \
      --melbo_dir results/melbo_diverse_capped/gemma-2-9b_target1.0 \
      --target gemma --layer 2
"""
import os, sys, pickle, argparse
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

OUT_DIR = 'results/fig4_robustness'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--melbo_dir', required=True)
    ap.add_argument('--target', default='gemma')
    ap.add_argument('--layer', type=int, default=2)
    args = ap.parse_args()

    src = os.path.join(args.melbo_dir, 'checkpoint.pkl')
    if not os.path.exists(src):
        src = os.path.join(args.melbo_dir, 'results.pkl')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'No checkpoint.pkl or results.pkl in {args.melbo_dir}')
    print(f'reading {src}')
    with open(src, 'rb') as f:
        d = pickle.load(f)
    raw = d['directions']
    directions = [F.normalize(t.cpu().float(), dim=0) for t in raw]
    print(f'  {len(directions)} directions, d_model={directions[0].numel()}')

    out_path = os.path.join(OUT_DIR,
                            f'v2_directions_melbo_{args.target}_L{args.layer}.pkl')
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        'directions': directions,
        'layer': args.layer,
        'model': d.get('args', {}).get('model', 'unknown'),
        'target_kl': d.get('args', {}).get('target_kl', None),
        'n_steps': d.get('args', {}).get('n_steps', None),
        'n_restarts': d.get('args', {}).get('n_restarts', None),
        'rewards': d.get('rewards', None),
        'source_pkl': src,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'saved → {out_path}')


if __name__ == '__main__':
    main()
