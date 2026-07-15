#!/usr/bin/env python3
"""Identify, per concept, the single best k=1 direction for dict and SAE.

The k=1 sparse-probe direction = the one whose per-sample value best SEPARATES
the concept's positives from negatives (standard sparse-probing selection):
  - dict : value_j = (a - abar) . d_j      (signed projection onto atom j)
  - SAE  : value_j = ReLU(encode(a))_j      (latent j's activation)
Selected atom/latent = argmax_j |mean_pos - mean_neg| / pooled_std; we also
report its 1-D AUC so we know whether it's actually a good separator.

(We deliberately do NOT use 'top-firing latent' for the SAE: one dominant
always-on latent is argmax for nearly every sample, so it carries no concept
signal. Best-separator is the meaningful, comparable selector.)

Outputs a JSON (concept -> selected indices for dict + SAE) + a printed table,
to then feed into build_neuronpedia_dashboard.py.

Run on the pod (cached acts + dicts on the volume; no model needed):

    python -u scripts/probes/select_k1_atoms.py \
      --acts results/probe_suite/suite_acts.pt \
      --dict results/dictionary/.../align1.0_k100_40k/dictionary.pt \
      --sae_ckpt /workspace/.../layer14.sae.pt \
      --out results/probe_suite/k1_selected.json
"""

import sys, os, argparse, json
sys.path.insert(0, '.')

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.probes.run_probe_dict_vs_sae import load_qwen_sae, load_dict


def select(scores, y):
    """scores [N, M] (signed values); y [N]. Pick the best class separator.

    top = argmax_j |mean_pos_j - mean_neg_j| / pooled_std_j; report its AUC.
    """
    y = np.asarray(y); pos, neg = y == 1, y == 0
    sp, sn = scores[pos], scores[neg]
    mp, mn = sp.mean(0), sn.mean(0)
    sd = scores.std(dim=0).clamp_min(1e-6)
    tstat = ((mp - mn).abs() / sd)
    top = int(tstat.argmax())
    col = scores[:, top].numpy()
    auc = roc_auc_score(y, col); auc = max(auc, 1.0 - auc)
    return {'top': top, 'tstat': float(tstat[top]), 'auc': float(auc),
            'sign': 'pos' if (mp[top] - mn[top]) > 0 else 'neg'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--dict', required=True)
    ap.add_argument('--sae_ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--enc_batch', type=int, default=1024)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    acts = torch.load(args.acts, weights_only=False)
    D = load_dict(args.dict).to(device)               # [K, d] unit rows
    sae = load_qwen_sae(args.sae_ckpt, device)

    out = {}
    print(f'{"concept":<18} {"dict_atom":>9} {"auc":>5} {"sgn":>4} | '
          f'{"sae_lat":>7} {"auc":>5} {"sgn":>4}')
    for name, a in acts.items():
        A = a['train'].float(); y = a['train_y']
        abar = A.mean(dim=0, keepdim=True)
        Pc = ((A.to(device) - abar.to(device)) @ D.T).cpu()         # [N, K] signed
        dstat = select(Pc, y)

        Z = []
        for i in range(0, len(A), args.enc_batch):
            with torch.no_grad():
                Z.append(torch.relu(sae.encode_pre(A[i:i+args.enc_batch].to(device))).cpu())
        Z = torch.cat(Z, dim=0)                                      # [N, K_sae]
        sstat = select(Z, y)

        out[name] = {'dict': dstat, 'sae': sstat}
        print(f'{name:<18} {dstat["top"]:>9} {dstat["auc"]:.2f} {dstat["sign"]:>4} | '
              f'{sstat["top"]:>7} {sstat["auc"]:.2f} {sstat["sign"]:>4}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    dict_atoms = sorted({v['dict']['top'] for v in out.values()})
    sae_lats = sorted({v['sae']['top'] for v in out.values()})
    print(f'\nUnique dict atoms: {",".join(map(str, dict_atoms))}')
    print(f'Unique SAE latents: {",".join(map(str, sae_lats))}')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
