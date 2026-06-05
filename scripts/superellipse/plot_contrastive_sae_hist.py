#!/usr/bin/env python3
"""Single combined histogram of best-match SAE-latent cosine for ALL contrastive
directions (semantic scaffold + language + code) vs the random-direction null.

Reads the two existing dict_sae_alignment pkls and merges them."""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--scaffold_pkl', required=True)
ap.add_argument('--langcode_pkl', required=True)
ap.add_argument('--n_lang', type=int, default=10)
ap.add_argument('--out', required=True)
args = ap.parse_args()

sc = pickle.load(open(args.scaffold_pkl, 'rb'))
lc = pickle.load(open(args.langcode_pkl, 'rb'))

semantic = np.asarray(sc['found_max_cos'])            # 17
langcode = np.asarray(lc['found_max_cos'])            # 19
language = langcode[:args.n_lang]                     # 10
code = langcode[args.n_lang:]                         # 9
allc = np.concatenate([semantic, language, code])     # 36
rand_max = np.asarray(sc['rand_max'])                 # 500 (same null)

r95 = np.quantile(rand_max, 0.95)
lo = min(allc.min(), rand_max.min())
hi = max(allc.max(), rand_max.max())
bins = np.linspace(lo, hi, 40)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(rand_max, bins=bins, density=True, color='gray', alpha=0.5,
        label=f'random null (n={rand_max.size})')
ax.hist(allc, bins=bins, density=True, color='C3', alpha=0.55,
        label=f'all contrastive (n={allc.size})')
# Per-family medians as vertical ticks for reference.
for vals, col, name in [(semantic, 'C0', 'semantic'),
                        (language, 'C2', 'language'),
                        (code, 'C1', 'code')]:
    ax.axvline(np.median(vals), color=col, ls='-', lw=1.8,
               label=f'{name} median={np.median(vals):.3f} (n={vals.size})')
ax.axvline(r95, color='k', ls='--', lw=1, label=f'random 95th pct={r95:.3f}')
ax.set_xlabel('max cosine with any SAE latent (best match)')
ax.set_ylabel('density')
ax.set_title('Contrastive directions vs SAE latents (Qwen3-1.7B L14, W32K SAE)\n'
             f'all contrastive median={np.median(allc):.3f}  |  '
             f'{int((allc > r95).sum())}/{allc.size} above random 95th pct')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches='tight')
print(f'all contrastive: median={np.median(allc):.3f} '
      f'mean={allc.mean():.3f} min={allc.min():.3f} max={allc.max():.3f}')
print(f'  semantic median={np.median(semantic):.3f}  '
      f'language median={np.median(language):.3f}  '
      f'code median={np.median(code):.3f}')
print(f'Saved {args.out}')
