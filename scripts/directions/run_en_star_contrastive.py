#!/usr/bin/env python3
"""Star-topology contrastive: directions = lang_i - en, sweep from English activations.

For each non-English language i, computes
    f̂_i = mean_p( h^{L}(lang_i prompt p) - h^{L}(en prompt p) )
(normalized), then perturbs along each direction starting from the English
activations and measures cos/L2/KL at a later layer. Random baseline included.

Usage:
    python scripts/directions/run_en_star_contrastive.py \
        --model qwen3-1.7b --layer 3 --bfloat16
"""
import sys, os, pickle, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from data.multilingual_prompts import PROMPT_SETS, LANGUAGES


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-1.7b')
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--measure_layer', type=int, default=None,
                    help='(default: n_layers - 2)')
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--out_dir', default=None)
args = parser.parse_args()

MODEL = args.model
LAYER = args.layer


def extract_prompt_activations(model, tokenizer, prompts, layer, device):
    """For each prompt, return (context_hidden_prefix, final_token_hidden) at `layer`.

    context_hidden_prefix: [seq_len-1, d]  (hidden states of all tokens except last)
    final_token_hidden:    [d]             (hidden state of the last token)

    Each prompt may have a different length, so we store as a list.
    """
    contexts = []
    activations = []
    seq_lens = []
    for p in prompts:
        toks = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
            hidden = out.hidden_states[layer + 1].squeeze(0).float()  # [L, d]
        contexts.append(hidden[:-1].cpu())
        activations.append(hidden[-1].cpu())
        seq_lens.append(hidden.shape[0])
    return contexts, torch.stack(activations), seq_lens


def forward_with_measure_single(model, ctx_prefix_d, act_d, layer, measure_layer):
    """Forward from `layer` onward starting at perturbed final-token activation.

    ctx_prefix_d: [L-1, d]  on device, original dtype
    act_d:        [d]       on device, original dtype
    Returns (measure_hidden [d], logits [V]).
    """
    # Reassemble full sequence: context + perturbed activation at final position
    hidden = torch.cat([ctx_prefix_d, act_d.unsqueeze(0)], dim=0).unsqueeze(0)  # [1, L, d]
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    measure_hidden = None
    for i, block in enumerate(blocks[layer + 1:], start=layer + 1):
        hidden = _run_block(block, hidden, model, pos_emb)
        if i == measure_layer:
            measure_hidden = hidden[:, -1, :].clone()
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return measure_hidden.squeeze(0), logits.squeeze(0)


def sweep_direction(model, contexts, activations, ref_measures, ref_log_probs,
                    direction, angles_rad, layer, measure_layer, device):
    """Sweep a fixed direction (per-activation Gram-Schmidt) across angles."""
    N = len(activations)
    n_angles = len(angles_rad)
    kl = np.zeros((N, n_angles))
    l2 = np.zeros((N, n_angles))
    cos = np.zeros((N, n_angles))
    model_dtype = next(model.parameters()).dtype
    d_vec = direction.to(device).float()
    for i in range(N):
        ctx = contexts[i].to(device)
        a = activations[i].to(device).float()
        a_norm = a.norm(); a_unit = a / a_norm
        d_orth = d_vec - (d_vec @ a_unit) * a_unit
        d_orth = F.normalize(d_orth, dim=0)
        ref_m = ref_measures[i]; ref_lp = ref_log_probs[i]
        for k, theta in enumerate(angles_rad):
            pert = a_norm * (np.cos(theta) * a_unit + np.sin(theta) * d_orth)
            with torch.no_grad():
                pm, pl = forward_with_measure_single(
                    model, ctx.to(model_dtype), pert.to(model_dtype),
                    layer, measure_layer)
            plog = F.log_softmax(pl.float(), dim=-1)
            l2[i, k] = (pm.float() - ref_m).norm().item()
            cos[i, k] = (1.0 - F.cosine_similarity(
                ref_m.unsqueeze(0), pm.float().unsqueeze(0), dim=-1)).item()
            kl[i, k] = (ref_lp.exp() * (ref_lp - plog)).sum().item()
    return kl, l2, cos


def sweep_random(model, contexts, activations, ref_measures, ref_log_probs,
                 angles_rad, layer, measure_layer, device, n_dirs=10, seed=0):
    N = len(activations)
    n_angles = len(angles_rad)
    kl = np.zeros((N, n_angles, n_dirs))
    l2 = np.zeros((N, n_angles, n_dirs))
    cos = np.zeros((N, n_angles, n_dirs))
    model_dtype = next(model.parameters()).dtype
    g = torch.Generator(device='cpu').manual_seed(seed)
    for i in range(N):
        ctx = contexts[i].to(device)
        a = activations[i].to(device).float()
        d_model = a.shape[0]
        a_norm = a.norm(); a_unit = a / a_norm
        rand = torch.randn(n_dirs, d_model, generator=g).to(device).float()
        rand = rand - (rand @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        rand = F.normalize(rand, dim=1)
        ref_m = ref_measures[i]; ref_lp = ref_log_probs[i]
        for j in range(n_dirs):
            for k, theta in enumerate(angles_rad):
                pert = a_norm * (np.cos(theta) * a_unit + np.sin(theta) * rand[j])
                with torch.no_grad():
                    pm, pl = forward_with_measure_single(
                        model, ctx.to(model_dtype), pert.to(model_dtype),
                        layer, measure_layer)
                plog = F.log_softmax(pl.float(), dim=-1)
                l2[i, k, j] = (pm.float() - ref_m).norm().item()
                cos[i, k, j] = (1.0 - F.cosine_similarity(
                    ref_m.unsqueeze(0), pm.float().unsqueeze(0), dim=-1)).item()
                kl[i, k, j] = (ref_lp.exp() * (ref_lp - plog)).sum().item()
    return kl, l2, cos


def main():
    print(f'Model: {MODEL}, layer: {LAYER}')
    load_dtype = torch.bfloat16 if args.bfloat16 else None
    model, tokenizer, device = load_model(MODEL, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    measure_layer = args.measure_layer if args.measure_layer is not None else n_layers - 2
    results_dir = args.out_dir or f'results/en_star_contrastive/{MODEL}_L{LAYER}'
    os.makedirs(results_dir, exist_ok=True)
    print(f'  n_layers={n_layers}, measure_layer={measure_layer}')

    # Gather activations for English + 10 other languages at layer L
    print('\n=== Phase 1: extracting activations for 11 languages ===')
    acts_en = None
    acts = {}
    en_prompts = [ps['en'] for ps in PROMPT_SETS]
    for lang in ['en'] + list(LANGUAGES):
        prompts = [ps[lang] for ps in PROMPT_SETS]
        ctxs, finals, _ = extract_prompt_activations(
            model, tokenizer, prompts, LAYER, device)
        if lang == 'en':
            acts_en = finals
            en_contexts = ctxs
        else:
            acts[lang] = finals
        print(f'  {lang}: {finals.shape}')

    # Compute per-language direction f̂_i = mean(h^i - h^en) over paired prompts, then normalize
    print('\n=== Phase 2: directions = lang_i - en ===')
    features_raw = {}; features = {}
    for lang in LANGUAGES:
        diff = (acts[lang] - acts_en).mean(dim=0)  # [d]
        features_raw[lang] = diff
        features[lang] = F.normalize(diff.unsqueeze(0), dim=-1).squeeze(0)
        print(f'  {lang}: ||Δ|| = {diff.norm().item():.3f}')

    # Cosine similarity matrix
    k = len(LANGUAGES)
    cos_matrix = np.zeros((k, k))
    for i, li in enumerate(LANGUAGES):
        for j, lj in enumerate(LANGUAGES):
            cos_matrix[i, j] = F.cosine_similarity(
                features[li].unsqueeze(0), features[lj].unsqueeze(0)).item()
    print('\n  Pairwise |cos| among 10 directions:')
    off = np.abs(cos_matrix) - np.eye(k)
    print(f'    mean: {off.mean():.3f}, max: {off.max():.3f}')

    # Reference forward from English activations: measure layer hidden + final log-probs
    print('\n=== Phase 3: reference forward on 36 English prompts ===')
    ref_measures = []; ref_log_probs = []
    model_dtype = next(model.parameters()).dtype
    for i in range(len(acts_en)):
        ctx = en_contexts[i].to(device=device, dtype=model_dtype)
        a = acts_en[i].to(device=device, dtype=model_dtype)
        with torch.no_grad():
            pm, pl = forward_with_measure_single(
                model, ctx, a, LAYER, measure_layer)
        ref_measures.append(pm.float())
        ref_log_probs.append(F.log_softmax(pl.float(), dim=-1))

    # Phase 4: sweep +f̂_i and -f̂_i for each language; random baseline
    angles_deg = np.linspace(3, 180, 60)
    angles_rad = np.radians(angles_deg)
    print(f'\n=== Phase 4: sweeps, {len(angles_deg)} angles up to 180° ===')

    sweep_results = {}
    for lang in LANGUAGES:
        print(f'  +{lang}...', end=' ', flush=True)
        kl, l2, cos = sweep_direction(
            model, en_contexts, acts_en, ref_measures, ref_log_probs,
            features[lang], angles_rad, LAYER, measure_layer, device)
        sweep_results[f'+{lang}'] = {'kl': kl, 'l2': l2, 'cos': cos}
        print(f'peak KL {kl.mean(0).max():.2f}')
    for lang in LANGUAGES:
        print(f'  -{lang}...', end=' ', flush=True)
        kl, l2, cos = sweep_direction(
            model, en_contexts, acts_en, ref_measures, ref_log_probs,
            -features[lang], angles_rad, LAYER, measure_layer, device)
        sweep_results[f'-{lang}'] = {'kl': kl, 'l2': l2, 'cos': cos}
        print(f'peak KL {kl.mean(0).max():.2f}')

    print('  random baseline (10 dirs / activation)...', end=' ', flush=True)
    rand_kl, rand_l2, rand_cos = sweep_random(
        model, en_contexts, acts_en, ref_measures, ref_log_probs,
        angles_rad, LAYER, measure_layer, device, n_dirs=10, seed=0)
    sweep_results['random'] = {'kl': rand_kl, 'l2': rand_l2, 'cos': rand_cos}
    print('done')

    # Save
    pickle.dump({
        'features': {l: features[l].cpu() for l in LANGUAGES},
        'features_raw': {l: features_raw[l].cpu() for l in LANGUAGES},
        'cosine_matrix': cos_matrix, 'languages': LANGUAGES,
        'layer': LAYER, 'measure_layer': measure_layer, 'model': MODEL,
    }, open(os.path.join(results_dir, 'directions.pkl'), 'wb'))
    pickle.dump({
        'sweep_results': sweep_results,
        'angles_deg': angles_deg,
        'layer': LAYER, 'measure_layer': measure_layer,
        'languages': LANGUAGES, 'model': MODEL,
    }, open(os.path.join(results_dir, 'sweep_results.pkl'), 'wb'))

    # Plots: 3-metric figure (positive directions)
    cmap = plt.cm.tab10
    mask = angles_deg <= 60
    for kind, sign in [('positive', '+'), ('negative', '-')]:
        fig, (ax_c, ax2, ax1) = plt.subplots(1, 3, figsize=(18, 5))
        for i, lang in enumerate(LANGUAGES):
            name = f'{sign}{lang}'
            c = sweep_results[name]['cos'][:, mask].mean(axis=0)
            l = sweep_results[name]['l2'][:, mask].mean(axis=0)
            kk = sweep_results[name]['kl'][:, mask].mean(axis=0)
            ax_c.plot(angles_deg[mask], c, color=cmap(i), label=lang, lw=1.5)
            ax2.plot(angles_deg[mask], l, color=cmap(i), label=lang, lw=1.5)
            ax1.plot(angles_deg[mask], kk, color=cmap(i), label=lang, lw=1.5)
        rc = rand_cos[:, mask, :].mean(axis=2); rl = rand_l2[:, mask, :].mean(axis=2); rk = rand_kl[:, mask, :].mean(axis=2)
        for ax, arr in [(ax_c, rc), (ax2, rl), (ax1, rk)]:
            ax.plot(angles_deg[mask], arr.mean(axis=0), color='gray', label='random', lw=1.5)
            ax.fill_between(angles_deg[mask],
                            arr.mean(axis=0) - arr.std(axis=0),
                            arr.mean(axis=0) + arr.std(axis=0),
                            color='gray', alpha=0.15)
        ax_c.set(xlabel='Perturbation angle (°)', ylabel='Mean cosine distance',
                 title=f'Cosine distance (L{measure_layer})')
        ax2.set(xlabel='Perturbation angle (°)', ylabel='Mean L2 distance',
                title=f'L2 distance (L{measure_layer})')
        ax1.set(xlabel='Perturbation angle (°)', ylabel='Mean KL divergence',
                title='KL divergence (from final logits)')
        for ax in (ax_c, ax2, ax1):
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
        fig.suptitle(
            f'{sign}(f̂_i) = {sign}mean(h^{{lang}} - h^{{en}}) at L{LAYER}; '
            f'perturbed from 36 English activations.  {MODEL}', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f'sweeps_{kind}.png'),
                    dpi=150, bbox_inches='tight')
        print(f'Saved sweeps_{kind}.png')


if __name__ == '__main__':
    main()
