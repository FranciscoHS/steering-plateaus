#!/usr/bin/env python3
"""
Pareto plot: reconstruction quality vs plateau-breaking-ness of directions.

For each (method, n) combination:
  - y-axis: reconstruction KL-div (averaged over eval activations)
  - x-axis: average plateau-breaking angle of the directions used

Plateau-breaking angle of a direction d: the smallest angle theta such that
spherically perturbing a real activation along d causes cosine distance > 0.05
at the final layer. Averaged over 10 real activations. Cached to disk.

Methods: MP-pruned dict, FP-pruned dict, SAE, PCA, random dict, contrastive.

Usage:
    python -u scripts/dictionary_learning/pareto_plateau_vs_recon.py \
        --model meta-llama/Llama-3.2-1B --layer 2 \
        --mp_dict_path results/pruned_dictionary/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --fp_dict_path results/pruned_dictionary_fp/llama-3.2-1b_layer2/pruned_dictionary.pkl \
        --sae_path results/trained_sae/llama-3.2-1b_layer2_8x_k32/sae_weights.pt \
        --output_dir results/pareto_plateau_recon/llama-3.2-1b_layer2
"""

import argparse
import hashlib
import os
import pickle
import sys
import time
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer, forward_from_layer_to_final,
    _get_final_norm, _get_lm_head,
)
from src.data import load_fineweb_fixed_length


# ── Reconstruction helpers ──────────────────────────────────────────────

def matching_pursuit_with_indices(dictionary, activations, n_nonzero_coefs):
    """Matching pursuit returning both reconstruction and selected atom indices.

    Returns:
        recon: (n_val, d_model)
        selected: (n_val, n_nonzero_coefs) long tensor of atom indices
    """
    n_val = activations.shape[0]
    device = activations.device
    residuals = activations.clone()
    recon = torch.zeros_like(activations)
    selected = torch.zeros(n_val, n_nonzero_coefs, dtype=torch.long, device=device)

    for step in range(n_nonzero_coefs):
        corr = residuals @ dictionary.T
        best = corr.abs().argmax(dim=1)
        selected[:, step] = best
        coefs = corr[torch.arange(n_val, device=device), best]
        contribution = coefs.unsqueeze(1) * dictionary[best]
        recon = recon + contribution
        residuals = residuals - contribution

    return recon, selected


def sae_reconstruct_with_indices(sae, activations, n, batch_size=256):
    """SAE top-n reconstruction returning both reconstruction and feature indices.

    Returns:
        recon: (n_val, d_model)
        selected: (n_val, n) long tensor of feature indices
    """
    n_val = activations.shape[0]
    recon_chunks = []
    idx_chunks = []
    for i in range(0, n_val, batch_size):
        batch = activations[i:i+batch_size]
        _, _, pre_acts = sae.encode(batch)
        topn_vals, topn_idx = pre_acts.topk(n, dim=-1)
        recon_chunks.append(sae.decode(topn_vals, topn_idx))
        idx_chunks.append(topn_idx)
    return torch.cat(recon_chunks, 0), torch.cat(idx_chunks, 0)


def pca_adaptive_reconstruct_with_indices(activations, components, mean, n):
    """Adaptive PCA: per-activation top-n component selection by projection magnitude.

    For each activation, projects onto all components and picks the n with
    largest absolute coefficients. Since PCA components are orthonormal,
    reconstruction is just the sum of those projections.

    Args:
        activations: (n_val, d_model)
        components: (n_components, d_model) - all PCA directions (orthonormal rows)
        mean: (d_model,) - PCA mean
        n: number of components to select per activation

    Returns:
        recon: (n_val, d_model)
        selected: (n_val, n) long tensor of component indices
    """
    centered = activations - mean
    all_proj = centered @ components.T  # (n_val, n_components)
    topn_vals, topn_idx = all_proj.abs().topk(n, dim=-1)  # (n_val, n)

    # Reconstruct from selected components (use actual signed coefficients)
    actual_coeffs = all_proj.gather(1, topn_idx)  # (n_val, n)
    selected_dirs = components[topn_idx]  # (n_val, n, d_model)
    recon = (actual_coeffs.unsqueeze(-1) * selected_dirs).sum(dim=1) + mean  # (n_val, d_model)

    return recon, topn_idx


# ── Contrastive direction ───────────────────────────────────────────────

def compute_contrastive_direction(model, tokenizer, device, layer):
    """Compute EN→ZH contrastive direction (Mandarin - English).

    Returns unit direction pointing from English toward Mandarin.
    """
    from data.bilingual_prompts_hq import PROMPT_PAIRS

    en_prompts = [p[0] for p in PROMPT_PAIRS]
    zh_prompts = [p[1] for p in PROMPT_PAIRS]

    def get_acts(prompts):
        acts = []
        for p in prompts:
            toks = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**toks, output_hidden_states=True)
            acts.append(out.hidden_states[layer + 1][:, -1, :].float().squeeze(0))
        return torch.stack(acts).cpu()

    print(f"  Computing EN-ZH contrastive direction from {len(PROMPT_PAIRS)} pairs...")
    en_acts = get_acts(en_prompts)
    zh_acts = get_acts(zh_prompts)
    diff = (zh_acts - en_acts).mean(dim=0)
    direction = F.normalize(diff, dim=0)
    print(f"  ||mean diff|| = {diff.norm().item():.2f}")
    return direction


# ── Plateau-breaking angle measurement ──────────────────────────────────

def direction_hash(d):
    """Hash a direction tensor for caching."""
    # Use first 64 bytes of the raw data as hash key
    data = d.cpu().float().numpy().tobytes()[:256]
    return hashlib.sha256(data).hexdigest()[:16]


def measure_plateau_angles_batched(
    model, directions, contexts, activations, layer, device,
    n_angles=30, max_angle=60.0, cos_dist_threshold=0.05,
    chunk_size=256,
):
    """Measure plateau-breaking angle for many directions, batched.

    For each direction, perturbs each activation along that direction at
    increasing angles, finds smallest angle where cosine distance > threshold.
    Averages over all activations.

    Args:
        directions: (n_dirs, d_model) unit-normalized directions
        contexts: (n_act, seq-1, d_model) contexts for real activations
        activations: (n_act, d_model) real activations
        n_angles, max_angle, cos_dist_threshold: sweep parameters
        chunk_size: directions per batch

    Returns:
        plateau_angles: (n_dirs,) in degrees. 'max_angle' if threshold never exceeded.
    """
    n_dirs = directions.shape[0]
    n_act = activations.shape[0]
    d_model = directions.shape[1]

    angles_deg = np.linspace(0, max_angle, n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    model_dtype = next(model.parameters()).dtype

    # For each direction, track the min angle (over angles) where threshold exceeded,
    # accumulated over activations
    # Shape: (n_dirs,) — sum of plateau-breaking angles across activations
    angle_sum = np.zeros(n_dirs)
    angle_count = np.zeros(n_dirs)

    for act_idx in range(n_act):
        a = activations[act_idx].to(device=device, dtype=model_dtype)
        ctx = contexts[act_idx].unsqueeze(0).to(device=device, dtype=model_dtype)
        R = a.norm()
        if R < 1e-8:
            continue
        a_unit = a / R

        # Reference final-layer activation
        with torch.no_grad():
            ref_point = a.unsqueeze(0).unsqueeze(0)
            ref_final = forward_from_layer_to_final(model, ctx, ref_point, layer)

        # For each direction, compute orthogonal component w.r.t. this activation
        dirs_dev = directions.to(device=device, dtype=model_dtype)
        proj = (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)  # (n_dirs, d_model)
        d_perp = dirs_dev - proj
        d_perp_norm = d_perp.norm(dim=-1, keepdim=True)
        # If direction is parallel to activation, d_perp is zero — set angle to max
        parallel_mask = (d_perp_norm.squeeze() < 1e-6).cpu().numpy()
        # Normalize (avoid div by zero)
        d_perp = d_perp / (d_perp_norm + 1e-8)

        # Track whether threshold exceeded for each direction
        found_angle = np.full(n_dirs, max_angle)

        for ang_idx, theta in enumerate(angles_rad):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Process directions in chunks
            for c_start in range(0, n_dirs, chunk_size):
                c_end = min(c_start + chunk_size, n_dirs)
                chunk_d = d_perp[c_start:c_end]  # (chunk, d_model)

                perturbed = R * (cos_t * a_unit.unsqueeze(0) + sin_t * chunk_d)
                perturbed = perturbed.unsqueeze(1)  # (chunk, 1, d_model)

                with torch.no_grad():
                    pert_final = forward_from_layer_to_final(model, ctx, perturbed, layer)

                cos_sim = F.cosine_similarity(
                    pert_final.float(), ref_final.float(), dim=-1
                )  # (chunk,)
                cos_dist = (1 - cos_sim).cpu().numpy()

                # Update found_angle for directions in this chunk that just exceeded threshold
                chunk_indices = np.arange(c_start, c_end)
                exceeded = cos_dist > cos_dist_threshold
                first_time = found_angle[chunk_indices] == max_angle
                update_mask = exceeded & first_time
                found_angle[chunk_indices[update_mask]] = angles_deg[ang_idx]

        found_angle[parallel_mask] = max_angle
        angle_sum += found_angle
        angle_count += 1

    # Average over activations
    plateau_angles = angle_sum / np.maximum(angle_count, 1)
    return plateau_angles


def get_or_compute_plateau_angles(
    model, directions, direction_ids, contexts, activations, layer, device,
    cache_path, chunk_size=256, n_angles=30, max_angle=60.0,
    cos_dist_threshold=0.05,
):
    """Get plateau-breaking angles, using cache where possible.

    Args:
        directions: (n_dirs, d_model)
        direction_ids: list of string IDs for each direction (for caching)
        contexts, activations: real activations for measurement
        cache_path: path to cache pkl

    Returns:
        plateau_angles: (n_dirs,) in degrees
    """
    # Load existing cache
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    n_dirs = directions.shape[0]
    cached_angles = np.zeros(n_dirs)
    uncached_mask = []

    for i, did in enumerate(direction_ids):
        if did in cache:
            cached_angles[i] = cache[did]
        else:
            uncached_mask.append(i)

    if uncached_mask:
        print(f"    {len(uncached_mask)} uncached directions "
              f"(of {n_dirs} total, {n_dirs - len(uncached_mask)} cached)")
        uncached_idx = np.array(uncached_mask)
        uncached_dirs = directions[uncached_idx]

        new_angles = measure_plateau_angles_batched(
            model, uncached_dirs, contexts, activations, layer, device,
            n_angles=n_angles, max_angle=max_angle,
            cos_dist_threshold=cos_dist_threshold,
            chunk_size=chunk_size,
        )

        for j, idx in enumerate(uncached_idx):
            cached_angles[idx] = new_angles[j]
            cache[direction_ids[idx]] = float(new_angles[j])

        # Save updated cache
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"    Cache updated: {len(cache)} total entries")
    else:
        print(f"    All {n_dirs} directions cached")

    return cached_angles


# ── Reconstruction KL ───────────────────────────────────────────────────

def compute_recon_kl(model, contexts, recon_activations, clean_logits, layer, device,
                     batch_size=64):
    """Compute mean KL(clean || reconstructed) over activations."""
    n_val = recon_activations.shape[0]
    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
    all_kl = []

    for i in range(0, n_val, batch_size):
        batch_recon = recon_activations[i:i+batch_size]
        batch_ctx = contexts[i:i+batch_size]

        batch_logits = []
        for j in range(batch_recon.shape[0]):
            with torch.no_grad():
                logits_j = forward_from_layer(
                    model,
                    batch_ctx[j:j+1].to(device),
                    batch_recon[j:j+1].unsqueeze(1).to(device),
                    layer,
                )
            batch_logits.append(logits_j.cpu())
        patched_logits = torch.cat(batch_logits, dim=0)
        patched_log_probs = F.log_softmax(patched_logits.float(), dim=-1)
        batch_clean = clean_log_probs[i:i+batch_size]
        kl = F.kl_div(patched_log_probs, batch_clean.exp(), reduction='none').sum(dim=-1)
        all_kl.append(kl)

    return torch.cat(all_kl).mean().item()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--mp_dict_path', required=True)
    parser.add_argument('--fp_dict_path', required=True)
    parser.add_argument('--sae_path', default=None)
    parser.add_argument('--n_values', default='5,10,20,50',
                        help='Comma-separated sparsity levels')
    parser.add_argument('--pca_n_values', default=None,
                        help='Comma-separated sparsity levels for PCA (default: auto up to d_model)')
    parser.add_argument('--n_eval', type=int, default=200,
                        help='Eval activations for reconstruction KL')
    parser.add_argument('--n_plateau_act', type=int, default=10,
                        help='Real activations for plateau-breaking angle measurement')
    parser.add_argument('--n_angles', type=int, default=30,
                        help='Angle points for plateau sweep')
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--cos_dist_threshold', type=float, default=0.05)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_total', type=int, default=8000)
    parser.add_argument('--eval_start', type=int, default=7000)
    parser.add_argument('--pca_fit_start', type=int, default=2000)
    parser.add_argument('--pca_fit_end', type=int, default=7000)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    eval_end = args.eval_start + args.n_eval
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, 'plateau_angles_cache.pkl')

    # PCA n_values: determined after we know d_model (see below)
    pca_n_values_arg = args.pca_n_values

    print("Pareto plot: reconstruction quality vs plateau-breaking angle")
    print(f"  model={args.model}, layer={args.layer}")
    print(f"  n_values={n_values}, n_eval={args.n_eval}")
    print(f"  plateau measurement: {args.n_plateau_act} acts, {args.n_angles} angles, "
          f"max={args.max_angle} deg, threshold={args.cos_dist_threshold}")
    print()

    # ── Load dictionaries ───────────────────────────────────────────────
    print("Loading dictionaries...")
    with open(args.mp_dict_path, 'rb') as f:
        mp_dict = F.normalize(pickle.load(f)['dictionary'].float(), dim=-1)
    print(f"  MP-pruned: {mp_dict.shape}")

    with open(args.fp_dict_path, 'rb') as f:
        fp_dict = F.normalize(pickle.load(f)['dictionary'].float(), dim=-1)
    print(f"  FP-pruned: {fp_dict.shape}")

    d_model = mp_dict.shape[1]
    n_atoms = mp_dict.shape[0]
    torch.manual_seed(args.seed + 999)
    rand_dict = F.normalize(torch.randn(n_atoms, d_model), dim=-1)
    print(f"  Random: {rand_dict.shape}")

    # Resolve PCA n_values now that d_model is known
    if pca_n_values_arg:
        pca_n_values = [int(x) for x in pca_n_values_arg.split(',')]
    else:
        pca_n_values = n_values
    pca_n_values = [n for n in pca_n_values if n <= d_model]
    pca_max_n = max(pca_n_values)
    print(f"  PCA n_values: {pca_n_values} (max={pca_max_n})")

    # ── Load SAE ────────────────────────────────────────────────────────
    sae = None
    sae_decoder = None
    if args.sae_path:
        print(f"Loading SAE from {args.sae_path}...")
        from src.autoencoders import TopKSAE
        sae_data = torch.load(args.sae_path, map_location='cpu', weights_only=False)
        config = sae_data['config']
        sae = TopKSAE(d_in=config['d_in'], d_sae=config['d_sae'], k=config['k'])
        sae.load_state_dict(sae_data['state_dict'])
        sae.eval()
        # Decoder columns as directions (unit-normalized)
        sae_decoder = F.normalize(sae.W_dec.data.float(), dim=-1)  # (d_sae, d_model)
        print(f"  SAE: d_sae={config['d_sae']}, k={config['k']}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer, device = load_model(args.model)

    # ── Contrastive direction ───────────────────────────────────────────
    print("\nComputing contrastive direction...")
    contrastive_dir = compute_contrastive_direction(
        model, tokenizer, device, args.layer
    )  # (d_model,) unit vector, points EN→ZH

    # ── Extract activations ─────────────────────────────────────────────
    print(f"\nLoading {args.n_total} FineWeb samples...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_total, tokenizer, seq_len=args.seq_len, seed=args.seed)

    # Eval activations (for reconstruction KL and direction selection)
    print(f"Extracting {args.n_eval} eval activations [{args.eval_start}:{eval_end}]...")
    eval_tokens = token_ids_list[args.eval_start:eval_end]
    all_contexts = []
    all_activations = []
    all_logits = []
    batch_size = 64
    for i in range(0, len(eval_tokens), batch_size):
        batch = torch.stack(eval_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        all_contexts.append(hidden[:, :-1, :].float().cpu())
        all_activations.append(hidden[:, -1, :].float().cpu())
        all_logits.append(outputs.logits[:, -1, :].float().cpu())
    eval_contexts = torch.cat(all_contexts, 0)
    eval_activations = torch.cat(all_activations, 0)
    eval_logits = torch.cat(all_logits, 0)
    print(f"  Shape: {eval_activations.shape}, mean norm: {eval_activations.norm(dim=-1).mean():.2f}")

    # Plateau measurement activations (first n_plateau_act of eval set)
    plateau_contexts = eval_contexts[:args.n_plateau_act]
    plateau_activations = eval_activations[:args.n_plateau_act]

    # PCA fitting activations
    print(f"Extracting PCA fitting activations [{args.pca_fit_start}:{args.pca_fit_end}]...")
    pca_tokens = token_ids_list[args.pca_fit_start:args.pca_fit_end]
    pca_acts = []
    for i in range(0, len(pca_tokens), batch_size):
        batch = torch.stack(pca_tokens[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1]
        pca_acts.append(hidden[:, -1, :].float().cpu())
    pca_fit_activations = torch.cat(pca_acts, 0)
    print(f"  PCA fit: {pca_fit_activations.shape}")

    # ── Phase 1: Run all reconstructions and collect direction usage ─────
    print("\n" + "=" * 60)
    print("Phase 1: Reconstructions and direction collection")
    print("=" * 60)

    # Move to device
    mp_dict_gpu = mp_dict.to(device)
    fp_dict_gpu = fp_dict.to(device)
    rand_dict_gpu = rand_dict.to(device)
    if sae is not None:
        sae = sae.to(device)
    eval_act_gpu = eval_activations.to(device)

    max_n = max(n_values)

    # PCA: compute full SVD to get all d_model principal components
    print(f"Computing PCA (full SVD)...")
    pca_mean = pca_fit_activations.mean(dim=0)
    centered_fit = pca_fit_activations - pca_mean
    _, _, Vh = torch.linalg.svd(centered_fit, full_matrices=False)
    pca_all_components = Vh  # (d_model, d_model) — rows are principal components
    print(f"  PCA components: {pca_all_components.shape}")

    # For each method, run MP/SAE at max_n and record selections
    # (smaller n values reuse the first n selections from the greedy MP)
    print("Running MP at max_n for each dictionary...")
    mp_recon_maxn, mp_selected = matching_pursuit_with_indices(mp_dict_gpu, eval_act_gpu, max_n)
    fp_recon_maxn, fp_selected = matching_pursuit_with_indices(fp_dict_gpu, eval_act_gpu, max_n)
    rand_recon_maxn, rand_selected = matching_pursuit_with_indices(rand_dict_gpu, eval_act_gpu, max_n)

    sae_selected = None
    if sae is not None:
        print("Running SAE at max_n...")
        _, sae_selected = sae_reconstruct_with_indices(sae, eval_act_gpu, max_n)

    # PCA adaptive: select top-n components per activation at max_n
    print("Running adaptive PCA at max_n...")
    pca_max_n_eval = max(pca_n_values)
    _, pca_selected = pca_adaptive_reconstruct_with_indices(
        eval_activations, pca_all_components, pca_mean, pca_max_n_eval
    )  # pca_selected: (n_eval, pca_max_n_eval)
    unique_pca = torch.unique(pca_selected).cpu()

    # Collect all unique direction indices per method
    unique_mp = torch.unique(mp_selected).cpu()
    unique_fp = torch.unique(fp_selected).cpu()
    unique_rand = torch.unique(rand_selected).cpu()
    unique_sae = torch.unique(sae_selected).cpu() if sae_selected is not None else torch.tensor([], dtype=torch.long)

    print(f"\nUnique directions used:")
    print(f"  MP-pruned: {len(unique_mp)}")
    print(f"  FP-pruned: {len(unique_fp)}")
    print(f"  Random: {len(unique_rand)}")
    if sae is not None:
        print(f"  SAE: {len(unique_sae)}")
    print(f"  PCA (adaptive): {len(unique_pca)}")
    print(f"  Contrastive: 1")

    # ── Phase 2: Measure plateau-breaking angles ────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Plateau-breaking angle measurement")
    print("=" * 60)

    # Build direction sets with IDs for caching
    # IDs: "method:index" to distinguish same-index atoms from different dicts
    direction_sets = {}

    # MP-pruned dictionary directions
    direction_sets['mp'] = {
        'directions': mp_dict[unique_mp],
        'ids': [f"mp:{i}" for i in unique_mp.tolist()],
    }
    direction_sets['fp'] = {
        'directions': fp_dict[unique_fp],
        'ids': [f"fp:{i}" for i in unique_fp.tolist()],
    }
    direction_sets['rand'] = {
        'directions': rand_dict[unique_rand],
        'ids': [f"rand:{i}" for i in unique_rand.tolist()],
    }
    if sae_decoder is not None and len(unique_sae) > 0:
        direction_sets['sae'] = {
            'directions': sae_decoder[unique_sae],
            'ids': [f"sae:{i}" for i in unique_sae.tolist()],
        }
    direction_sets['pca'] = {
        'directions': pca_all_components[unique_pca],
        'ids': [f"pca:{i}" for i in unique_pca.tolist()],
    }
    direction_sets['contrastive'] = {
        'directions': contrastive_dir.unsqueeze(0),
        'ids': ['contrastive:enzh'],
    }

    # Measure plateau angles for each set
    plateau_angle_lookup = {}  # id -> angle
    total_to_measure = sum(len(ds['ids']) for ds in direction_sets.values())
    print(f"Total unique directions to evaluate: {total_to_measure}")

    for method_key, ds in direction_sets.items():
        n_ds = len(ds['ids'])
        print(f"\n  [{method_key}] {n_ds} directions")
        t0 = time.time()
        angles = get_or_compute_plateau_angles(
            model, ds['directions'], ds['ids'],
            plateau_contexts, plateau_activations,
            args.layer, device, cache_path,
            chunk_size=args.chunk_size,
            n_angles=args.n_angles,
            max_angle=args.max_angle,
            cos_dist_threshold=args.cos_dist_threshold,
        )
        for did, angle in zip(ds['ids'], angles):
            plateau_angle_lookup[did] = angle
        print(f"    Time: {time.time() - t0:.1f}s")
        print(f"    Angles: min={angles.min():.1f}, median={np.median(angles):.1f}, "
              f"max={angles.max():.1f}")

    # Contrastive angle
    contrastive_angle = plateau_angle_lookup['contrastive:enzh']
    print(f"\nContrastive (EN→ZH) plateau-breaking angle: {contrastive_angle:.1f} deg")

    # ── Phase 3: Compute Pareto points ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Pareto points (reconstruction KL + avg plateau angle)")
    print("=" * 60)

    # Helper to get average plateau angle for selected directions
    def avg_plateau_angle(selected_indices, method_prefix):
        """selected_indices: (n_eval, n) on CPU. method_prefix: e.g. 'mp'"""
        total = 0.0
        count = 0
        for i in range(selected_indices.shape[0]):
            for j in range(selected_indices.shape[1]):
                idx = selected_indices[i, j].item()
                did = f"{method_prefix}:{idx}"
                total += plateau_angle_lookup[did]
                count += 1
        return total / count

    # For reconstruction at each n, we re-run MP with that n
    # (can't just truncate max_n MP since MP is greedy — first n atoms from max_n run
    # are the same as running with n_nonzero_coefs=n)
    # Actually, MP IS greedy and the first n selections are identical to running with n.
    # So we can truncate.

    pareto_points = {}  # method -> list of (avg_angle, kl, n)

    for method_name, dict_gpu, selected, prefix in [
        ('MP-pruned', mp_dict_gpu, mp_selected, 'mp'),
        ('FP-pruned', fp_dict_gpu, fp_selected, 'fp'),
        ('Random dict', rand_dict_gpu, rand_selected, 'rand'),
    ]:
        pareto_points[method_name] = []
        print(f"\n  {method_name}:")
        for n in n_values:
            # Truncate to first n selections (MP is greedy)
            sel_n = selected[:, :n].cpu()
            avg_angle = avg_plateau_angle(sel_n, prefix)

            # Reconstruct with n atoms
            recon = torch.zeros_like(eval_act_gpu)
            residuals = eval_act_gpu.clone()
            for step in range(n):
                best = selected[:, step]
                coefs = (residuals * dict_gpu[best]).sum(dim=-1)
                contribution = coefs.unsqueeze(1) * dict_gpu[best]
                recon = recon + contribution
                residuals = residuals - contribution

            kl = compute_recon_kl(model, eval_contexts, recon.cpu(), eval_logits,
                                  args.layer, device)
            pareto_points[method_name].append((avg_angle, kl, n))
            print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, KL={kl:.4f}")

    # SAE
    if sae is not None and sae_selected is not None:
        pareto_points['SAE'] = []
        print(f"\n  SAE:")
        for n in n_values:
            sel_n = sae_selected[:, :n].cpu()
            avg_angle = avg_plateau_angle(sel_n, 'sae')

            # Reconstruct with top-n
            recon, _ = sae_reconstruct_with_indices(sae, eval_act_gpu, n)
            kl = compute_recon_kl(model, eval_contexts, recon.cpu(), eval_logits,
                                  args.layer, device)
            pareto_points['SAE'].append((avg_angle, kl, n))
            print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, KL={kl:.4f}")

    # PCA (adaptive: per-activation top-n by projection magnitude)
    pareto_points['PCA'] = []
    print(f"\n  PCA (adaptive):")
    for n in pca_n_values:
        # Adaptive selection: top-n components per activation
        recon, sel = pca_adaptive_reconstruct_with_indices(
            eval_activations, pca_all_components, pca_mean, n
        )
        sel_cpu = sel.cpu()
        avg_angle = avg_plateau_angle(sel_cpu, 'pca')

        recon = recon.to(device)
        kl = compute_recon_kl(model, eval_contexts, recon.cpu(), eval_logits,
                              args.layer, device)
        pareto_points['PCA'].append((avg_angle, kl, n))
        print(f"    n={n:3d}: avg_angle={avg_angle:.1f} deg, KL={kl:.4f}")

    # Contrastive: single point (n=1, angle = contrastive_angle, KL from using just that direction)
    print(f"\n  Contrastive (EN→ZH):")
    contrastive_gpu = contrastive_dir.to(device=device).unsqueeze(0)  # (1, d_model)
    # Reconstruct: project each eval activation onto contrastive direction
    coefs = (eval_act_gpu * contrastive_gpu).sum(dim=-1, keepdim=True)  # (n_eval, 1)
    contrastive_recon = coefs * contrastive_gpu  # (n_eval, d_model)
    contrastive_kl = compute_recon_kl(
        model, eval_contexts, contrastive_recon.cpu(), eval_logits, args.layer, device
    )
    pareto_points['Contrastive (EN→ZH)'] = [(contrastive_angle, contrastive_kl, 1)]
    print(f"    angle={contrastive_angle:.1f} deg, KL={contrastive_kl:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    save_data = {
        'args': vars(args),
        'pareto_points': pareto_points,
        'contrastive_angle': contrastive_angle,
        'plateau_angle_lookup': plateau_angle_lookup,
        'n_values': n_values,
        'pca_n_values': pca_n_values,
    }
    pkl_path = os.path.join(args.output_dir, 'pareto_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved: {pkl_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    colors = {
        'MP-pruned': 'C0',
        'FP-pruned': 'C1',
        'SAE': 'C2',
        'PCA': 'C4',
        'Random dict': 'C3',
        'Contrastive (EN→ZH)': 'red',
    }
    markers = {
        'MP-pruned': 'o',
        'FP-pruned': 's',
        'SAE': '^',
        'PCA': 'D',
        'Random dict': 'x',
        'Contrastive (EN→ZH)': '*',
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for method, points in pareto_points.items():
        angles = [p[0] for p in points]
        kls = [p[1] for p in points]
        ns = [p[2] for p in points]
        color = colors.get(method, 'gray')
        marker = markers.get(method, 'o')

        ax.plot(angles, kls, marker=marker, color=color, label=method,
                linewidth=2, markersize=8)

        # Annotate n values
        for angle, kl, n in points:
            ax.annotate(f'n={n}', (angle, kl), textcoords='offset points',
                        xytext=(5, 5), fontsize=7, color=color)

    ax.set_xlabel('Avg plateau-breaking angle (degrees)\n← more plateau-breaking | less plateau-breaking →')
    ax.set_ylabel('Reconstruction KL-divergence (nats)\n← better reconstruction')
    ax.set_title(f'Plateau-breaking vs reconstruction quality\n'
                 f'{args.model}, layer {args.layer}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'pareto_plot.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
