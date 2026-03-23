#!/usr/bin/env python3
"""Generate all figures for the LessWrong post.

Two modes:
  --sweep   Run perturbation sweeps on GPU, saving pkl data files.
  --plot    Generate PNG plots from existing pkl data (no GPU needed).

Usage:
    # Full pipeline on GPU
    python scripts/perturbation/generate_lesswrong_figures.py --sweep --plot

    # Just regenerate plots from existing data
    python scripts/perturbation/generate_lesswrong_figures.py --plot

All contrastive directions use 30 paired prompts.
Direction types: Random (improved baseline), Contrastive (-DoM), SAE features.
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Figure configurations
# ---------------------------------------------------------------------------

N_PROMPTS = 30        # Number of paired prompts for contrastive directions
N_ACT = 30            # Number of activations to perturb
N_CURVES = 10         # Number of individual curves per direction type
N_DIRS = 10           # Top SAE features per activation
MAX_ANGLE = 90.0
N_ANGLES = 90
SEQ_LEN = 5
SEED = 123

FIGURES = [
    {
        'name': 'gemma-9b-en-zh',
        'model': 'gemma-2-9b',
        'contrast': 'language',
        'layer': 2,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete', 'simplified'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-en-zh',
    },
    {
        'name': 'gemma-9b-male-female',
        'model': 'gemma-2-9b',
        'contrast': 'gender',
        'layer': 2,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-male-female',
    },
    {
        'name': 'gemma-9b-python-haskell',
        'model': 'gemma-2-9b',
        'contrast': 'programming',
        'layer': 2,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-python-haskell',
    },
    {
        'name': 'llama-8b-en-zh',
        'model': 'llama-3.1-8b',
        'contrast': 'language',
        'layer': 2,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '32k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/llama-8b-en-zh',
    },
    {
        'name': 'qwen-1.7b-en-zh',
        'model': 'qwen3-1.7b',
        'contrast': 'language',
        'layer': 7,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '65k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/qwen-1.7b-en-zh',
    },
    {
        'name': 'gemma-9b-en-zh-additive',
        'model': 'gemma-2-9b',
        'contrast': 'language',
        'layer': 2,
        'metric': 'l2',
        'perturbation': 'additive',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-en-zh-additive',
    },
    {
        'name': 'gemma-9b-en-zh-kl',
        'model': 'gemma-2-9b',
        'contrast': 'language',
        'layer': 2,
        'metric': 'kl',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-en-zh-kl',
    },
    {
        'name': 'gemma-9b-en-zh-cosine',
        'model': 'gemma-2-9b',
        'contrast': 'language',
        'layer': 2,
        'metric': 'cosine',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-en-zh-cosine',
    },
    {
        'name': 'gemma-9b-en-zh-L8',
        'model': 'gemma-2-9b',
        'contrast': 'language',
        'layer': 8,
        'metric': 'l2',
        'perturbation': 'spherical',
        'dtype': 'bfloat16',
        'sae_width': '16k',
        'plot_styles': ['complete'],
        'results_dir': 'results/lesswrong_figures/gemma-9b-en-zh-L8',
    },
]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_contrast_prompts(contrast, n_prompts=N_PROMPTS):
    """Load paired prompts for contrastive direction computation.

    Returns:
        prompts_a: list of strings (group A, e.g. English / Male / Python)
        prompts_b: list of strings (group B, e.g. Chinese / Female / Haskell)
        label_a, label_b: human-readable labels
    """
    if contrast == 'language':
        from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS
        return (ENGLISH_PROMPTS[:n_prompts], CHINESE_PROMPTS[:n_prompts],
                'English', 'Mandarin')
    elif contrast == 'gender':
        from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
        return (MALE_PROMPTS[:n_prompts], FEMALE_PROMPTS[:n_prompts],
                'Male', 'Female')
    elif contrast == 'programming':
        from data.programming_prompts import PYTHON_PROMPTS, HASKELL_PROMPTS
        return (PYTHON_PROMPTS[:n_prompts], HASKELL_PROMPTS[:n_prompts],
                'Python', 'Haskell')
    else:
        raise ValueError(f"Unknown contrast: {contrast}")


def load_holdout_prompts(contrast, n_act=N_ACT):
    """Load held-out prompts for the 'A' group (used as perturbation targets).

    Returns prompts from group A that are independent of the paired prompts
    used to compute the DoM direction.
    """
    if contrast == 'language':
        from data.english_prompts_holdout import ENGLISH_PROMPTS_HOLDOUT
        return ENGLISH_PROMPTS_HOLDOUT[:n_act]
    elif contrast == 'gender':
        from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT
        return MALE_PROMPTS_HOLDOUT[:n_act]
    elif contrast == 'programming':
        from data.python_prompts_holdout import PYTHON_PROMPTS_HOLDOUT
        return PYTHON_PROMPTS_HOLDOUT[:n_act]
    else:
        raise ValueError(f"Unknown contrast: {contrast}")


# ---------------------------------------------------------------------------
# Direction computation
# ---------------------------------------------------------------------------

def compute_contrastive_direction(model, tokenizer, device, layer,
                                  prompts_a, prompts_b):
    """Compute negated difference-of-means direction (pointing toward group B).

    Returns:
        neg_dom: [d_model] unit vector, direction = -(mean_A - mean_B)
    """
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :]
                        .squeeze(0).float().cpu())
        return torch.stack(acts)

    a_acts = get_acts(prompts_a)
    b_acts = get_acts(prompts_b)
    mean_diff = (a_acts - b_acts).mean(0)
    dom = F.normalize(mean_diff, dim=0)
    neg_dom = -dom
    print(f"    DoM magnitude (pre-norm): {mean_diff.norm():.2f}")
    return neg_dom


def compute_improved_random_directions(activations, n_dirs, seed):
    """Random baseline: normalized difference of two random activations."""
    rng = np.random.RandomState(seed)
    n = len(activations)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = activations[i] - activations[j]
        dirs.append(F.normalize(diff, dim=0))
    return torch.stack(dirs)


def compute_sae_directions(sae, activations, n_dirs, device):
    """Top-active SAE decoder directions per activation.

    Returns:
        directions: [n_act, n_dirs, d_model]
        feature_ids: [n_act, n_dirs]
    """
    n_act, d_model = activations.shape
    all_directions = torch.zeros(n_act, n_dirs, d_model)
    all_feature_ids = torch.zeros(n_act, n_dirs, dtype=torch.long)

    W_dec = sae.W_dec.float() if isinstance(sae.W_dec, torch.Tensor) else sae.W_dec.data.float()

    for i in range(n_act):
        act = activations[i:i+1].to(device)
        feature_acts = sae.encode(act)
        topk = feature_acts[0].topk(n_dirs)
        feat_ids = topk.indices
        all_feature_ids[i] = feat_ids.cpu()
        dirs = W_dec[feat_ids.cpu()]
        dirs = F.normalize(dirs, dim=-1)
        all_directions[i] = dirs.cpu()

    return all_directions, all_feature_ids


# ---------------------------------------------------------------------------
# SAE loading (dispatch to existing loaders)
# ---------------------------------------------------------------------------

def load_sae(model_name, layer, device, sae_width):
    """Load the appropriate SAE for the model."""
    if 'gemma' in model_name:
        from scripts.perturbation.sweep_sae_features import load_gemma_scope_sae
        model_size = '9b' if '9b' in model_name else '2b'
        return load_gemma_scope_sae(layer, width=sae_width, device=device,
                                     model_size=model_size)
    elif 'qwen' in model_name:
        from scripts.perturbation.sweep_lesswrong_replication import load_qwen_sae
        return load_qwen_sae(layer, width=sae_width, device=device)
    elif 'llama' in model_name:
        from scripts.perturbation.sweep_lesswrong_replication import load_llama_scope_sae
        expansion = {'32k': '8x', '65k': '8x', '128k': '32x'}.get(sae_width, '8x')
        return load_llama_scope_sae(layer, expansion=expansion, device=device)
    else:
        raise ValueError(f"No SAE loader for model: {model_name}")


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations_holdout(model, tokenizer, device, layer, prompts):
    """Extract activations from held-out prompts."""
    contexts = []
    acts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]
        contexts.append(hidden[:, :-1, :].float().cpu())
        acts.append(hidden[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(acts)


# ---------------------------------------------------------------------------
# Perturbation sweeps
# ---------------------------------------------------------------------------

def sweep_one_spherical(model, ctx, act, direction, layer, measure_layer,
                        device, angles_rad, metric='l2', batch_size=64):
    """Spherical perturbation: rotate activation on the sphere toward direction."""
    from src.model import forward_from_layer, forward_from_layer_to_layer
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    R = a_d.float().norm()
    a_unit = a_d.float() / R
    d_d = direction.to(device=device, dtype=torch.float32)
    d_orth = d_d - (d_d @ a_unit) * a_unit
    d_orth = F.normalize(d_orth, dim=0)

    thetas = torch.tensor(angles_rad, dtype=torch.float32, device=device)
    cos_t = thetas.cos().unsqueeze(1)
    sin_t = thetas.sin().unsqueeze(1)
    all_points = (R * (cos_t * a_unit.unsqueeze(0)
                       + sin_t * d_orth.unsqueeze(0))).to(model_dtype)

    return _measure_response(model, ctx_d, a_d, all_points, layer,
                             measure_layer, device, metric, batch_size,
                             len(angles_rad))


def sweep_one_additive(model, ctx, act, direction, layer, measure_layer,
                       device, magnitudes, metric='l2', batch_size=64):
    """Additive perturbation: add magnitude * direction (changes norm)."""
    from src.model import forward_from_layer, forward_from_layer_to_layer
    model_dtype = next(model.parameters()).dtype
    ctx_d = ctx.to(device=device, dtype=model_dtype)
    a_d = act.to(device=device, dtype=model_dtype)
    d_d = direction.to(device=device, dtype=model_dtype)

    mags = torch.tensor(magnitudes, dtype=torch.float32, device=device)
    all_points = (a_d.unsqueeze(0)
                  + mags.unsqueeze(1) * d_d.unsqueeze(0)).to(model_dtype)

    return _measure_response(model, ctx_d, a_d, all_points, layer,
                             measure_layer, device, metric, batch_size,
                             len(magnitudes))


def _measure_response(model, ctx_d, a_d, all_points, layer, measure_layer,
                      device, metric, batch_size, n_points):
    """Measure downstream response for a batch of perturbed activations."""
    from src.model import forward_from_layer, forward_from_layer_to_layer
    use_logits = (metric == 'kl')
    fwd_fn = forward_from_layer if use_logits else forward_from_layer_to_layer

    result = np.zeros(n_points)
    with torch.no_grad():
        if use_logits:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer).float()
            ref_logp = F.log_softmax(ref, dim=-1)
        else:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer,
                         measure_layer).float()

        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch_points = all_points[start:end].unsqueeze(1)
            if use_logits:
                pf = fwd_fn(model, ctx_d, batch_points, layer).float()
                pf_logp = F.log_softmax(pf, dim=-1)
                kl = F.kl_div(pf_logp, ref_logp.expand_as(pf_logp),
                              log_target=True, reduction='none').sum(-1)
                result[start:end] = kl.cpu().numpy()
            elif metric == 'cosine':
                pf = fwd_fn(model, ctx_d, batch_points, layer,
                            measure_layer).float()
                result[start:end] = (1.0 - F.cosine_similarity(
                    ref, pf, dim=-1)).cpu().numpy()
            else:
                pf = fwd_fn(model, ctx_d, batch_points, layer,
                            measure_layer).float()
                result[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
    return result


# ---------------------------------------------------------------------------
# Run one figure's sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg):
    """Run the perturbation sweep for one figure configuration."""
    from src.model import load_model, _get_blocks

    model_name = cfg['model']
    layer = cfg['layer']
    contrast = cfg['contrast']
    metric = cfg['metric']
    perturbation = cfg['perturbation']
    results_dir = cfg['results_dir']
    sae_width = cfg['sae_width']
    os.makedirs(results_dir, exist_ok=True)

    dtype_map = {'bfloat16': torch.bfloat16, 'float32': torch.float32}
    load_dtype = dtype_map.get(cfg.get('dtype'), None)

    print(f"\n{'='*70}")
    print(f"SWEEP: {cfg['name']}")
    print(f"  model={model_name}, layer={layer}, contrast={contrast}")
    print(f"  metric={metric}, perturbation={perturbation}")
    print(f"{'='*70}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print(f"\n  Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = n_layers - 2
    print(f"  d_model={d_model}, n_layers={n_layers}, "
          f"perturb@L{layer}, measure@L{measure_layer}")

    # Load contrastive prompts
    prompts_a, prompts_b, label_a, label_b = load_contrast_prompts(
        contrast, n_prompts=N_PROMPTS)
    print(f"\n  Contrastive: {label_a} vs {label_b} ({len(prompts_a)} pairs)")

    # Compute -DoM direction (toward group B)
    print(f"  Computing -{label_a}→{label_b} DoM direction...")
    neg_dom = compute_contrastive_direction(
        model, tokenizer, device, layer, prompts_a, prompts_b)

    # Extract activations to perturb (holdout prompts from group A)
    holdout = load_holdout_prompts(contrast, n_act=N_ACT)
    print(f"\n  Extracting activations from {len(holdout)} "
          f"{label_a} holdout prompts...")
    perturb_contexts, perturb_acts = extract_activations_holdout(
        model, tokenizer, device, layer, holdout)
    # FineWeb pool for improved random baseline directions
    from src.data import load_fineweb_fixed_length
    from scripts.perturbation.sweep_random_points import (
        extract_context_and_activation,
    )
    n_pool = max(N_ACT, N_CURVES * 2)
    pool_ids = load_fineweb_fixed_length(
        n_pool, tokenizer, seq_len=SEQ_LEN, seed=SEED)
    _, pool_acts = extract_context_and_activation(
        model, tokenizer, pool_ids, layer, device)
    print(f"  Activation norms: mean={perturb_acts.norm(dim=-1).mean():.2f}")

    # Improved random directions
    print(f"\n  Computing {N_CURVES} improved random directions...")
    rand_dirs = compute_improved_random_directions(pool_acts, N_CURVES, SEED)

    # SAE directions
    print(f"  Loading SAE ({sae_width}) for layer {layer}...")
    sae = load_sae(model_name, layer, device, sae_width)
    sae_dirs, feature_ids = compute_sae_directions(
        sae, perturb_acts, N_DIRS, device)
    print(f"  Top feature IDs: {feature_ids.min().item()}-"
          f"{feature_ids.max().item()}")
    del sae
    torch.cuda.empty_cache()

    # Select curve indices
    rng = np.random.RandomState(SEED)
    act_indices = rng.randint(0, len(perturb_acts), size=N_CURVES)
    dir_indices = rng.randint(0, N_DIRS, size=N_CURVES)

    # Set up x-axis values
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    if perturbation == 'additive':
        mean_norm = perturb_acts.norm(dim=-1).mean().item()
        eff_angle = min(MAX_ANGLE, 80.0)
        max_mag = mean_norm * np.tan(np.radians(eff_angle))
        magnitudes = np.linspace(0, max_mag, N_ANGLES + 1)[1:]
        equiv_angles = np.degrees(np.arctan(magnitudes / mean_norm))
        sweep_fn = lambda m, ctx, act, d: sweep_one_additive(
            m, ctx, act, d, layer, measure_layer, device, magnitudes,
            metric=metric)
    else:
        sweep_fn = lambda m, ctx, act, d: sweep_one_spherical(
            m, ctx, act, d, layer, measure_layer, device, angles_rad,
            metric=metric)

    # Run sweeps
    curves = {}

    print(f"\n  Sweeping {N_CURVES} random curves...")
    t0 = time.time()
    for k in range(N_CURVES):
        curves[f'random_{k}'] = sweep_fn(
            model, perturb_contexts[act_indices[k]],
            perturb_acts[act_indices[k]], rand_dirs[k])
    print(f"    {time.time()-t0:.1f}s")

    n_dom = min(N_CURVES, len(perturb_acts))
    print(f"  Sweeping {n_dom} contrastive curves...")
    t0 = time.time()
    for k in range(n_dom):
        curves[f'contrastive_{k}'] = sweep_fn(
            model, perturb_contexts[k], perturb_acts[k], neg_dom)
    print(f"    {time.time()-t0:.1f}s")

    print(f"  Sweeping {N_CURVES} SAE curves...")
    t0 = time.time()
    for k in range(N_CURVES):
        curves[f'sae_{k}'] = sweep_fn(
            model, perturb_contexts[act_indices[k]],
            perturb_acts[act_indices[k]],
            sae_dirs[act_indices[k], dir_indices[k]])
    print(f"    {time.time()-t0:.1f}s")

    # Save pkl
    save_data = {
        'config': cfg,
        'angles_deg': angles_deg,
        'curves': curves,
        'feature_ids': feature_ids.numpy(),
    }
    if perturbation == 'additive':
        save_data['magnitudes'] = magnitudes
        save_data['equiv_angles'] = equiv_angles
        save_data['mean_norm'] = mean_norm

    pkl_path = os.path.join(results_dir, 'curves.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n  Saved: {pkl_path}")

    # Free GPU memory before next figure
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {'random': '#4477AA', 'contrastive': '#228833', 'sae': '#EE6677'}
LABELS = {'random': 'Random', 'contrastive': 'Contrastive', 'sae': 'SAE'}


def _group_curves(curves):
    """Group curve keys by prefix (random, contrastive, sae)."""
    groups = {}
    for prefix in ['random', 'contrastive', 'sae']:
        keys = sorted([k for k in curves if k.rsplit('_', 1)[0] == prefix],
                      key=lambda k: int(k.rsplit('_', 1)[1]))
        if keys:
            groups[prefix] = {
                'keys': keys,
                'data': np.array([curves[k] for k in keys]),
            }
    return groups


def plot_complete(x_values, groups, labels, xlabel, ylabel, title, save_path):
    """Complete plot: individual curves (alpha=0.5) + thick median."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix in ['random', 'contrastive', 'sae']:
        if prefix not in groups:
            continue
        g = groups[prefix]
        color = COLORS[prefix]
        for k in g['keys']:
            ax.plot(x_values, g['data'][g['keys'].index(k)],
                    color=color, alpha=0.5, lw=0.8)
        med = np.median(g['data'], axis=0)
        ax.plot(x_values, med, color=color, lw=2.5,
                label=labels.get(prefix, prefix))

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_simplified(x_values, groups, labels, title, save_path):
    """Simplified plot: median + IQR shading, no tick labels, generic axes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for prefix in ['random', 'contrastive', 'sae']:
        if prefix not in groups:
            continue
        g = groups[prefix]
        color = COLORS[prefix]
        data = g['data']
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        ax.fill_between(x_values, q25, q75, color=color, alpha=0.15)
        ax.plot(x_values, med, color=color, lw=2.5,
                label=labels.get(prefix, prefix))

    ax.set_xlabel('Perturbation', fontsize=14)
    ax.set_ylabel('Response', fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def make_title(cfg):
    """Generate a human-readable plot title from config."""
    model_names = {
        'gemma-2-9b': 'Gemma2-9B',
        'llama-3.1-8b': 'Llama3.1-8B',
        'qwen3-1.7b': 'Qwen3-1.7B',
    }
    contrast_names = {
        'language': 'English/Mandarin',
        'gender': 'Male/Female',
        'programming': 'Python/Haskell',
    }
    model_str = model_names.get(cfg['model'], cfg['model'])
    contrast_str = contrast_names.get(cfg['contrast'], cfg['contrast'])
    title = f"{model_str} ({contrast_str})"

    # Add suffixes for non-default settings
    suffixes = []
    if cfg['perturbation'] == 'additive':
        suffixes.append('additive')
    if cfg['metric'] == 'kl':
        suffixes.append('KL divergence')
    elif cfg['metric'] == 'cosine':
        suffixes.append('cosine distance')
    if cfg['layer'] != 2:
        suffixes.append(f'layer {cfg["layer"]}')
    if suffixes:
        title += f" — {', '.join(suffixes)}"
    return title


def plot_figure(cfg):
    """Generate plots for one figure configuration from saved pkl data."""
    results_dir = cfg['results_dir']
    pkl_path = os.path.join(results_dir, 'curves.pkl')

    if not os.path.exists(pkl_path):
        print(f"  SKIP {cfg['name']}: {pkl_path} not found")
        return

    print(f"\n  Plotting: {cfg['name']}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    curves = data['curves']
    groups = _group_curves(curves)
    labels = LABELS
    title = make_title(cfg)

    # Determine x-axis
    if cfg['perturbation'] == 'additive':
        x_values = data['magnitudes']
        xlabel = 'Perturbation norm'
    else:
        x_values = data['angles_deg']
        xlabel = 'Perturbation angle (degrees)'

    # Determine y-axis label
    if cfg['metric'] == 'kl':
        ylabel = 'KL divergence'
    elif cfg['metric'] == 'cosine':
        ylabel = 'Cosine distance'
    else:
        ylabel = 'L2 distance'

    for style in cfg['plot_styles']:
        if style == 'complete':
            save_path = os.path.join(results_dir, 'plot_complete.png')
            plot_complete(x_values, groups, labels, xlabel, ylabel,
                          title, save_path)
        elif style == 'simplified':
            save_path = os.path.join(results_dir, 'plot_simplified.png')
            plot_simplified(x_values, groups, labels, title, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate all LessWrong post figures')
    parser.add_argument('--sweep', action='store_true',
                        help='Run perturbation sweeps (needs GPU)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots from existing pkl data')
    parser.add_argument('--figures', default=None,
                        help='Comma-separated figure names to process '
                             '(default: all)')
    args = parser.parse_args()

    if not args.sweep and not args.plot:
        parser.error("Specify --sweep and/or --plot")

    # Filter figures if requested
    if args.figures:
        names = set(args.figures.split(','))
        figures = [f for f in FIGURES if f['name'] in names]
        if not figures:
            print(f"No matching figures. Available: "
                  f"{', '.join(f['name'] for f in FIGURES)}")
            return
    else:
        figures = FIGURES

    if args.sweep:
        print("=" * 70)
        print("RUNNING SWEEPS")
        print("=" * 70)
        for cfg in figures:
            run_sweep(cfg)

    if args.plot:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)
        for cfg in figures:
            plot_figure(cfg)

    print("\nDone!")


if __name__ == '__main__':
    main()
