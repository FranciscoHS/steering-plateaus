#!/usr/bin/env python3
"""Perturbation sweep for LessWrong replication: GPT-2 small and Qwen3-1.7B.

Three direction types:
  1. Random (improved): normalized diff of two random FineWeb activations
  2. -DoM (contrastive): negated English↔Mandarin difference-of-means
  3. SAE features: top-active SAE decoder directions per activation

Usage:
    # GPT-2 small (layer 2, no dtype needed)
    python scripts/perturbation/sweep_lesswrong_replication.py --model gpt2 --layer 2

    # Qwen3-1.7B (layer 7, bfloat16)
    python scripts/perturbation/sweep_lesswrong_replication.py --model qwen3-1.7b --layer 7 --dtype bfloat16
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer, forward_from_layer_to_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS
from data.sentiment_prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS
from data.positive_prompts_holdout import POSITIVE_PROMPTS_HOLDOUT
from data.programming_prompts import PYTHON_PROMPTS, HASKELL_PROMPTS
from data.python_prompts_holdout import PYTHON_PROMPTS_HOLDOUT
from data.gender_prompts import MALE_PROMPTS, FEMALE_PROMPTS
from data.male_prompts_holdout import MALE_PROMPTS_HOLDOUT


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_gpt2_sae(layer, device='cpu'):
    """Load Joseph Bloom's GPT-2 small residual SAE via SAELens."""
    from sae_lens import SAE
    release = "gpt2-small-res-jb"
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"  Loading SAE: {release} / {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    print(f"  SAE loaded: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")
    return sae


def load_qwen_sae(layer, width='65k', device='cpu', dtype=torch.float32):
    """Load adamkarvonen batch-top-k SAE for Qwen3-1.7B."""
    trainer = {'16k': 0, '65k': 2}[width]
    from huggingface_hub import hf_hub_download
    import json

    repo_id = "adamkarvonen/qwen3-1.7b-saes"
    base = f"saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}"

    cfg_path = hf_hub_download(repo_id, f"{base}/config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"  SAE config: {config}")

    ae_path = hf_hub_download(repo_id, f"{base}/ae.pt")
    state = torch.load(ae_path, map_location=device, weights_only=True)

    key_map = {
        'encoder.weight': 'W_enc', 'encoder.bias': 'b_enc',
        'decoder.weight': 'W_dec', 'decoder.bias': 'b_dec',
    }
    remapped = {}
    for k, v in state.items():
        remapped[key_map.get(k, k)] = v

    W_dec = remapped['W_dec'].to(dtype=dtype)
    if W_dec.shape[0] < W_dec.shape[1]:
        W_dec = W_dec.T  # ensure [n_features, d_model]

    class SimpleSAE:
        def __init__(self, W_enc, b_enc, W_dec, b_dec):
            self.W_enc = W_enc.to(dtype=dtype)
            self.b_enc = b_enc.to(dtype=dtype)
            self.W_dec = W_dec.to(dtype=dtype)
            self.b_dec = b_dec.to(dtype=dtype)

        def encode(self, x):
            x_centered = x - self.b_dec
            return F.relu(x_centered @ self.W_enc + self.b_enc)

    W_enc = remapped['W_enc'].to(dtype=dtype)
    if W_enc.shape[1] < W_enc.shape[0]:
        W_enc = W_enc.T  # ensure [d_model, n_features]
    sae = SimpleSAE(W_enc, remapped['b_enc'].to(dtype=dtype),
                    W_dec, remapped['b_dec'].to(dtype=dtype))
    print(f"  SAE loaded manually: W_dec shape {W_dec.shape}")
    return sae


def load_llama_scope_sae(layer, expansion='8x', device='cpu', dtype=torch.float32):
    """Load Llama Scope JumpReLU SAE for Llama 3.1-8B."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    repo_id = f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{expansion}"
    subdir = f"Llama3_1-8B-Base-L{layer}R-{expansion}"

    print(f"  Loading Llama Scope SAE: {repo_id}/{subdir}")

    cfg_path = hf_hub_download(repo_id, f"{subdir}/hyperparams.json")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"  SAE config: d_model={config['d_model']}, d_sae={config['d_sae']}, act_fn={config.get('act_fn')}")

    weights_path = hf_hub_download(repo_id, f"{subdir}/checkpoints/final.safetensors")
    state = load_file(weights_path, device=str(device))

    # Print keys so we can see the weight naming
    print(f"  Weight keys: {list(state.keys())}")

    # Llama Scope SAEs use JumpReLU activation
    threshold = config.get('threshold', 0.0)

    class LlamaScopeSAE:
        def __init__(self, state, config, threshold, dtype):
            self.W_enc = state['encoder.weight'].to(dtype=dtype)  # [d_sae, d_model]
            self.b_enc = state['encoder.bias'].to(dtype=dtype)    # [d_sae]
            self.W_dec = state['decoder.weight'].to(dtype=dtype)  # [d_sae, d_model]
            self.b_dec = state['decoder.bias'].to(dtype=dtype)    # [d_model]
            self.threshold = threshold
            # Ensure W_dec is [n_features, d_model]
            if self.W_dec.shape[1] > self.W_dec.shape[0]:
                self.W_dec = self.W_dec.T
            # Ensure W_enc is [d_model, n_features]
            if self.W_enc.shape[0] > self.W_enc.shape[1]:
                self.W_enc = self.W_enc.T

        def encode(self, x):
            x_centered = x - self.b_dec
            pre_acts = x_centered @ self.W_enc + self.b_enc
            # JumpReLU: zero out activations below threshold
            return F.relu(pre_acts) * (pre_acts > self.threshold).float()

    sae = LlamaScopeSAE(state, config, threshold, dtype)
    print(f"  SAE loaded: W_dec shape {sae.W_dec.shape}")
    return sae


def load_gemma_scope_sae(layer, width='16k', device='cpu', model_size='9b'):
    """Load Gemma Scope SAE via SAELens."""
    from sae_lens import SAE
    release = f"gemma-scope-{model_size}-pt-res-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"
    print(f"  Loading SAE: {release} / {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    print(f"  SAE loaded: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")
    return sae


def load_sae_for_model(model_name, layer, device, sae_width='65k'):
    """Dispatch to the right SAE loader."""
    if model_name.startswith('gpt2'):
        return load_gpt2_sae(layer, device=device)
    elif 'qwen' in model_name:
        return load_qwen_sae(layer, width=sae_width, device=device)
    elif 'llama' in model_name:
        expansion = {'32k': '8x', '128k': '32x'}.get(sae_width, '8x')
        return load_llama_scope_sae(layer, expansion=expansion, device=device)
    elif 'gemma' in model_name:
        model_size = '9b' if '9b' in model_name else '2b'
        return load_gemma_scope_sae(layer, width=sae_width, device=device, model_size=model_size)
    else:
        raise ValueError(f"No SAE loader for model: {model_name}")


# ---------------------------------------------------------------------------
# Direction computation
# ---------------------------------------------------------------------------

def get_top_sae_directions(sae, activations, n_dirs, device):
    """Encode activations through SAE, return top-n decoder directions per activation."""
    n_act, d_model = activations.shape
    all_directions = torch.zeros(n_act, n_dirs, d_model)
    all_feature_ids = torch.zeros(n_act, n_dirs, dtype=torch.long)

    W_dec = sae.W_dec.float() if isinstance(sae.W_dec, torch.Tensor) else sae.W_dec.data.float()

    for i in range(n_act):
        act = activations[i:i+1].to(device)
        feature_acts = sae.encode(act)  # [1, n_features]
        topk = feature_acts[0].topk(n_dirs)
        feat_ids = topk.indices
        all_feature_ids[i] = feat_ids.cpu()
        dirs = W_dec[feat_ids.cpu()]
        dirs = F.normalize(dirs, dim=-1)
        all_directions[i] = dirs.cpu()

    return all_directions, all_feature_ids


def get_contrastive_direction(model, tokenizer, device, layer, contrastive_type='language'):
    """Compute contrastive DoM direction.

    Args:
        contrastive_type: 'language' for EN↔ZH, 'sentiment' for positive↔negative,
                          'programming' for Python↔Haskell

    Returns:
        neg_dom: [d_model] unit direction (negated DoM)
    """
    if contrastive_type == 'sentiment':
        prompts_a, prompts_b = POSITIVE_PROMPTS, NEGATIVE_PROMPTS
        label_a, label_b = 'positive', 'negative'
    elif contrastive_type == 'programming':
        prompts_a, prompts_b = PYTHON_PROMPTS, HASKELL_PROMPTS
        label_a, label_b = 'Python', 'Haskell'
    elif contrastive_type == 'gender':
        prompts_a, prompts_b = MALE_PROMPTS, FEMALE_PROMPTS
        label_a, label_b = 'male', 'female'
    else:
        prompts_a, prompts_b = ENGLISH_PROMPTS, CHINESE_PROMPTS
        label_a, label_b = 'EN', 'ZH'

    print(f"  Computing {label_a}↔{label_b} DoM at layer {layer}...")
    def get_acts(prompts):
        acts = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
        return torch.stack(acts)

    a_acts = get_acts(prompts_a)
    b_acts = get_acts(prompts_b)
    mean_vec = (a_acts - b_acts).mean(0)
    dom = F.normalize(mean_vec, dim=0)
    print(f"  DoM pre-norm magnitude: {mean_vec.norm():.2f}")
    neg_dom = -dom  # toward B (Chinese / negative)
    print(f"  Using NEGATED DoM (toward {label_b})")
    return neg_dom


def make_improved_random_directions(activations, n_curves, seed):
    """Improved random baseline: normalized diff of two random FineWeb activations.

    For each of n_curves directions, sample two distinct activations and return
    the normalized difference as the direction.
    """
    rng = np.random.RandomState(seed)
    n_act = len(activations)
    directions = []
    for _ in range(n_curves):
        i, j = rng.choice(n_act, size=2, replace=False)
        diff = activations[i] - activations[j]
        directions.append(F.normalize(diff, dim=0))
    return torch.stack(directions)  # [n_curves, d_model]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_one(model, ctx, act, direction, layer, measure_layer, device, angles_rad,
              metric='l2', batch_size=64):
    """Sweep a single (activation, direction) pair. Returns array of response values."""
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
    all_points = (R * (cos_t * a_unit.unsqueeze(0) + sin_t * d_orth.unsqueeze(0))).to(model_dtype)

    use_logits = (metric == 'kl')
    fwd_fn = forward_from_layer if use_logits else forward_from_layer_to_layer

    result = np.zeros(len(angles_rad))
    with torch.no_grad():
        if use_logits:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer).float()
            ref_logp = F.log_softmax(ref, dim=-1)
        else:
            ref = fwd_fn(model, ctx_d, a_d[None, None], layer, measure_layer).float()

        for start in range(0, len(angles_rad), batch_size):
            end = min(start + batch_size, len(angles_rad))
            batch_points = all_points[start:end].unsqueeze(1)
            if use_logits:
                pf = fwd_fn(model, ctx_d, batch_points, layer).float()
                pf_logp = F.log_softmax(pf, dim=-1)
                kl = F.kl_div(pf_logp, ref_logp.expand_as(pf_logp),
                              log_target=True, reduction='none').sum(-1)
                result[start:end] = kl.cpu().numpy()
            else:
                pf = fwd_fn(model, ctx_d, batch_points, layer, measure_layer).float()
                result[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_individual_plus_median(curves, angles_deg, title, save_path):
    """Individual curves at alpha=0.5, thick median on top. LessWrong style."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = [
        ('random_', 'C0', 'Random'),
        ('neg_dom_', 'C2', '-DoM (contrastive)'),
        ('sae_', 'C3', 'SAE features'),
    ]

    for prefix, color, label in groups:
        keys = sorted([k for k in curves if k.startswith(prefix)])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        # Individual curves
        for k in keys:
            ax.plot(angles_deg, curves[k], color=color, alpha=0.5, lw=0.8)
        # Thick median
        med = np.median(data, axis=0)
        ax.plot(angles_deg, med, color=color, lw=3.0, label=f'{label} ({len(keys)})')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel('L2 distance', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='LessWrong replication sweep')
    parser.add_argument('--model', required=True,
                        choices=['gpt2', 'qwen3-1.7b', 'llama-3.1-8b', 'gemma-2-2b', 'gemma-2-9b'])
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--sae_width', default='65k')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--n_dirs', type=int, default=10,
                        help='Top SAE features per activation')
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--contrastive', default='auto',
                        choices=['language', 'sentiment', 'programming', 'gender', 'auto'],
                        help='Contrastive direction type (auto = gender for gpt2, language for others)')
    parser.add_argument('--metric', default='l2', choices=['l2', 'kl'],
                        help='Response metric: l2 (hidden state) or kl (logit KL-div)')
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Defaults per model
    if args.results_dir is None:
        suffix_map = {'gpt2': 'gpt2-small', 'qwen3-1.7b': 'qwen3-1.7b-final',
                      'llama-3.1-8b': 'llama-3.1-8b'}
        suffix = suffix_map.get(args.model, args.model)
        args.results_dir = f'results/lesswrong_plot/{suffix}'
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    load_dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}.get(args.dtype)
    print(f"Loading model: {args.model}" + (f" (dtype={args.dtype})" if args.dtype else ""))
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = n_layers - 2
    print(f"d_model={d_model}, n_layers={n_layers}, perturb@L{args.layer}, measure@L{measure_layer}")

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # Extract activations
    # For sentiment contrastive: perturb positive-sentiment activations (holdout set)
    # For language contrastive: perturb FineWeb (English) activations
    contrastive_type = args.contrastive
    if contrastive_type == 'auto':
        contrastive_type = 'gender' if args.model.startswith('gpt2') else 'language'
    print(f"Contrastive type: {contrastive_type}")

    if contrastive_type in ('programming', 'sentiment', 'gender'):
        if contrastive_type == 'programming':
            prompts = PYTHON_PROMPTS_HOLDOUT[:args.n_act]
            label = 'Python holdout'
        elif contrastive_type == 'gender':
            prompts = MALE_PROMPTS_HOLDOUT[:args.n_act]
            label = 'male holdout'
        else:
            prompts = POSITIVE_PROMPTS_HOLDOUT[:args.n_act]
            label = 'positive-sentiment holdout'
        print(f"\nExtracting layer-{args.layer} activations from {len(prompts)} {label} prompts...")
        all_contexts = []
        acts_list = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[args.layer + 1]
            all_contexts.append(hidden[:, :-1, :].float().cpu())
            acts_list.append(hidden[:, -1, :].squeeze(0).float().cpu())
        all_activations = torch.stack(acts_list)
    else:
        print(f"\nLoading {args.n_act} FineWeb sequences (seq_len={args.seq_len})...")
        token_ids_list = load_fineweb_fixed_length(
            args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
        print(f"Extracting layer-{args.layer} activations...")
        all_contexts, all_activations = extract_context_and_activation(
            model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

    # Contrastive direction
    neg_dom = get_contrastive_direction(model, tokenizer, device, args.layer,
                                        contrastive_type=contrastive_type)

    # SAE directions
    print(f"\nLoading SAE for {args.model}, layer {args.layer}...")
    sae = load_sae_for_model(args.model, args.layer, device, sae_width=args.sae_width)
    sae_dirs, feature_ids = get_top_sae_directions(sae, all_activations, args.n_dirs, device)
    print(f"Top feature IDs range: {feature_ids.min().item()}-{feature_ids.max().item()}")
    del sae
    torch.cuda.empty_cache()

    # Improved random directions: normalized diff of two FineWeb activations
    print(f"\nGenerating {args.n_curves} improved random directions...")
    rand_directions = make_improved_random_directions(all_activations, args.n_curves, args.seed)

    # Run sweeps
    rng = np.random.RandomState(args.seed)
    n_act = len(all_activations)
    act_indices = rng.randint(0, n_act, size=args.n_curves)
    dir_indices = rng.randint(0, args.n_dirs, size=args.n_curves)

    curves = {}

    # Random (improved baseline)
    print(f"\n  Sweeping {args.n_curves} random curves...")
    t0 = time.time()
    for k in range(args.n_curves):
        curves[f'random_{k}'] = sweep_one(
            model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
            rand_directions[k], args.layer, measure_layer, device, angles_rad,
            metric=args.metric)
    print(f"    {time.time()-t0:.1f}s")

    # -DoM (contrastive)
    n_cont = min(args.n_curves, n_act)
    print(f"  Sweeping {n_cont} -DoM curves...")
    t0 = time.time()
    for k in range(n_cont):
        curves[f'neg_dom_{k}'] = sweep_one(
            model, all_contexts[k], all_activations[k],
            neg_dom, args.layer, measure_layer, device, angles_rad,
            metric=args.metric)
    print(f"    {time.time()-t0:.1f}s")

    # SAE features
    print(f"  Sweeping {args.n_curves} SAE curves...")
    t0 = time.time()
    for k in range(args.n_curves):
        curves[f'sae_{k}'] = sweep_one(
            model, all_contexts[act_indices[k]], all_activations[act_indices[k]],
            sae_dirs[act_indices[k], dir_indices[k]],
            args.layer, measure_layer, device, angles_rad,
            metric=args.metric)
    print(f"    {time.time()-t0:.1f}s")

    # Save data
    save_data = {
        'args': vars(args),
        'angles_deg': angles_deg,
        'curves': curves,
        'feature_ids': feature_ids.numpy(),
    }
    metric_label = args.metric
    pkl_path = os.path.join(args.results_dir, f'curves_{metric_label}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n  Saved: {pkl_path}")

    # Plot
    title = f'{args.model}: inject@L{args.layer}, measure@L{measure_layer}'
    plot_individual_plus_median(
        curves, angles_deg, title,
        os.path.join(args.results_dir, f'plot_{metric_label}.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
