#!/usr/bin/env python3
"""Quick random-direction sweep at a given layer. Outputs KL + L2 curves."""

import sys, os, pickle, argparse
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import (
    load_model, _get_blocks, _get_final_norm, _get_lm_head,
    _is_rope_model, _get_position_embeddings, _run_block,
)
from data.coding_prompts_agnostic import CODING_PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-xl')
parser.add_argument('--layer', type=int, default=2)
parser.add_argument('--n_act', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--n_dirs', type=int, default=10)
args = parser.parse_args()

model, tokenizer, device = load_model(args.model)
n_layers = len(_get_blocks(model))
measure_layer = n_layers - 2
print(f"{n_layers} layers, perturb at {args.layer}, measure at {measure_layer}")

# Tokenize coding prompts
all_ids = [tokenizer.encode(p) for p in CODING_PROMPTS]
token_ids_list = [torch.tensor(ids[:args.seq_len]) for ids in all_ids if len(ids) >= args.seq_len]
token_ids_list = token_ids_list[:args.n_act]
print(f"Using {len(token_ids_list)} coding prompts")

# Extract activations
contexts_list, activations_list = [], []
for i in range(0, len(token_ids_list), 64):
    batch = torch.stack(token_ids_list[i:i+64]).to(device)
    with torch.no_grad():
        outputs = model(batch, output_hidden_states=True)
    hidden = outputs.hidden_states[args.layer + 1]
    contexts_list.append(hidden[:, :-1, :].cpu())
    activations_list.append(hidden[:, -1, :].cpu())
contexts = torch.cat(contexts_list, 0).to(device)
activations = torch.cat(activations_list, 0).to(device)


def forward_with_measure(model, hidden_states, start_layer, meas_layer):
    hidden = hidden_states
    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    pos_emb = _get_position_embeddings(model, hidden) if rope else None
    measure_hidden = None
    for i, block in enumerate(blocks[start_layer + 1:], start=start_layer + 1):
        hidden = _run_block(block, hidden, model, pos_emb)
        if i == meas_layer:
            measure_hidden = hidden[:, -1, :].clone()
    final_hidden = hidden[:, -1, :]
    normed = _get_final_norm(model)(final_hidden)
    logits = _get_lm_head(model)(normed)
    return measure_hidden, logits


# Reference
with torch.no_grad():
    ref_hidden = torch.cat([contexts, activations.unsqueeze(1)], dim=1)
    ref_measure, ref_logits = forward_with_measure(model, ref_hidden, args.layer, measure_layer)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

angles_deg = np.linspace(3, 90, 30)
angles_rad = np.radians(angles_deg)

# Random sweep
N, d = activations.shape
n_dirs = args.n_dirs
n_angles = len(angles_rad)
kl_divs = np.zeros((N, n_angles, n_dirs))
l2_dists = np.zeros((N, n_angles, n_dirs))

a_norms = activations.norm(dim=-1, keepdim=True)
a_unit = activations / a_norms
act_dtype = activations.dtype

# Generate random perpendicular directions
v_perps = torch.zeros(N, n_dirs, d, device=device, dtype=act_dtype)
for i in range(N):
    v = torch.randn(n_dirs, d, device=device, dtype=act_dtype)
    v = v - (v @ a_unit[i]).unsqueeze(1) * a_unit[i].unsqueeze(0)
    v_perps[i] = F.normalize(v, dim=-1)

print("Sweeping random directions...")
for ang_idx, theta in enumerate(angles_rad):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for dir_idx in range(n_dirs):
        v = v_perps[:, dir_idx, :]
        a_pert = a_norms * (cos_t * a_unit + sin_t * v)
        with torch.no_grad():
            hidden = torch.cat([contexts, a_pert.unsqueeze(1)], dim=1)
            pert_measure, pert_logits = forward_with_measure(model, hidden, args.layer, measure_layer)
            pert_log_probs = F.log_softmax(pert_logits, dim=-1)
            l2 = (pert_measure - ref_measure).norm(dim=-1)
            kl = (ref_log_probs.exp() * (ref_log_probs - pert_log_probs)).sum(dim=-1)
            l2_dists[:, ang_idx, dir_idx] = l2.float().cpu().numpy()
            kl_divs[:, ang_idx, dir_idx] = kl.float().cpu().numpy()
    if (ang_idx + 1) % 10 == 0:
        print(f"  {ang_idx + 1}/{n_angles}")

# Save
out_dir = f'results/programming_contrastive/{args.model}'
os.makedirs(out_dir, exist_ok=True)
save = {
    'angles_deg': angles_deg,
    'kl': kl_divs,
    'l2': l2_dists,
    'layer': args.layer,
    'measure_layer': measure_layer,
}
out_path = f'{out_dir}/random_sweep_layer{args.layer}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(save, f)
print(f"Saved: {out_path}")

# Print summary
kl_mean = kl_divs.mean(axis=(0, 2))
l2_mean = l2_dists.mean(axis=(0, 2))
print(f"\nMean KL at select angles:")
for deg, kl_val, l2_val in zip(angles_deg[::5], kl_mean[::5], l2_mean[::5]):
    print(f"  {deg:.0f}°: KL={kl_val:.4f}, L2={l2_val:.1f}")
