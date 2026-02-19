#!/usr/bin/env python3
"""Quick interpretation of MELBO directions found on dog breed prompts."""
import pickle, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '.')
from src.model import load_model, forward_from_layer_allpos
from data.animal_prompts import DOGS

# Load results
with open('results/melbo_dogs/qwen3-0.6b/results.pkl', 'rb') as f:
    r = pickle.load(f)
melbo_dirs = r['melbo_directions']
melbo_kls = r['melbo_kls']
prompts = DOGS

# Load model
model, tokenizer, device = load_model('qwen3-0.6b')

# Extract activations
contexts = []
activations = []
for p in prompts:
    tokens = tokenizer(p, return_tensors='pt').to(device)
    n_toks = tokens['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[15]  # layer 14
    contexts.append(hidden[:, :-1, :])
    activations.append(hidden[:, -1, :].squeeze(0))
    print(f"  '{p}': {n_toks} tokens, norm={activations[-1].norm():.1f}")

# Reference
ref_log_probs_list = []
for i in range(len(prompts)):
    with torch.no_grad():
        hidden = torch.cat([contexts[i], activations[i].unsqueeze(0).unsqueeze(0)], dim=1)
        logits = forward_from_layer_allpos(model, hidden, 14)
        ref_log_probs_list.append(F.log_softmax(logits, dim=-1))

# Baseline predictions
print("\n=== BASELINE PREDICTIONS ===")
for i, p in enumerate(prompts):
    top_vals, top_ids = ref_log_probs_list[i].squeeze().topk(5)
    toks = [tokenizer.decode([tid]).strip() for tid in top_ids]
    probs = [v.exp().item() for v in top_vals]
    tok_str = ", ".join(f"'{t}' ({p:.2f})" for t, p in zip(toks, probs))
    print(f"  {p:25s} -> {tok_str}")

# Interpret each direction at 15 degrees
for d_idx in range(3):
    direction = melbo_dirs[d_idx].to(device)
    kl_opt = melbo_kls[d_idx]
    print(f"\n=== MELBO_{d_idx+1} (opt KL={kl_opt:.2f}) ===")
    for sign, label in [(1.0, "+"), (-1.0, "-")]:
        print(f"\n  Direction {label} at 15 deg:")
        angle_rad = sign * np.deg2rad(15.0)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        for i, p in enumerate(prompts):
            a = activations[i]
            a_norm = a.norm()
            a_unit = a / a_norm
            proj = (direction * a_unit).sum()
            u_perp = direction - proj * a_unit
            u_perp = u_perp / (u_perp.norm() + 1e-10)
            a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
            with torch.no_grad():
                hidden = torch.cat([contexts[i], a_pert.unsqueeze(0).unsqueeze(0)], dim=1)
                logits = forward_from_layer_allpos(model, hidden, 14)
                pert_lp = F.log_softmax(logits, dim=-1).squeeze()
            ref_lp = ref_log_probs_list[i].squeeze()
            diff = pert_lp - ref_lp

            # Top promoted and demoted
            top_vals, top_ids = diff.topk(3)
            bot_vals, bot_ids = (-diff).topk(3)
            promoted = [(tokenizer.decode([tid]).strip(), v.item())
                        for tid, v in zip(top_ids, top_vals)]
            demoted = [(tokenizer.decode([tid]).strip(), -v.item())
                       for tid, v in zip(bot_ids, bot_vals)]
            kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()

            prom = ", ".join(f"'{t}' (+{v:.1f})" for t, v in promoted)
            dem = ", ".join(f"'{t}' (-{v:.1f})" for t, v in demoted)
            print(f"    {p:25s} KL={kl:7.2f} | up: {prom} | down: {dem}")

# Also check: per-prompt KL distribution (mean vs median discrepancy)
print("\n=== PER-PROMPT KL AT 15 DEG (positive direction) ===")
for d_idx in range(3):
    direction = melbo_dirs[d_idx].to(device)
    angle_rad = np.deg2rad(15.0)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    kls = []
    for i in range(len(prompts)):
        a = activations[i]
        a_norm = a.norm()
        a_unit = a / a_norm
        proj = (direction * a_unit).sum()
        u_perp = direction - proj * a_unit
        u_perp = u_perp / (u_perp.norm() + 1e-10)
        a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)
        with torch.no_grad():
            hidden = torch.cat([contexts[i], a_pert.unsqueeze(0).unsqueeze(0)], dim=1)
            logits = forward_from_layer_allpos(model, hidden, 14)
            pert_lp = F.log_softmax(logits, dim=-1).squeeze()
        ref_lp = ref_log_probs_list[i].squeeze()
        kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()
        kls.append(kl)
    kls = np.array(kls)
    print(f"  MELBO_{d_idx+1}: mean={kls.mean():.2f}, median={np.median(kls):.2f}, "
          f"min={kls.min():.2f}, max={kls.max():.2f}, std={kls.std():.2f}")
    # Show which prompts have highest/lowest KL
    order = np.argsort(kls)[::-1]
    print(f"    Highest: {prompts[order[0]]} ({kls[order[0]]:.1f}), "
          f"{prompts[order[1]]} ({kls[order[1]]:.1f}), "
          f"{prompts[order[2]]} ({kls[order[2]]:.1f})")
    print(f"    Lowest:  {prompts[order[-1]]} ({kls[order[-1]]:.1f}), "
          f"{prompts[order[-2]]} ({kls[order[-2]]:.1f}), "
          f"{prompts[order[-3]]} ({kls[order[-3]]:.1f})")
