#!/usr/bin/env python3
"""
Steer dog prompts along robust MELBO_1 (red) and MELBO_2 (yellow) directions.

Tests both +/- at several angles around the onset region.
Shows baseline predictions, then top promoted/demoted tokens after steering.
"""
import pickle, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '.')
from src.model import load_model, forward_from_layer_allpos
from data.animal_prompts import DOGS

# Load robust results
with open('results/melbo_animal/qwen3-0.6b_robust_r20_s500/results.pkl', 'rb') as f:
    r = pickle.load(f)
melbo_dirs = r['melbo_directions']
melbo_kls = r['melbo_kls']

# Load model
model, tokenizer, device = load_model('qwen3-0.6b')
layer = 14

# Extract activations for dog prompts
contexts = []
activations = []
for p in DOGS:
    tokens = tokenizer(p, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]
    contexts.append(hidden[:, :-1, :])
    activations.append(hidden[:, -1, :].squeeze(0))

# Reference predictions
ref_log_probs_list = []
for i in range(len(DOGS)):
    with torch.no_grad():
        hidden = torch.cat([contexts[i], activations[i].unsqueeze(0).unsqueeze(0)], dim=1)
        logits = forward_from_layer_allpos(model, hidden, layer)
        ref_log_probs_list.append(F.log_softmax(logits, dim=-1))

# Baseline
print("=== BASELINE PREDICTIONS (dog prompts) ===")
for i, p in enumerate(DOGS):
    top_vals, top_ids = ref_log_probs_list[i].squeeze().topk(5)
    toks = [tokenizer.decode([tid]).strip() for tid in top_ids]
    probs = [v.exp().item() for v in top_vals]
    tok_str = ", ".join(f"'{t}' ({pr:.2f})" for t, pr in zip(toks, probs))
    print(f"  {p:25s} -> {tok_str}")


def steer_and_report(direction, angle_deg, sign, label):
    """Steer all dog prompts and report top promoted/demoted tokens."""
    angle_rad = np.deg2rad(angle_deg)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad) * sign

    for i, p in enumerate(DOGS):
        a = activations[i]
        a_norm = a.norm()
        a_unit = a / a_norm
        proj = (direction * a_unit).sum()
        u_perp = direction - proj * a_unit
        u_perp = u_perp / (u_perp.norm() + 1e-10)
        a_pert = a_norm * (cos_t * a_unit + sin_t * u_perp)

        with torch.no_grad():
            hidden = torch.cat([contexts[i], a_pert.unsqueeze(0).unsqueeze(0)], dim=1)
            logits = forward_from_layer_allpos(model, hidden, layer)
            pert_lp = F.log_softmax(logits, dim=-1).squeeze()

        ref_lp = ref_log_probs_list[i].squeeze()
        kl = (ref_lp.exp() * (ref_lp - pert_lp)).sum().item()

        # Top-5 after steering
        top_vals, top_ids = pert_lp.topk(5)
        toks = [tokenizer.decode([tid]).strip() for tid in top_ids]
        probs = [v.exp().item() for v in top_vals]
        top_str = ", ".join(f"'{t}' ({pr:.2f})" for t, pr in zip(toks, probs))

        # Top promoted/demoted
        diff = pert_lp - ref_lp
        prom_vals, prom_ids = diff.topk(3)
        dem_vals, dem_ids = (-diff).topk(3)
        prom = ", ".join(f"'{tokenizer.decode([tid]).strip()}' (+{v:.1f})"
                         for tid, v in zip(prom_ids, prom_vals))
        dem = ", ".join(f"'{tokenizer.decode([tid]).strip()}' (-{v:.1f})"
                        for tid, v in zip(dem_ids, dem_vals))

        print(f"  {p:25s} KL={kl:6.3f} | top: {top_str}")
        print(f"  {'':25s}         | up: {prom}")
        print(f"  {'':25s}         | down: {dem}")


# === MELBO_1 (red) ===
d1 = melbo_dirs[0].to(device)
d1 = d1 / d1.norm()

for angle in [15, 20, 25, 30]:
    for sign, sign_label in [(1.0, "+"), (-1.0, "-")]:
        print(f"\n=== MELBO_1 (red), {sign_label}dir, {angle} deg ===")
        steer_and_report(d1, angle, sign, f"MELBO_1 {sign_label}")

# === MELBO_2 (yellow) ===
d2 = melbo_dirs[1].to(device)
d2 = d2 / d2.norm()

for angle in [20, 25, 30, 35]:
    for sign, sign_label in [(1.0, "+"), (-1.0, "-")]:
        print(f"\n=== MELBO_2 (yellow), {sign_label}dir, {angle} deg ===")
        steer_and_report(d2, angle, sign, f"MELBO_2 {sign_label}")

print("\nDone!")
