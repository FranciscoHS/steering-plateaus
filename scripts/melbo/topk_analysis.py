#!/usr/bin/env python3
"""Analyze top-k token distributions for each MELBO direction."""
import sys, pickle, argparse
import numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks


def get_logits_with_steering(model, tokenizer, device, prompt, direction, steer_layer, angle_deg):
    cos_t = np.cos(np.radians(angle_deg))
    sin_t = np.sin(np.radians(angle_deg))
    d = direction.to(device)
    d = d / d.norm()

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        pos = hidden[0, -1:, :]
        R = pos.norm(dim=-1, keepdim=True)
        a_unit = pos / (R + 1e-10)
        proj = (a_unit * d).sum(dim=-1, keepdim=True)
        v_perp = d.unsqueeze(0) - proj * a_unit
        v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
        steered = R * (cos_t * a_unit + sin_t * v_perp)
        hidden[0, -1:, :] = steered
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layers = _get_blocks(model)
    handle = layers[steer_layer].register_forward_hook(steering_hook)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
    handle.remove()
    return outputs.logits[0, -1, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='The old man sat by the')
    parser.add_argument('--angle', type=float, default=5.0)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()

    model, tokenizer, device = load_model('qwen3-1.7b')

    with open(args.results_pkl, 'rb') as f:
        data = pickle.load(f)

    directions = data['directions']
    steer_layer = data['args']['inject_layer']
    n_dirs = len(directions)
    K = args.topk

    # Unsteered
    tokens = tokenizer(args.prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        base_logits = model(tokens, use_cache=False).logits[0, -1, :]
    base_probs = F.softmax(base_logits.float(), dim=-1).cpu()
    base_log_probs = F.log_softmax(base_logits.float(), dim=-1).cpu()
    base_topk = torch.topk(base_probs, K)

    lines = []

    header = "=== UNSTEERED top-%d ===" % K
    print(header)
    lines.append(header)
    for i in range(K):
        tok = tokenizer.decode([base_topk.indices[i].item()])
        line = "  %.4f  %r" % (base_topk.values[i].item(), tok)
        print(line)
        lines.append(line)

    header2 = "\n=== STEERED at %.1f deg, top-%d for each direction ===" % (args.angle, K)
    print(header2)
    lines.append(header2)

    all_steered_probs = []
    for di in range(n_dirs):
        logits = get_logits_with_steering(
            model, tokenizer, device, args.prompt,
            directions[di], steer_layer, args.angle
        )
        probs = F.softmax(logits.float(), dim=-1).cpu()
        log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
        all_steered_probs.append(probs)
        topk = torch.topk(probs, K)

        # KL from base
        kl = (base_probs * (base_log_probs - log_probs)).sum().item()

        dir_header = "\nDir %d (KL=%.4f):" % (di, kl)
        print(dir_header)
        lines.append(dir_header)
        for i in range(K):
            tok = tokenizer.decode([topk.indices[i].item()])
            base_p = base_probs[topk.indices[i]].item()
            new_p = topk.values[i].item()
            delta = new_p - base_p
            line = "  %.4f (%+.4f)  %r" % (new_p, delta, tok)
            print(line)
            lines.append(line)

    # Pairwise KL between river dirs
    river_dirs = [di for di in range(n_dirs)]
    header3 = "\n=== Pairwise KL between all directions ==="
    print(header3)
    lines.append(header3)
    for i in range(len(river_dirs)):
        for j in range(i+1, len(river_dirs)):
            di, dj = river_dirs[i], river_dirs[j]
            p = all_steered_probs[di]
            q = all_steered_probs[dj]
            # Symmetrized KL
            kl_pq = (p * (p.log() - q.log())).sum().item()
            kl_qp = (q * (q.log() - p.log())).sum().item()
            skl = (kl_pq + kl_qp) / 2
            line = "  Dir %d vs Dir %d: KL=%.4f (sym=%.4f)" % (di, dj, kl_pq, skl)
            print(line)
            lines.append(line)

    if args.outfile:
        with open(args.outfile, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print("\nSaved to %s" % args.outfile)


if __name__ == '__main__':
    main()
