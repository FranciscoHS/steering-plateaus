#!/usr/bin/env python3
"""
Experiment 2: steering with the learned (d1, d2) pair.

For a couple of optimized (training) prompts and a couple of held-out prompts,
generate 50 tokens:
  - unsteered;
  - steered along d1 at 15/30/60 deg;
  - steered along d2 at 15/30/60 deg.

Steering = geodesic rotation (exp-map, norm-preserving) of the last-token
residual at the directions' layer, applied at every decode step (sustained):
    a' = R (cos theta * a_hat + sin theta * d_perp)
with d_perp = direction orthogonalized against a_hat (matches the perturbation
chart). Greedy decoding for reproducibility.

Dumps everything (prompt text, condition, angle, generated text) to JSON + a
readable TXT.

Usage:
    python scripts/superellipse/steer_pair_generate.py \
        --dict_pkl results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl
"""
import os, sys, json, pickle, argparse
sys.path.insert(0, '.')
import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length


def make_steer_hook(direction, angle_rad):
    """Forward hook: geodesic-rotate the last-token residual by angle along direction."""
    cos_t, sin_t = float(np.cos(angle_rad)), float(np.sin(angle_rad))

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        a = h[:, -1, :]                                  # [B, d]
        R = a.norm(dim=-1, keepdim=True)
        a_hat = a / R
        d = direction.to(device=a.device, dtype=a.dtype)
        proj = (a_hat * d).sum(-1, keepdim=True)
        d_perp = d.unsqueeze(0) - proj * a_hat
        d_perp = d_perp / (d_perp.norm(dim=-1, keepdim=True) + 1e-10)
        steered = R * (cos_t * a_hat + sin_t * d_perp)
        h = h.clone()
        h[:, -1, :] = steered.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


@torch.no_grad()
def generate(model, tokenizer, input_ids, layer, device, direction=None,
             angle_rad=0.0, max_new_tokens=50, temperature=0.0, gen_seed=0):
    handle = None
    if direction is not None and angle_rad > 0:
        block = _get_blocks(model)[layer]
        handle = block.register_forward_hook(make_steer_hook(direction, angle_rad))
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        torch.manual_seed(gen_seed)  # reproducible single sample
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)
    try:
        out = model.generate(input_ids, **gen_kwargs)
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_pkl', type=str,
                    default='results/superellipse_dict/qwen3-1.7b_L2_a30/results.pkl')
    ap.add_argument('--n_prompts', type=int, default=2,
                    help='prompts per set (optimized / held-out)')
    ap.add_argument('--prompts', type=str, nargs='+', default=None,
                    help='custom open-ended text prompts; overrides the '
                         'FineWeb optimized/held-out sets')
    ap.add_argument('--angles', type=float, nargs='+', default=[15.0, 30.0, 60.0])
    ap.add_argument('--max_new_tokens', type=int, default=50)
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='>0 enables sampling (one sample per prompt); 0 = greedy')
    ap.add_argument('--gen_seed', type=int, default=0)
    ap.add_argument('--fresh_seed', type=int, default=999)
    ap.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--results_dir', type=str, default=None)
    args = ap.parse_args()

    with open(args.dict_pkl, 'rb') as f:
        dd = pickle.load(f)
    directions = dd['directions'].float()
    da = dd['args']
    model_name = da['model']; layer = da['layer']; seq_len = da['seq_len']
    train_seed = da['seed']; n_act_train = da['n_act']

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.dict_pkl), 'steering')
    os.makedirs(args.results_dir, exist_ok=True)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    print(f'Loading {model_name} (dtype={args.dtype})...')
    model, tokenizer, device = load_model(model_name, dtype=dtype_map[args.dtype])
    d1, d2 = directions[0], directions[1]

    if args.prompts:
        # Custom open-ended prompts (not tied to a specific anchor set).
        prompt_sets = {
            'custom': [tokenizer(p, return_tensors='pt').input_ids[0] for p in args.prompts],
        }
    else:
        # Prompt sets: same FineWeb sequences whose last-token L2 activations
        # are the optimized anchors (seed=train_seed) vs held-out (fresh_seed).
        tok_opt = load_fineweb_fixed_length(max(100, n_act_train), tokenizer, seq_len, seed=train_seed)
        tok_fresh = load_fineweb_fixed_length(max(args.n_prompts, 30), tokenizer, seq_len, seed=args.fresh_seed)
        prompt_sets = {
            'optimized': tok_opt[:args.n_prompts],
            'held_out': tok_fresh[:args.n_prompts],
        }

    runs = []
    for set_name, toks in prompt_sets.items():
        for pi, tok in enumerate(toks):
            input_ids = tok.unsqueeze(0).to(device)
            prompt_text = tokenizer.decode(tok, skip_special_tokens=True)
            print(f'\n[{set_name} #{pi}] prompt: {prompt_text!r}')

            base = generate(model, tokenizer, input_ids, layer, device,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, gen_seed=args.gen_seed)
            runs.append({'set': set_name, 'prompt_idx': pi, 'prompt': prompt_text,
                         'direction': None, 'angle_deg': 0.0, 'generated': base})
            print(f'  unsteered: {base!r}')

            for dname, d in [('d1', d1), ('d2', d2)]:
                for ang in args.angles:
                    g = generate(model, tokenizer, input_ids, layer, device,
                                 direction=d, angle_rad=np.deg2rad(ang),
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, gen_seed=args.gen_seed)
                    runs.append({'set': set_name, 'prompt_idx': pi, 'prompt': prompt_text,
                                 'direction': dname, 'angle_deg': ang, 'generated': g})
                    print(f'  {dname} @ {ang:.0f}deg: {g!r}')

    out_json = os.path.join(args.results_dir, 'steering_generations.json')
    with open(out_json, 'w') as f:
        json.dump({'args': vars(args), 'model': model_name, 'layer': layer,
                   'cos_d1_d2': float((d1 @ d2).item()), 'runs': runs}, f, indent=2)

    out_txt = os.path.join(args.results_dir, 'steering_generations.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Steering with learned pair  ({model_name}, layer {layer})\n')
        f.write(f'cos(d1,d2) = {(d1 @ d2).item():.4f}\n\n')
        cur = None
        for r in runs:
            key = (r['set'], r['prompt_idx'])
            if key != cur:
                cur = key
                f.write(f'\n{"="*80}\n[{r["set"]} #{r["prompt_idx"]}] PROMPT: {r["prompt"]!r}\n{"="*80}\n')
            tag = 'UNSTEERED' if r['direction'] is None else f'{r["direction"]} @ {r["angle_deg"]:.0f}deg'
            f.write(f'\n--- {tag} ---\n{r["generated"]}\n')

    print(f'\nSaved: {out_json}')
    print(f'Saved: {out_txt}')


if __name__ == '__main__':
    main()
