#!/usr/bin/env python3
"""Spherical steering generation along dictionary atoms.

For each atom and each angle, rotate the layer-L residual toward the atom
direction (norm-preserving, orthogonal component) at ALL positions during
generation, and greedily decode from a single open-ended prompt. Saves all
rollouts to a .txt (readable) and .json.

    python scripts/perturbation/steer_dict_atoms.py \
      --dict_path .../dictionary.pt --atoms 1784,8660,8777,13090,15476 \
      --model qwen3-1.7b --layer 14 --angles 0,20,40,60 \
      --prompt "Let me tell you about" --max_new_tokens 60 --out .../steer
"""
import sys, os, argparse, json
sys.path.insert(0, '.')
import numpy as np
import torch
from src.model import load_model, _get_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict_path', default=None)
    ap.add_argument('--atoms', default=None)
    ap.add_argument('--direction_pkls', nargs='+', default=None,
                    help="steer along directions from pkls ({'directions':[vec]}) "
                         "instead of dict atoms")
    ap.add_argument('--model', default='qwen3-1.7b')
    ap.add_argument('--layer', type=int, default=14)
    ap.add_argument('--angles', default='0,20,40,60')
    ap.add_argument('--prompt', default='Let me tell you about')
    ap.add_argument('--prompts', default=None,
                    help='pipe-separated prompts; overrides --prompt')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='0 = greedy; >0 = sample at this temperature')
    ap.add_argument('--n_samples', type=int, default=1,
                    help='samples per (prompt,atom,angle) when temperature>0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_new_tokens', type=int, default=60)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = args.prompts.split('|') if args.prompts else [args.prompt]

    angles = [float(x) for x in args.angles.split(',')]

    model, tokenizer, device = load_model(args.model)
    blocks = _get_blocks(model)

    # named_dirs: list of (label, unit_vec) from either direction pkls or dict atoms
    named_dirs = []
    if args.direction_pkls:
        import pickle
        for p in args.direction_pkls:
            v = pickle.load(open(p, 'rb'))['directions'][0].float()
            named_dirs.append((os.path.splitext(os.path.basename(p))[0], v / v.norm()))
    else:
        blob = torch.load(args.dict_path, map_location='cpu', weights_only=False)
        D = blob['dictionary'].float()
        D = D / D.norm(dim=-1, keepdim=True)
        for idx in [int(x) for x in args.atoms.split(',')]:
            named_dirs.append((f'atom{idx}', D[idx]))

    state = {'d': None, 'theta': 0.0}

    def hook(module, inp, out):
        if state['theta'] == 0.0:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        d = state['d'].to(hidden.device, hidden.dtype)
        R = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        a_unit = hidden / R
        proj = (a_unit * d).sum(-1, keepdim=True)
        v_perp = d - proj * a_unit
        v_perp = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ct, st = np.cos(state['theta']), np.sin(state['theta'])
        new = R * (ct * a_unit + st * v_perp)
        if isinstance(out, tuple):
            return (new,) + out[1:]
        return new

    sampling = args.temperature > 0
    n_samples = args.n_samples if sampling else 1

    def gen_one(prompt, pi, s):
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        if sampling:
            torch.manual_seed(args.seed + 9973 * s + 31 * pi)
            gk = dict(do_sample=True, temperature=args.temperature, top_p=0.95)
        else:
            gk = dict(do_sample=False)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id, **gk)
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    nonzero = [a for a in angles if a != 0]
    handle = blocks[args.layer].register_forward_hook(hook)
    rollouts = []
    try:
        # baseline (theta=0 -> hook is a no-op) generated ONCE per (prompt, sample)
        if any(a == 0 for a in angles):
            state['theta'] = 0.0
            for pi, prompt in enumerate(prompts):
                for s in range(n_samples):
                    text = gen_one(prompt, pi, s)
                    rollouts.append({'dir': 'baseline', 'angle': 0.0, 'prompt': prompt,
                                     'sample': s, 'text': text})
                    print(f'[baseline p{pi} s{s}] {text!r}', flush=True)
        for label, dvec in named_dirs:
            state['d'] = dvec
            for ang in nonzero:
                state['theta'] = np.deg2rad(ang)
                for pi, prompt in enumerate(prompts):
                    for s in range(n_samples):
                        text = gen_one(prompt, pi, s)
                        rollouts.append({'dir': label, 'angle': ang, 'prompt': prompt,
                                         'sample': s, 'text': text})
                        print(f'[{label} @ {ang:.0f}deg p{pi} s{s}] {text!r}', flush=True)
    finally:
        handle.remove()

    with open(os.path.join(args.out, 'steer_rollouts.json'), 'w') as f:
        json.dump({'args': vars(args), 'rollouts': rollouts}, f, indent=2)
    with open(os.path.join(args.out, 'steer_rollouts.txt'), 'w') as f:
        f.write(f'model={args.model} L{args.layer} temperature={args.temperature}\n'
                f'prompts: {prompts}\n')
        for r in rollouts:
            f.write(f'\n=== {r["dir"]} @ {r["angle"]:.0f}deg | '
                    f'prompt={r["prompt"]!r} | sample {r["sample"]} ===\n{r["text"]}\n')
    print(f'\nSaved {args.out}/steer_rollouts.{{json,txt}}')


if __name__ == '__main__':
    main()
