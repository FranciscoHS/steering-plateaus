#!/usr/bin/env python3
"""1D perturbation sweeps along contrastive directions, with the starting
activations matched to each direction's prompt source.

Covers:
  - 12 single-axis directions (Gender, Sentiment, Age, Era, Tense,
    TensePresent, Number, Person, Status, Honesty, HonestyShort, Refusal)
  - 2 register directions (Formal from casual, Literary from slang)

For each direction we:
  1. Compute the DoM direction at perturb_layer (target_side − start_side)
  2. Extract activations at perturb_layer from start_side prompts (clean
     holdout if available, otherwise in-sample training prompts)
  3. Run a 1D spherical sweep along the direction (0..MAX_ANGLE, N_STEPS)
  4. Run N_RAND random unit vectors swept from the same activations

Output: results/fig4_robustness/v2_S1d_<target>_L<layer>.pkl
  {
    'angles_deg': (n_steps,),
    'layer': perturb_layer, 'measure_layer': measure_layer,
    'sources': { dir_name: {
        'l2_dir':  (n_act, n_steps),
        'l2_rand': (n_rand, n_act, n_steps),
        'starting_module': str, 'starting_attr': str, 'in_sample': bool,
    }},
  }

Usage:
  python scripts/perturbation/multisource_1d_sweeps.py --target gemma --layer 2
"""
import os, sys, pickle, argparse, time, importlib
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks, forward_from_layer_to_layer

OUT_DIR = 'results/fig4_robustness'

TARGETS = {
    'gemma':   dict(model='gemma-2-9b'),
    'qwen':    dict(model='qwen3-1.7b'),
    'llama':   dict(model='llama-3.1-8b'),
    'mistral': dict(model='mistral-7b-v0.3'),
    'aya':     dict(model='aya-expanse-8b'),
    'yi':      dict(model='yi-1.5-9b'),
}

# (dir_name, dom_module, start_attr, target_attr, starting_prompts_module,
#  starting_prompts_attr, in_sample)
# starting_prompts_* may be None to default to dom_module/start_attr.
SINGLE_AXIS = [
    # name, dom_module,                       start_attr,             target_attr,             holdout_module,                  holdout_attr,                   in_sample
    ('Gender',       'data.gender_prompts',         'MALE_PROMPTS',         'FEMALE_PROMPTS',        'data.male_prompts_holdout',     'MALE_PROMPTS_HOLDOUT',         False),
    ('Sentiment',    'data.sentiment_prompts',      'POSITIVE_PROMPTS',     'NEGATIVE_PROMPTS',      'data.positive_prompts_holdout', 'POSITIVE_PROMPTS_HOLDOUT',     False),
    ('Age',          'data.age_prompts',            'YOUNG_PROMPTS',        'OLD_PROMPTS',           None,                            None,                           True),
    ('Era',          'data.era_prompts',            'MEDIEVAL_PROMPTS',     'MODERN_PROMPTS',        None,                            None,                           True),
    ('Tense',        'data.tense_prompts',          'PAST_PROMPTS',         'FUTURE_PROMPTS',        None,                            None,                           True),
    ('TensePresent', 'data.tense_present_prompts',  'PRESENT_PROMPTS',      'PAST_PROMPTS',          None,                            None,                           True),
    ('Number',       'data.number_prompts',         'SINGULAR_PROMPTS',     'PLURAL_PROMPTS',        None,                            None,                           True),
    ('Person',       'data.person_prompts',         'FIRST_PROMPTS',        'THIRD_PROMPTS',         None,                            None,                           True),
    ('Status',       'data.status_prompts',         'ROYAL_PROMPTS',        'COMMONER_PROMPTS',      None,                            None,                           True),
    ('Honesty',      'data.honesty_prompts',        'HONEST_PROMPTS',       'DISHONEST_PROMPTS',     None,                            None,                           True),
    ('HonestyShort', 'data.honesty_short_prompts',  'HONEST_SHORT_PROMPTS', 'DISHONEST_SHORT_PROMPTS', None,                          None,                           True),
    ('Refusal',      'data.refusal_prompts',        'COMPLY_PROMPTS',       'REFUSE_PROMPTS',        None,                            None,                           True),
]

# Multi-key register directions: pick start key + target key from PROMPT_SETS dicts
REGISTER = [
    # name, dom_module, start_key, target_key
    ('Formal',   'data.register_prompts', 'casual', 'formal'),
    ('Literary', 'data.register_prompts', 'slang',  'literary'),
]

N_ACT = 30
N_STEPS = 51
MAX_ANGLE = 75.0
N_RAND = 10
SEED = 42
BATCH = 64


def ts(): return time.strftime('%H:%M:%S')


def get_acts_lastpos(model, tokenizer, device, layer, prompts):
    """Return (list of context tensors, (n,D) last-pos activations) at L+1."""
    contexts, activations = [], []
    for p in prompts:
        tok = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        activations.append(h[:, -1, :].squeeze(0).float().cpu())
    return contexts, torch.stack(activations)


def compute_dom_from_prompts(model, tokenizer, device, layer, a_prompts, b_prompts):
    """direction = b_mean − a_mean (i.e. points from a-side toward b-side)."""
    def _acts(prompts):
        out = []
        for p in prompts:
            tok = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                o = model(**tok, output_hidden_states=True)
            out.append(o.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
        return torch.stack(out)
    a = _acts(a_prompts); b = _acts(b_prompts)
    return F.normalize((b - a).mean(0), dim=0)


def sweep_1d(model, ctx_d, a_dev, d_dev, layer, measure_layer,
             angles_rad, model_dtype, device):
    R = a_dev.norm(); a_unit = a_dev / R
    d_o = F.normalize(d_dev - (d_dev @ a_unit) * a_unit, dim=0)
    s = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    c = torch.sqrt(torch.clamp(1.0 - s**2, min=0.0))
    pts = (R * (c.unsqueeze(-1) * a_unit.unsqueeze(0)
                + s.unsqueeze(-1) * d_o.unsqueeze(0))).to(model_dtype)
    n = len(angles_rad)
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_dev[None, None].to(model_dtype),
            layer, measure_layer).float()
        for st in range(0, n, BATCH):
            ed = min(st + BATCH, n)
            pf = forward_from_layer_to_layer(
                model, ctx_d, pts[st:ed].unsqueeze(1),
                layer, measure_layer).float()
            out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


def run_one_source(model, tokenizer, device, perturb_layer, measure_layer,
                   model_dtype, name, direction, start_prompts,
                   angles_rad, rng):
    print(f'[{ts()}] {name}: extracting {len(start_prompts)} starting acts',
          flush=True)
    contexts, activations = get_acts_lastpos(
        model, tokenizer, device, perturb_layer, start_prompts)
    d_model = activations.shape[-1]
    d_dev = direction.to(device)

    l2_dir = np.zeros((N_ACT, len(angles_rad)), dtype=np.float32)
    t0 = time.time()
    for ai in range(N_ACT):
        ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
        a_dev = activations[ai].to(device=device, dtype=torch.float32)
        l2_dir[ai] = sweep_1d(model, ctx_d, a_dev, d_dev, perturb_layer,
                               measure_layer, angles_rad, model_dtype, device)
    rand_dirs = F.normalize(
        torch.tensor(rng.randn(N_RAND, d_model), dtype=torch.float32), dim=-1)
    l2_rand = np.zeros((N_RAND, N_ACT, len(angles_rad)), dtype=np.float32)
    for ri in range(N_RAND):
        d_dev_r = rand_dirs[ri].to(device)
        for ai in range(N_ACT):
            ctx_d = contexts[ai].to(device=device, dtype=model_dtype)
            a_dev = activations[ai].to(device=device, dtype=torch.float32)
            l2_rand[ri, ai] = sweep_1d(model, ctx_d, a_dev, d_dev_r,
                                        perturb_layer, measure_layer,
                                        angles_rad, model_dtype, device)
    print(f'  [{ts()}] {name}: dir+random done ({time.time()-t0:.1f}s)',
          flush=True)
    return l2_dir, l2_rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(TARGETS))
    ap.add_argument('--layer', type=int, default=2)
    args = ap.parse_args()
    cfg = TARGETS[args.target]

    out_path = os.path.join(OUT_DIR,
                            f'v2_S1d_{args.target}_L{args.layer}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} exists, exiting'); return

    print(f'[{ts()}] loading {cfg["model"]}', flush=True)
    model, tokenizer, device = load_model(cfg['model'], dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    model_dtype = next(model.parameters()).dtype
    print(f'  n_layers={n_layers}  perturb={args.layer}  measure={measure_layer}',
          flush=True)

    angles = np.linspace(0, MAX_ANGLE, N_STEPS)
    angles_rad = np.deg2rad(angles)
    rng = np.random.RandomState(SEED)

    sources_out = {}

    # Single-axis directions
    for name, dom_mod, a_attr, b_attr, hold_mod, hold_attr, in_sample in SINGLE_AXIS:
        try:
            mod = importlib.import_module(dom_mod)
            a_prompts = getattr(mod, a_attr)[:N_ACT]
            b_prompts = getattr(mod, b_attr)[:N_ACT]
        except Exception as e:
            print(f'  SKIP {name}: cannot load DoM prompts ({e})')
            continue
        d = compute_dom_from_prompts(model, tokenizer, device, args.layer,
                                      a_prompts, b_prompts)
        if hold_mod is not None:
            try:
                hm = importlib.import_module(hold_mod)
                start_prompts = getattr(hm, hold_attr)[:N_ACT]
                start_module, start_attr_used = hold_mod, hold_attr
            except Exception as e:
                print(f'  WARN {name}: holdout missing ({e}), falling back to in-sample')
                start_prompts = a_prompts
                start_module, start_attr_used = dom_mod, a_attr
                in_sample = True
        else:
            start_prompts = a_prompts
            start_module, start_attr_used = dom_mod, a_attr
        l2_dir, l2_rand = run_one_source(
            model, tokenizer, device, args.layer, measure_layer, model_dtype,
            name, d, start_prompts, angles_rad, rng)
        sources_out[name] = {
            'l2_dir': l2_dir, 'l2_rand': l2_rand,
            'starting_module': start_module, 'starting_attr': start_attr_used,
            'in_sample': in_sample,
        }
        # save incrementally so plotting can pick up partial results
        out = {'angles_deg': angles, 'layer': args.layer,
               'measure_layer': measure_layer, 'model': args.target,
               'sources': sources_out}
        os.makedirs(OUT_DIR, exist_ok=True)
        tmp = out_path + '.tmp'
        with open(tmp, 'wb') as f: pickle.dump(out, f)
        os.replace(tmp, out_path)

    # Register directions (multi-key)
    for name, dom_mod, a_key, b_key in REGISTER:
        try:
            mod = importlib.import_module(dom_mod)
            sets = getattr(mod, 'PROMPT_SETS')
            a_prompts = [s[a_key] for s in sets][:N_ACT]
            b_prompts = [s[b_key] for s in sets][:N_ACT]
        except Exception as e:
            print(f'  SKIP {name}: cannot load REG prompts ({e})')
            continue
        d = compute_dom_from_prompts(model, tokenizer, device, args.layer,
                                      a_prompts, b_prompts)
        l2_dir, l2_rand = run_one_source(
            model, tokenizer, device, args.layer, measure_layer, model_dtype,
            name, d, a_prompts, angles_rad, rng)
        sources_out[name] = {
            'l2_dir': l2_dir, 'l2_rand': l2_rand,
            'starting_module': dom_mod, 'starting_attr': a_key,
            'in_sample': True,
        }
        out = {'angles_deg': angles, 'layer': args.layer,
               'measure_layer': measure_layer, 'model': args.target,
               'sources': sources_out}
        tmp = out_path + '.tmp'
        with open(tmp, 'wb') as f: pickle.dump(out, f)
        os.replace(tmp, out_path)

    print(f'[{ts()}] all done → {out_path}  ({len(sources_out)} sources)',
          flush=True)


if __name__ == '__main__':
    main()
