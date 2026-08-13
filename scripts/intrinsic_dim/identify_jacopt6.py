#!/usr/bin/env python3
"""What is jac-opt point 6? Nearest token activations, output distribution,
and generation behavior.

Three probes (qwen3-1.7b, L2, jac-opt point 6, context = FineWeb sample 6):
  1. Activation-space neighbors: nearest real token activations (cosine and
     L2) from a bank of 2000 FineWeb seq-10 sequences x all positions,
     with the tokens they belong to. Baseline: same stats for real act 6.
  2. Output distribution at the point (patched into ctx 6): top-10 tokens,
     entropy, JS divergence to the real-activation-6 distribution.
  3. Generation: greedy 40-token continuations with L2 last-prompt-position
     patched by (a) nothing, (b) real act 6 (sanity), (c) jac-opt 6.

Output: results/intrinsic_dim/identify_jacopt6/{results.pkl, report.txt}
"""

import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/steering-plateaus')

import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, forward_from_layer, _get_blocks
from src.data import load_fineweb_fixed_length

OUT_DIR = 'results/intrinsic_dim/identify_jacopt6'
MODEL = 'qwen3-1.7b'
LAYER = 2
IDX = 6
BANK_N = 2000
BANK_SEQ = 10
TOPK = 10
GEN_TOKENS = 40


def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(a, b):
        return float((a * ((a + eps) / (b + eps)).log()).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(s=''):
        print(s, flush=True)
        lines.append(s)

    model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    # context 6 + real act 6
    toks30 = load_fineweb_fixed_length(30, tokenizer, seq_len=5, seed=42,
                                       cache_dir='/workspace/fineweb_cache')
    tids6 = toks30[IDX]
    prompt_text = tokenizer.decode(tids6)
    with torch.no_grad():
        outs = model(tids6.unsqueeze(0).to(device), output_hidden_states=True)
    h = outs.hidden_states[LAYER + 1]
    ctx6 = h[:, :-1, :].to(model_dtype)
    a_real = h[:, -1, :].squeeze(0).float()

    opt = torch.load('results/intrinsic_dim/jac_optimized_data.pt',
                     map_location='cpu', weights_only=False)
    b_opt = opt['X_opt'].float()[IDX].to(device)
    log(f'prompt (sample {IDX}): {prompt_text!r}')
    log(f'||real||={a_real.norm():.2f}  ||jacopt||={b_opt.norm():.2f}  '
        f'cos(real, jacopt)={F.cosine_similarity(a_real, b_opt, dim=0):.3f}')

    # --- 1. activation bank: all positions of 2000 seq-10 FineWeb samples
    bank_tok_lists = load_fineweb_fixed_length(
        BANK_N, tokenizer, seq_len=BANK_SEQ, seed=42,
        cache_dir='/workspace/fineweb_cache')
    accs, tok_ids = [], []
    with torch.no_grad():
        for st in range(0, BANK_N, 64):
            batch = torch.stack(bank_tok_lists[st:st + 64]).to(device)
            o = model(batch, output_hidden_states=True)
            accs.append(o.hidden_states[LAYER + 1].float().cpu())
            tok_ids.append(batch.cpu())
    bank = torch.cat(accs).reshape(-1, a_real.shape[0])       # [N*seq, d]
    bank_toks = torch.cat(tok_ids).reshape(-1)                # [N*seq]
    log(f'bank: {bank.shape[0]} activations '
        f'({BANK_N} seqs x {BANK_SEQ} positions)')

    for name, q in [('jac-opt 6', b_opt.cpu()), ('real 6', a_real.cpu())]:
        cos = F.cosine_similarity(bank, q[None], dim=-1)
        l2 = (bank - q[None]).norm(dim=-1)
        log(f'\n=== nearest bank activations to {name} ===')
        for metric, vals, best in [('cos', cos, cos.topk(TOPK).indices),
                                   ('L2', l2, (-l2).topk(TOPK).indices)]:
            items = []
            for ix in best:
                tok = tokenizer.decode([int(bank_toks[ix])])
                items.append(f'{tok!r}({vals[ix]:.2f})')
            log(f'  top-{TOPK} by {metric}: ' + '  '.join(items))
        log(f'  best cos={cos.max():.3f}  best L2={l2.min():.2f}  '
            f'(bank self-scale: median NN L2 of a real act ~ see pkl)')

    # --- 2. output distributions at ctx6
    with torch.no_grad():
        pts = torch.stack([a_real, b_opt.float()])
        logits = forward_from_layer(model, ctx6, pts.unsqueeze(1).to(model_dtype),
                                    LAYER).float()
    p_real = F.softmax(logits[0], dim=-1).cpu()
    p_opt = F.softmax(logits[1], dim=-1).cpu()
    jsd = js_divergence(p_opt, p_real)
    log(f'\n=== output distributions (patched into ctx {IDX}) ===')
    for name, p in [('real 6', p_real), ('jac-opt 6', p_opt)]:
        ent = float(-(p * (p + 1e-12).log()).sum())
        top = p.topk(TOPK)
        toks = '  '.join(f'{tokenizer.decode([int(i)])!r}({v:.3f})'
                         for v, i in zip(top.values, top.indices))
        log(f'  {name}: entropy={ent:.2f} nats;  top-{TOPK}: {toks}')
    log(f'  JSD(jac-opt || real) = {jsd:.4f}  (max ln2={np.log(2):.3f})')

    # --- 3. greedy generations with L2 patch at last prompt position
    blocks = _get_blocks(model)
    patch_vec = {'v': None}
    prompt_len = tids6.shape[0]

    def patch_hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] == prompt_len and patch_vec['v'] is not None:
            hs = hs.clone()
            hs[:, -1, :] = patch_vec['v'].to(hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return output

    handle = blocks[LAYER].register_forward_hook(patch_hook)
    gens = {}
    try:
        for name, vec in [('unpatched', None), ('real 6', a_real),
                          ('jac-opt 6', b_opt.float())]:
            patch_vec['v'] = vec
            with torch.no_grad():
                out_ids = model.generate(
                    tids6.unsqueeze(0).to(device),
                    max_new_tokens=GEN_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            gens[name] = tokenizer.decode(out_ids[0][prompt_len:])
    finally:
        handle.remove()
    log(f'\n=== greedy continuations of {prompt_text!r} ===')
    for name, g in gens.items():
        log(f'  [{name}]: {g!r}')

    with open(os.path.join(OUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump({'p_real_top': p_real.topk(50), 'p_opt_top': p_opt.topk(50),
                     'jsd': jsd, 'generations': gens,
                     'prompt': prompt_text}, f)
    with open(os.path.join(OUT_DIR, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved {os.path.join(OUT_DIR, 'results.pkl')} and report.txt")


if __name__ == '__main__':
    main()
