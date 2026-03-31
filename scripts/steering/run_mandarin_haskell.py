#!/usr/bin/env python3
"""
Mandarin × Haskell composition experiment on Gemma-2-9B.

Tests whether two independently-extracted steering directions (Mandarin and
Haskell) compose additively. Feed the model Python code, steer simultaneously
toward Mandarin and Haskell via sequential hooks, check whether the output
becomes Mandarin Haskell.

Steps:
  1. Load Gemma-2-9B
  2. Extract contrastive directions at layer 2 (EN→ZH and Python→Haskell)
  3. Run perturbation sweeps to validate both directions
  4. 2×2 steered generation: baseline, +Mandarin, +Haskell, +both

Usage:
    python scripts/steering/run_mandarin_haskell.py [--skip-sweeps] [--angle 30]
"""

import sys, os, json, pickle, time, argparse
import numpy as np
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.model import load_model, _get_blocks, forward_from_layer_to_layer


# ── Config ──────────────────────────────────────────────────────────────────
LAYER = 2
N_TOKENS = 60
TEMPERATURE = 0.7
SEED = 42
N_PROMPTS = 10  # how many holdout prompts to use for generation


# ── Direction extraction ────────────────────────────────────────────────────

def compute_contrastive_direction(model, tokenizer, device, layer,
                                  prompts_a, prompts_b):
    """Compute negated diff-of-means direction (pointing A → B).

    Returns unit vector in the direction -(mean_A - mean_B) = mean_B - mean_A.
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
    neg_dom = -dom  # points A → B
    print(f"    DoM magnitude (pre-norm): {mean_diff.norm():.2f}")
    return neg_dom


# ── Perturbation sweep ──────────────────────────────────────────────────────

def sweep_one_spherical(model, ctx, act, direction, layer, measure_layer,
                        device, angles_rad, batch_size=64):
    """Spherical perturbation sweep, returns L2 distances at measure_layer."""
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

    n_points = len(angles_rad)
    result = np.zeros(n_points)
    with torch.no_grad():
        ref = forward_from_layer_to_layer(
            model, ctx_d, a_d[None, None], layer, measure_layer).float()
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch_points = all_points[start:end].unsqueeze(1)
            pf = forward_from_layer_to_layer(
                model, ctx_d, batch_points, layer, measure_layer).float()
            result[start:end] = (pf - ref).norm(dim=-1).cpu().numpy()
    return result


def extract_activations_holdout(model, tokenizer, device, layer, prompts):
    """Extract context and last-token activations from prompts."""
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


# ── Steered generation ──────────────────────────────────────────────────────

def generate_steered_composed(model, tokenizer, device, prompt,
                              directions_and_angles, n_tokens, temperature,
                              layer):
    """Generate with multiple spherical steering hooks applied sequentially.

    Args:
        directions_and_angles: list of (direction, angle_deg) tuples.
            Each hook rotates the activation toward its direction by angle_deg.
            Hooks are applied sequentially on the same layer.
    """
    # Filter out zero-angle directions
    active = [(d, a) for d, a in directions_and_angles if a != 0]

    if not active:
        # No steering — plain generation
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        generated_ids = []
        for step in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids,
                                   torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Precompute trig for each direction
    hooks_info = []
    for direction, angle_deg in active:
        cos_theta = np.cos(np.radians(angle_deg))
        sin_theta = np.sin(np.radians(angle_deg))
        hooks_info.append((direction, cos_theta, sin_theta))

    tokens = tokenizer(prompt, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    first_steer_pos = prompt_len - 1

    generated_ids = []
    for step in range(n_tokens):
        handles = []

        # Register one hook per direction, all on the same layer.
        # PyTorch forward hooks execute in registration order, so these
        # are applied sequentially: first Mandarin, then Haskell, etc.
        for d, cos_theta, sin_theta in hooks_info:
            def make_hook(d_vec, ct, st):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    steer_slice = hidden[0, first_steer_pos:, :]
                    if steer_slice.shape[0] == 0:
                        return output
                    R = steer_slice.norm(dim=-1, keepdim=True)
                    a_unit = steer_slice / (R + 1e-10)
                    proj = (a_unit * d_vec).sum(dim=-1, keepdim=True)
                    v_perp = d_vec.unsqueeze(0) - proj * a_unit
                    v_perp = v_perp / (v_perp.norm(dim=-1, keepdim=True) + 1e-10)
                    steered = R * (ct * a_unit + st * v_perp)
                    hidden[0, first_steer_pos:, :] = steered
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook

            layers = _get_blocks(model)
            handle = layers[layer].register_forward_hook(make_hook(d, cos_theta, sin_theta))
            handles.append(handle)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids,
                               torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Character classifiers ───────────────────────────────────────────────────

def is_cjk_char(ch):
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: return True
    if 0x3400 <= cp <= 0x4DBF: return True
    if 0x20000 <= cp <= 0x2A6DF: return True
    if 0xF900 <= cp <= 0xFAFF: return True
    if 0x3000 <= cp <= 0x303F: return True
    if 0xFF00 <= cp <= 0xFFEF: return True
    return False


def cjk_fraction(text):
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) == 0: return 0.0
    return sum(1 for ch in chars if is_cjk_char(ch)) / len(chars)


def haskell_keywords_fraction(text):
    """Fraction of words that are Haskell-ish keywords or syntax."""
    hs_keywords = {
        'module', 'where', 'import', 'data', 'type', 'newtype', 'class',
        'instance', 'deriving', 'let', 'in', 'case', 'of', 'do', 'if',
        'then', 'else', 'qualified', 'as', 'hiding', 'forall', 'infixl',
        'infixr', 'infix', 'IO', 'Int', 'String', 'Bool', 'Maybe', 'Just',
        'Nothing', 'Either', 'Left', 'Right', 'True', 'False', 'putStrLn',
        'show', 'map', 'filter', 'foldr', 'foldl', 'pure', 'return',
    }
    # Also count :: and -> as Haskell markers
    hs_syntax_count = text.count('::') + text.count('->')
    words = text.split()
    if len(words) == 0: return 0.0
    keyword_count = sum(1 for w in words if w in hs_keywords)
    return (keyword_count + hs_syntax_count) / len(words)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-sweeps', action='store_true',
                        help='Skip perturbation sweep validation')
    parser.add_argument('--angle', type=float, default=30,
                        help='Steering angle in degrees (default: 30)')
    parser.add_argument('--n-prompts', type=int, default=N_PROMPTS,
                        help='Number of holdout prompts for generation')
    args = parser.parse_args()

    results_dir = 'results/mandarin_haskell/gemma-2-9b'
    os.makedirs(results_dir, exist_ok=True)

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading Gemma-2-9B")
    print("=" * 60)
    model, tokenizer, device = load_model('gemma-2-9b', dtype=torch.bfloat16)
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  Layers: {n_layers}, measure_layer: {measure_layer}")

    # ── Step 2: Extract contrastive directions at layer 2 ───────────────
    print("\n" + "=" * 60)
    print("Step 2: Extracting contrastive directions at layer 2")
    print("=" * 60)

    from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS
    from data.programming_prompts import PYTHON_PROMPTS, HASKELL_PROMPTS

    print("\n  Mandarin direction (EN → ZH):")
    d_mandarin = compute_contrastive_direction(
        model, tokenizer, device, LAYER,
        ENGLISH_PROMPTS, CHINESE_PROMPTS)

    print("\n  Haskell direction (Python → Haskell):")
    d_haskell = compute_contrastive_direction(
        model, tokenizer, device, LAYER,
        PYTHON_PROMPTS, HASKELL_PROMPTS)

    cos_sim = (d_mandarin @ d_haskell).item()
    print(f"\n  Cosine similarity between directions: {cos_sim:.4f}")

    # Save directions
    dir_data = {
        'd_mandarin': d_mandarin,
        'd_haskell': d_haskell,
        'cosine_similarity': cos_sim,
        'layer': LAYER,
    }
    dir_path = os.path.join(results_dir, 'directions.pkl')
    with open(dir_path, 'wb') as f:
        pickle.dump(dir_data, f)
    print(f"  Saved: {dir_path}")

    # ── Step 3: Perturbation sweeps ─────────────────────────────────────
    if not args.skip_sweeps:
        print("\n" + "=" * 60)
        print("Step 3: Perturbation sweeps to validate directions")
        print("=" * 60)

        from data.python_prompts_holdout import PYTHON_PROMPTS_HOLDOUT
        sweep_prompts = PYTHON_PROMPTS_HOLDOUT[:10]

        print(f"  Extracting activations for {len(sweep_prompts)} holdout prompts...")
        contexts, acts = extract_activations_holdout(
            model, tokenizer, device, LAYER, sweep_prompts)

        angles_deg = np.linspace(0, 90, 91)
        angles_rad = np.radians(angles_deg)

        sweep_results = {'angles_deg': angles_deg}

        for dir_name, direction in [('mandarin', d_mandarin), ('haskell', d_haskell)]:
            print(f"\n  Sweeping {dir_name} direction...")
            all_curves = []
            for i in range(len(sweep_prompts)):
                curve = sweep_one_spherical(
                    model, contexts[i], acts[i], direction,
                    LAYER, measure_layer, device, angles_rad)
                all_curves.append(curve)
                if (i + 1) % 5 == 0:
                    print(f"    {i+1}/{len(sweep_prompts)} prompts done")
            sweep_results[dir_name] = np.array(all_curves)
            mean_curve = np.mean(all_curves, axis=0)
            print(f"    Mean L2 at 30°: {mean_curve[30]:.2f}, "
                  f"at 60°: {mean_curve[60]:.2f}, "
                  f"at 90°: {mean_curve[90]:.2f}")

        sweep_path = os.path.join(results_dir, 'sweep_curves.pkl')
        with open(sweep_path, 'wb') as f:
            pickle.dump(sweep_results, f)
        print(f"\n  Saved: {sweep_path}")
    else:
        print("\n  Skipping perturbation sweeps (--skip-sweeps)")

    # ── Step 4: 2×2 steered generation ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 4: 2×2 steered generation (angle={args.angle}°)")
    print("=" * 60)

    from data.python_prompts_holdout import PYTHON_PROMPTS_HOLDOUT
    gen_prompts = PYTHON_PROMPTS_HOLDOUT[:args.n_prompts]

    d_mandarin_dev = d_mandarin.to(device)
    d_haskell_dev = d_haskell.to(device)

    conditions = {
        'baseline': [],
        'mandarin_only': [(d_mandarin_dev, args.angle)],
        'haskell_only': [(d_haskell_dev, args.angle)],
        'mandarin_haskell': [(d_mandarin_dev, args.angle),
                             (d_haskell_dev, args.angle)],
    }

    all_generations = []
    total = len(gen_prompts) * len(conditions)
    count = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(gen_prompts):
        for cond_name, dirs_and_angles in conditions.items():
            torch.manual_seed(SEED + prompt_idx * 100 + hash(cond_name) % 50)

            text = generate_steered_composed(
                model, tokenizer, device, prompt,
                dirs_and_angles, N_TOKENS, TEMPERATURE, LAYER)

            result = {
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'condition': cond_name,
                'angle': args.angle,
                'text': text,
                'cjk_fraction': cjk_fraction(text),
                'haskell_fraction': haskell_keywords_fraction(text),
            }
            all_generations.append(result)

            count += 1
            elapsed = time.time() - t0
            if count % 4 == 0:
                eta = elapsed / count * (total - count)
                print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    for cond_name in conditions:
        cond_results = [r for r in all_generations if r['condition'] == cond_name]
        mean_cjk = np.mean([r['cjk_fraction'] for r in cond_results])
        mean_hs = np.mean([r['haskell_fraction'] for r in cond_results])
        print(f"\n  {cond_name}:")
        print(f"    CJK fraction: {mean_cjk:.1%}")
        print(f"    Haskell fraction: {mean_hs:.1%}")
        # Show first 2 examples
        for r in cond_results[:2]:
            text_preview = r['text'][:120].replace('\n', '\\n')
            print(f"    [{r['prompt'][:30]}] {text_preview}")

    # Save generations (angle-specific filenames to avoid overwriting)
    angle_tag = f"_{int(args.angle)}deg"
    gen_path = os.path.join(results_dir, f'generations{angle_tag}.json')
    with open(gen_path, 'w') as f:
        json.dump(all_generations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {gen_path}")

    # Also save a readable text version
    txt_path = os.path.join(results_dir, f'generations{angle_tag}.txt')
    with open(txt_path, 'w') as f:
        for cond_name in conditions:
            f.write(f"\n{'='*60}\n")
            angle_str = "no steering" if cond_name == 'baseline' else f"angle={args.angle}°"
            f.write(f"Condition: {cond_name} ({angle_str})\n")
            f.write(f"{'='*60}\n")
            cond_results = [r for r in all_generations if r['condition'] == cond_name]
            for r in cond_results:
                f.write(f"\nPrompt: {r['prompt']}\n")
                f.write(f"CJK: {r['cjk_fraction']:.1%}, Haskell: {r['haskell_fraction']:.1%}\n")
                f.write(f"Output: {r['text']}\n")
                f.write("-" * 40 + "\n")
    print(f"Saved: {txt_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
