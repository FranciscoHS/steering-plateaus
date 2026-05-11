"""Find the coherence boundary in the (alpha_1, alpha_2) plane.

Protocol:
- 16 rays at evenly-spaced angles phi in [0, pi/2].
- Per ray, bisection in r over 6 iterations starting from [0, 85] deg
  (giving ~1.3 deg resolution).
- At each (alpha_1, alpha_2), apply persistent exp_map_2d steering at
  perturb_layer to 10 FineWeb prompts (seq_len=20), generate 20 new
  tokens each, judge each via Claude Haiku ("Is this a coherent English
  continuation of the prompt?"), and call the point coherent if >=8/10
  judgments are YES (>75%).
- Save per-ray boundary, all generations, and all judge calls.

Run on a pod with the model loaded. Needs ANTHROPIC_API_KEY env var.

Usage:
  ANTHROPIC_API_KEY=... HF_HOME=/workspace/hf_cache \\
      python -u scripts/exp_map/coherence_boundary.py \\
          --target gemma --layer 2 --d1 Gender --d2 Tense
"""
from __future__ import annotations
import os, sys, time, json, math, argparse
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model
from src.data import load_fineweb_fixed_length

from scripts.exp_map.lib import registry, directions as dirlib
from scripts.exp_map.comp_steer import make_steer_hook_persistent, get_block

OUT_DIR = "results/exp_map/data/coherence"


JUDGE_TEMPLATE = (
    "You are evaluating an LLM continuation. The continuation is limited "
    "to about 20 tokens and may end mid-sentence; the model may also make "
    "factual errors. NEITHER of those is a reason to say NO.\n\n"
    "Reply with only YES or NO.\n\n"
    "Say YES if the continuation is grammatical English text on a topic "
    "plausibly related to the prompt — even if factually wrong, even if "
    "cut off mid-sentence.\n\n"
    "Say NO only if the continuation is gibberish, word-salad, repetitive "
    "nonsense, mostly non-English where English was expected, or makes "
    "no grammatical or semantic sense.\n\n"
    "PROMPT: {prompt}\n"
    "CONTINUATION: {continuation}"
)


def parse_yes_no(text: str) -> bool:
    """Return True if the response indicates YES (coherent), False otherwise."""
    t = text.upper()
    yes_idx = t.find("YES")
    no_idx = t.find("NO")
    if yes_idx == -1 and no_idx == -1:
        return False  # default to incoherent on ambiguous
    if yes_idx == -1:
        return False
    if no_idx == -1:
        return True
    return yes_idx < no_idx


def judge_batch(client, prompts: list[str], continuations: list[str],
                model_id: str = "claude-haiku-4-5") -> list[dict]:
    """Sequentially call Anthropic API on each (prompt, continuation) pair.
    Returns list of dicts with 'yes' (bool), 'raw' (str), 'error' (str|None)."""
    out = []
    for p, c in zip(prompts, continuations):
        msg = JUDGE_TEMPLATE.format(prompt=p, continuation=c)
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=8,
                messages=[{"role": "user", "content": msg}],
            )
            raw = resp.content[0].text if resp.content else ""
            out.append({"yes": parse_yes_no(raw), "raw": raw, "error": None})
        except Exception as e:
            out.append({"yes": False, "raw": "", "error": str(e)})
    return out


@torch.no_grad()
def generate_steered(model, tokenizer, prompts: list[str],
                     d1: torch.Tensor, d2: torch.Tensor,
                     alpha1_rad: float, alpha2_rad: float,
                     perturb_layer: int, max_new: int,
                     temperature: float, top_p: float,
                     device, seed: int) -> list[str]:
    """Generate one continuation per prompt under persistent rotation
    steering at (alpha1_rad, alpha2_rad). Returns list of strings (just
    the new tokens, decoded)."""
    # Tokenize batched. Prompts are FineWeb-decoded sequences of identical
    # length so no padding needed.
    enc = tokenizer(prompts, return_tensors="pt", padding=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    block = get_block(model, perturb_layer)
    hook, fired = make_steer_hook_persistent(d1, d2, alpha1_rad, alpha2_rad)
    handle = block.register_forward_hook(hook)
    torch.manual_seed(seed)
    try:
        out_ids = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )
    finally:
        handle.remove()
    new_tokens = out_ids[:, input_ids.shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]


def evaluate_point(model, tokenizer, client, prompts: list[str],
                   d1: torch.Tensor, d2: torch.Tensor,
                   alpha1_deg: float, alpha2_deg: float, args, device,
                   seed: int) -> dict:
    """Evaluate one (alpha_1, alpha_2): generate, judge, return summary."""
    a1r = math.radians(alpha1_deg)
    a2r = math.radians(alpha2_deg)
    t0 = time.time()
    cont = generate_steered(
        model, tokenizer, prompts, d1, d2, a1r, a2r,
        args.layer, args.max_new, args.temperature, args.top_p,
        device, seed)
    t_gen = time.time() - t0
    t0 = time.time()
    judgments = judge_batch(client, prompts, cont, args.judge_model)
    t_judge = time.time() - t0
    n_yes = sum(j["yes"] for j in judgments)
    return {
        "alpha1_deg": float(alpha1_deg),
        "alpha2_deg": float(alpha2_deg),
        "n_yes": int(n_yes),
        "n": len(prompts),
        "rate": n_yes / len(prompts) if prompts else 0.0,
        "coherent": n_yes >= args.coherent_threshold,
        "t_gen_s": t_gen,
        "t_judge_s": t_judge,
        "continuations": cont,
        "judgments": judgments,
    }


def bisect_ray(model, tokenizer, client, prompts, d1, d2,
               phi_deg, args, device, log_writer):
    """Bisect r along one ray. Returns dict with boundary r and trace."""
    phi = math.radians(phi_deg)
    cphi, sphi = math.cos(phi), math.sin(phi)

    r_low, r_high = float(args.r_low), float(args.r_high)
    trace = []
    for it in range(args.n_iter):
        r_mid = 0.5 * (r_low + r_high)
        a1 = r_mid * cphi
        a2 = r_mid * sphi
        seed = args.seed + 1000 * int(round(phi_deg)) + it
        ev = evaluate_point(
            model, tokenizer, client, prompts, d1, d2,
            a1, a2, args, device, seed=seed)
        ev["iter"] = it
        ev["r"] = r_mid
        ev["phi_deg"] = phi_deg
        trace.append(ev)
        # flush incremental log
        log_writer.write(json.dumps({"phi_deg": phi_deg, "iter": it,
                                     "r": r_mid, "alpha1": a1, "alpha2": a2,
                                     "rate": ev["rate"],
                                     "coherent": ev["coherent"],
                                     "t_gen_s": ev["t_gen_s"],
                                     "t_judge_s": ev["t_judge_s"]}) + "\n")
        log_writer.flush()
        print(f"  phi={phi_deg:5.1f}  it={it}  r={r_mid:6.3f}  "
              f"a1={a1:6.3f} a2={a2:6.3f}  yes={ev['n_yes']}/{ev['n']}  "
              f"{'C' if ev['coherent'] else 'I'}  "
              f"gen={ev['t_gen_s']:4.1f}s judge={ev['t_judge_s']:4.1f}s",
              flush=True)
        if ev["coherent"]:
            r_low = r_mid
        else:
            r_high = r_mid
    return {"phi_deg": phi_deg, "r_low": r_low, "r_high": r_high,
            "boundary_r": r_low,
            "alpha1_deg": r_low * cphi, "alpha2_deg": r_low * sphi,
            "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--d1", default="Gender")
    ap.add_argument("--d2", default="Tense")
    ap.add_argument("--n_prompts", type=int, default=25)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--n_rays", type=int, default=16)
    ap.add_argument("--n_iter", type=int, default=6)
    ap.add_argument("--r_low", type=float, default=0.0)
    ap.add_argument("--r_high", type=float, default=85.0)
    ap.add_argument("--coherent_threshold", type=int, default=19,
                    help="number of YES out of n_prompts to call coherent. "
                         "Default 19/25 = 76%% (just over 75%% per protocol). "
                         "If you change --n_prompts, update this too.")
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge_model", default="claude-haiku-4-5")
    ap.add_argument("--dtype", default="float32",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    # Anthropic client (fail fast if missing key)
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic()

    cfg = registry.TARGETS[args.target]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    print(f"loading {cfg['model']} dtype={args.dtype}", flush=True)
    model, tokenizer, device = load_model(cfg["model"],
                                          dtype=dtype_map[args.dtype])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Directions
    print("loading directions", flush=True)
    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, args.target, args.layer,
        names=[args.d1, args.d2])
    all_dirs = dirs_blob["directions"]
    d1 = all_dirs[args.d1].to(device=device, dtype=torch.float32)
    d2 = all_dirs[args.d2].to(device=device, dtype=torch.float32)

    # 10 fresh FineWeb prompts at seq_len=20 (different seed than sweep
    # so we don't reuse the 5-token anchors).
    print(f"loading {args.n_prompts} FineWeb prompts at seq_len={args.seq_len}",
          flush=True)
    cache_dir = f"/workspace/fineweb_cache_{args.target}"
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        args.n_prompts, tokenizer,
        seq_len=args.seq_len, seed=12345,
        cache_dir=cache_dir)
    prompts = [tokenizer.decode(t, skip_special_tokens=True)
               for t in token_lists]
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p!r}", flush=True)

    # Output paths
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_path = os.path.join(
        OUT_DIR,
        f"{args.target}_L{args.layer}_{args.d1}__{args.d2}_"
        f"coh_n{args.n_prompts}_r{args.n_rays}{tag}.json")
    log_path = out_path.replace(".json", ".log.jsonl")

    print(f"will write {out_path}\n         {log_path}", flush=True)
    rays_phi = np.linspace(0.0, 90.0, args.n_rays)
    print(f"rays (deg): {rays_phi}", flush=True)

    boundaries = []
    with open(log_path, "w") as logf:
        for phi_deg in rays_phi:
            t0 = time.time()
            print(f"\n=== ray phi={phi_deg:.2f} deg ===", flush=True)
            res = bisect_ray(model, tokenizer, client, prompts, d1, d2,
                             phi_deg, args, device, logf)
            print(f"  -> r_boundary = {res['boundary_r']:.3f} deg "
                  f"(elapsed {time.time()-t0:.1f}s)", flush=True)
            boundaries.append(res)

    # Final save
    out = {
        "target": args.target,
        "model": cfg["model"],
        "layer": args.layer,
        "d1": args.d1, "d2": args.d2,
        "n_prompts": args.n_prompts, "seq_len": args.seq_len,
        "n_rays": args.n_rays, "n_iter": args.n_iter,
        "r_low_init": args.r_low, "r_high_init": args.r_high,
        "coherent_threshold": args.coherent_threshold,
        "max_new": args.max_new, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed,
        "judge_model": args.judge_model, "dtype": args.dtype,
        "judge_template": JUDGE_TEMPLATE,
        "prompts": prompts,
        "rays": boundaries,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
