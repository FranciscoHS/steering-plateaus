"""Quick fp64 sanity check before launching the full microdegree sweep.

Loads Gemma-2-9b in fp64, takes the (Gender, Tense) pair on a single
FineWeb anchor, and runs a 5x5 single-batch grid at angles
{0, 1e-7, 1e-6, 1e-5, 1e-4} deg along each axis. Reports VRAM, the
exact L^2 at the (0,0) reference, and the per-angle L^2 along each axis
so we can see whether the signal at 1e-6 deg is above the fp64 floor
and whether scaling stays linear.

Usage:
  python scripts/exp_map/verify_fp64.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import _get_blocks
from scripts.exp_map.lib import registry, directions as dirlib, activations as actlib
from scripts.exp_map.lib.sweep_core import (
    forward_from_layer_to_layer, build_perturbed_pts_2d, orthonormalize,
)

TARGET = "gemma"
PERTURB_LAYER = 2
ANGLES_DEG = np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
PAIR = ("Gender", "Tense")


def main():
    cfg = registry.TARGETS[TARGET]
    hf_name = cfg["model"]
    # registry stores short names — translate via aliases like load_model does
    from src.model import MODEL_ALIASES
    hf_name = MODEL_ALIASES.get(hf_name, hf_name)
    print(f"loading {hf_name} in bf16 → cuda → cast .double()", flush=True)
    t0 = time.time()
    # Load in native bf16 (fits, fast), move to cuda, then cast to fp64
    # in-place. Avoids any device_map / dtype-promotion path quirks.
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.bfloat16, device_map=None)
    model = model.to("cuda:0").double().eval()
    # load_model sets _plateau_info at the end; we bypass it, so set
    # the attribute manually. Gemma-2 uses the llama-architecture path.
    model._plateau_info = {"model_type": "llama",
                            "d_model": model.config.hidden_size}
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    device = "cuda:0"
    cpu_params = [n for n, p in model.named_parameters()
                  if p.device.type == "cpu"]
    cpu_bufs = [n for n, b in model.named_buffers()
                if b.device.type == "cpu"]
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"vram={torch.cuda.memory_allocated()/1e9:.1f} GB; "
          f"cpu_params={len(cpu_params)} cpu_bufs={len(cpu_bufs)}",
          flush=True)

    n_layers = len(_get_blocks(model))
    measure_layer = n_layers - 2
    print(f"  n_layers={n_layers} measure_layer={measure_layer}", flush=True)

    # Directions — move to cuda+fp64 before use
    a, b = PAIR
    dirs_blob = dirlib.extract_all(
        model, tokenizer, device, TARGET, PERTURB_LAYER, names=[a, b])
    all_dirs = dirs_blob["directions"]
    d1 = all_dirs[a].to(torch.float64).to("cuda:0")
    d2 = all_dirs[b].to(torch.float64).to("cuda:0")
    print(f"  ||d1||={d1.norm().item():.6f}  ||d2||={d2.norm().item():.6f} "
          f"d1.device={d1.device}", flush=True)

    # One FineWeb anchor — move to cuda+fp64 before use
    fw = actlib.fineweb_acts(model, tokenizer, device, TARGET,
                              PERTURB_LAYER, n=1)
    ctx = fw["contexts"][0].to("cuda:0")
    a_act = fw["activations"][0].to(torch.float64).to("cuda:0")
    print(f"  ||a||={a_act.norm().item():.4f}  a.device={a_act.device} "
          f"ctx.device={ctx.device}", flush=True)

    a64 = a_act
    a_hat = a64 / a64.norm()
    d1p, d2p = orthonormalize(a_hat, d1, d2)

    angles_rad = np.deg2rad(ANGLES_DEG)
    alphas_t = torch.from_numpy(angles_rad).to(device).to(torch.float64)
    pts = build_perturbed_pts_2d(a64, d1p, d2p, alphas_t, mode="geodesic")
    print(f"  pts.shape={tuple(pts.shape)}  pts.dtype={pts.dtype} "
          f"pts.device={pts.device}", flush=True)
    # Diagnostic: spot-check device + dtype of layer-2 input layernorm
    try:
        ln = _get_blocks(model)[3].input_layernorm  # block run after layer 2
        print(f"  layer-3 input_layernorm.weight.device={ln.weight.device} "
              f"dtype={ln.weight.dtype}", flush=True)
    except Exception as e:
        print(f"  (layernorm probe failed: {e})", flush=True)

    n = len(angles_rad)
    pf = forward_from_layer_to_layer(
        model, ctx, pts.unsqueeze(1),
        PERTURB_LAYER, measure_layer).double()
    print(f"  pf.shape={tuple(pf.shape)}  pf.dtype={pf.dtype}", flush=True)
    ref = pf[0:1]
    l2 = (pf - ref).norm(dim=-1).detach().cpu().numpy()
    l2_grid = l2.reshape(n, n)

    print()
    print(f"angles_deg = {ANGLES_DEG}")
    print(f"L^2(0,0) = {l2_grid[0, 0]:.3e}  (should be exactly 0)")
    print()
    print("L^2 along axis 1 (alpha2 = 0):")
    for i, ang in enumerate(ANGLES_DEG):
        print(f"  alpha1={ang:.1e} deg  L2={l2_grid[i, 0]:.6e}")
    print()
    print("L^2 along axis 2 (alpha1 = 0):")
    for j, ang in enumerate(ANGLES_DEG):
        print(f"  alpha2={ang:.1e} deg  L2={l2_grid[0, j]:.6e}")
    print()
    # Linear-scaling check: ratio L2/alpha along each axis (skip alpha=0)
    print("Linearity check (L2/alpha, should be ~constant if linear):")
    for i in range(1, n):
        r1 = l2_grid[i, 0] / ANGLES_DEG[i]
        r2 = l2_grid[0, i] / ANGLES_DEG[i]
        print(f"  alpha={ANGLES_DEG[i]:.1e}  axis1: {r1:.4e}  axis2: {r2:.4e}")
    print()
    print(f"vram peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
