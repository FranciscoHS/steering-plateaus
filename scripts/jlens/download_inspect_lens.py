#!/usr/bin/env python3
"""Download the pre-fitted Jacobian Lens for Qwen3-8B from neuronpedia/jacobian-lens
and inspect its format (keys, shapes, dtypes).

Lens files live under qwen3-8b/jlens/Salesforce-wikitext/ in the HF repo.
Writes an inspection report next to the downloaded files.
"""
import os
import torch
from huggingface_hub import snapshot_download

OUT_DIR = 'results/jlens/qwen3-8b'
REPORT = os.path.join(OUT_DIR, 'lens_inspection.txt')


def describe(obj, name, lines, depth=0):
    pad = '  ' * depth
    if isinstance(obj, torch.Tensor):
        lines.append(f"{pad}{name}: Tensor {tuple(obj.shape)} {obj.dtype} "
                     f"norm={obj.float().norm():.4g}")
    elif isinstance(obj, dict):
        lines.append(f"{pad}{name}: dict with {len(obj)} keys")
        for k, v in obj.items():
            describe(v, str(k), lines, depth + 1)
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:3]):
            describe(v, f"[{i}]", lines, depth + 1)
        if len(obj) > 3:
            lines.append(f"{pad}  ... ({len(obj) - 3} more)")
    else:
        r = repr(obj)
        lines.append(f"{pad}{name}: {type(obj).__name__} = "
                     f"{r if len(r) < 200 else r[:200] + '...'}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = snapshot_download(
        'neuronpedia/jacobian-lens',
        allow_patterns=['qwen3-8b/**'],
        local_dir=OUT_DIR,
    )
    print(f"Downloaded to: {path}")

    lens_dir = os.path.join(OUT_DIR, 'qwen3-8b/jlens/Salesforce-wikitext')
    lines = [f"Files in {lens_dir}:"]
    for f in sorted(os.listdir(lens_dir)):
        full = os.path.join(lens_dir, f)
        lines.append(f"  {f}  ({os.path.getsize(full) / 1e6:.1f} MB)")

    cfg = os.path.join(lens_dir, 'config.yaml')
    lines.append("\n--- config.yaml ---")
    lines.append(open(cfg).read())

    conv = os.path.join(lens_dir, 'Qwen3-8B_convergence.csv')
    lines.append("--- convergence.csv (first 5 lines) ---")
    lines.extend(open(conv).read().splitlines()[:5])

    pt = os.path.join(lens_dir, 'Qwen3-8B_jacobian_lens.pt')
    lines.append("\n--- Qwen3-8B_jacobian_lens.pt structure ---")
    obj = torch.load(pt, map_location='cpu', weights_only=False)
    describe(obj, 'root', lines)

    report = '\n'.join(lines)
    with open(REPORT, 'w') as f:
        f.write(report)
    print(report)
    print(f"\nReport saved: {REPORT}")


if __name__ == '__main__':
    main()
