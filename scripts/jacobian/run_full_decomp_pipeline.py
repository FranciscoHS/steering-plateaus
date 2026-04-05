#!/usr/bin/env python3
"""
Full pipeline: decompose an activation, then run perturbation sweeps.

1. Run response decomposition with --response_self
2. Split directions into individual pkls
3. Perturbation sweep on the target activation
4. Perturbation sweep on 10 OOD activations

Usage:
    python scripts/jacobian/run_full_decomp_pipeline.py --sample_idx 0
    python scripts/jacobian/run_full_decomp_pipeline.py --sample_idx 3 --k 20
"""

import argparse
import os
import pickle
import subprocess
import sys
sys.path.insert(0, '.')


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-1.7b')
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--sample_idx', type=int, required=True)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_ood', type=int, default=10)
    parser.add_argument('--base_dir', default='results/decomp_pipeline')
    args = parser.parse_args()

    out_dir = os.path.join(args.base_dir, f'sample_{args.sample_idx}')
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Decomposition
    print(f"\n{'='*60}")
    print(f"STEP 1: Decompose sample {args.sample_idx}")
    print(f"{'='*60}")
    run(f"python -u scripts/jacobian/response_decomp.py "
        f"--model {args.model} --layer {args.layer} --k {args.k} "
        f"--seed {args.seed} --sample_idx {args.sample_idx} "
        f"--response_self --results_dir {out_dir}/decomp")

    # Step 2: Split directions
    print(f"\n{'='*60}")
    print(f"STEP 2: Split directions")
    print(f"{'='*60}")
    with open(f'{out_dir}/decomp/results.pkl', 'rb') as f:
        res = pickle.load(f)
    n_dirs = len(res['directions'])
    for i in range(n_dirs):
        d = {'directions': [res['directions'][i]]}
        with open(f'{out_dir}/decomp/dir_{i}.pkl', 'wb') as f:
            pickle.dump(d, f)
    print(f"Split {n_dirs} directions")

    # Get the target prompt for sweep
    from src.model import load_model
    from src.data import load_fineweb_fixed_length
    model, tokenizer, device = load_model(args.model)
    n_total = 11  # matches default n_response_acts + 1
    token_ids_list = load_fineweb_fixed_length(
        n_total, tokenizer, seq_len=20, seed=args.seed
    )
    target_prompt = tokenizer.decode(token_ids_list[args.sample_idx], skip_special_tokens=True)
    # Escape for shell
    target_prompt_escaped = target_prompt.replace("'", "'\\''")
    del model, tokenizer  # free GPU memory

    # Build direction pkl args
    dir_pkls = ' '.join(f'{out_dir}/decomp/dir_{i}.pkl' for i in range(n_dirs))
    dir_labels = ' '.join(f'd{i+1}' for i in range(n_dirs))

    # Step 3: Sweep on target activation
    print(f"\n{'='*60}")
    print(f"STEP 3: Sweep on target activation")
    print(f"{'='*60}")
    run(f"python -u scripts/perturbation/run_feature_vs_random_sweep.py "
        f"--model {args.model} --layer {args.layer} "
        f"--n_act 1 --n_random_dirs 15 --max_angle 90 --measure_layer -2 "
        f"--custom_prompts '{target_prompt_escaped}' "
        f"--direction_pkl {dir_pkls} --dir_labels {dir_labels} "
        f"--results_dir {out_dir}/sweep_target")

    # Step 4: Sweep on OOD activations
    print(f"\n{'='*60}")
    print(f"STEP 4: Sweep on {args.n_ood} OOD activations")
    print(f"{'='*60}")
    run(f"python -u scripts/perturbation/run_feature_vs_random_sweep.py "
        f"--model {args.model} --layer {args.layer} "
        f"--n_act {args.n_ood} --n_random_dirs 15 --max_angle 90 --measure_layer -2 "
        f"--seed 999 "
        f"--direction_pkl {dir_pkls} --dir_labels {dir_labels} "
        f"--results_dir {out_dir}/sweep_ood")

    print(f"\n{'='*60}")
    print(f"DONE: sample {args.sample_idx}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
