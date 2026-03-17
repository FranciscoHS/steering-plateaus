#!/usr/bin/env python3
"""Perturbation sweeps: how DoM quality scales with number of prompt pairs.

Experiment 3 (matched): DoM from subsets of matched EN-ZH bilingual pairs (n=1,2,5,10,20,50).
Experiment 4 (unmatched): DoM from unmatched EN/ZH prompts (random pairing, same n values).

Both compared against "improved random baseline" (diff of random FineWeb activations).

Usage:
    python scripts/perturbation/sweep_scaling_dom.py --model gemma-2-9b --layer 2
    python scripts/perturbation/sweep_scaling_dom.py --model gemma-2-9b --layer 2 --mode matched
    python scripts/perturbation/sweep_scaling_dom.py --model gemma-2-9b --layer 2 --mode unmatched
"""
import argparse, os, sys, pickle, time
sys.path.insert(0, '.')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import (
    load_model, forward_from_layer_to_layer, _get_blocks,
)
from src.data import load_fineweb_fixed_length
from scripts.perturbation.sweep_random_points import extract_context_and_activation
from scripts.perturbation.sweep_qwen_sae import sweep_one
from data.bilingual_prompts import ENGLISH_PROMPTS, CHINESE_PROMPTS


# Unmatched Chinese prompts: simple sentences with DIFFERENT content from ENGLISH_PROMPTS.
# These cover food, weather, emotions, colors, animals, etc. — deliberately NOT matching
# the English prompts' topics (science, geography, tech, etc.)
UNMATCHED_CHINESE = [
    "今天的天气非常",
    "我最喜欢的食物是",
    "这只猫看起来很",
    "春天的花开得非常",
    "他每天早上都会去",
    "这个城市的夜景很",
    "妈妈做的菜总是很",
    "下雨天我喜欢在家",
    "小狗在公园里快乐地",
    "红色的苹果看起来很",
    "冬天的雪花很",
    "她的笑容让人感到",
    "这本小说的结局让我",
    "爸爸教我如何",
    "夏天的西瓜特别",
    "这条河流非常",
    "月亮今晚看起来特别",
    "弟弟喜欢玩",
    "这家餐厅的服务很",
    "秋天的落叶让人感到",
    "奶奶的故事总是很",
    "蓝色的天空下",
    "我们一起去",
    "这首歌让人感到",
    "绿色的草地上有很多",
    "姐姐在大学学习",
    "早上的咖啡让我",
    "这个周末我打算",
    "海边的风吹得很",
    "他们在操场上开心地",
    "黄色的向日葵长得",
    "图书馆里很",
    "我的生日愿望是",
    "这道数学题让我",
    "晚上的星星很",
    "爷爷喜欢在院子里",
    "白色的雪覆盖了",
    "她在厨房里忙着",
    "这部电影的音乐很",
    "紫色的薰衣草闻起来",
    "哥哥带我去了",
    "下午茶时间我喜欢吃",
    "这个公园里有很多",
    "橙色的夕阳非常",
    "我的梦想是成为",
    "同学们一起在教室里",
    "这条街上有很多",
    "粉色的樱花开满了",
    "周末的早晨我喜欢",
    "这只小鸟唱得很",
]

# Simple unmatched English prompts (different content from CHINESE_PROMPTS)
UNMATCHED_ENGLISH = [
    "The old cat sat on the",
    "My favorite color has always been",
    "Yesterday I went to the",
    "The cake in the oven smells",
    "He always wears a blue",
    "The garden needs more",
    "She plays the piano every",
    "The train arrives at exactly",
    "My grandmother's recipe for",
    "The bird outside my window is",
    "I forgot to bring my",
    "The library closes at",
    "His new bicycle is very",
    "We should plant more",
    "The movie was really",
    "Her painting of the sunset was",
    "The soup needs a little more",
    "They decided to walk to the",
    "The baby laughed when she saw the",
    "My brother collects old",
    "The snow covered everything in",
    "She bought a new pair of",
    "The dog chased the ball across the",
    "We had pasta for dinner with",
    "The autumn leaves turned",
    "He fixed the broken",
    "The concert starts at eight in the",
    "My sister loves to read about",
    "The flowers in the vase are",
    "They celebrated by going to the",
    "The coffee shop on the corner makes",
    "She always carries an umbrella when",
    "The children played in the",
    "His favorite sport is",
    "The cookies are almost ready to",
    "We watched the fireworks from the",
    "The postman delivers letters every",
    "Her dress was a beautiful shade of",
    "The mountain trail was quite",
    "They served fresh bread with",
    "The kitten fell asleep on the",
    "My uncle works at a",
    "The sunset painted the sky in shades of",
    "She finished her homework before",
    "The river flows gently through the",
    "He ordered a large plate of",
    "The school bus arrives at",
    "We went fishing at the",
    "The stars were particularly bright on",
    "My friend brought me a gift from",
]


def get_acts_for_prompts(model, tokenizer, prompts, device, layer):
    """Get last-token hidden states at a given layer for a list of prompts."""
    acts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        acts.append(outputs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu())
    return torch.stack(acts)


def compute_dom_from_subsets(en_acts, zh_acts, n_values):
    """Compute negated DoM direction for each subset size.

    Returns dict: {n: unit_direction} where direction points TOWARD Chinese.
    """
    doms = {}
    for n in n_values:
        D = en_acts[:n] - zh_acts[:n]
        mean_vec = D.mean(0)
        dom = F.normalize(mean_vec, dim=0)
        dom = -dom  # negate: point toward Chinese
        doms[n] = dom
        print(f"    n={n:3d}: pre-norm magnitude={mean_vec.norm():.2f}")
    return doms


def get_diff_random_directions(all_activations, n_dirs, seed=99):
    """Normalized differences between random pairs of FineWeb activations."""
    n = len(all_activations)
    rng = np.random.RandomState(seed)
    dirs = []
    for _ in range(n_dirs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = all_activations[i] - all_activations[j]
        dirs.append(F.normalize(diff, dim=0))
    return torch.stack(dirs)


def run_experiment(model, all_contexts, all_activations, dom_dict,
                   diff_random_dirs, layer, measure_layer, device,
                   angles_rad, n_curves, metric='l2'):
    """Run sweeps for all DoM subset sizes + improved random baseline."""
    n_act = len(all_activations)
    curves = {}
    rng = np.random.RandomState(42)

    # Improved random baseline (diff of random FineWeb activations)
    act_indices_rand = rng.randint(0, n_act, size=n_curves)
    print(f"  Sweeping {n_curves} diff_random curves...")
    t0 = time.time()
    for k in range(n_curves):
        curves[f'diff_random_{k}'] = sweep_one(
            model, all_contexts[act_indices_rand[k]], all_activations[act_indices_rand[k]],
            diff_random_dirs[k % len(diff_random_dirs)],
            layer, measure_layer, device, angles_rad, metric=metric)
    print(f"    {time.time()-t0:.1f}s")

    # DoM for each n
    for n, dom_dir in dom_dict.items():
        n_sweep = min(n_curves, n_act)
        print(f"  Sweeping {n_sweep} DoM(n={n}) curves...")
        t0 = time.time()
        for k in range(n_sweep):
            curves[f'dom_n{n}_{k}'] = sweep_one(
                model, all_contexts[k], all_activations[k],
                dom_dir, layer, measure_layer, device, angles_rad, metric=metric)
        print(f"    {time.time()-t0:.1f}s")

    return curves


def plot_scaling(curves, angles_deg, n_values, ylabel, title, save_path):
    """Plot median+IQR for each n, plus improved random baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colormap for n values
    cmap = plt.cm.viridis
    n_colors = len(n_values)

    # Improved random baseline in C0
    rand_keys = sorted([k for k in curves if k.startswith('diff_random_')])
    if rand_keys:
        data = np.array([curves[k] for k in rand_keys])
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color='C0', alpha=0.1)
        ax.plot(angles_deg, med, color='C0', lw=2.5, label='Improved random', linestyle='--')

    # DoM curves for each n
    for i, n in enumerate(n_values):
        color = cmap(i / max(n_colors - 1, 1))
        keys = sorted([k for k in curves if k.startswith(f'dom_n{n}_')])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        med = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        ax.fill_between(angles_deg, q25, q75, color=color, alpha=0.1)
        # Highlight the full DoM (n=50) with thicker line
        lw = 3.0 if n == n_values[-1] else 2.0
        ax.plot(angles_deg, med, color=color, lw=lw,
                label=f'-DoM (n={n})')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_individual(curves, angles_deg, n_values, ylabel, title, save_path):
    """Plot individual curves (alpha=0.5) with thick median on top."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    n_colors = len(n_values)

    # Improved random baseline
    rand_keys = sorted([k for k in curves if k.startswith('diff_random_')])
    if rand_keys:
        data = np.array([curves[k] for k in rand_keys])
        for k in rand_keys:
            ax.plot(angles_deg, curves[k], color='C0', alpha=0.3, lw=0.8)
        med = np.median(data, axis=0)
        ax.plot(angles_deg, med, color='C0', lw=2.5, label='Improved random', linestyle='--')

    # DoM curves
    for i, n in enumerate(n_values):
        color = cmap(i / max(n_colors - 1, 1))
        keys = sorted([k for k in curves if k.startswith(f'dom_n{n}_')])
        if not keys:
            continue
        data = np.array([curves[k] for k in keys])
        for k in keys:
            ax.plot(angles_deg, curves[k], color=color, alpha=0.3, lw=0.8)
        med = np.median(data, axis=0)
        lw = 3.0 if n == n_values[-1] else 2.0
        ax.plot(angles_deg, med, color=color, lw=lw, label=f'-DoM (n={n})')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(matched_curves, unmatched_curves, angles_deg, n_values,
                    ylabel, title, save_path):
    """Compare matched vs unmatched DoM at each n on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    n_colors = len(n_values)

    for i, n in enumerate(n_values):
        color = cmap(i / max(n_colors - 1, 1))

        # Matched: solid
        m_keys = sorted([k for k in matched_curves if k.startswith(f'dom_n{n}_')])
        if m_keys:
            m_data = np.array([matched_curves[k] for k in m_keys])
            m_med = np.median(m_data, axis=0)
            ax.plot(angles_deg, m_med, color=color, lw=2.0, linestyle='-',
                    label=f'Matched n={n}')

        # Unmatched: dashed
        u_keys = sorted([k for k in unmatched_curves if k.startswith(f'dom_n{n}_')])
        if u_keys:
            u_data = np.array([unmatched_curves[k] for k in u_keys])
            u_med = np.median(u_data, axis=0)
            ax.plot(angles_deg, u_med, color=color, lw=2.0, linestyle='--',
                    label=f'Unmatched n={n}')

    ax.set_xlabel('Perturbation angle (degrees)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, angles_deg[-1])
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemma-2-9b')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--n_act', type=int, default=30)
    parser.add_argument('--n_curves', type=int, default=10)
    parser.add_argument('--max_angle', type=float, default=90.0)
    parser.add_argument('--n_angles', type=int, default=90)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--metric', default='l2')
    parser.add_argument('--mode', default='both', choices=['matched', 'unmatched', 'both'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_values = [1, 2, 5, 10, 20, 50]

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, tokenizer, device = load_model(args.model, dtype=load_dtype)
    n_layers = len(_get_blocks(model))
    d_model = model._plateau_info['d_model']
    measure_layer = n_layers - 2
    print(f"d_model={d_model}, n_layers={n_layers}, perturb@L{args.layer}, measure@L{measure_layer}")

    angles_deg = np.linspace(0, args.max_angle, args.n_angles + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)

    # FineWeb activations (the ones we perturb)
    print(f"\nLoading {args.n_act} FineWeb sequences...")
    token_ids_list = load_fineweb_fixed_length(
        args.n_act, tokenizer, seq_len=args.seq_len, seed=args.seed)
    print(f"Extracting layer-{args.layer} activations...")
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, args.layer, device)
    print(f"Activation norms: mean={all_activations.norm(dim=-1).mean():.2f}")

    # Improved random baseline (diff of random FineWeb activations)
    print("\nComputing improved random baseline directions...")
    diff_random_dirs = get_diff_random_directions(all_activations, n_dirs=args.n_curves, seed=99)

    ylabel = f'L2 distance (layer {measure_layer})' if args.metric == 'l2' else \
             f'Cosine distance (layer {measure_layer})' if args.metric == 'cosine' else \
             'KL divergence (logits)'

    matched_curves = None
    unmatched_curves = None

    # ===== EXPERIMENT 3: Matched bilingual pairs =====
    if args.mode in ('matched', 'both'):
        results_dir = f'results/lesswrong_plot/{args.model}-scaling-matched'
        os.makedirs(results_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Matched bilingual pairs")
        print("=" * 60)

        # Get activations for matched EN/ZH prompts
        print("  Extracting matched EN activations...")
        en_acts = get_acts_for_prompts(model, tokenizer, ENGLISH_PROMPTS, device, args.layer)
        print("  Extracting matched ZH activations...")
        zh_acts = get_acts_for_prompts(model, tokenizer, CHINESE_PROMPTS, device, args.layer)

        # DoM from subsets
        print("  Computing DoM for subset sizes:")
        matched_doms = compute_dom_from_subsets(en_acts, zh_acts, n_values)

        # Check cosine similarities between subset DoMs and full DoM
        full_dom = matched_doms[50]
        print("  Cosine similarity with full DoM (n=50):")
        for n in n_values[:-1]:
            cos = (matched_doms[n] @ full_dom).item()
            print(f"    n={n:3d}: cos={cos:.4f}")

        # Run sweeps
        matched_curves = run_experiment(
            model, all_contexts, all_activations, matched_doms,
            diff_random_dirs, args.layer, measure_layer, device,
            angles_rad, args.n_curves, metric=args.metric)

        # Save
        pkl_path = os.path.join(results_dir, f'curves_{args.metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'angles_deg': angles_deg,
                'curves': matched_curves,
                'n_values': n_values,
                'mode': 'matched',
            }, f)
        print(f"  Saved: {pkl_path}")

        title = f'{args.model} L{args.layer}: -DoM scaling (matched pairs)'
        plot_scaling(matched_curves, angles_deg, n_values, ylabel, title,
                     os.path.join(results_dir, f'plot_{args.metric}_median.png'))
        plot_individual(matched_curves, angles_deg, n_values, ylabel, title,
                        os.path.join(results_dir, f'plot_{args.metric}_individual.png'))

    # ===== EXPERIMENT 4: Unmatched prompts =====
    if args.mode in ('unmatched', 'both'):
        results_dir = f'results/lesswrong_plot/{args.model}-scaling-unmatched'
        os.makedirs(results_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Unmatched prompts")
        print("=" * 60)

        # Get activations for unmatched EN/ZH prompts
        print("  Extracting unmatched EN activations...")
        en_acts_um = get_acts_for_prompts(model, tokenizer, UNMATCHED_ENGLISH, device, args.layer)
        print("  Extracting unmatched ZH activations...")
        zh_acts_um = get_acts_for_prompts(model, tokenizer, UNMATCHED_CHINESE, device, args.layer)

        # DoM from subsets
        print("  Computing DoM for subset sizes (unmatched):")
        unmatched_doms = compute_dom_from_subsets(en_acts_um, zh_acts_um, n_values)

        # Compare to matched DoM if available
        if matched_curves is not None:
            en_acts_m = get_acts_for_prompts(model, tokenizer, ENGLISH_PROMPTS, device, args.layer)
            zh_acts_m = get_acts_for_prompts(model, tokenizer, CHINESE_PROMPTS, device, args.layer)
            matched_full = -F.normalize((en_acts_m.mean(0) - zh_acts_m.mean(0)), dim=0)
            print("  Cosine similarity (unmatched DoM vs matched full DoM):")
            for n in n_values:
                cos = (unmatched_doms[n] @ matched_full).item()
                print(f"    n={n:3d}: cos={cos:.4f}")

        # Run sweeps
        unmatched_curves = run_experiment(
            model, all_contexts, all_activations, unmatched_doms,
            diff_random_dirs, args.layer, measure_layer, device,
            angles_rad, args.n_curves, metric=args.metric)

        # Save
        pkl_path = os.path.join(results_dir, f'curves_{args.metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'angles_deg': angles_deg,
                'curves': unmatched_curves,
                'n_values': n_values,
                'mode': 'unmatched',
            }, f)
        print(f"  Saved: {pkl_path}")

        title = f'{args.model} L{args.layer}: -DoM scaling (unmatched prompts)'
        plot_scaling(unmatched_curves, angles_deg, n_values, ylabel, title,
                     os.path.join(results_dir, f'plot_{args.metric}_median.png'))
        plot_individual(unmatched_curves, angles_deg, n_values, ylabel, title,
                        os.path.join(results_dir, f'plot_{args.metric}_individual.png'))

    # ===== Comparison plot (if both modes ran) =====
    if matched_curves is not None and unmatched_curves is not None:
        comp_dir = f'results/lesswrong_plot/{args.model}-scaling-comparison'
        os.makedirs(comp_dir, exist_ok=True)
        title = f'{args.model} L{args.layer}: matched vs unmatched DoM scaling'
        plot_comparison(matched_curves, unmatched_curves, angles_deg, n_values,
                        ylabel, title,
                        os.path.join(comp_dir, f'plot_{args.metric}_comparison.png'))
        # Save combined data
        pkl_path = os.path.join(comp_dir, f'curves_{args.metric}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'angles_deg': angles_deg,
                'matched_curves': matched_curves,
                'unmatched_curves': unmatched_curves,
                'n_values': n_values,
            }, f)
        print(f"  Saved: {pkl_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
