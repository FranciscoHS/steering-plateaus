#!/usr/bin/env python3
"""Verify activation plateaus exist in Llama-3.2-1B.

Loads Llama-3.2-1B, samples FineWeb-like text via the tokenizer,
extracts activations at a configurable layer, perturbs along random
directions on the sphere, and measures cosine distance downstream.

This is a prerequisite check before running GLP projection experiments.

Usage:
    python scripts/perturbation/sweep_llama_plateau.py [--inject_layer 8] [--measure_layer 14]
"""

import os, sys, pickle, time, argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.model import load_model, forward_from_layer_to_layer

# Defaults
SEED = 42
N_TEXTS = 60       # number of text samples
N_SWEEP = 50       # activations to sweep
N_DIRS = 20        # random directions per activation
MAX_ANGLE = 60.0
N_ANGLES = 240
RESULTS_DIR = 'results/llama_plateau'

# Short prompts for activation extraction (diverse topics)
PROMPTS = [
    "The capital of France is Paris, which is known for",
    "In quantum mechanics, the wave function describes",
    "To make a good pasta sauce, you need to start with",
    "The stock market crashed in 1929, leading to",
    "Machine learning models are trained by minimizing",
    "The Great Wall of China was built to protect against",
    "Photosynthesis converts sunlight into chemical energy through",
    "Shakespeare wrote many famous plays including Hamlet and",
    "The theory of relativity states that the speed of light",
    "Climate change is caused primarily by the emission of",
    "Neural networks consist of layers of interconnected nodes that",
    "The human genome contains approximately three billion base pairs",
    "Democracy is a form of government where power is held by",
    "Water boils at 100 degrees Celsius at standard atmospheric",
    "The Renaissance was a period of cultural rebirth in Europe",
    "Antibiotics work by killing bacteria or preventing them from",
    "The Amazon rainforest produces about 20 percent of the world's",
    "Einstein received the Nobel Prize for his work on the",
    "Artificial intelligence has been advancing rapidly in recent years",
    "The periodic table organizes elements by their atomic number",
    "Black holes are regions of spacetime where gravity is so strong",
    "The Industrial Revolution began in Britain in the late 18th",
    "Vaccines work by training the immune system to recognize and fight",
    "The speed of sound depends on the medium through which it travels",
    "Fibonacci numbers appear frequently in nature, from sunflower",
    "The French Revolution began in 1789 with the storming of",
    "DNA carries the genetic instructions for the development of all",
    "The internet was originally developed as a military communication",
    "Plate tectonics explains how the Earth's crust is divided into",
    "The human brain contains roughly 86 billion neurons connected by",
    "Gravity is the weakest of the four fundamental forces but acts",
    "The Pythagorean theorem states that in a right triangle",
    "Global warming has led to rising sea levels and more extreme",
    "Beethoven composed his famous ninth symphony while he was",
    "The solar system formed approximately 4.6 billion years ago from",
    "Encryption algorithms protect data by converting it into an",
    "The mitochondria are often called the powerhouse of the cell",
    "Ancient Rome was founded according to legend by Romulus and",
    "The Doppler effect explains why the pitch of a siren changes",
    "Quantum entanglement allows particles to be correlated regardless",
    "The printing press invented by Gutenberg revolutionized the spread",
    "Evolution by natural selection was proposed independently by Darwin",
    "Superconductors are materials that conduct electricity with zero",
    "The United Nations was established in 1945 after the end of",
    "Calculus was developed independently by Newton and Leibniz in the",
    "The Voyager spacecraft have traveled beyond the boundary of our",
    "Coral reefs support more species per unit area than any other",
    "The Turing test measures a machine's ability to exhibit intelligent",
    "Magnetic resonance imaging uses strong magnetic fields to create",
    "The discovery of penicillin by Alexander Fleming in 1928 marked",
    "Dark matter makes up approximately 27 percent of the universe",
    "The Apollo 11 mission successfully landed humans on the Moon in",
    "CRISPR technology allows scientists to edit genes with unprecedented",
    "The Silk Road was an ancient network of trade routes connecting",
    "Chaos theory studies systems that are highly sensitive to initial",
    "The Hubble Space Telescope has provided some of the most detailed",
    "Photons are the fundamental particles of light and carry",
    "The Treaty of Versailles was signed in 1919 ending World War",
    "Deep learning has achieved breakthrough results in image recognition",
    "The human heart beats approximately 100,000 times per day pumping",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject_layer', type=int, default=8)
    parser.add_argument('--measure_layer', type=int, default=14)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--tag', type=str, default=None, help='Output file tag (default: auto from model name)')
    args = parser.parse_args()

    INJECT_LAYER = args.inject_layer
    MEASURE_LAYER = args.measure_layer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    load_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else None
    print(f"Loading {args.model} (dtype={args.dtype})...")
    model, tokenizer, device = load_model(args.model, device=device, dtype=load_dtype)
    blocks = model.model.layers
    n_layers = len(blocks)
    d_model = model._plateau_info['d_model']
    print(f"d_model={d_model}, n_layers={n_layers}, device={device}")
    print(f"Inject layer={INJECT_LAYER}, measure layer={MEASURE_LAYER}")

    if MEASURE_LAYER >= n_layers:
        print(f"WARNING: measure_layer {MEASURE_LAYER} >= n_layers {n_layers}, capping to {n_layers-1}")
        MEASURE_LAYER = n_layers - 1

    # Extract activations
    print(f"\nExtracting layer-{INJECT_LAYER} activations from {len(PROMPTS)} prompts...")
    contexts = []
    activations = []

    for text in PROMPTS[:N_TEXTS]:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[INJECT_LAYER + 1]
        contexts.append(hidden[:, :-1, :].cpu())
        activations.append(hidden[:, -1, :].squeeze(0).cpu())

    N_SWEEP_ACTUAL = min(N_SWEEP, len(activations))
    activations_t = torch.stack(activations[:N_SWEEP_ACTUAL])
    print(f"Got {N_SWEEP_ACTUAL} activations")
    print(f"Activation norms: mean={activations_t.norm(dim=-1).mean():.2f}, "
          f"std={activations_t.norm(dim=-1).std():.2f}")

    # Generate random activations (Gaussian with matched mean + covariance)
    print(f"\nGenerating {N_SWEEP_ACTUAL} random activations (matched Gaussian)...")
    act_matrix = activations_t.float().numpy()  # [N_SWEEP_ACTUAL, d_model]
    mean = act_matrix.mean(axis=0)
    centered = act_matrix - mean
    cov = (centered.T @ centered) / (N_SWEEP_ACTUAL - 1)
    rng_np = np.random.default_rng(SEED + 1)
    random_acts = rng_np.multivariate_normal(mean, cov, size=N_SWEEP_ACTUAL)
    random_acts_t = torch.tensor(random_acts, dtype=torch.float32)
    print(f"Random activation norms: mean={random_acts_t.norm(dim=-1).mean():.2f}, "
          f"std={random_acts_t.norm(dim=-1).std():.2f}")

    # Angle grid
    model_dtype = next(model.parameters()).dtype
    angles_deg = np.linspace(0, MAX_ANGLE, N_ANGLES + 1)[1:]
    angles_rad = np.deg2rad(angles_deg)
    cos_t = torch.tensor(np.cos(angles_rad), dtype=model_dtype, device=device)
    sin_t = torch.tensor(np.sin(angles_rad), dtype=model_dtype, device=device)

    def sweep_activations(acts, ctxs, label):
        """Sweep perturbations for a set of activations. Returns [n, N_DIRS, N_ANGLES]."""
        n = len(acts)
        curves = np.zeros((n, N_DIRS, N_ANGLES))
        t0 = time.time()
        for i in range(n):
            ctx = ctxs[i].to(device=device, dtype=model_dtype)
            act = acts[i].to(device=device, dtype=model_dtype)
            R = act.norm()
            a_unit = act / R

            with torch.no_grad():
                ref = forward_from_layer_to_layer(model, ctx, act[None, None], INJECT_LAYER, MEASURE_LAYER)
                ref = ref.float()

            for j in range(N_DIRS):
                d = torch.randn(d_model, device=device, dtype=model_dtype)
                d = d - (d @ a_unit) * a_unit
                d = F.normalize(d, dim=0)
                perturbed = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
                with torch.no_grad():
                    measured = forward_from_layer_to_layer(
                        model, ctx, perturbed.unsqueeze(1), INJECT_LAYER, MEASURE_LAYER
                    ).float()
                    cos_dist = 1 - F.cosine_similarity(ref, measured, dim=-1)
                    curves[i, j] = cos_dist.cpu().numpy()

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] {i+1}/{n}  ({(i+1)/elapsed:.2f} act/s)")
        elapsed = time.time() - t0
        print(f"  [{label}] Done in {elapsed:.1f}s")
        return curves

    # Sweep real activations
    print(f"\n=== Real activations ({N_SWEEP_ACTUAL} acts x {N_DIRS} dirs x {N_ANGLES} angles) ===")
    real_curves = sweep_activations(
        [activations[i] for i in range(N_SWEEP_ACTUAL)],
        [contexts[i] for i in range(N_SWEEP_ACTUAL)],
        "Real"
    )

    # Sweep random activations (use random context from the real set for each)
    print(f"\n=== Random activations ({N_SWEEP_ACTUAL} acts x {N_DIRS} dirs x {N_ANGLES} angles) ===")
    rand_ctx_indices = rng_np.integers(0, N_SWEEP_ACTUAL, size=N_SWEEP_ACTUAL)
    rand_curves = sweep_activations(
        [random_acts_t[i] for i in range(N_SWEEP_ACTUAL)],
        [contexts[rand_ctx_indices[i]] for i in range(N_SWEEP_ACTUAL)],
        "Random"
    )

    # Save
    model_tag = args.tag or args.model.split('/')[-1].lower().replace('-', '').replace('.', '')
    tag = f"{model_tag}_L{INJECT_LAYER}_L{MEASURE_LAYER}"
    results = {
        'config': {
            'model': args.model,
            'inject_layer': INJECT_LAYER,
            'measure_layer': MEASURE_LAYER,
            'n_sweep': N_SWEEP_ACTUAL,
            'n_dirs': N_DIRS,
            'max_angle': MAX_ANGLE,
            'n_angles': N_ANGLES,
            'seed': SEED,
        },
        'angles_deg': angles_deg,
        'real_curves': real_curves,
        'rand_curves': rand_curves,
        'norms': activations_t.float().norm(dim=-1).numpy(),
        'rand_norms': random_acts_t.float().norm(dim=-1).numpy(),
    }
    pkl_path = os.path.join(RESULTS_DIR, f'plateau_{tag}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    # Plot: real vs random
    real_per_act = np.median(real_curves, axis=1)  # [n_sweep, n_angles]
    rand_per_act = np.median(rand_curves, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Real
    ax.fill_between(angles_deg,
                    np.percentile(real_per_act, 25, axis=0),
                    np.percentile(real_per_act, 75, axis=0),
                    color='C0', alpha=0.15)
    ax.plot(angles_deg, np.median(real_per_act, axis=0),
            color='C0', lw=2.5, label=f'Real activations (n={N_SWEEP_ACTUAL})')
    # Random
    ax.fill_between(angles_deg,
                    np.percentile(rand_per_act, 25, axis=0),
                    np.percentile(rand_per_act, 75, axis=0),
                    color='C1', alpha=0.15)
    ax.plot(angles_deg, np.median(rand_per_act, axis=0),
            color='C1', lw=2.5, label=f'Random activations (n={N_SWEEP_ACTUAL})')

    ax.set_xlabel('Perturbation angle (degrees)')
    ax.set_ylabel(f'Cosine distance (at layer {MEASURE_LAYER})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ANGLE)
    ax.set_title(f'{args.model.split("/")[-1]}: real vs random, inject@L{INJECT_LAYER} measure@L{MEASURE_LAYER}')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'plateau_{tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Numerical summary
    for label, per_act in [("Real", real_per_act), ("Random", rand_per_act)]:
        med = np.median(per_act, axis=0)
        for thresh in [0.01, 0.05, 0.1]:
            crossings = np.where(med > thresh)[0]
            if len(crossings):
                print(f"[{label}] Median crosses {thresh} at {angles_deg[crossings[0]]:.1f}°")
            else:
                print(f"[{label}] Median never crosses {thresh}")


if __name__ == '__main__':
    main()
