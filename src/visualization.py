"""
Visualization functions for steering activation plateaus.

Standard plotting utilities for velocity profiles, path analysis, etc.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_profiles(velocities_dict, title=None, save_path=None):
    """
    Plot velocity profiles for multiple paths.

    Args:
        velocities_dict: dict mapping path names to velocity arrays
            e.g., {'Slerp': vel_slerp, 'Optimized': vel_opt}
        title: Plot title (optional)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['--', '-', '-.', ':']

    for i, (name, velocities) in enumerate(velocities_dict.items()):
        par = np.max(velocities) / np.mean(velocities)
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]

        plt.plot(velocities, label=f'{name} (PAR={par:.2f})',
                 color=color, linestyle=ls, linewidth=2 if i > 0 else 1.5,
                 alpha=0.6 if i == 0 else 1.0)

        # Mean line
        plt.axhline(y=np.mean(velocities), color=color, linestyle=':', alpha=0.3)

    plt.xlabel("Step Index $t$")
    plt.ylabel("Output Step Size $||f_{t+1} - f_t||$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if title:
        plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_path_lengths(step_lengths_dict, n_steps, straight_line_dist=None,
                      title=None, save_path=None):
    """
    Plot step lengths along paths.

    Args:
        step_lengths_dict: dict mapping path names to step length arrays
        n_steps: Total number of steps (for uniform line)
        straight_line_dist: Euclidean distance (optional, for reference line)
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['red', 'blue', 'green', 'orange']

    # Plot 1: Step lengths
    for i, (name, lengths) in enumerate(step_lengths_dict.items()):
        ax1.plot(lengths, label=name, color=colors[i % len(colors)],
                 linestyle='--' if i == 0 else '-', linewidth=2)

    if straight_line_dist is not None:
        ax1.axhline(y=straight_line_dist / (n_steps - 1), color='gray',
                    linestyle=':', alpha=0.5, label='Uniform steps')

    ax1.set_xlabel("Step Index")
    ax1.set_ylabel("Step Length")
    ax1.set_title("Step Lengths Along Paths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative distance
    for i, (name, lengths) in enumerate(step_lengths_dict.items()):
        cumulative = np.cumsum(np.concatenate([[0], lengths]))
        ax2.plot(cumulative, label=name, color=colors[i % len(colors)],
                 linestyle='--' if i == 0 else '-', linewidth=2)

    ax2.set_xlabel("Step Index")
    ax2.set_ylabel("Cumulative Distance")
    ax2.set_title("Cumulative Path Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_singular_values(singular_values, title=None, save_path=None, log_scale=True):
    """
    Plot singular value spectrum.

    Args:
        singular_values: array of singular values (descending)
        title: Plot title
        save_path: Path to save figure
        log_scale: Use log scale for y-axis
    """
    plt.figure(figsize=(10, 5))

    plt.plot(singular_values, 'b.-', linewidth=1, markersize=4)

    if log_scale:
        plt.yscale('log')

    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True, alpha=0.3)

    if title:
        plt.title(title)
    else:
        plt.title(f"Top {len(singular_values)} Singular Values of Jacobian")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_optimization_history(history, save_path=None):
    """
    Plot optimization loss curves.

    Args:
        history: dict with 'loss', 'energy_input', 'energy_output' lists
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['loss'], 'b-')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['energy_input'], 'g-')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Input Energy")
    axes[1].set_title("Input Energy (Path Length²)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['energy_output'], 'r-')
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Output Energy")
    axes[2].set_title("Output Energy")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_sensitivity_histogram(sensitivities, slerp_sensitivity=None,
                               title=None, save_path=None):
    """
    Plot histogram of random direction sensitivities.

    Args:
        sensitivities: array of sensitivities from random directions
        slerp_sensitivity: sensitivity along slerp direction (for reference)
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))

    plt.hist(sensitivities, bins=30, alpha=0.7, edgecolor='black')

    if slerp_sensitivity is not None:
        plt.axvline(slerp_sensitivity, color='red', linestyle='--',
                    linewidth=2, label=f'Slerp direction: {slerp_sensitivity:.2f}')
        plt.legend()

    plt.xlabel("Sensitivity")
    plt.ylabel("Count")

    if title:
        plt.title(title)
    else:
        plt.title("Sensitivity Distribution (Random Directions)")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_distribution_comparison(real_values, random_values, metric_name,
                                  title=None, save_path=None):
    """
    Plot overlaid histograms comparing real vs random activation metrics.

    Args:
        real_values: array of metric values for real activations
        random_values: array of metric values for random activations
        metric_name: Name of metric (for axis label)
        title: Plot title (auto-generated if None)
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))

    # Compute shared bins
    all_values = np.concatenate([real_values, random_values])
    bins = np.linspace(np.min(all_values), np.max(all_values), 30)

    plt.hist(real_values, bins=bins, alpha=0.6, label='Real', color='blue', edgecolor='black')
    plt.hist(random_values, bins=bins, alpha=0.6, label='Random', color='red', edgecolor='black')

    # Add vertical lines for means
    real_mean = np.mean(real_values)
    random_mean = np.mean(random_values)
    plt.axvline(real_mean, color='blue', linestyle='--', linewidth=2,
                label=f'Real mean: {real_mean:.2f}')
    plt.axvline(random_mean, color='red', linestyle='--', linewidth=2,
                label=f'Random mean: {random_mean:.2f}')

    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if title:
        plt.title(title)
    else:
        plt.title(f"{metric_name}: Real vs Random Activations")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
