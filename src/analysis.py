"""
Analysis functions for steering activation plateaus.

Includes:
- Velocity profiles (output change rates along paths)
- Jacobian/sensitivity analysis
- Eigenvalue computation for J^T J
- Entropy and confidence metrics
"""

import torch
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from .model import forward_from_layer1, get_logits_at_point


def compute_velocity_profile(model, prefix_resid, path):
    """
    Compute output velocity profile ||f(t+1) - f(t)|| along a path.

    This measures how fast the output distribution changes as we move
    along the path in activation space.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        path: [n_steps, 1, hidden_dim]

    Returns:
        velocities: numpy array of shape [n_steps-1]
    """
    with torch.no_grad():
        logits = forward_from_layer1(model, prefix_resid, path)
        diffs = logits[1:] - logits[:-1]
        velocities = diffs.norm(dim=1).cpu().numpy()
    return velocities


def peak_to_average_ratio(velocities):
    """Compute Peak-to-Average Ratio (PAR) of velocity profile."""
    return np.max(velocities) / np.mean(velocities)


def compute_entropy(probs):
    """
    Compute entropy of a probability distribution.

    Lower entropy = more confident/peaked distribution.

    Args:
        probs: probability tensor

    Returns:
        entropy value (scalar)
    """
    p = probs[probs > 1e-10]
    return -torch.sum(p * torch.log(p)).item()


def get_top_k_predictions(model, tokenizer, prefix_resid, point, k=5):
    """
    Get top-k token predictions at a path point.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        k: Number of top predictions

    Returns:
        dict with 'tokens', 'probs', 'entropy'
    """
    with torch.no_grad():
        logits = get_logits_at_point(model, prefix_resid, point)
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

        entropy = compute_entropy(probs)

    return {
        'tokens': top_tokens,
        'probs': top_probs.cpu().numpy(),
        'entropy': entropy
    }


def compute_directional_sensitivity(model, prefix_resid, point, direction, epsilon=0.01):
    """
    Compute sensitivity ||∂f/∂x · v|| in a given direction.

    Uses finite differences to estimate the directional derivative.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim] - Point to analyze
        direction: [1, 1, hidden_dim] - Direction vector (should be unit norm)
        epsilon: Finite difference step size

    Returns:
        sensitivity: scalar (norm of output change per unit input change)
    """
    with torch.no_grad():
        logits_center = get_logits_at_point(model, prefix_resid, point)
        perturbed = point + epsilon * direction
        logits_perturbed = get_logits_at_point(model, prefix_resid, perturbed)

        diff = (logits_perturbed - logits_center) / epsilon
        sensitivity = torch.norm(diff).item()

    return sensitivity


def sample_random_sensitivities(model, prefix_resid, point, n_samples=100):
    """
    Sample sensitivity in random directions to characterize the local landscape.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        n_samples: Number of random directions to sample

    Returns:
        sensitivities: numpy array of sensitivities
    """
    device = point.device
    hidden_dim = point.shape[-1]
    sensitivities = []

    for _ in range(n_samples):
        random_dir = torch.randn(1, 1, hidden_dim, device=device)
        random_dir = random_dir / torch.norm(random_dir)
        sens = compute_directional_sensitivity(model, prefix_resid, point, random_dir)
        sensitivities.append(sens)

    return np.array(sensitivities)


def compute_top_singular_values(model, prefix_resid, point, k=100):
    """
    Compute top-k singular values of the Jacobian J = d(logits)/d(activation).

    Uses Lanczos iteration to find top eigenvalues of J^T J, then takes sqrt.
    This characterizes the local sensitivity structure.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        k: Number of singular values to compute

    Returns:
        singular_values: numpy array of top-k singular values (descending)
        eigenvectors: [k, hidden_dim] array of corresponding directions
    """
    device = point.device
    d = point.shape[-1]

    # Precompute base logits
    with torch.no_grad():
        logits_base = get_logits_at_point(model, prefix_resid, point)

    def jtj_matvec(v_np):
        """Compute J^T J v using finite differences + backprop."""
        v = torch.tensor(v_np, dtype=torch.float32, device=device).reshape(1, 1, d)

        # Jv via finite differences
        eps = 1e-4
        with torch.no_grad():
            logits_pert = get_logits_at_point(model, prefix_resid, point + eps * v)
        Jv = (logits_pert - logits_base) / eps

        # J^T(Jv) via backprop
        act = point.clone().requires_grad_(True)
        logits = get_logits_at_point(model, prefix_resid, act)
        (logits * Jv.detach()).sum().backward()

        return act.grad.reshape(-1).cpu().numpy().astype(np.float64)

    # Lanczos iteration
    JTJ_op = LinearOperator((d, d), matvec=jtj_matvec, dtype=np.float64)
    eigenvalues, eigenvectors = eigsh(JTJ_op, k=k, which='LM')

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert to singular values
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    return singular_values, eigenvectors.T


def analyze_path_semantics(model, tokenizer, prefix_resid, path, sample_indices=None):
    """
    Analyze semantic coherence of predictions along a path.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prefix_resid: [1, seq_len-1, hidden_dim]
        path: [n_steps, 1, hidden_dim]
        sample_indices: Which path indices to analyze (default: evenly spaced)

    Returns:
        list of dicts with 'index', 'tokens', 'probs', 'entropy' for each sample point
    """
    n_steps = path.shape[0]

    if sample_indices is None:
        # Sample ~7 evenly spaced points
        sample_indices = [0, n_steps // 5, 2 * n_steps // 5, n_steps // 2,
                          3 * n_steps // 5, 4 * n_steps // 5, n_steps - 1]

    results = []
    for idx in sample_indices:
        point = path[idx:idx + 1]
        preds = get_top_k_predictions(model, tokenizer, prefix_resid, point)
        preds['index'] = idx
        results.append(preds)

    return results
