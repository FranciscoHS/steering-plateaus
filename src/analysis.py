"""
Analysis functions for steering activation plateaus.

Includes:
- Velocity profiles (output change rates along paths)
- Jacobian/sensitivity analysis
- Eigenvalue computation for J^T J
- Entropy and confidence metrics
"""

import torch
from torch.func import jvp
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from .model import (forward_from_layer1, forward_from_layer, forward_from_layer_allpos,
                     forward_from_layer_to_final, get_logits_at_point)


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


def compute_directional_sensitivity(model, prefix_resid, point, direction, layer=0, epsilon=0.01):
    """
    Compute sensitivity ||∂f/∂x · v|| in a given direction.

    Uses finite differences to estimate the directional derivative.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim] - Point to analyze
        direction: [1, 1, hidden_dim] - Direction vector (should be unit norm)
        layer: Layer index (0-indexed)
        epsilon: Finite difference step size

    Returns:
        sensitivity: scalar (norm of output change per unit input change)
    """
    with torch.no_grad():
        logits_center = get_logits_at_point(model, prefix_resid, point, layer=layer)
        perturbed = point + epsilon * direction
        logits_perturbed = get_logits_at_point(model, prefix_resid, perturbed, layer=layer)

        diff = (logits_perturbed - logits_center) / epsilon
        sensitivity = torch.norm(diff).item()

    return sensitivity


def sample_random_sensitivities(model, prefix_resid, point, layer=0, n_samples=100):
    """
    Sample sensitivity in random directions to characterize the local landscape.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed)
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
        sens = compute_directional_sensitivity(model, prefix_resid, point, random_dir, layer=layer)
        sensitivities.append(sens)

    return np.array(sensitivities)


def compute_top_singular_values(model, prefix_resid, point, layer=0, k=100):
    """
    Compute top-k singular values of the Jacobian J = d(logits)/d(activation).

    Uses Lanczos iteration to find top eigenvalues of J^T J, then takes sqrt.
    This characterizes the local sensitivity structure.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed)
        k: Number of singular values to compute

    Returns:
        singular_values: numpy array of top-k singular values (descending)
        eigenvectors: [k, hidden_dim] array of corresponding directions
    """
    device = point.device
    d = point.shape[-1]

    # Precompute base logits
    with torch.no_grad():
        logits_base = get_logits_at_point(model, prefix_resid, point, layer=layer)

    def jtj_matvec(v_np):
        """Compute J^T J v using finite differences + backprop."""
        v = torch.tensor(v_np, dtype=torch.float32, device=device).reshape(1, 1, d)

        # Jv via finite differences
        eps = 1e-4
        with torch.no_grad():
            logits_pert = get_logits_at_point(model, prefix_resid, point + eps * v, layer=layer)
        Jv = (logits_pert - logits_base) / eps

        # J^T(Jv) via backprop
        act = point.clone().requires_grad_(True)
        logits = get_logits_at_point(model, prefix_resid, act, layer=layer)
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


def estimate_frobenius_norm_hutchinson(model, activations, layer, n_probes=100, seed=None):
    """
    Estimate Frobenius norm of Jacobian via Hutchinson's trace estimator.

    Uses forward-mode AD (torch.func.jvp) for exact Jacobian-vector products.
    Estimates ||J||_F^2 = E[||Jv||^2] where v ~ N(0, I).

    The function f maps layer-L activations (all positions, flattened) to
    final-position logits.

    Temporarily disables SDPA (scaled dot product attention) because
    torch.func.jvp doesn't support it yet.

    Args:
        model: GPT-2 model
        activations: [1, seq_len, hidden_dim] - Full residual stream at layer L
        layer: Layer index (0-indexed)
        n_probes: Number of random probe vectors
        seed: Random seed (optional)

    Returns:
        frobenius_norm: estimated ||J||_F
        frobenius_sq_std: std of the per-probe estimates of ||J||_F^2 (for diagnostics)
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = activations.device
    seq_len, d_model = activations.shape[1], activations.shape[2]
    D = seq_len * d_model

    # Define the function from flat activations to logits
    # Disable SDPA since forward-mode AD doesn't support it
    def f(flat_act):
        act = flat_act.reshape(1, seq_len, d_model)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return forward_from_layer_allpos(model, act, layer)  # [1, vocab]

    flat_act = activations.reshape(D)

    estimates = []
    for i in range(n_probes):
        # Random Gaussian probe vector
        v = torch.randn(D, device=device)

        # Exact Jv via forward-mode AD
        _, Jv = jvp(f, (flat_act,), (v,))

        # ||Jv||^2 is an unbiased estimate of ||J||_F^2 (when v ~ N(0, I))
        estimates.append(Jv.squeeze().norm().item() ** 2)

    estimates = np.array(estimates)
    frobenius_sq = np.mean(estimates)
    frobenius_norm = np.sqrt(max(frobenius_sq, 0))
    frobenius_sq_std = np.std(estimates)

    return frobenius_norm, frobenius_sq_std


def estimate_frobenius_norm(model, context, point, layer, n_samples=100, epsilon=0.01):
    """
    Estimate Frobenius norm of Jacobian via random projections.

    Uses the identity: E[||Jv||^2] = ||J||_F^2 / d for unit random v.
    So ||J||_F^2 = d * E[||Jv||^2].

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed)
        n_samples: Number of random directions
        epsilon: Finite difference step size

    Returns:
        frobenius_norm: estimated ||J||_F
    """
    sensitivities = sample_random_sensitivities(
        model, context, point, layer=layer, n_samples=n_samples
    )
    d = point.shape[-1]
    # E[||Jv||^2] where v is unit vector
    mean_sq = np.mean(sensitivities ** 2)
    frobenius_sq = d * mean_sq
    return np.sqrt(frobenius_sq)


def compute_jacobian_norm_vjp(model, context, point, layer, n_probes=100):
    """
    Estimate ||J||_F for J mapping layer L last-token → final layer last-token.

    Uses VJP (backward-mode AD) with Hutchinson estimator. Standard autograd,
    no torch.func needed.

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim] - Prefix at layer L (detached)
        point: [hidden_dim] tensor - Last-token activation (detached)
        layer: Start layer index
        n_probes: Number of Hutchinson probes

    Returns:
        frobenius_norm: estimated ||J||_F
        frobenius_sq_std: std of per-probe ||J||_F^2 estimates
    """
    d_model = point.shape[-1]
    device = point.device

    estimates = []
    for k in range(n_probes):
        point_var = point.clone().reshape(1, 1, d_model).requires_grad_(True)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            final_act = forward_from_layer_to_final(model, context, point_var, layer)

        # VJP: random probe in output space
        u = torch.randn_like(final_act)
        s = (final_act * u).sum()
        g = torch.autograd.grad(s, point_var)[0]

        estimates.append((g ** 2).sum().item())

    estimates = np.array(estimates)
    frob_sq = np.mean(estimates)
    frob_norm = np.sqrt(max(frob_sq, 0))
    frob_sq_std = np.std(estimates)

    return frob_norm, frob_sq_std


def compute_logit_entropy(model, context, point, layer):
    """
    Compute entropy of the output distribution at a point.

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed)

    Returns:
        entropy: scalar
    """
    with torch.no_grad():
        logits = get_logits_at_point(model, context, point, layer=layer)
        probs = torch.softmax(logits, dim=-1)
        return compute_entropy(probs)
