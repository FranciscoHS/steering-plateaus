"""
Path generation and optimization for steering activation plateaus.

Key functions:
- slerp_path: Generate spherical linear interpolation between two points
- optimize_path: Find geodesic-like paths that minimize output velocity variations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .model import forward_from_layer1


def slerp_path(v0, v1, n_steps, device=None):
    """
    Generate a Slerp (spherical linear interpolation) path between two vectors.

    Slerp maintains constant angular velocity along a great circle on the hypersphere.
    The magnitude is linearly interpolated.

    Args:
        v0: Start vector [1, 1, hidden_dim]
        v1: End vector [1, 1, hidden_dim]
        n_steps: Number of points along the path
        device: torch device (inferred from v0 if None)

    Returns:
        path: [n_steps, 1, hidden_dim] tensor
    """
    if device is None:
        device = v0.device

    t_values = torch.linspace(0, 1, n_steps, device=device)

    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)
    v0_unit = v0_flat / norm0
    v1_unit = v1_flat / norm1

    # Angle between vectors
    dot = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = torch.acos(dot)

    path = []
    for t in t_values:
        if theta.abs() < 1e-6:
            # Vectors are nearly parallel - use linear interpolation
            direction = v0_unit
        else:
            sin_theta = torch.sin(theta)
            direction = (torch.sin((1 - t) * theta) / sin_theta) * v0_unit + \
                        (torch.sin(t * theta) / sin_theta) * v1_unit

        # Linearly interpolate magnitude
        mag = (1 - t) * norm0 + t * norm1
        path.append(direction * mag)

    # Reshape to [n_steps, 1, hidden_dim]
    return torch.stack(path).reshape(n_steps, v0.shape[-2], v0.shape[-1])


def linear_path(v0, v1, n_steps, device=None):
    """
    Generate a simple linear interpolation path.

    Args:
        v0: Start vector [1, 1, hidden_dim]
        v1: End vector [1, 1, hidden_dim]
        n_steps: Number of points

    Returns:
        path: [n_steps, 1, hidden_dim] tensor
    """
    if device is None:
        device = v0.device

    t_values = torch.linspace(0, 1, n_steps, device=device)

    path = []
    for t in t_values:
        point = (1 - t) * v0 + t * v1
        path.append(point.squeeze(0))

    return torch.stack(path)


def optimize_path(model, prefix_resid, start_vec, end_vec, n_steps=50,
                  lambda_output=1.0, lr=0.05, iterations=200,
                  init_method='slerp', verbose=True):
    """
    Optimize a path to minimize output velocity variations (find geodesic-like paths).

    The optimization minimizes:
        L = E_input + λ * E_output

    Where:
        E_input  = Σ ||P[t+1] - P[t]||²  (input kinetic energy / path length)
        E_output = Σ ||f(P[t+1]) - f(P[t])||²  (output kinetic energy)

    This finds paths where the output changes smoothly along the path.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim] - Fixed context
        start_vec: [1, 1, hidden_dim] - Starting activation
        end_vec: [1, 1, hidden_dim] - Ending activation
        n_steps: Number of points in path
        lambda_output: Weight on output energy term
        lr: Learning rate
        iterations: Number of optimization steps
        init_method: 'slerp' or 'linear'
        verbose: Print progress

    Returns:
        optimized_path: [n_steps, 1, hidden_dim] tensor (detached)
        history: dict with 'loss', 'energy_input', 'energy_output' lists
    """
    device = start_vec.device

    # Initialize path
    if init_method == 'slerp':
        initial_path = slerp_path(start_vec, end_vec, n_steps, device)
    else:
        initial_path = linear_path(start_vec, end_vec, n_steps, device)

    # Make trainable
    optimized_path = nn.Parameter(initial_path.clone().detach())
    optimizer = optim.Adam([optimized_path], lr=lr)

    history = {'loss': [], 'energy_input': [], 'energy_output': []}

    for i in range(iterations):
        optimizer.zero_grad()

        # Forward pass
        logits = forward_from_layer1(model, prefix_resid, optimized_path)

        # Input energy: ||P[t+1] - P[t]||²
        p_flat = optimized_path.squeeze(1)
        diff_input = p_flat[1:] - p_flat[:-1]
        energy_input = torch.sum(diff_input.norm(dim=1) ** 2)

        # Output energy: ||f(P[t+1]) - f(P[t])||²
        diff_output = logits[1:] - logits[:-1]
        energy_output = torch.sum(diff_output.norm(dim=1) ** 2)

        loss = energy_input + lambda_output * energy_output

        loss.backward()
        optimizer.step()

        # Enforce boundary conditions
        with torch.no_grad():
            optimized_path[0].copy_(start_vec.squeeze(0))
            optimized_path[-1].copy_(end_vec.squeeze(0))

        history['loss'].append(loss.item())
        history['energy_input'].append(energy_input.item())
        history['energy_output'].append(energy_output.item())

        if verbose and i % 20 == 0:
            print(f"Iter {i:03d}: Loss {loss.item():.2f} "
                  f"(In: {energy_input.item():.2f}, Out: {energy_output.item():.2f})")

    return optimized_path.detach(), history


def compute_path_length(path):
    """
    Compute total arc length of a path in activation space.

    Args:
        path: [n_steps, 1, hidden_dim] or [n_steps, hidden_dim]

    Returns:
        total_length: scalar
        step_lengths: [n_steps-1] array of individual step lengths
    """
    if path.dim() == 3:
        path = path.squeeze(1)

    path = path.detach()
    diffs = path[1:] - path[:-1]
    step_lengths = torch.norm(diffs, dim=1)
    total_length = torch.sum(step_lengths).item()

    return total_length, step_lengths.cpu().numpy()
