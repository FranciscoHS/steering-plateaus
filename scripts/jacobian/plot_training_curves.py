#!/usr/bin/env python3
"""Plot training loss curves for AE vs SAE final comparison."""

import pickle
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_ae_log(path):
    """Parse step/loss/cos from AE training log text."""
    steps, losses, cos_sims = [], [], []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s+Step\s+(\d+): loss=([\d.]+), cos=([-\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                cos_sims.append(float(m.group(3)))
    return np.array(steps), np.array(losses), np.array(cos_sims)


def main():
    base = 'results/final_comparison'

    # SAE
    with open(f'{base}/sae_k32_50k/training_log.pkl', 'rb') as f:
        sae_log = pickle.load(f)
    sae_steps = np.array([d['step'] for d in sae_log])
    sae_loss = np.array([d['loss'] for d in sae_log])

    # AE MSE+cos
    ae_mc_steps, ae_mc_loss, _ = parse_ae_log(f'{base}/ae_mse_cos_training.log')

    # AE MSE
    ae_m_steps, ae_m_loss, _ = parse_ae_log(f'{base}/ae_mse_training.log')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: MSE losses (SAE + AE-MSE)
    ax1.plot(sae_steps, sae_loss, label='SAE (MSE loss)', alpha=0.8)
    ax1.plot(ae_m_steps, ae_m_loss, label='AE — MSE loss', alpha=0.8)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('MSE loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: MSE+cos loss (AE-MSE+cos)
    ax2.plot(ae_mc_steps, ae_mc_loss, label='AE — MSE+cos (α=5)', color='C2', alpha=0.8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('MSE + 5·(1−cos) Loss')
    ax2.set_title('MSE + cosine loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Training loss vs step (k=32, 50k steps)', fontsize=13)

    out = f'{base}/training_loss_curves.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
