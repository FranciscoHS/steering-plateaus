#!/usr/bin/env python3
"""Quick check: does AE cosine drop on held-out portion of same cache?"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from src.autoencoders import FlexAutoencoder, TopKSAE

device = "cuda" if torch.cuda.is_available() else "cpu"

acts = torch.load("results/trained_sae/qwen3-1.7b_layer10_k32/cached_activations.pt",
                  weights_only=True)
print(f"Cache shape: {acts.shape}")

train_sample = acts[:10000].to(device)
heldout_sample = acts[-10000:].to(device)

# Load AE
ae_data = torch.load("results/trained_autoencoder/qwen3-1.7b_layer10_k32/final_mse_model.pt",
                     weights_only=False, map_location=device)
config = ae_data["config"]
ae = FlexAutoencoder(d_in=config["d_in"], k=config["k"], hidden_dims=config["hidden_dims"],
                     activation=config.get("activation", "gelu"),
                     norm=config.get("norm", "batchnorm"))
ae.load_state_dict(ae_data["state_dict"])
ae = ae.to(device).eval()

# Load SAE
sae_data = torch.load("results/trained_sae/qwen3-1.7b_layer10_k32/sae_weights.pt",
                      weights_only=False, map_location=device)
sc = sae_data["config"]
sae = TopKSAE(d_in=sc["d_in"], d_sae=sc["d_sae"], k=sc["k"])
sae.load_state_dict(sae_data["state_dict"])
sae = sae.to(device).eval()

with torch.no_grad():
    for label, sample in [("train[:10k]", train_sample), ("cache[-10k:]", heldout_sample)]:
        ae_recon, _ = ae(sample)
        ae_cos = F.cosine_similarity(sample, ae_recon, dim=-1).mean().item()
        sae_recon, _, _ = sae(sample)
        sae_cos = F.cosine_similarity(sample, sae_recon, dim=-1).mean().item()
        print(f"{label}:  AE cos={ae_cos:.4f}  SAE cos={sae_cos:.4f}")
