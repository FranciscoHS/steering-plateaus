"""
Autoencoder and sparse autoencoder classes used across training, probe, and
steering scripts.

Three incompatible AE state-dict layouts coexist in this repo because they
were developed at different times. Keep them all so existing trained
checkpoints continue to load:

- ``FlexAutoencoder`` — configurable Block-based AE (hillclimb training pipeline).
  Parameter names: ``encoder_blocks.{i}.linear.weight``, ``encoder_out.weight``, ...
  ``forward()`` returns ``(recon, z)``.

- ``DeepAutoencoder`` — simple ``nn.Sequential`` AE with ReLU (older training
  pipeline). Parameter names: ``encoder.{i}.weight``.
  ``forward()`` returns ``(recon, z)``.

- ``Autoencoder`` — simple ``nn.Sequential`` AE with GELU+BatchNorm
  (default 4096/4096/2048 hidden dims). Used by the steering scripts that
  load AE checkpoints trained with this layout. Parameter names:
  ``encoder.{i}.weight``. ``forward()`` returns ``(recon, z)``.

- ``TopKSAE`` — TopK sparse autoencoder. ``forward()`` returns
  ``(recon, top_vals, top_idx)``.

Plus ``load_ae`` / ``load_sae`` helpers for the FlexAutoencoder and TopKSAE
on-disk formats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Activations ─────────────────────────────────────────────────────────

ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


class SwiGLU(nn.Module):
    """SwiGLU activation: takes 2*d input, splits into value and gate."""
    def forward(self, x):
        v, g = x.chunk(2, dim=-1)
        return v * F.silu(g)


class GEGLU(nn.Module):
    """GEGLU activation: takes 2*d input, splits into value and gate."""
    def forward(self, x):
        v, g = x.chunk(2, dim=-1)
        return v * F.gelu(g)


# ── FlexAutoencoder ─────────────────────────────────────────────────────

class Block(nn.Module):
    """Building block for FlexAutoencoder: linear + activation + (norm) + (dropout) + (residual)."""

    def __init__(self, d_in, d_out, activation='relu', norm='none', residual=False,
                 dropout=0.0):
        super().__init__()
        self.residual = residual and (d_in == d_out)

        # Gated activations (SwiGLU/GEGLU) need 2x intermediate width
        if activation in ('swiglu', 'geglu'):
            self.linear = nn.Linear(d_in, d_out * 2)
            self.act = SwiGLU() if activation == 'swiglu' else GEGLU()
        else:
            self.linear = nn.Linear(d_in, d_out)
            self.act = ACTIVATIONS[activation]()

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm1d(d_out)
        elif norm == 'layernorm':
            self.norm = nn.LayerNorm(d_out)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.act(self.linear(x))
        out = self.norm(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out


class FlexAutoencoder(nn.Module):
    """Configurable MLP autoencoder with k-dimensional bottleneck.

    Supports SwiGLU/GEGLU, dropout, bottleneck noise, spectral norm, and
    asymmetric encoder/decoder shapes.
    """

    def __init__(self, d_in, k, hidden_dims, activation='relu',
                 norm='none', residual=False, bottleneck_noise=0.0,
                 encoder_dims=None, decoder_dims=None, spectral_norm=False,
                 dropout=0.0):
        super().__init__()
        self.d_in = d_in
        self.k = k
        self.bottleneck_noise = bottleneck_noise

        enc_dims = encoder_dims if encoder_dims is not None else hidden_dims
        dec_dims = decoder_dims if decoder_dims is not None else list(reversed(hidden_dims))

        maybe_sn = nn.utils.spectral_norm if spectral_norm else (lambda x: x)

        # Encoder
        encoder_blocks = []
        prev = d_in
        for h in enc_dims:
            block = Block(prev, h, activation, norm, residual, dropout=dropout)
            if spectral_norm:
                block.linear = maybe_sn(block.linear)
            encoder_blocks.append(block)
            prev = h
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.encoder_out = maybe_sn(nn.Linear(prev, k))

        # Decoder
        decoder_blocks = []
        prev = k
        for h in dec_dims:
            block = Block(prev, h, activation, norm, residual, dropout=dropout)
            if spectral_norm:
                block.linear = maybe_sn(block.linear)
            decoder_blocks.append(block)
            prev = h
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_out = maybe_sn(nn.Linear(prev, d_in))

    def encode(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_out(x)

    def decode(self, z):
        x = z
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_out(x)

    def forward(self, x):
        z = self.encode(x)
        if self.training and self.bottleneck_noise > 0:
            z = z + torch.randn_like(z) * self.bottleneck_noise
        recon = self.decode(z)
        return recon, z


# ── DeepAutoencoder ─────────────────────────────────────────────────────

class DeepAutoencoder(nn.Module):
    """Older Sequential-based AE with ReLU. Used by train_autoencoder.py.

    Architecture:
        Encoder: d_in → h1 → h2 → k
        Decoder: k → h2 → h1 → d_in
    """

    def __init__(self, d_in, k, hidden_dims=None, use_batchnorm=False):
        super().__init__()
        self.d_in = d_in
        self.k = k
        self.use_batchnorm = use_batchnorm

        if hidden_dims is None:
            hidden_dims = [d_in // 2, d_in // 4]

        # Encoder
        encoder_layers = []
        prev_dim = d_in
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, k))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = k
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, d_in))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ── Autoencoder (simple GELU+BatchNorm) ─────────────────────────────────

class Autoencoder(nn.Module):
    """Simple Sequential MLP autoencoder with GELU + BatchNorm.

    Default architecture (hidden_dims=[4096, 4096, 2048]) matches the
    best config from the hillclimb sweep, but the state dict uses
    Sequential-indexed parameter names rather than FlexAutoencoder's
    Block-based names. Kept as a distinct class because existing
    on-disk weights (under results/on_manifold_steering/ etc.) are
    saved in this layout.
    """

    def __init__(self, d_in, k, hidden_dims=None):
        super().__init__()
        self.d_in = d_in
        self.k = k
        if hidden_dims is None:
            hidden_dims = [4096, 4096, 2048]

        # Encoder
        enc = []
        prev = d_in
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU()])
            prev = h
        enc.append(nn.Linear(prev, k))
        self.encoder = nn.Sequential(*enc)

        # Decoder (mirror)
        dec = []
        prev = k
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU()])
            prev = h
        dec.append(nn.Linear(prev, d_in))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ── TopKSAE ─────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """TopK sparse autoencoder with unit-norm decoder columns.

    forward(x) returns (recon, top_vals, top_idx) — three values, matching
    train_sae.py. Callers that only need the reconstruction should unpack:
        recon, _, _ = sae(x)
    """

    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_in, d_sae)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x):
        """Encode activations to sparse codes.

        Returns:
            top_vals: (batch, k) top-k activation values
            top_idx:  (batch, k) top-k feature indices
            pre_acts: (batch, d_sae) full pre-top-k activations (post-ReLU)
        """
        x_centered = x - self.b_dec
        pre_acts = F.relu(self.encoder(x_centered))
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        return top_vals, top_idx, pre_acts

    def decode(self, top_vals, top_idx):
        """Decode sparse codes to reconstructed activations."""
        W_selected = self.W_dec[top_idx]
        recon = (top_vals.unsqueeze(-1) * W_selected).sum(dim=1) + self.b_dec
        return recon

    def encode_full(self, x):
        """Encode to the full pre-activation vector (post-ReLU, pre-top-k)."""
        x_centered = x - self.b_dec
        return F.relu(self.encoder(x_centered))

    def decode_from_full(self, pre_acts):
        """Decode from a full pre-activation vector via top-k."""
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        W_selected = self.W_dec[top_idx]
        recon = (top_vals.unsqueeze(-1) * W_selected).sum(dim=1) + self.b_dec
        return recon

    def forward(self, x):
        top_vals, top_idx, _ = self.encode(x)
        recon = self.decode(top_vals, top_idx)
        return recon, top_vals, top_idx

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


# ── Loading helpers ─────────────────────────────────────────────────────

def load_sae(path, device):
    """Load a TopKSAE checkpoint saved as {'state_dict': ..., 'config': ...}."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    sae = TopKSAE(config['d_in'], config['d_sae'], config['k'])
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()
    return sae.to(device)


def load_ae(path, device):
    """Load a FlexAutoencoder checkpoint saved as {'state_dict': ..., 'config': ...}."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    ae = FlexAutoencoder(
        d_in=config['d_in'], k=config['k'],
        hidden_dims=config.get('hidden_dims', [4096, 4096, 2048]),
        activation=config.get('activation', 'gelu'),
        norm=config.get('norm', 'batchnorm'),
        residual=config.get('residual', False),
        encoder_dims=config.get('encoder_dims'),
        decoder_dims=config.get('decoder_dims'),
    )
    ae.load_state_dict(checkpoint['state_dict'])
    ae.eval()
    return ae.to(device)
