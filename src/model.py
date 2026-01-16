"""
Model loading and layer-wise forward pass utilities for GPT-2.

Uses HuggingFace transformers directly (can switch to transformer_lens later).
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name='gpt2', device=None):
    """
    Load GPT-2 model and tokenizer.

    Args:
        model_name: HuggingFace model name ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        device: torch device (auto-detected if None)

    Returns:
        model, tokenizer, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()

    # Freeze model weights - we only optimize paths, not model
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, device


def get_layer0_output(model, tokenizer, prompt, device):
    """
    Get the Layer 0 output (after block 0) for a prompt.

    This gives us the residual stream state after the first transformer block,
    which we use as the starting point for path optimization.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text
        device: torch device

    Returns:
        hidden_states: [1, seq_len, hidden_dim] tensor
        input_ids: [1, seq_len] tensor
    """
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    # Compute embeddings: token embeddings + position embeddings
    wte = model.transformer.wte(input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    wpe = model.transformer.wpe(positions)
    hidden_states = wte + wpe

    # Apply dropout and first transformer block
    hidden_states = model.transformer.drop(hidden_states)
    hidden_states = model.transformer.h[0](hidden_states)[0]

    return hidden_states, input_ids


def forward_from_layer1(model, prefix_resid, path_vectors):
    """
    Forward pass from layer 1 onwards, batched over path points.

    This runs the model from layer 1 to the output, allowing us to
    optimize paths in the post-layer-0 residual stream space.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim] - Fixed context (all but last token)
        path_vectors: [n_steps, 1, hidden_dim] - Path points for the last token position

    Returns:
        logits: [n_steps, vocab_size] - Output logits for last token at each path point
    """
    n_steps = path_vectors.shape[0]

    # Expand prefix to batch size
    prefix_batch = prefix_resid.expand(n_steps, -1, -1)

    # Concatenate: [context tokens] + [path point]
    sequence_batch = torch.cat([prefix_batch, path_vectors], dim=1)

    hidden = sequence_batch

    # Pass through layers 1 to n-1
    for block in model.transformer.h[1:]:
        hidden = block(hidden)[0]

    # Final layer norm
    hidden = model.transformer.ln_f(hidden)

    # Project to vocab (last token only)
    last_token_hidden = hidden[:, -1, :]
    logits = model.lm_head(last_token_hidden)

    return logits


def get_logits_at_point(model, prefix_resid, point):
    """
    Get output logits at a single activation point.

    Convenience wrapper around forward_from_layer1 for single points.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]

    Returns:
        logits: [vocab_size] tensor
    """
    logits = forward_from_layer1(model, prefix_resid, point)
    return logits.squeeze(0)
