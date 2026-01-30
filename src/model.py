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


def forward_from_layer(model, context, point, start_layer):
    """
    Forward pass from an arbitrary layer, batched over points.

    Generalizes forward_from_layer1 to start from any layer.

    Args:
        model: GPT-2 model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n_points, 1, hidden_dim] - Final-token activation(s)
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        logits: [n_points, vocab_size] - Output logits for final token
    """
    n_points = point.shape[0]

    context_batch = context.expand(n_points, -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    for block in model.transformer.h[start_layer + 1:]:
        hidden = block(hidden)[0]

    hidden = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden[:, -1, :])

    return logits


def get_logits_at_point(model, prefix_resid, point, layer=0):
    """
    Get output logits at a single activation point.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed). Default 0 for backward compatibility.

    Returns:
        logits: [vocab_size] tensor
    """
    logits = forward_from_layer(model, prefix_resid, point, layer)
    return logits.squeeze(0)


def extract_activations(model, tokenizer, texts, layer, device, batch_size=1):
    """
    Extract residual stream activations at a given layer for multiple texts.

    For each text, returns the context (prefix positions) and the final-token
    activation at the specified layer. Stores results on CPU to save GPU memory.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        texts: list of strings
        layer: transformer block index (0-indexed)
        device: torch device
        batch_size: texts per batch (1 for variable-length texts)

    Returns:
        contexts: list of [1, seq_len-1, hidden_dim] tensors (CPU)
        activations: [n_texts, hidden_dim] tensor (CPU)
    """
    contexts = []
    activations = []

    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        # hidden_states[layer+1] = residual stream after transformer block `layer`
        hidden = outputs.hidden_states[layer + 1]

        contexts.append(hidden[:, :-1, :].cpu())
        activations.append(hidden[:, -1, :].squeeze(0).cpu())

    activations = torch.stack(activations)

    return contexts, activations
