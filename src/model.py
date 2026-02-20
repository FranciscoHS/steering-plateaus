"""
Model loading and layer-wise forward pass utilities.

Supports GPT-2 family and Llama-architecture models (Qwen3, etc.).
Uses HuggingFace transformers directly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


# Short names → HuggingFace model IDs
MODEL_ALIASES = {
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
    'qwen3-4b': 'Qwen/Qwen3-4B',
    'gemma-1-2b': 'google/gemma-2b',
    'gemma-2-2b': 'google/gemma-2-2b',
    'gemma-3-270m': 'google/gemma-3-270m',
    'gemma-3-1b': 'google/gemma-3-1b-pt',
    'gemma-3-4b': 'google/gemma-3-4b-pt',
    'gemma-3-12b': 'google/gemma-3-12b-pt',
}

# Multimodal Gemma 3 models (4B+) need Gemma3ForCausalLM to extract the text decoder
GEMMA3_MULTIMODAL = {'google/gemma-3-4b-pt', 'google/gemma-3-12b-pt', 'google/gemma-3-27b-pt'}


def _get_blocks(model):
    """Get the list of transformer blocks."""
    if hasattr(model, 'transformer'):  # GPT-2
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama/Qwen
        return model.model.layers
    raise ValueError(f"Unknown model architecture: {type(model)}")


def _get_final_norm(model):
    """Get the final layer norm."""
    if hasattr(model, 'transformer'):  # GPT-2
        return model.transformer.ln_f
    elif hasattr(model, 'model') and hasattr(model.model, 'norm'):  # Llama/Qwen
        return model.model.norm
    raise ValueError(f"Unknown model architecture: {type(model)}")


def _get_lm_head(model):
    """Get the language model head (unembedding)."""
    return model.lm_head


def _is_rope_model(model):
    """True for models that need explicit position_embeddings (Llama/Qwen/Gemma)."""
    return hasattr(model, '_plateau_info') and model._plateau_info['model_type'] in ('llama', 'gemma3')


def _get_position_embeddings(model, hidden_states):
    """Compute position embeddings for RoPE models.

    Args:
        model: Llama/Qwen/Gemma model
        hidden_states: [batch, seq_len, d_model] - used to infer seq_len and device

    Returns:
        For Gemma 3: ((cos_global, sin_global), (cos_local, sin_local))
        For others: (cos, sin) tuple
    """
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    if model._plateau_info['model_type'] == 'gemma3':
        global_emb = model.model.rotary_emb(hidden_states, position_ids)
        local_emb = model.model.rotary_emb_local(hidden_states, position_ids)
        return (global_emb, local_emb)
    return model.model.rotary_emb(hidden_states, position_ids)


def _run_block(block, hidden, model, pos_emb=None):
    """Run a single transformer block, handling different architectures.

    GPT-2 blocks return a tuple (hidden_states, ...).
    Llama/Qwen/Gemma blocks return a plain tensor.
    """
    model_type = model._plateau_info['model_type']
    if model_type == 'gemma3':
        global_emb, local_emb = pos_emb
        result = block(hidden, position_embeddings_global=global_emb,
                       position_embeddings_local=local_emb)
    elif model_type == 'llama':
        result = block(hidden, position_embeddings=pos_emb)
    else:
        result = block(hidden)
    # Unpack tuple if needed (GPT-2 style)
    if isinstance(result, tuple):
        return result[0]
    return result


def load_model(model_name='gpt2', device=None, dtype=None):
    """
    Load model and tokenizer.

    Args:
        model_name: Short name or HuggingFace model name.
            GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            Qwen: 'qwen3-0.6b'
        device: torch device (auto-detected if None)
        dtype: torch dtype for model weights (e.g. torch.bfloat16).
            None = float32 for most models, bfloat16 for Gemma3 multimodal.

    Returns:
        model, tokenizer, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_name = MODEL_ALIASES.get(model_name, model_name)

    # Detect architecture type
    is_gpt2 = model_name.startswith('gpt2')

    if is_gpt2:
        model = GPT2LMHeadModel.from_pretrained(hf_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(hf_name)
        model_type = 'gpt2'
        d_model = model.config.n_embd
    elif hf_name in GEMMA3_MULTIMODAL:
        # Multimodal Gemma 3 (4B+): use Gemma3ForCausalLM to extract text decoder
        from transformers import Gemma3ForCausalLM
        load_dtype = dtype if dtype is not None else torch.bfloat16
        model = Gemma3ForCausalLM.from_pretrained(hf_name, torch_dtype=load_dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model_type = 'gemma3'
        d_model = model.config.hidden_size
    else:
        load_dtype = dtype if dtype is not None else torch.float32
        model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=load_dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        config_type = getattr(model.config, 'model_type', '')
        if config_type.startswith('gemma3'):
            model_type = 'gemma3'
        else:
            model_type = 'llama'  # Llama-architecture (Qwen3, Llama, Gemma 1/2, etc.)
        d_model = model.config.hidden_size

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Attach metadata for dispatch
    model._plateau_info = {'model_type': model_type, 'd_model': d_model}

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
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n_points, 1, hidden_dim] - Final-token activation(s)
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        logits: [n_points, vocab_size] - Output logits for final token
    """
    n_points = point.shape[0]

    context_batch = context.expand(n_points, -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    if rope:
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb if rope else None)

    hidden = _get_final_norm(model)(hidden)
    logits = _get_lm_head(model)(hidden[:, -1, :])

    return logits


def forward_from_layer_to_layer(model, context, point, start_layer, end_layer):
    """
    Forward from start_layer to end_layer (inclusive). Returns last-token residual stream.

    Like forward_from_layer_to_final but stops at a specific intermediate layer
    instead of running through all remaining blocks.

    Args:
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n, 1, hidden_dim] - Last-token activation (may carry gradients)
        start_layer: Layer index (0-indexed). Runs blocks starting from start_layer+1.
        end_layer: Layer index (0-indexed, inclusive). Runs blocks through end_layer.

    Returns:
        final_act: [n, hidden_dim] - Last-token residual stream after end_layer block
    """
    context_batch = context.expand(point.shape[0], -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    if rope:
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb if rope else None)

    return hidden[:, -1, :]


def forward_from_layer_to_final(model, context, point, start_layer):
    """
    Forward from layer L to final transformer block (no ln_f, no unembedding).

    Returns last-token residual stream after the final transformer block.
    Differentiable w.r.t. point — uses only out-of-place ops.

    Args:
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n, 1, hidden_dim] - Last-token activation (may carry gradients)
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        final_act: [n, hidden_dim] - Last-token residual stream after final block
    """
    context_batch = context.expand(point.shape[0], -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    if rope:
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb if rope else None)

    return hidden[:, -1, :]


def forward_from_layer_allpos(model, hidden_states, start_layer):
    """
    Forward pass from an arbitrary layer, patching ALL positions.

    Unlike forward_from_layer (which takes separate context + final-token),
    this takes the full residual stream at all positions and runs the
    remaining layers + ln_f + lm_head.

    Args:
        model: GPT-2 or Llama-arch model
        hidden_states: [batch, seq_len, hidden_dim] - Full residual stream at start_layer
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        logits: [batch, vocab_size] - Output logits for final token only
    """
    hidden = hidden_states

    blocks = _get_blocks(model)
    rope = _is_rope_model(model)
    if rope:
        pos_emb = _get_position_embeddings(model, hidden)

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb if rope else None)

    hidden = _get_final_norm(model)(hidden)
    logits = _get_lm_head(model)(hidden[:, -1, :])

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
        model: GPT-2 or Llama-arch model
        tokenizer: tokenizer
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


def extract_activations_allpos(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """
    Extract residual stream activations at ALL positions for fixed-length token sequences.

    Args:
        model: GPT-2 or Llama-arch model
        tokenizer: tokenizer
        token_ids_list: list of [seq_len] tensors (all same length)
        layer: transformer block index (0-indexed)
        device: torch device
        batch_size: sequences per batch

    Returns:
        activations: [n_sequences, seq_len * hidden_dim] tensor (CPU) - concatenated
    """
    all_activations = []

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)

        # hidden_states[layer+1] = residual stream after block `layer`
        hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d]

        # Flatten positions: [batch, seq_len * d]
        batch_flat = hidden.reshape(hidden.shape[0], -1).cpu()
        all_activations.append(batch_flat)

        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {min(i + batch_size, len(token_ids_list))}/{len(token_ids_list)}", end='\r')

    print()
    return torch.cat(all_activations, dim=0)
