# Implementation Plan: Real vs Random Activation Comparison

## Goal
Compare geometric properties (Jacobian structure, entropy) of real activations vs random activations in GPT-2 Large at layer 6.

## Key Design Decisions
- **Model:** GPT-2 Large, residual stream at layer 6, final token position only
- **Real activations:** 100 samples from FineWeb, each evaluated with its own native context (prefix residual stream at layer L)
- **Random activations:** 100 samples from N(μ, Σ) where mean and full covariance are estimated from ~2000 FineWeb activations (to ensure full-rank covariance, since d=1280). All paired with a single fixed real context.
- **Context pairing rationale:** Real activations keep their native context because the final-token vector and context co-evolved through the forward pass. Random activations use one fixed context since they have no natural context. Causal attention means prefix tokens don't attend to final token, so context at layer L is independent of final token.
- **Metrics per point:** Top-5 singular values of Jacobian (via Lanczos on J^T J), Jacobian Frobenius norm (estimated via random projections), logit entropy
- **Framework:** HuggingFace transformers (extending existing codebase, not switching to TransformerLens)

## Files to Modify/Create

### `src/model.py` — add two functions
- `forward_from_layer(model, context, point, start_layer)` — generalized forward pass from arbitrary layer. Takes context [1, seq_len-1, d] and point [batch, 1, d], concatenates, runs blocks [start_layer+1:] + ln_f + lm_head, returns final-token logits.
- `extract_activations(model, tokenizer, texts, layer, device)` — run full forward with output_hidden_states=True. Return list of (context_at_layer, final_token_activation) pairs. context = hidden_states[layer+1][:, :-1, :], activation = hidden_states[layer+1][:, -1, :].

### `src/data.py` — new file, two functions
- `load_fineweb_samples(n, tokenizer, min_length)` — load N texts from FineWeb (HuggingFace datasets), filter for minimum token length, return list of token tensors.
- `sample_random_activations(real_activations, n)` — takes tensor of real activations [N, d], computes mean and full covariance, samples n vectors from multivariate Gaussian. Expects N > d for full-rank covariance.

### `src/analysis.py` — make layer-aware
- Update `compute_top_singular_values` to accept `layer` param, use `forward_from_layer` instead of `forward_from_layer1`.
- Update `compute_directional_sensitivity` similarly.
- Update `sample_random_sensitivities` similarly.
- Add `estimate_frobenius_norm(model, context, point, layer, n_samples)` — sample random unit vectors, compute E[||Jv||^2] * d to estimate ||J||_F^2.

### `src/visualization.py` — add one function
- `plot_distribution_comparison(real_values, random_values, metric_name, save_path)` — overlaid histograms for real vs random.

### `scripts/run_plateau_analysis.py` — new experiment script
- Load GPT-2 Large
- Extract activations from ~2000 FineWeb texts at layer 6
- Select 100 as "real" samples (keep their contexts)
- Pick one context as the fixed context for random activations
- Generate 100 random activations from estimated N(μ, Σ)
- For each of 200 points: top-5 singular values, Frobenius norm, logit entropy
- Checkpoint after: activation extraction, random sampling, each batch of Jacobian computations
- Save results dict + generate comparison plots

## Future Extensions (not implemented now)
- Boundary points via slerp interpolation (requires designing meaningful prompt pairs differing only on final token)
- Distribution comparison across layers
- Other models (Qwen, etc.)
