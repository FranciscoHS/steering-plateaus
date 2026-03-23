# Plan: Shareable Repo for LessWrong Post / Preprint

**Post title:** "Contrastive features elicit different perturbation responses than SAE features"

## Repo Name Options

- `contrastive-perturbation-responses` (concise, searchable)
- `contrastive-vs-sae-perturbation` (more descriptive)
- `perturbation-responses` (short, maybe too generic)

Francisco to pick.

## Repo Structure

```
<repo-name>/
├── README.md              # Overview, abstract, how to reproduce figures
├── requirements.txt       # Minimal deps (torch, transformers, sae_lens, etc.)
├── setup_gpu.sh           # GPU setup script (from setup_runpod.sh, generalized)
├── generate_figures.py    # Main entry point: --sweep and --plot modes
├── src/
│   ├── model.py           # Model loading + layer-wise forward passes (trimmed)
│   ├── sae.py             # SAE loading for all 3 model families (extracted)
│   ├── sweep.py           # Perturbation sweep logic (extracted from generate_figures.py)
│   └── data.py            # FineWeb loading (just load_fineweb_fixed_length)
├── data/
│   ├── bilingual_prompts.py
│   ├── english_prompts_holdout.py
│   ├── gender_prompts.py
│   ├── male_prompts_holdout.py
│   ├── programming_prompts.py
│   └── python_prompts_holdout.py
└── results/               # Pre-computed .pkl files + plots (optional, ~few MB)
```

## What to Port

### `generate_figures.py` (from `scripts/perturbation/generate_lesswrong_figures.py`)
- Move to repo root as the main entry point
- Fix imports: `src.model`, `src.sae`, `src.sweep`, `src.data`
- Extract sweep functions (`sweep_one_spherical`, `sweep_one_additive`, `_measure_response`) into `src/sweep.py`
- Extract SAE loading into `src/sae.py`
- Extract `extract_context_and_activation` into `src/data.py` (or `src/sweep.py`)
- Keep: figure configs, plotting, main CLI
- The `extract_context_and_activation` function from `sweep_random_points.py` is imported — extract the relevant parts

### `src/model.py` (trim heavily)
**Keep:**
- `MODEL_ALIASES` — only entries for gemma-2-9b, llama-3.1-8b, qwen3-1.7b
- `load_model()` — simplified, no chess/CSP/circuitgpt/Gemma3-multimodal paths
- `_get_blocks()`, `_get_final_norm()`, `_get_lm_head()` — only gpt2 + llama branches
- `_is_rope_model()`, `_get_position_embeddings()`, `_run_block()` — only gpt2 + llama
- `forward_from_layer()`, `forward_from_layer_to_layer()` — the two used by sweeps

**Remove:**
- All chess/CSP/circuitgpt/Othello/TinyStories/OLMo/Pythia/Deepseek aliases and code paths
- `GEMMA3_MULTIMODAL` and the multimodal extraction logic
- `forward_from_layer1()`, `get_layer0_output()`, `get_logits_at_point()` (legacy)
- `forward_from_layer_allpos()`, `extract_activations_allpos()` (not used in figures)
- `forward_from_layer_to_final()` (not used in figures)
- `extract_activations()` (replaced by inline `extract_activations_holdout` in the figure script)

### `src/sae.py` (new, extracted from existing loaders)
- `load_gemma_scope_sae()` — from `sweep_lesswrong_replication.py` (uses sae_lens)
- `load_qwen_sae()` — from `sweep_lesswrong_replication.py` (manual HF download + SimpleSAE wrapper)
- `load_llama_scope_sae()` — from `sweep_lesswrong_replication.py` (safetensors + LlamaScopeSAE wrapper)
- `load_sae()` — dispatcher (currently in generate_figures.py)
- Each SAE loader class (SimpleSAE, LlamaScopeSAE) included inline

### `src/data.py` (minimal)
- `load_fineweb_fixed_length()` — from current `src/data.py`
- `extract_context_and_activation()` — from `sweep_random_points.py`, simplified (remove circuitgpt path)
- Remove: `generate_chess_games`, `generate_random_token_sequences`, `load_python_code_fixed_length`, `sample_random_activations`, `sample_random_activations_v2`, `load_fineweb_samples`

### `data/` prompt files (copy as-is)
All 6 files listed above, unchanged.

### `setup_gpu.sh` (adapt from `setup_runpod.sh`)
- Generalize: don't assume RunPod paths, use `./venv` as default
- Keep torch cu124 install + requirements
- Optional HF login

## What NOT to Include
- `src/` files not needed: anything related to steering, MELBO, boundary, jacobian, anisotropy, etc.
- All `scripts/` subdirectories (melbo, directions, steering, boundary, etc.)
- `notebooks/`
- `writeup/`
- `Plans/`
- CLAUDE.md (project-specific)
- RunPod CLI stuff

## README Content
- Title + 1-paragraph abstract
- Link to LessWrong post
- "How to reproduce" section: install deps, run `python generate_figures.py --sweep --plot`
- Brief description of what each figure shows
- Citation / BibTeX if applicable
- License (MIT or similar — Francisco to decide)

## Pre-computed Results
- Optional: include the 9 `curves.pkl` files (~few MB each) so people can `--plot` without GPU
- Include the PNGs in the repo for quick browsing
- Or just link to them / leave them out and let people regenerate

## Steps to Execute
1. Create the repo locally (or on GitHub)
2. Copy + trim `src/model.py`
3. Extract `src/sae.py` from existing loaders
4. Extract `src/sweep.py` from `generate_lesswrong_figures.py`
5. Trim `src/data.py`
6. Copy `data/` prompt files
7. Adapt `generate_figures.py` with new imports
8. Write `requirements.txt`
9. Adapt `setup_gpu.sh`
10. Write `README.md`
11. Test: `--plot` mode locally with existing pkl files
12. Test: full `--sweep --plot` on GPU
