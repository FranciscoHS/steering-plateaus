# Mandarin Haskell Composition Experiment on Gemma-2-9B

## Context
We want to test whether two independently-extracted steering directions (Mandarin and Haskell) compose additively on Gemma-2-9B. The experiment: feed the model Python code, steer simultaneously toward Mandarin and Haskell, and check whether the output becomes Mandarin Haskell.

## Plan

### Step 1: Write `scripts/steering/run_mandarin_haskell.py`
A single self-contained script (run on RunPod with a GPU) that:

#### 1a. Load Gemma-2-9B
- Use `src/model.load_model('gemma-2-9b', dtype=torch.bfloat16)`

#### 1b. Extract contrastive directions at layer 2
- **Mandarin direction:** Use `data/bilingual_prompts.py` (EN-ZH pairs). Compute diff-of-means at layer 2, negate to point EN->ZH. Same logic as `generate_lesswrong_figures.py:compute_contrastive_direction()`.
- **Haskell direction:** Use `data/programming_prompts.py` (Python-Haskell pairs). Compute diff-of-means at layer 2, negate to point Python->Haskell.
- Report cosine similarity between the two directions (hope: near-orthogonal, like the Russian x C case which was cos=0.014).

#### 1c. Perturbation sweep to validate directions
- Run a quick perturbation sweep along each direction separately at layer 2 (measure at layer n-2) to verify they produce plateaus (i.e., the model discriminates Python/Haskell and EN/ZH at this layer).
- Reuse `src/model.forward_from_layer` / `forward_from_layer_to_layer` and the spherical sweep pattern from `generate_lesswrong_figures.py`.
- Save sweep curves to results file.

#### 1d. Steered generation: 2x2 grid
- Input prompts: Python function signatures from `data/python_prompts_holdout.py` (or a subset, e.g. 10).
- Four conditions, all using spherical steering at layer 2:
  1. **Baseline** (angle=0): no steering
  2. **+Mandarin only** (angle=30): steer toward Mandarin direction
  3. **+Haskell only** (angle=30): steer toward Haskell direction
  4. **+Mandarin + Haskell** (angle=30 each): steer toward both simultaneously
- For combined steering: apply both hooks sequentially (one per direction) on the same layer. First hook rotates toward Mandarin, second hook rotates the result toward Haskell. This preserves independent angle control for each direction.
- Generate ~60 tokens per prompt, temperature=0.7.
- Save all generations to a text/json results file.

### Key files to reuse
- `src/model.py`: `load_model`, `_get_blocks`, `forward_from_layer`, `forward_from_layer_to_layer`
- `data/bilingual_prompts.py`: `ENGLISH_PROMPTS`, `CHINESE_PROMPTS`
- `data/programming_prompts.py`: `PYTHON_PROMPTS`, `HASKELL_PROMPTS`
- `data/python_prompts_holdout.py`: `PYTHON_PROMPTS_HOLDOUT`
- `generate_lesswrong_figures.py`: patterns for `compute_contrastive_direction`, `sweep_one_spherical`
- `validate_programming_steering.py`: pattern for `steer_and_generate` with HF hooks
- `plot_crossgroup_composition.py`: inspiration for 2x2 visualization (if we want a figure later)

### Output
- `results/mandarin_haskell/gemma-2-9b/directions.pkl` — the two direction vectors + cosine similarity
- `results/mandarin_haskell/gemma-2-9b/sweep_curves.pkl` — perturbation sweep data for both directions
- `results/mandarin_haskell/gemma-2-9b/generations.json` — all steered text outputs (4 conditions x N prompts)

### Verification
1. Check that the perturbation sweeps show plateaus (indicating the directions are meaningful at layer 2)
2. Read `generations.json` to see if:
   - Baseline produces Python completions
   - +Mandarin produces Mandarin text
   - +Haskell produces Haskell-like code
   - +Both produces Mandarin Haskell
