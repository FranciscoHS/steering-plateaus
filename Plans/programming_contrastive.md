# Multi-Contrastive Steering for Programming Languages

## Context
We want to reproduce the multi-contrastive feature isolation method (from `multi_contrastive_projection.pdf`) for programming languages instead of natural languages. The goal is to isolate "programming language features" (e.g., a "Python direction", a "Rust direction") from the residual stream of Qwen3-0.6B using Section 8's all-pairwise contrasts approach, which gives O(εC) error vs O(C/√k) from star topology.

## Languages
Python, JavaScript, Java, C, Go, Rust, Ruby, TypeScript (8 languages — syntactically distinct function signature styles)

## Step 1: Create `data/programming_prompts.py`
~30 matched function signature sets across 8 programming languages. Each set expresses the same concept (fibonacci, factorial, is_prime, reverse_string, etc.) as a function definition opening.

Structure mirrors `data/multilingual_prompts.py`:
```python
LANGUAGES = ['python', 'javascript', 'java', 'c', 'go', 'rust', 'ruby', 'typescript']

PROMPT_SETS = [
    {
        'python': "def fibonacci(n):",
        'javascript': "function fibonacci(n) {",
        'java': "public static int fibonacci(int n) {",
        'c': "int fibonacci(int n) {",
        'go': "func fibonacci(n int) int {",
        'rust': "fn fibonacci(n: u32) -> u32 {",
        'ruby': "def fibonacci(n)",
        'typescript': "function fibonacci(n: number): number {",
    },
    # ... ~30 total
]
```

Concepts to cover (~30):
- Math: fibonacci, factorial, is_prime, gcd, power, absolute_value, is_even, to_binary
- Strings: reverse_string, is_palindrome, count_words, capitalize_words, string_length
- Arrays/lists: sort_list, max_value, min_value, binary_search, contains, remove_duplicates, find_index, sum_list, average, flatten
- Misc: read_file, validate_email, calculate_distance, matrix_multiply, merge_sort, depth_first_search

## Step 2: Test Qwen3-0.6B autocompletion
Script: `scripts/directions/test_programming_autocompletion.py`
- Feed each prompt to Qwen3-0.6B, generate 30-50 tokens
- Print completions grouped by language for manual inspection
- Heuristic checks: does the completion contain syntax markers of the right language?
- Report per-language coherence rates
- This tells us which languages to keep and which to drop

Reuses: `load_model()` from `src/model.py`

## Step 3: Implement all-pairwise contrastive procedure (Section 8)
Script: `scripts/directions/run_programming_contrastive.py`

### Phase A: Extract activations
For each language pair (i,j), extract final-token activations for matched prompts:
- k=8 languages → 28 pairs
- ~30 prompts per pair
- Reuse `get_acts()` pattern from `run_multi_contrastive.py`

### Phase B: Compute pairwise mean differences
For each pair (i,j): v̄_ij = mean(a_i - a_j) across matched prompts
→ 28 mean difference vectors

### Phase C: Recover feature vectors (Section 8 formula)
All-pairwise least-squares estimator:
  f̂_i = (1/k) Σ_{j≠i} v̄_ij  for each language i

Also compute star-topology estimators (Section 4) for comparison:
  For target language t: f̂_t^star = (1/(k-1)) Σ_{j≠t} v̄_tj

### Phase D: SVD estimator
Stack all individual (not pre-averaged) difference vectors:
  V = [all (a_i^(m) - a_j^(m)) for all pairs (i,j) and all prompts m]
Leading right singular vector ≈ dominant shared direction

### Phase E: Diagnostics
- σ₂/σ₁ ratio (should be small if single direction)
- Left singular vector u₁^(left) weights (should be ~uniform if assumptions hold)
- Cosine similarity matrix between all recovered features
- Sanity check: f̂_i should sum to ~0 (translation ambiguity)

### Phase F: Perturbation sweeps
- Sweep each f̂_i direction on FineWeb activations
- Compare with random directions
- Reuse `sweep_single_direction()` from `run_multi_contrastive.py`

### Phase G: Steering validation
- Steer FineWeb activations along f̂_python, generate text
- Check if output looks like Python code
- Test generalization: steer toward language not heavily used in training prompts

## Files to create
- `data/programming_prompts.py` — prompt data
- `scripts/directions/test_programming_autocompletion.py` — Step 2
- `scripts/directions/run_programming_contrastive.py` — Steps 3-7

## Files to reuse
- `src/model.py` — `load_model()`, `forward_from_layer_allpos()`, internal helpers
- `scripts/directions/run_multi_contrastive.py` — `get_acts()`, `extract_context_and_activation()`, `build_perturbed_hidden()`, `sweep_single_direction()`, `forward_to_hidden_and_logits()`
- `data/multilingual_prompts.py` — structural template

## Verification
1. Run autocompletion test locally (or on RunPod for GPU)
2. Inspect completions — do they stay in the right language?
3. Run contrastive procedure, check:
   - σ₂/σ₁ < 0.5 (ideally)
   - Features sum to ~0
   - Cosine similarities between same-family languages (e.g., JS/TS) flagged
   - Perturbation sweeps show programming-language directions have more effect than random
4. Steering test: does steering toward Python actually produce Python-like output?

## Execution order
1. Create prompts data file (can do locally, no GPU)
2. Create + run autocompletion test (needs GPU — RunPod)
3. Filter languages based on results, adjust prompts if needed
4. Create + run contrastive procedure (needs GPU — RunPod)
