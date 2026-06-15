# AxBench concept-detection eval: dictionary vs SAE vs contrastive

Goal: evaluate our plateau-breaking dictionary
(`results/dictionary/qwen3-1.7b_L14_scaffoldcontrast_align1.0_k100_40k/dictionary.pt`,
network-volume copy, ~12.7k/40k atoms at snapshot time — build still in progress)
on AxBench-style concept detection, against (a) the TopK SAE
(`qwen3-1.7b_L14_K8000/sae_L0_50_layer14.pt`, 32768 latents, decoder rows as
directions) and (b) per-concept contrastive (difference-of-means) vectors.

## Data (pyvene/axbench-concept500, subset 2b/l20)
- 500 concepts (GemmaScope labels; text is model-agnostic — we compute Qwen3-1.7B L14 activations on it).
- Train split: 72 positives per concept + 216 shared negatives (`concept_id == -1`, 72 per genre).
  Per-concept training set = 72 positives + the 72 genre-matched shared negatives → 144 examples.
- Test split: per concept ≈ 36 positives + 36 negatives + ~2 hard negatives.
  AUROC labels: positive = 1, everything else = 0 (AxBench convention).

## Scoring (identical for all three methods)
- Text scored: the `output` field only (the response contains the concept; Qwen3-1.7B is a base model so no chat template). Truncate at 512 tokens.
- Per token: s = |cos(h − μ, d)| with h = resid post layer 14, d unit-norm.
- μ = corpus mean over FineWeb token activations (2000 seqs × 64 tokens, skip pos 0; same loader/cache as the dictionary build).
- Skip position 0 of each example (attention sink, repo convention).
- Pool to one score per example with **max over tokens** (AxBench convention).

## Procedure
1. Snapshot the dictionary on the pod (`cp` to the output dir) so results refer to a frozen n_done.
2. Train pass: forward all 36,216 unique train examples; store max-pooled scores vs all dict atoms (~12.7k) and all 32,768 SAE decoder dirs (fp16, CPU). Also accumulate per-example mean-pooled activations for contrastive.
3. Selection: per concept, vectorized rank-based AUROC of every direction on its 144-example train set; pick the argmax direction for dict and for SAE separately.
4. Contrastive: d_c = mean(example-mean acts of 72 positives) − mean(example-mean acts of 72 genre-matched negatives), unit-normalized.
5. Test pass: forward the 38,037 test examples; per example score only its concept's 3 selected directions; per-concept test AUROC for each method.

## Outputs (results/axbench_detection/qwen3-1.7b_L14_2b-l20/)
- `dictionary_snapshot.pt` — frozen copy used for the eval
- `selection.pkl` — per concept: best dict/SAE index + per-dir train AUROC matrices
- `contrastive_dirs.pt` — [500, 2048]
- `test_scores.pkl` — raw per-example test scores for the 3 methods
- `results.csv` — per concept: concept name, genre, test AUROC for dict / SAE / contrastive
- `summary.txt` — mean/median test AUROC per method

## Execution
- Run on fran9 pod (RTX 4090, idle). Model + dictionary + SAE already on the volume.
- Checkpoint intermediate score matrices after the train pass.
