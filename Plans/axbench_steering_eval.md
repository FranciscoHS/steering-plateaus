# AxBench steering eval (50-concept subset): dictionary vs SAE

Follow-up to `Plans/axbench_detection_eval.md`. Evaluate steering (AxBench
protocol) for our dictionary vs the SAE, on a 50-concept random subset
(seed 0) of concept500 2b/l20. Contrastive excluded for now to cut cost
(Francisco: ~20x cheaper than the full 3-method 500-concept eval).

## Directions
- Per concept: the detection-selected directions (`best_dict_idx`,
  `best_sae_idx` from `results.csv`) — same contenders as detection.
- Sign fix: detection used |cos|, so selected directions have arbitrary sign.
  Steer with sign(d · d_contrastive) · d, using the concept's contrastive
  vector (natural positive sign = toward concept).

## Generation (on fran9 pod)
- Qwen3-1.7B in chat mode (Qwen/Qwen3-1.7B is the post-trained hybrid model;
  enable_thinking=False), so the AxBench instruction-following judge applies.
- Steering: h += factor · d at resid post layer 14, all token positions,
  during the whole generation (AxBench addition intervention).
- 10 fixed generic instructions (sampled seed 1 from the shared train
  negative pool, genre text), same for all concepts.
- Factor sweep: calibrate first with a mini run (3 concepts × wide factor
  range × 3 instructions), inspect samples, then pick ~5 factors for the
  full run.
- Sampling: temperature 0.7, top_p 0.8 (Qwen3 non-thinking recommended),
  max_new_tokens 128, 1 sample per (concept, method, factor, instruction).
- Full run: 50 concepts × 2 methods × ~5 factors × 10 instructions = 5,000
  generations → `results/axbench_steering/.../generations.jsonl`.

## Judging (local, Anthropic API)
- Haiku 4.5 via the Batches API (50% discount; ~15k calls ≈ $15).
- AxBench's three 0-2 judge prompts per generation: concept relevance,
  instruction relevance, fluency.
- Per-generation score = harmonic mean, 0 if any rating is 0 (AxBench rule).

## Aggregation
- Per concept × method: best factor by mean score over the 10 instructions.
- Report mean (and median) over the 50 concepts per method.
