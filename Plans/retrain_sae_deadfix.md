# Retrain L10 SAEs with standard dead-latent fixes

**Goal:** redo the Qwen3-1.7B layer-10 SAE trainings (baseline + high-p popt) with
the standard dead-latent mitigations from Gao et al. 2024 ("Scaling and evaluating
sparse autoencoders"), so the dictionaries actually use their 32768 latents.
Current runs have ~75% latents that never fire (measured over 200k cached rows;
see `results/sae_alignment_popt_vs_baseline/`), i.e. effective expansion ~4x
instead of the nominal 16x.

## Why the current runs ended up 75% dead
- The trainer's "fired" criterion is index-membership in the top-32 (`top_idx.unique()`),
  with no value>0 check. `topk` runs on post-ReLU pre-acts, so zero-valued latents
  occupy top-k slots and every latent "fires" within ~1000 steps → the log shows
  `dead=0` throughout → the one resampling event (step 25000) reinitialized nothing.
- No aux-k loss, no tied encoder init.

## Fixes to implement (all standard, from Gao et al. 2024)
1. **Fix the fired criterion:** a latent fired iff it is in the top-k AND its
   post-ReLU value > 0. This alone makes the existing resampler functional, and is
   needed for correct dead tracking/logging regardless.
2. **Tied init:** initialize `encoder.weight = W_dec` (they're both stored
   [d_sae, d_in], so numerically `W_enc := W_dec` with W_dec rows unit-norm),
   `encoder.bias = 0`. Also init `b_dec` to the mean of a sample of cached
   activations (cheap stand-in for the geometric median).
3. **AuxK loss:** auxiliary reconstruction of the main-loss residual using the
   top-`k_aux` *dead* latents:
   - `e = batch - recon` (detached recon path is fine; standard is to use the
     main residual), `recon_aux = decode(topk_aux over dead latents' pre-acts)`,
     `L_aux = ||e - recon_aux||^2 / ||e - e.mean||^2` (normalized MSE),
     total loss `L = L_main + alpha * L_aux` with **alpha = 1/32**, **k_aux = 512**.
   - "Dead" for aux purposes: hasn't fired (criterion in fix 1) in the last
     `dead_steps_threshold` steps; use ~2000 steps (~8M samples) with a warmup
     (no latent counts as dead before step 2000).
   - Guard: if fewer than k_aux dead latents, use however many there are; if none,
     skip the term.
4. Keep the existing resampler available but it should rarely trigger once aux-k
   works; don't fight over it.

## Where the code lives
- `scripts/dictionary_learning/train_sae.py` — base trainer (TopKSAE from
  `src/autoencoders.py`), owns `cache_activations`, `resample_dead_features`,
  the training loop.
- `scripts/dictionary_learning/train_sae_popt.py` — same loop + differentiable
  p-term (`ploss = mean R_mix/(R_d+R_c)` at inject L10 / measure L26), imports
  helpers from the base via `train_sae_pcull.py`.
- Implement the fixes in the base loop behind flags (`--auxk`, `--tied_init`,
  defaulting ON for these runs), then port the same few lines into
  `train_sae_popt.py`'s loop (it has its own copy of the loop). Do NOT overwrite
  the old scripts' behavior for existing result dirs — new flags, new output dirs.
- Log per-1000-steps: mse, val_cos, **true dead count** (fix-1 criterion over the
  logging window), aux loss. Save `fired counts` array in the final checkpoint so
  dead fractions can be checked without re-encoding.

## Runs (3 total, same base config as before)
Shared config: Qwen3-1.7B, layer 10 resid_post, d_sae=32768, k=32, batch 4096,
50k steps, lr/schedule as before, cache reuse:
`/workspace/steering-plateaus/results/trained_autoencoder/l10_cache_10M/cached_activations.pt`
(memmap + `.meta.json` sidecar, loads via `--reuse_cache --cache_path ...`).

1. **Baseline v2** (plain MSE + fixes) → `results/trained_sae_v2/l10_auxk_baseline`
2. **popt λ=1 + fixes** → `results/trained_sae_popt_v2/l10_lam1`
3. **popt λ=0.1 + fixes** → `results/trained_sae_popt_v2/l10_lam0p1`

**Seeds:** give each run a DIFFERENT init seed (e.g. 0/1/2) and record it. The
previous runs shared init, which contaminated the decoder-alignment analysis
(dead latents matched their identical init twins at cos≈0.83). If Francisco
prefers shared init for controlled comparison, flag before running — default to
distinct seeds.

## Compute
- One RTX 4090 pod is enough; popt runs took ~1.6 h each before (112 s / 1000
  steps), aux-k adds little. Run the three sequentially under `nohup python -u`,
  logs + checkpoints on `/workspace` (pod-local venv, HF_HOME=/workspace/hf_cache).
- check with francisco actually might be better to spin up 3 pods and run in parallel!
- Note the p-term needs the model forward (inject L10 → measure L26), so popt
  runs need the model loaded, as before.

## After training (sanity evals, before any new analysis)
- Report per run: val_cos, MSE, **dead fraction over 200k cache rows** (target:
  well under 20%; Gao et al. get ~7%), and median full-p via
  `scripts/dictionary_learning/eval_sae_full_p.py` (320 random pairs) to confirm
  the popt runs still reach p≈2.8 and baseline stays ≈2.1 with the fixes on.
- Do NOT delete or overwrite the old runs (`results/trained_sae/l10_doubled_k32_2M`,
  `results/trained_sae_popt/l10_lam*`).

## Success criteria
- Dead fraction (200k-row criterion) drops from ~75% to <~20% on all three runs
  at no material val_cos cost vs the old runs (0.919–0.920).
- popt λ=1 median p stays ≈2.8 (or we learn that high-p was partly living off the
  dead capacity — also a result worth knowing).
