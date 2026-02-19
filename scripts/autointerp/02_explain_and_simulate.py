#!/usr/bin/env python3
"""
Auto-interpretability pipeline: explain MELBO features and simulate activations.

For each feature direction:
1. Show top-30 activating prompts to Claude → get description
2. Show 100 held-out prompts to Claude with the description → get 0-10 predictions

Uses async Anthropic client with concurrency limit.

Usage:
    python scripts/autointerp/02_explain_and_simulate.py
    python scripts/autointerp/02_explain_and_simulate.py --features 0,1,2  # test on 3 features
    python scripts/autointerp/02_explain_and_simulate.py --concurrency 3   # slower but safer
"""

import argparse
import asyncio
import json
import os
import pickle
import re
import sys
import time

import anthropic
import numpy as np


EXPLAIN_PROMPT = """\
You are helping to interpret a feature direction found in the residual stream \
of a language model (Qwen3 0.6B, layer 14).

Below are the {n_examples} text fragments where this feature activates most strongly, \
ranked from strongest to weakest. Each is a 5-token prompt; activation is \
the absolute projection of the last-token hidden state onto the feature direction.

{examples}

Based on these examples, provide a concise description (1-3 sentences) of what \
this feature responds to. Be specific. If no clear pattern emerges, say so."""

SIMULATE_PROMPT = """\
You are helping to evaluate a feature direction in a language model (Qwen3 0.6B, \
layer 14).

This feature has been described as: "{description}"

For each prompt below, predict how strongly this feature activates on a scale \
from 0 (not at all) to 10 (very strongly, comparable to top activating examples).

Respond with one prediction per line in the format "N: score" where N is the \
prompt number and score is an integer 0-10. Do not include any other text.

{prompts}"""


def format_examples(prompts, activations, top_indices):
    """Format top-activating prompts for the explain prompt."""
    lines = []
    for rank, idx in enumerate(top_indices):
        prompt = prompts[idx]
        act = activations[idx]
        lines.append(f'{rank+1}. "{prompt}" (activation: {act:.2f})')
    return '\n'.join(lines)


def format_simulate_prompts(prompts, indices):
    """Format prompts for the simulate prompt."""
    lines = []
    for rank, idx in enumerate(indices):
        lines.append(f'{rank+1}. "{prompts[idx]}"')
    return '\n'.join(lines)


def select_simulate_indices(n_total, top_indices, abs_projections_for_feature, rng):
    """Select 100 prompts for simulation: 50 random + 50 evenly spaced by activation.

    Excludes the top_indices used for explanation.

    Args:
        n_total: total number of prompts
        top_indices: set of indices used for explanation
        abs_projections_for_feature: [n_total] array of absolute projections
        rng: numpy random generator

    Returns:
        list of 100 indices
    """
    # Pool: everything not in explain set
    pool = np.array([i for i in range(n_total) if i not in top_indices])

    # 50 random
    random_indices = rng.choice(pool, size=min(50, len(pool)), replace=False)

    # 50 evenly spaced across the activation range
    pool_acts = abs_projections_for_feature[pool]
    max_act = pool_acts.max()
    # Create bins from 0 to max, pick one prompt from each bin
    n_bins = 50
    bin_edges = np.linspace(0, max_act * 1.001, n_bins + 1)
    spaced_indices = []
    for b in range(n_bins):
        in_bin = pool[(pool_acts >= bin_edges[b]) & (pool_acts < bin_edges[b+1])]
        if len(in_bin) > 0:
            spaced_indices.append(rng.choice(in_bin))
    spaced_indices = np.array(spaced_indices)

    # Combine, deduplicate
    combined = np.unique(np.concatenate([random_indices, spaced_indices]))

    # If we have more than 100, subsample; if fewer, that's fine
    if len(combined) > 100:
        combined = rng.choice(combined, size=100, replace=False)

    return combined.tolist()


def parse_predictions(text, n_expected):
    """Parse "N: score" predictions from Claude's response.

    Returns:
        list of (prompt_number, score) tuples — prompt_number is 1-indexed
    """
    predictions = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Match patterns like "1: 7", "1: 7/10", "1. 7", etc.
        m = re.match(r'(\d+)[.:]\s*(\d+)', line)
        if m:
            num = int(m.group(1))
            score = int(m.group(2))
            score = max(0, min(10, score))  # clamp to 0-10
            predictions.append((num, score))
    return predictions


async def explain_feature(client, prompts, abs_projections_for_feature, feature_idx,
                          n_top=30, semaphore=None):
    """Call Claude to explain a single feature."""
    top_indices = np.argsort(abs_projections_for_feature)[-n_top:][::-1]
    examples_text = format_examples(prompts, abs_projections_for_feature, top_indices)

    prompt = EXPLAIN_PROMPT.format(n_examples=n_top, examples=examples_text)

    for attempt in range(3):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=300,
                        messages=[{"role": "user", "content": prompt}],
                    )
            else:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
            description = response.content[0].text.strip()
            return description, top_indices.tolist()
        except anthropic.APIError as e:
            wait = 2 ** (2 * attempt + 1)  # 2s, 8s, 32s
            print(f"  API error on explain (feature {feature_idx}, attempt {attempt+1}): {e}")
            await asyncio.sleep(wait)

    return None, top_indices.tolist()


async def simulate_feature(client, prompts, description, simulate_indices, feature_idx,
                           semaphore=None):
    """Call Claude to predict activations for held-out prompts."""
    prompts_text = format_simulate_prompts(prompts, simulate_indices)

    prompt = SIMULATE_PROMPT.format(
        description=description,
        prompts=prompts_text,
    )

    for attempt in range(3):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": prompt}],
                    )
            else:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
            text = response.content[0].text
            predictions = parse_predictions(text, len(simulate_indices))

            if len(predictions) < len(simulate_indices) * 0.5:
                # Too few parsed — retry with clarification
                if attempt == 0:
                    print(f"  Only parsed {len(predictions)}/{len(simulate_indices)} "
                          f"predictions for feature {feature_idx}, retrying...")
                    continue
                else:
                    print(f"  Warning: only {len(predictions)} predictions for feature {feature_idx}")

            return predictions, text
        except anthropic.APIError as e:
            wait = 2 ** (2 * attempt + 1)
            print(f"  API error on simulate (feature {feature_idx}, attempt {attempt+1}): {e}")
            await asyncio.sleep(wait)

    return None, None


async def process_feature(client, prompts, abs_projections, feature_idx, checkpoint_dir,
                          semaphore, rng):
    """Full explain + simulate pipeline for one feature."""
    checkpoint_path = os.path.join(checkpoint_dir, f'feature_{feature_idx:03d}.json')

    # Skip if already done
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        if data.get('status') == 'completed':
            return data

    abs_proj = abs_projections[:, feature_idx]

    # Explain
    description, top_indices = await explain_feature(
        client, prompts, abs_proj, feature_idx, n_top=30, semaphore=semaphore)

    if description is None:
        result = {
            'feature_idx': feature_idx,
            'status': 'failed',
            'error': 'explain step failed after 3 retries',
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # Select simulate prompts
    top_set = set(top_indices)
    simulate_indices = select_simulate_indices(
        len(prompts), top_set, abs_proj, rng)

    # Simulate
    predictions, raw_response = await simulate_feature(
        client, prompts, description, simulate_indices, feature_idx, semaphore=semaphore)

    if predictions is None:
        result = {
            'feature_idx': feature_idx,
            'status': 'failed',
            'description': description,
            'error': 'simulate step failed after 3 retries',
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # Build prediction map: prompt_index → predicted_score
    pred_map = {}
    for prompt_num, score in predictions:
        if 1 <= prompt_num <= len(simulate_indices):
            actual_idx = simulate_indices[prompt_num - 1]
            pred_map[str(actual_idx)] = score

    # Actual activations for simulate prompts
    actual_map = {}
    for idx in simulate_indices:
        actual_map[str(idx)] = float(abs_proj[idx])

    result = {
        'feature_idx': feature_idx,
        'status': 'completed',
        'description': description,
        'top_indices': top_indices,
        'simulate_indices': simulate_indices,
        'predictions': pred_map,       # {prompt_idx_str: predicted_score}
        'actuals': actual_map,          # {prompt_idx_str: actual_abs_projection}
        'n_predictions_parsed': len(predictions),
        'raw_simulate_response': raw_response,
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


async def main_async(args):
    """Async main — runs all features with concurrency limit."""
    # Load projections
    print(f"Loading projections from {args.projections_path}")
    with open(args.projections_path, 'rb') as f:
        data = pickle.load(f)

    prompts = data['prompts_text']
    abs_projections = data['abs_projections']  # [n_prompts, n_dirs]
    n_prompts, n_dirs = abs_projections.shape
    print(f"  {n_prompts} prompts, {n_dirs} directions")

    # Determine which features to process
    if args.features:
        feature_indices = [int(x) for x in args.features.split(',')]
    else:
        feature_indices = list(range(n_dirs))
    print(f"  Processing {len(feature_indices)} features")

    # Setup
    checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    client = anthropic.AsyncAnthropic()  # uses ANTHROPIC_API_KEY env var
    semaphore = asyncio.Semaphore(args.concurrency)
    rng = np.random.default_rng(args.seed)

    # Count already completed
    already_done = 0
    for fi in feature_indices:
        cp = os.path.join(checkpoint_dir, f'feature_{fi:03d}.json')
        if os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            if d.get('status') == 'completed':
                already_done += 1

    print(f"  Already completed: {already_done}/{len(feature_indices)}")
    if already_done == len(feature_indices):
        print("All features already processed. Exiting.")
        return

    # Process features
    t0 = time.time()
    tasks = []
    for fi in feature_indices:
        task = process_feature(
            client, prompts, abs_projections, fi, checkpoint_dir, semaphore, rng)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Summary
    elapsed = time.time() - t0
    completed = sum(1 for r in results if r.get('status') == 'completed')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    skipped = already_done  # already counted above

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Completed: {completed} (including {skipped} from previous run)")
    print(f"  Failed: {failed}")

    # Show a few descriptions
    print(f"\nSample descriptions:")
    for r in results[:5]:
        if r.get('description'):
            desc = r['description'][:100] + ('...' if len(r.get('description', '')) > 100 else '')
            print(f"  Feature {r['feature_idx']}: {desc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projections_path', default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--features', default=None,
                        help='Comma-separated feature indices (default: all)')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Max concurrent API calls')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.projections_path is None:
        args.projections_path = 'results/autointerp/qwen3-0.6b_layer14_k100/projections.pkl'
    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100'

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
