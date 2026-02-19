#!/usr/bin/env python3
"""
Long-context auto-interpretability: explain and simulate with per-token predictions.

For each MELBO feature direction:
1. Show top-10 passages with activating tokens highlighted → get description
2. Show 5 held-out passages → predict per-token activation (comma-separated 0-10)

Usage:
    python scripts/autointerp/02_explain_and_simulate_longctx.py
    python scripts/autointerp/02_explain_and_simulate_longctx.py --features 0,1,2
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
You are interpreting a feature direction in the residual stream of Qwen3 0.6B \
(layer 14).

Below are the {n_examples} passages where this feature activates most strongly. \
Tokens with high activation are marked with <<double brackets>>. The peak \
activation value is shown for each passage.

{examples}

Based on these examples, describe what this feature responds to (1-3 sentences). \
Focus on the highlighted tokens and their role in context. Be specific. If no \
clear pattern emerges, say so."""

SIMULATE_PROMPT = """\
You are evaluating a feature direction in Qwen3 0.6B (layer 14).

This feature has been described as: "{description}"

For each passage below, predict how strongly this feature activates on each \
token on a scale from 0 (not at all) to 10 (very strongly, comparable to top \
activating examples). Respond with comma-separated integers, one line per \
passage in the format "N: score1,score2,score3,...".

{passages}"""


def highlight_passage(token_texts, abs_projections_for_passage, threshold_pct=90):
    """Format a passage with high-activation tokens highlighted in <<brackets>>.

    Args:
        token_texts: list of decoded token strings for this passage
        abs_projections_for_passage: [seq_len] array of |projections| for one feature
        threshold_pct: percentile threshold for highlighting

    Returns:
        highlighted text string
    """
    threshold = np.percentile(abs_projections_for_passage, threshold_pct)
    parts = []
    for tok_text, act in zip(token_texts, abs_projections_for_passage):
        if act >= threshold and threshold > 0:
            parts.append(f'<<{tok_text}>>')
        else:
            parts.append(tok_text)
    return ''.join(parts)


def format_explain_examples(token_texts_list, abs_proj, feature_idx, top_passage_indices,
                            n_examples=10):
    """Format top-activating passages with highlighted tokens for the explain prompt."""
    lines = []
    for rank, pi in enumerate(top_passage_indices[:n_examples]):
        passage_proj = abs_proj[pi, :, feature_idx]
        peak = passage_proj.max()
        highlighted = highlight_passage(token_texts_list[pi], passage_proj)
        lines.append(f'{rank+1}. (peak: {peak:.2f}) "{highlighted}"')
    return '\n\n'.join(lines)


def format_simulate_passages(token_texts_list, passage_indices):
    """Format passages for the simulate prompt with |delimited| tokens."""
    blocks = []
    for rank, pi in enumerate(passage_indices):
        full_text = ''.join(token_texts_list[pi])
        token_list = '|'.join(token_texts_list[pi]) + '|'
        blocks.append(
            f'Passage {rank+1}: "{full_text}"\n'
            f'Tokens: |{token_list}'
        )
    return '\n\n'.join(blocks)


def parse_comma_predictions(text, expected_counts):
    """Parse comma-separated predictions from Claude's response.

    Args:
        text: raw response text
        expected_counts: list of expected token counts per passage

    Returns:
        list of lists of (int) predictions, one per passage
    """
    predictions = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Match "N: 1,2,3,..." or just "1,2,3,..."
        m = re.match(r'(?:\d+:\s*)?([\d,\s]+)', line)
        if m:
            scores_str = m.group(1).strip()
            try:
                scores = [max(0, min(10, int(s.strip()))) for s in scores_str.split(',') if s.strip()]
                if scores:
                    predictions.append(scores)
            except ValueError:
                continue
    return predictions


def select_simulate_passages(n_total, explain_indices, max_proj_per_passage, rng, n_simulate=5):
    """Select passages stratified by max activation, excluding explain set.

    Args:
        n_total: total number of passages
        explain_indices: set of passage indices used for explanation
        max_proj_per_passage: [n_total] array of max |projection| per passage
        rng: numpy random generator
        n_simulate: number of passages to select

    Returns:
        list of passage indices
    """
    pool = np.array([i for i in range(n_total) if i not in explain_indices])
    pool_maxes = max_proj_per_passage[pool]

    # Stratify: evenly space across the range
    max_val = pool_maxes.max()
    min_val = pool_maxes.min()
    bin_edges = np.linspace(min_val, max_val * 1.001, n_simulate + 1)

    selected = []
    for b in range(n_simulate):
        in_bin = pool[(pool_maxes >= bin_edges[b]) & (pool_maxes < bin_edges[b+1])]
        if len(in_bin) > 0:
            selected.append(rng.choice(in_bin))

    # If some bins were empty, fill with random from pool
    if len(selected) < n_simulate:
        remaining = [i for i in pool if i not in selected]
        extra = rng.choice(remaining, size=min(n_simulate - len(selected), len(remaining)),
                          replace=False)
        selected.extend(extra.tolist())

    return selected[:n_simulate]


async def explain_feature(client, token_texts_list, abs_proj, feature_idx,
                          n_top=10, semaphore=None):
    """Call Claude to explain a single feature from highlighted passages."""
    # Rank passages by max token activation
    max_per_passage = abs_proj[:, :, feature_idx].max(axis=1)
    top_indices = np.argsort(max_per_passage)[-n_top:][::-1]

    examples_text = format_explain_examples(
        token_texts_list, abs_proj, feature_idx, top_indices, n_examples=n_top)

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
            wait = 2 ** (2 * attempt + 1)
            print(f"  API error on explain (feature {feature_idx}, attempt {attempt+1}): {e}")
            await asyncio.sleep(wait)

    return None, top_indices.tolist()


async def simulate_feature(client, token_texts_list, description, simulate_indices,
                           feature_idx, seq_len, semaphore=None):
    """Call Claude to predict per-token activations for held-out passages."""
    passages_text = format_simulate_passages(token_texts_list, simulate_indices)

    prompt = SIMULATE_PROMPT.format(
        description=description,
        passages=passages_text,
    )

    expected_counts = [seq_len] * len(simulate_indices)

    for attempt in range(3):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}],
                    )
            else:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}],
                )
            text = response.content[0].text
            predictions = parse_comma_predictions(text, expected_counts)

            if len(predictions) < len(simulate_indices):
                if attempt == 0:
                    print(f"  Only parsed {len(predictions)}/{len(simulate_indices)} "
                          f"passages for feature {feature_idx}, retrying...")
                    continue
                else:
                    print(f"  Warning: only {len(predictions)} passages for feature {feature_idx}")

            return predictions, text
        except anthropic.APIError as e:
            wait = 2 ** (2 * attempt + 1)
            print(f"  API error on simulate (feature {feature_idx}, attempt {attempt+1}): {e}")
            await asyncio.sleep(wait)

    return None, None


async def process_feature(client, token_texts_list, abs_proj, feature_idx,
                          seq_len, checkpoint_dir, semaphore, rng):
    """Full explain + simulate pipeline for one feature."""
    checkpoint_path = os.path.join(checkpoint_dir, f'feature_{feature_idx:03d}.json')

    # Skip if already done
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        if data.get('status') == 'completed':
            return data

    # Explain
    description, top_indices = await explain_feature(
        client, token_texts_list, abs_proj, feature_idx, n_top=10, semaphore=semaphore)

    if description is None:
        result = {
            'feature_idx': feature_idx,
            'status': 'failed',
            'error': 'explain step failed after 3 retries',
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # Select simulate passages (stratified by max activation)
    explain_set = set(top_indices)
    max_per_passage = abs_proj[:, :, feature_idx].max(axis=1)
    simulate_indices = select_simulate_passages(
        abs_proj.shape[0], explain_set, max_per_passage, rng, n_simulate=5)

    # Simulate
    predictions, raw_response = await simulate_feature(
        client, token_texts_list, description, simulate_indices,
        feature_idx, seq_len, semaphore=semaphore)

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

    # Build per-passage prediction/actual pairs
    # Truncate or pad predictions to match seq_len
    passage_results = []
    for p_rank, pi in enumerate(simulate_indices):
        actual_proj = abs_proj[pi, :, feature_idx].tolist()  # [seq_len]
        if p_rank < len(predictions):
            pred = predictions[p_rank]
            # Truncate if too long, pad with 0 if too short
            if len(pred) > seq_len:
                pred = pred[:seq_len]
            elif len(pred) < seq_len:
                pred = pred + [0] * (seq_len - len(pred))
        else:
            pred = [0] * seq_len  # missing passage prediction

        passage_results.append({
            'passage_idx': int(pi),
            'predicted': pred,
            'actual': actual_proj,
            'n_tokens_predicted': len(predictions[p_rank]) if p_rank < len(predictions) else 0,
        })

    result = {
        'feature_idx': feature_idx,
        'status': 'completed',
        'description': description,
        'top_indices': top_indices,
        'simulate_indices': [int(i) for i in simulate_indices],
        'passage_results': passage_results,
        'n_passages_parsed': len(predictions),
        'raw_simulate_response': raw_response,
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


async def main_async(args):
    """Async main — runs all features with concurrency limit."""
    print(f"Loading projections from {args.projections_path}")
    with open(args.projections_path, 'rb') as f:
        data = pickle.load(f)

    token_texts_list = data['token_texts']
    signed_proj = data['projections']       # [n_passages, seq_len, n_dirs]
    n_passages, seq_len, n_dirs = signed_proj.shape
    print(f"  {n_passages} passages, {seq_len} tokens, {n_dirs} directions")

    if args.center_by_position:
        # Mean-center projections per position to remove positional bias.
        # Since projection is linear: centering activations then projecting
        # is equivalent to centering projections directly.
        pos_mean = signed_proj.mean(axis=0, keepdims=True)  # [1, seq_len, n_dirs]
        signed_proj = signed_proj - pos_mean
        print(f"  Centered by position (removed per-position mean)")

    abs_proj = np.abs(signed_proj)

    # Determine which features to process
    if args.features:
        feature_indices = [int(x) for x in args.features.split(',')]
    else:
        feature_indices = list(range(n_dirs))
    print(f"  Processing {len(feature_indices)} features")

    # Setup
    checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    client = anthropic.AsyncAnthropic()
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
            client, token_texts_list, abs_proj, fi, seq_len,
            checkpoint_dir, semaphore, rng)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Summary
    elapsed = time.time() - t0
    completed = sum(1 for r in results if r.get('status') == 'completed')
    failed = sum(1 for r in results if r.get('status') == 'failed')

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Completed: {completed} (including {already_done} from previous run)")
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
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--center_by_position', action='store_true', default=True,
                        help='Mean-center projections per position (removes positional bias)')
    parser.add_argument('--no_center', dest='center_by_position', action='store_false',
                        help='Disable per-position centering')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.projections_path is None:
        args.projections_path = 'results/autointerp/qwen3-0.6b_layer14_k100_longctx/projections.pkl'
    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100_longctx'

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
