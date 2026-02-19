#!/usr/bin/env python3
"""
Score long-context auto-interp results: per-token Spearman correlation.

Loads per-feature checkpoint JSONs with per-token predictions and actual projections,
computes rank correlations, and optionally compares with short-context results.

Usage:
    python scripts/autointerp/03_score_longctx.py
"""

import argparse
import json
import os
import pickle

import numpy as np
from scipy import stats


def load_feature_results(checkpoint_dir):
    """Load all per-feature checkpoint JSONs."""
    results = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(checkpoint_dir, fname)) as f:
            results.append(json.load(f))
    return results


def compute_spearman(feature_result):
    """Compute per-token Spearman correlation for a single feature.

    Pools all tokens across all simulate passages.

    Returns:
        correlation, p_value, n_tokens — or None if insufficient data
    """
    if feature_result.get('status') != 'completed':
        return None

    all_predicted = []
    all_actual = []

    for pr in feature_result.get('passage_results', []):
        predicted = pr['predicted']
        actual = pr['actual']
        # Use the minimum length (in case of truncation)
        n = min(len(predicted), len(actual))
        all_predicted.extend(predicted[:n])
        all_actual.extend(actual[:n])

    if len(all_predicted) < 10:
        return None

    corr, pval = stats.spearmanr(all_predicted, all_actual)
    return corr, pval, len(all_predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--short_context_results', default=None,
                        help='Path to short-context results.pkl for comparison')
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100_longctx'
    if args.short_context_results is None:
        short_path = 'results/autointerp/qwen3-0.6b_layer14_k100/results.pkl'
        if os.path.exists(short_path):
            args.short_context_results = short_path

    checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoints found at {checkpoint_dir}")
        return

    # Load results
    feature_results = load_feature_results(checkpoint_dir)
    print(f"Loaded {len(feature_results)} feature results")

    # Compute correlations
    scored = []
    failed = []
    for fr in feature_results:
        result = compute_spearman(fr)
        if result is not None:
            corr, pval, n = result
            scored.append({
                'feature_idx': fr['feature_idx'],
                'correlation': corr,
                'p_value': pval,
                'n_tokens': n,
                'description': fr.get('description', ''),
            })
        else:
            failed.append(fr.get('feature_idx', '?'))

    if not scored:
        print("No features scored successfully.")
        return

    correlations = np.array([s['correlation'] for s in scored])

    # Summary
    print(f"\n{'='*60}")
    print("LONG-CONTEXT AUTO-INTERPRETABILITY RESULTS")
    print(f"{'='*60}")
    print(f"Features scored: {len(scored)}")
    print(f"Features failed: {len(failed)}")
    if failed:
        print(f"  Failed indices: {failed}")
    print(f"\nSpearman correlation (per-token):")
    print(f"- Mean: {correlations.mean():.4f}")
    print(f"- Median: {np.median(correlations):.4f}")
    print(f"- Std: {correlations.std():.4f}")
    print(f"- Min: {correlations.min():.4f}")
    print(f"- Max: {correlations.max():.4f}")
    print(f"- Fraction > 0: {(correlations > 0).mean():.2f}")
    print(f"- Fraction > 0.3: {(correlations > 0.3).mean():.2f}")

    # Top 5
    scored_sorted = sorted(scored, key=lambda x: x['correlation'], reverse=True)
    print(f"\nTop 5 features:")
    for s in scored_sorted[:5]:
        desc = s['description'][:80] + ('...' if len(s['description']) > 80 else '')
        print(f"- Feature {s['feature_idx']}: r={s['correlation']:.4f} (n={s['n_tokens']})")
        print(f"  {desc}")

    # Bottom 5
    print(f"\nBottom 5 features:")
    for s in scored_sorted[-5:]:
        desc = s['description'][:80] + ('...' if len(s['description']) > 80 else '')
        print(f"- Feature {s['feature_idx']}: r={s['correlation']:.4f} (n={s['n_tokens']})")
        print(f"  {desc}")

    # Compare with short-context if available
    if args.short_context_results:
        print(f"\n{'='*60}")
        print("COMPARISON WITH SHORT-CONTEXT RESULTS")
        print(f"{'='*60}")
        with open(args.short_context_results, 'rb') as f:
            short_results = pickle.load(f)
        short_corrs = {s['feature_idx']: s['correlation']
                      for s in short_results['scored_features']}
        long_corrs = {s['feature_idx']: s['correlation'] for s in scored}

        common = set(short_corrs.keys()) & set(long_corrs.keys())
        if common:
            short_vals = np.array([short_corrs[k] for k in sorted(common)])
            long_vals = np.array([long_corrs[k] for k in sorted(common)])
            print(f"Features in common: {len(common)}")
            print(f"Short-context mean r: {short_vals.mean():.4f}")
            print(f"Long-context mean r: {long_vals.mean():.4f}")
            print(f"Improvement: {(long_vals - short_vals).mean():.4f}")
            print(f"Features improved: {(long_vals > short_vals).sum()}/{len(common)}")

            # Correlation between short and long scores
            rr, pp = stats.spearmanr(short_vals, long_vals)
            print(f"Correlation between short/long scores: r={rr:.4f} (p={pp:.4f})")

    # Save
    output = {
        'scored_features': scored,
        'failed_features': failed,
        'summary': {
            'n_scored': len(scored),
            'n_failed': len(failed),
            'mean_correlation': float(correlations.mean()),
            'median_correlation': float(np.median(correlations)),
            'std_correlation': float(correlations.std()),
            'min_correlation': float(correlations.min()),
            'max_correlation': float(correlations.max()),
            'frac_positive': float((correlations > 0).mean()),
            'frac_above_0.3': float((correlations > 0.3).mean()),
        },
    }
    output_path = os.path.join(args.results_dir, 'results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
