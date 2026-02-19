#!/usr/bin/env python3
"""
Score auto-interpretability results: Spearman correlation between predicted and actual.

Loads per-feature checkpoint JSONs, computes rank correlation, prints summary.

Usage:
    python scripts/autointerp/03_score.py
    python scripts/autointerp/03_score.py --results_dir results/autointerp/qwen3-0.6b_layer14_k100
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
    """Compute Spearman correlation for a single feature.

    Returns:
        correlation, p_value, n_pairs — or None if insufficient data
    """
    if feature_result.get('status') != 'completed':
        return None

    predictions = feature_result.get('predictions', {})
    actuals = feature_result.get('actuals', {})

    # Match indices present in both
    common_keys = set(predictions.keys()) & set(actuals.keys())
    if len(common_keys) < 5:
        return None

    pred_vals = [predictions[k] for k in sorted(common_keys)]
    actual_vals = [actuals[k] for k in sorted(common_keys)]

    corr, pval = stats.spearmanr(pred_vals, actual_vals)
    return corr, pval, len(common_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = 'results/autointerp/qwen3-0.6b_layer14_k100'

    checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoints found at {checkpoint_dir}")
        return

    # Load all results
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
                'n_pairs': n,
                'description': fr.get('description', ''),
            })
        else:
            failed.append(fr.get('feature_idx', '?'))

    if not scored:
        print("No features scored successfully.")
        return

    correlations = [s['correlation'] for s in scored]
    correlations_arr = np.array(correlations)

    # Summary statistics
    print(f"\n{'='*60}")
    print("AUTO-INTERPRETABILITY RESULTS")
    print(f"{'='*60}")
    print(f"Features scored: {len(scored)}")
    print(f"Features failed: {len(failed)}")
    if failed:
        print(f"  Failed indices: {failed}")
    print(f"\nSpearman correlation:")
    print(f"- Mean: {correlations_arr.mean():.4f}")
    print(f"- Median: {np.median(correlations_arr):.4f}")
    print(f"- Std: {correlations_arr.std():.4f}")
    print(f"- Min: {correlations_arr.min():.4f}")
    print(f"- Max: {correlations_arr.max():.4f}")
    print(f"- Fraction > 0: {(correlations_arr > 0).mean():.2f}")
    print(f"- Fraction > 0.3: {(correlations_arr > 0.3).mean():.2f}")

    # Top 5
    scored_sorted = sorted(scored, key=lambda x: x['correlation'], reverse=True)
    print(f"\nTop 5 features:")
    for s in scored_sorted[:5]:
        desc = s['description'][:80] + ('...' if len(s['description']) > 80 else '')
        print(f"- Feature {s['feature_idx']}: r={s['correlation']:.4f} (n={s['n_pairs']})")
        print(f"  {desc}")

    # Bottom 5
    print(f"\nBottom 5 features:")
    for s in scored_sorted[-5:]:
        desc = s['description'][:80] + ('...' if len(s['description']) > 80 else '')
        print(f"- Feature {s['feature_idx']}: r={s['correlation']:.4f} (n={s['n_pairs']})")
        print(f"  {desc}")

    # Save aggregated results
    output = {
        'scored_features': scored,
        'failed_features': failed,
        'summary': {
            'n_scored': len(scored),
            'n_failed': len(failed),
            'mean_correlation': float(correlations_arr.mean()),
            'median_correlation': float(np.median(correlations_arr)),
            'std_correlation': float(correlations_arr.std()),
            'min_correlation': float(correlations_arr.min()),
            'max_correlation': float(correlations_arr.max()),
            'frac_positive': float((correlations_arr > 0).mean()),
            'frac_above_0.3': float((correlations_arr > 0.3).mean()),
        },
    }
    output_path = os.path.join(args.results_dir, 'results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
