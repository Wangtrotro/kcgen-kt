"""
Preliminary Validation: Same Problem, Different Path Analysis
==============================================================
Tests the fundamental premise: do students who use different solution paths
on the same problem show different subsequent performance patterns?

This analysis does NOT require GPU or LLM — it only needs the dataset
and path assignments from path_discovery.py.

If this test is negative (no difference between paths), the entire
path-aware direction loses its foundation.

Usage:
    python analyze_path_difference.py --path_dir path_cache --data_path data/dataset_time.pkl
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats


def load_data_with_paths(data_path, path_dir):
    """Load dataset and merge with path assignments."""
    df = pd.read_pickle(data_path)

    if 'Score_x' in df.columns:
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)

    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    # Load path assignments
    with open(os.path.join(path_dir, "path_assignments.json")) as f:
        raw = json.load(f)

    path_map = {}
    for key, val in raw.items():
        sid, pid = key.split("||")
        try:
            pid = int(pid)
        except ValueError:
            pass
        path_map[(sid, pid)] = val['hard']

    df['path_id'] = df.apply(lambda r: path_map.get((r['SubjectID'], r['ProblemID']), -1), axis=1)

    return df


def analyze_subsequent_performance(df):
    """
    Core analysis: For each problem with multiple paths, compare the
    subsequent performance of students who took different paths.
    
    'Subsequent performance' = average score on the NEXT N problems after this one.
    """
    results = {}
    problems_with_paths = df[df['path_id'] >= 0]['ProblemID'].unique()

    print(f"\nAnalyzing {len(problems_with_paths)} problems with path assignments...")

    significant_count = 0
    total_tested = 0

    for pid in problems_with_paths:
        problem_df = df[df['ProblemID'] == pid]
        paths_present = problem_df['path_id'].unique()

        if len(paths_present) < 2:
            continue

        # For each student who did this problem, compute their score on next 3 problems
        subsequent_scores_by_path = defaultdict(list)

        students_on_problem = problem_df[['SubjectID', 'path_id']].values

        for sid, path in students_on_problem:
            student_df = df[df['SubjectID'] == sid].sort_values('ServerTimestamp')
            student_problems = student_df['ProblemID'].tolist()

            if pid not in student_problems:
                continue

            idx = student_problems.index(pid)
            # Get scores on next 1-3 problems
            next_scores = student_df.iloc[idx+1:idx+4]['Score'].tolist()
            if next_scores:
                subsequent_scores_by_path[path].append(np.mean(next_scores))

        # Statistical test: do different paths lead to different subsequent performance?
        path_groups = {p: scores for p, scores in subsequent_scores_by_path.items() if len(scores) >= 5}

        if len(path_groups) >= 2:
            total_tested += 1
            groups = list(path_groups.values())

            # Kruskal-Wallis test (non-parametric, works for >2 groups)
            if len(groups) == 2:
                stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                stat, p_val = stats.kruskal(*groups)

            means = {p: np.mean(s) for p, s in path_groups.items()}
            sizes = {p: len(s) for p, s in path_groups.items()}

            results[pid] = {
                'p_value': p_val,
                'significant': p_val < 0.05,
                'means_by_path': means,
                'sizes_by_path': sizes,
                'effect_size': max(means.values()) - min(means.values()),
            }

            if p_val < 0.05:
                significant_count += 1

    return results, significant_count, total_tested


def analyze_path_diversity(df):
    """Analyze how diverse paths are within each problem."""
    problems = df[df['path_id'] >= 0]['ProblemID'].unique()

    diversity_stats = []
    for pid in problems:
        problem_df = df[df['ProblemID'] == pid]
        path_counts = problem_df['path_id'].value_counts()
        n_paths = len(path_counts)
        entropy = stats.entropy(path_counts.values)
        majority_ratio = path_counts.iloc[0] / len(problem_df)

        diversity_stats.append({
            'problem_id': pid,
            'n_paths_used': n_paths,
            'entropy': entropy,
            'majority_ratio': majority_ratio,
            'total_submissions': len(problem_df),
        })

    return pd.DataFrame(diversity_stats)


def analyze_path_score_correlation(df):
    """Check if path choice correlates with score on the SAME problem."""
    problems = df[df['path_id'] >= 0]['ProblemID'].unique()

    correlations = []
    for pid in problems:
        problem_df = df[df['ProblemID'] == pid]
        paths = problem_df['path_id'].unique()
        if len(paths) < 2:
            continue

        score_by_path = problem_df.groupby('path_id')['Score'].mean()
        if len(score_by_path) >= 2:
            correlations.append({
                'problem_id': pid,
                'score_spread': score_by_path.max() - score_by_path.min(),
                'scores': score_by_path.to_dict(),
            })

    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir", type=str, default="path_cache")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    args = parser.parse_args()

    print("=" * 60)
    print("  Same-Problem, Different-Path Analysis")
    print("  Testing: Do different paths predict different outcomes?")
    print("=" * 60)

    df = load_data_with_paths(args.data_path, args.path_dir)
    assigned = df[df['path_id'] >= 0]
    print(f"\nDataset: {len(df)} submissions, {len(assigned)} with path assignments")
    print(f"Students: {df['SubjectID'].nunique()}")
    print(f"Problems: {df['ProblemID'].nunique()}")

    # Analysis 1: Path diversity
    print("\n--- Path Diversity ---")
    diversity = analyze_path_diversity(df)
    print(f"  Problems with 2+ paths used: {(diversity['n_paths_used'] >= 2).sum()}")
    print(f"  Mean entropy: {diversity['entropy'].mean():.3f}")
    print(f"  Mean majority ratio: {diversity['majority_ratio'].mean():.3f}")

    # Analysis 2: Path-score correlation (same problem)
    print("\n--- Path vs Score on Same Problem ---")
    correlations = analyze_path_score_correlation(df)
    if correlations:
        spreads = [c['score_spread'] for c in correlations]
        print(f"  Problems tested: {len(correlations)}")
        print(f"  Mean score spread between paths: {np.mean(spreads):.3f}")
        print(f"  Max score spread: {np.max(spreads):.3f}")

        # Show top examples
        top = sorted(correlations, key=lambda x: x['score_spread'], reverse=True)[:5]
        print(f"  Top 5 problems with largest path-score difference:")
        for c in top:
            print(f"    Problem {c['problem_id']}: spread={c['score_spread']:.3f}, scores={c['scores']}")

    # Analysis 3: Subsequent performance difference (KEY TEST)
    print("\n--- Subsequent Performance by Path (KEY TEST) ---")
    results, sig_count, total_tested = analyze_subsequent_performance(df)
    print(f"  Problems tested: {total_tested}")
    print(f"  Significant differences (p<0.05): {sig_count} ({sig_count/max(total_tested,1)*100:.1f}%)")

    if results:
        effect_sizes = [r['effect_size'] for r in results.values()]
        print(f"  Mean effect size: {np.mean(effect_sizes):.3f}")
        print(f"  Median effect size: {np.median(effect_sizes):.3f}")

        # Show significant examples
        sig_results = {k: v for k, v in results.items() if v['significant']}
        if sig_results:
            print(f"\n  Significant examples:")
            for pid, r in sorted(sig_results.items(), key=lambda x: x[1]['effect_size'], reverse=True)[:5]:
                print(f"    Problem {pid}: effect={r['effect_size']:.3f}, "
                      f"means={r['means_by_path']}, sizes={r['sizes_by_path']}")

    # Conclusion
    print("\n" + "=" * 60)
    if total_tested > 0 and sig_count / total_tested > 0.15:
        print("  CONCLUSION: Premise SUPPORTED")
        print("  Different solution paths are associated with different subsequent outcomes.")
        print("  → Path-conditioned KC activation is justified.")
    elif total_tested > 0 and sig_count / total_tested > 0.05:
        print("  CONCLUSION: Premise WEAKLY SUPPORTED")
        print("  Some evidence that paths predict different outcomes, but effect may be small.")
        print("  → Proceed with caution; path-aware model may show modest gains.")
    else:
        print("  CONCLUSION: Premise NOT SUPPORTED (or insufficient data)")
        print("  No clear evidence that path choice predicts subsequent performance.")
        print("  → Reconsider whether path-level KC activation adds value in this dataset.")
    print("=" * 60)

    # Save results
    output = {
        'total_problems_tested': int(total_tested),
        'significant_count': int(sig_count),
        'significance_rate': float(sig_count / max(total_tested, 1)),
        'mean_effect_size': float(np.mean(effect_sizes)) if results else 0,
        'detailed_results': {str(k): {
            'p_value': float(v['p_value']),
            'significant': bool(v['significant']),
            'effect_size': float(v['effect_size']),
            'means_by_path': {str(pk): float(pv) for pk, pv in v['means_by_path'].items()},
            'sizes_by_path': {str(pk): int(pv) for pk, pv in v['sizes_by_path'].items()},
        } for k, v in results.items()}
    }
    output_file = os.path.join(args.path_dir, "path_difference_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
