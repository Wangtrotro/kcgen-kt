#!/usr/bin/env python
"""
Run Path-aware KT Ablation Experiment
======================================
Automates the core validation: real-path vs random-path comparison.

This script runs the experiment in three modes and compares results:
  1. no_path:     Original kcgen-kt (all KCs activated equally)
  2. real_path:   Path-conditioned KC activation (with real path IDs)
  3. random_path: Same model, but path IDs randomly shuffled per-problem

If real_path > random_path > no_path (or real_path > no_path ≈ random_path):
  → Path identity carries diagnostic signal beyond just "having more parameters"

Usage:
    # Full ablation (requires GPU + dataset)
    python run_ablation.py --mode all
    
    # Only path discovery (CPU, no LLM needed)
    python run_ablation.py --mode discover
    
    # Only comparison after paths are discovered
    python run_ablation.py --mode compare
    
    # Quick test with minimal data
    python run_ablation.py --mode all --testing
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def run_path_discovery(args):
    """Step 1: Discover solution paths via code clustering."""
    print("\n" + "=" * 60)
    print("  STEP 1: Path Discovery")
    print("=" * 60)

    cmd = [
        sys.executable, "path_discovery.py",
        "--n_paths", str(args.n_paths),
        "--data_path", args.data_path,
        "--output_dir", args.path_dir,
        "--seed", str(args.seed),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("  ERROR: Path discovery failed!")
        return False
    return True


def run_path_kc_gen(args):
    """Step 2: Generate path-level KCs (or use fallback)."""
    print("\n" + "=" * 60)
    print("  STEP 2: Path-level KC Generation")
    print("=" * 60)

    cmd = [
        sys.executable, "path_kc_gen.py",
        "--path_dir", args.path_dir,
        "--data_path", args.data_path,
    ]

    if args.fallback_kc or not os.getenv("OPENAI_API_KEY"):
        cmd.append("--fallback")
        cmd.extend(["--existing_kc", "problem_kc.json"])
        print("  Using fallback heuristic (no LLM calls)")
    else:
        cmd.extend(["--model", args.llm_model])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("  WARNING: Path KC generation failed, will use problem-level KCs")
        return False
    return True


def run_kt_experiment(mode_name, path_mode, randomize, args):
    """Run a single KT experiment configuration."""
    print(f"\n  Running KT mode: {mode_name} (path_mode={path_mode}, randomize={randomize})")

    # Build hydra override args
    overrides = [
        f"path_mode={path_mode}",
        f"randomize_paths={str(randomize).lower()}",
        f"path_dir={args.path_dir}",
        f"n_paths={args.n_paths}",
        f"seed={args.seed}",
        f"epochs={args.epochs}",
    ]
    if args.testing:
        overrides.append("testing=true")
    if args.kc_path:
        overrides.append(f"kc_path={args.kc_path}")

    cmd = [sys.executable, "main_path_kt.py"] + overrides
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  ERROR in {mode_name}:")
        print(result.stderr[-1000:] if result.stderr else "No stderr")
        return None

    # Try to find results file
    results_dir = "checkpoints"
    results_files = [f for f in os.listdir(results_dir) if f.startswith(f"results_path_{path_mode}")]
    if randomize:
        results_files = [f for f in results_files if "RANDOM" in f]
    else:
        results_files = [f for f in results_files if "RANDOM" not in f]

    if results_files:
        latest = sorted(results_files)[-1]
        with open(os.path.join(results_dir, latest)) as f:
            return json.load(f)
    return None


def run_comparison(args):
    """Step 3: Run the three-way comparison experiment."""
    print("\n" + "=" * 60)
    print("  STEP 3: Ablation Comparison")
    print("=" * 60)

    results = {}

    # Mode 1: No path (baseline)
    print("\n--- Experiment 1/3: No Path (Baseline) ---")
    results['no_path'] = run_kt_experiment("no_path", "none", False, args)

    # Mode 2: Real path
    print("\n--- Experiment 2/3: Real Path ---")
    results['real_path'] = run_kt_experiment("real_path", args.path_activation, False, args)

    # Mode 3: Random path (ablation)
    print("\n--- Experiment 3/3: Random Path (Ablation) ---")
    results['random_path'] = run_kt_experiment("random_path", args.path_activation, True, args)

    return results


def print_comparison(results):
    """Print the final comparison table."""
    print("\n" + "=" * 60)
    print("  ABLATION RESULTS COMPARISON")
    print("=" * 60)

    header = f"{'Mode':<20} {'Loss':>8} {'ACC':>8} {'AUC':>8}"
    print(header)
    print("-" * len(header))

    for mode_name, res in results.items():
        if res is None:
            print(f"{mode_name:<20} {'FAILED':>8}")
            continue

        loss = res.get('best_valid_loss', float('nan'))
        test = res.get('best_test_metrics', {})
        acc = test.get('acc', float('nan'))
        auc = test.get('auc', float('nan'))

        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)[:8]
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)[:8]

        print(f"{mode_name:<20} {loss:>8.4f} {acc_str:>8} {auc_str:>8}")

    print("\n" + "=" * 60)
    print("  INTERPRETATION GUIDE:")
    print("  • If real_path > random_path: path identity carries signal")
    print("  • If real_path > no_path: path conditioning improves KT")
    print("  • If random_path ≈ no_path: gains are from path, not extra params")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Path-aware KT ablation experiment")
    parser.add_argument("--mode", choices=['all', 'discover', 'compare', 'kc_gen'],
                        default='all', help="Which steps to run")
    parser.add_argument("--n_paths", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--path_dir", type=str, default="path_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--testing", action="store_true", help="Quick test mode")
    parser.add_argument("--fallback_kc", action="store_true",
                        help="Use heuristic KC split instead of LLM")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--kc_path", type=str, default="problem_kc.json")
    parser.add_argument("--path_activation", choices=['hard', 'soft'], default='hard',
                        help="How to apply path conditioning")
    args = parser.parse_args()

    print(f"\nPath-aware KT Ablation Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}, Paths: {args.n_paths}, Activation: {args.path_activation}")

    if args.mode in ('all', 'discover'):
        success = run_path_discovery(args)
        if not success and args.mode == 'all':
            print("Path discovery failed. Aborting.")
            return

    if args.mode in ('all', 'kc_gen'):
        run_path_kc_gen(args)

    if args.mode in ('all', 'compare'):
        results = run_comparison(args)
        print_comparison(results)

        # Save comparison
        output_file = os.path.join(args.path_dir, "ablation_results.json")
        serializable = {}
        for k, v in results.items():
            serializable[k] = v if v else "FAILED"
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nComparison saved to {output_file}")


if __name__ == "__main__":
    main()
