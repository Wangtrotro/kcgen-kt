"""
Experiment Runner for Reasoning-Enhanced Code-KT

This script orchestrates all experiments described in the paper:
1. Baseline: KCGen-KT (original)
2. Method 1: KCGen-KT + Reasoning (trained)
3. Method 2: Thinking-Code-KT (training-free)
4. Method 3: Execution Verification
5. Ablations: thinking budget, prompt configs

Usage:
    # Run all experiments
    python run_experiments.py --experiment all --kc_path problem_kc_5_50_CodeWorkout.json
    
    # Run specific experiment
    python run_experiments.py --experiment baseline --kc_path problem_kc_5_50_CodeWorkout.json
    python run_experiments.py --experiment reasoning_trained --kc_path problem_kc_5_50_CodeWorkout.json
    python run_experiments.py --experiment thinking_free --kc_path problem_kc_5_50_CodeWorkout.json
    python run_experiments.py --experiment ablation_budget --kc_path problem_kc_5_50_CodeWorkout.json
"""

import os
import json
import argparse
import subprocess
import sys
from datetime import datetime


def run_command(cmd: str, description: str):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"  [WARNING] Command exited with code {result.returncode}")
    
    return result.returncode


def experiment_baseline(args):
    """Experiment 1: Run original KCGen-KT baseline."""
    cmd = (
        f"python main_kc_okt.py "
        f"kc_path={args.kc_path} "
        f"epochs={args.epochs} "
        f"use_reasoning=false "
        f"log_wandb={str(args.log_wandb).lower()} "
        f"seed={args.seed}"
    )
    return run_command(cmd, "Experiment 1: KCGen-KT Baseline (no reasoning)")


def experiment_reasoning_trained(args):
    """Experiment 2: KCGen-KT + Reasoning (trained)."""
    
    traces_path = args.reasoning_traces or "data/reasoning_traces.json"
    if not os.path.exists(traces_path):
        print("Generating condensed reasoning traces first...")
        gen_cmd = (
            f"python reasoning_gen.py "
            f"--data_path data/dataset_time.pkl "
            f"--kc_path {args.kc_path} "
            f"--output {traces_path} "
            f"--mode condensed"
        )
        run_command(gen_cmd, "Pre-step: Generating reasoning traces")
    
    cmd = (
        f"python main_kc_okt.py "
        f"kc_path={args.kc_path} "
        f"epochs={args.epochs} "
        f"use_reasoning=true "
        f"reasoning_mode=trained "
        f"reasoning_traces_path={traces_path} "
        f"log_wandb={str(args.log_wandb).lower()} "
        f"seed={args.seed}"
    )
    return run_command(cmd, "Experiment 2: KCGen-KT + Reasoning (trained)")


def experiment_thinking_free(args):
    """Experiment 3: Training-free Thinking-Code-KT."""
    cmd = (
        f"python thinking_code_kt.py "
        f"--kc_path {args.kc_path} "
        f"--model {args.thinking_model} "
        f"--thinking_budget {args.thinking_budget} "
        f"--prompt_config full "
        f"--max_predictions {args.max_predictions} "
        f"--output results/thinking_code_kt_full.json "
        f"--seed {args.seed}"
    )
    return run_command(cmd, "Experiment 3: Thinking-Code-KT (training-free)")


def experiment_ablation_budget(args):
    """Experiment 6: Thinking budget ablation."""
    cmd = (
        f"python thinking_code_kt.py "
        f"--kc_path {args.kc_path} "
        f"--model {args.thinking_model} "
        f"--max_predictions {args.max_predictions} "
        f"--output results/ablation_budget.json "
        f"--ablation "
        f"--seed {args.seed}"
    )
    return run_command(cmd, "Experiment 6: Thinking Budget Ablation (B=256,512,1024,2048)")


def experiment_compare_prompts(args):
    """Experiment 7: Code-specific vs generic prompt comparison."""
    cmd = (
        f"python thinking_code_kt.py "
        f"--kc_path {args.kc_path} "
        f"--model {args.thinking_model} "
        f"--max_predictions {args.max_predictions} "
        f"--output results/prompt_comparison.json "
        f"--compare_prompts "
        f"--seed {args.seed}"
    )
    return run_command(cmd, "Experiment 7: Prompt Configuration Comparison")


def experiment_exec_verify(args):
    """Experiment 4: Execution verification."""
    reasoning_results = "results/thinking_code_kt_full.json"
    
    if not os.path.exists(reasoning_results):
        print(f"[INFO] Reasoning results not found at {reasoning_results}")
        print("       Run 'thinking_free' experiment first.")
        return 1
    
    cmd = (
        f"python exec_verify.py "
        f"--reasoning_results {reasoning_results} "
        f"--output results/exec_verify_results.json"
    )
    return run_command(cmd, "Experiment 4: Execution Verification (RVS)")


EXPERIMENTS = {
    'baseline': experiment_baseline,
    'reasoning_trained': experiment_reasoning_trained,
    'thinking_free': experiment_thinking_free,
    'ablation_budget': experiment_ablation_budget,
    'compare_prompts': experiment_compare_prompts,
    'exec_verify': experiment_exec_verify,
}


def main():
    parser = argparse.ArgumentParser(description="Run Reasoning-Enhanced Code-KT Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--kc_path", type=str, required=True,
                        help="Path to KC-problem mapping JSON")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs for trained experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--reasoning_traces", type=str, default=None,
                        help="Path to pre-generated reasoning traces")
    parser.add_argument("--thinking_model", type=str, 
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model for training-free experiments")
    parser.add_argument("--thinking_budget", type=int, default=1024)
    parser.add_argument("--max_predictions", type=int, default=200,
                        help="Max predictions for training-free experiments")
    
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*60}")
    print(f"  Reasoning-Enhanced Code-KT Experiments")
    print(f"  Timestamp: {timestamp}")
    print(f"  Experiment: {args.experiment}")
    print(f"{'#'*60}")
    
    if args.experiment == "all":
        results = {}
        for name, func in EXPERIMENTS.items():
            try:
                ret = func(args)
                results[name] = "success" if ret == 0 else f"failed (code={ret})"
            except Exception as e:
                results[name] = f"error: {str(e)}"
        
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, status in results.items():
            print(f"  {name:25s} : {status}")
        
        with open(f"results/experiment_summary_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
    else:
        EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
