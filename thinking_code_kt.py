"""
Thinking-Code-KT: Training-Free Code-Specific Knowledge Tracing with Test-Time Scaling

This module implements Method 2 of the Reasoning-Enhanced Code-KT framework:
a training-free approach that uses structured code reasoning prompts with
Test-Time Scaling (TTS) to perform programming knowledge tracing.

Inspired by Thinking-KT (ACL ARR 2026), but specifically designed for
programming education with code-aware reasoning templates.

Usage:
    python thinking_code_kt.py --data_path data/dataset_time.pkl \
        --kc_path problem_kc_5_50_CodeWorkout.json \
        --model Qwen/Qwen2.5-3B-Instruct \
        --thinking_budget 1024
"""

import os
import json
import argparse
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
except ImportError:
    print("Warning: torch/transformers not installed")

from reasoning_prompt import (
    THINKING_CODE_KT_SYSTEM_PROMPT,
    build_thinking_code_kt_prompt,
    PROMPT_CONFIGS,
)
from reasoning_gen import load_dataset, load_kc_mapping, build_student_history, estimate_kc_mastery


class ThinkingCodeKT:
    """Training-free Code-KT using Test-Time Scaling with reasoning."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        thinking_budget: int = 1024,
        prompt_config: str = "full",
        device: str = "cuda",
        use_system_prompt: bool = True,
    ):
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.prompt_config = PROMPT_CONFIGS.get(prompt_config, PROMPT_CONFIGS["full"])
        self.device = device
        self.use_system_prompt = use_system_prompt
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load the inference model."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def build_chat_messages(self, user_prompt: str) -> List[Dict]:
        """Build chat messages for the model."""
        messages = []
        if self.use_system_prompt:
            messages.append({"role": "system", "content": THINKING_CODE_KT_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": user_prompt})
        return messages
    
    def predict_with_reasoning(
        self,
        problem_history: List[Dict],
        kc_mastery: Dict[str, float],
        next_problem: str,
        next_kcs: List[str],
    ) -> Dict:
        """Predict student performance with reasoning trace.
        
        Returns:
            Dict with keys: prediction, diagnosis, reasoning_trace, raw_output
        """
        user_prompt = build_thinking_code_kt_prompt(
            problem_history=problem_history,
            kc_mastery=kc_mastery,
            next_problem=next_problem,
            next_kcs=next_kcs,
            include_code=self.prompt_config["include_code"],
            max_history=self.prompt_config["max_history"],
        )
        
        messages = self.build_chat_messages(user_prompt)
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.thinking_budget,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        result = self.parse_output(raw_output)
        result['raw_output'] = raw_output
        result['reasoning_length'] = len(generated_ids)
        
        return result
    
    def parse_output(self, output: str) -> Dict:
        """Parse the model output to extract prediction and diagnosis."""
        prediction = None
        diagnosis = ""
        reasoning_trace = output
        
        pred_match = re.search(
            r'Prediction:\s*(correct|wrong)', output, re.IGNORECASE
        )
        if pred_match:
            prediction = 1 if pred_match.group(1).lower() == 'correct' else 0
            reasoning_trace = output[:pred_match.start()].strip()
        
        diag_match = re.search(
            r'Diagnosis:\s*(.+?)(?:\n|$)', output, re.IGNORECASE
        )
        if diag_match:
            diagnosis = diag_match.group(1).strip()
        
        if prediction is None:
            output_lower = output.lower()
            if 'correct' in output_lower[-100:]:
                prediction = 1
            elif 'wrong' in output_lower[-100:] or 'incorrect' in output_lower[-100:]:
                prediction = 0
            else:
                prediction = -1  # Unparseable
        
        return {
            'prediction': prediction,
            'diagnosis': diagnosis,
            'reasoning_trace': reasoning_trace,
        }
    
    def classify_reasoning_episodes(self, reasoning_trace: str) -> Dict[str, int]:
        """Classify reasoning trace segments into Code-Specific Episodes."""
        episodes = {
            'CodeRead': 0,
            'PatternRecall': 0,
            'ExecTrace': 0,
            'MisconceptionID': 0,
            'Verify': 0,
        }
        
        markers = {
            'CodeRead': [r'\[CodeRead\]', r'problem requires', r'need to implement'],
            'PatternRecall': [r'\[PatternRecall\]', r'has shown mastery', r'struggles with', r'demonstrated'],
            'ExecTrace': [r'\[ExecTrace\]', r'would likely write', r'execute', r'trace', r'loop.*iteration'],
            'MisconceptionID': [r'\[MisconceptionID\]', r'misconception', r'error', r'mistake', r'forget', r'off-by-one'],
            'Verify': [r'\[Verify\]', r'consistent', r'aligns with', r'confirms'],
        }
        
        for episode, patterns in markers.items():
            for pattern in patterns:
                if re.search(pattern, reasoning_trace, re.IGNORECASE):
                    episodes[episode] += 1
                    break
        
        return episodes


def evaluate_thinking_code_kt(
    model: ThinkingCodeKT,
    df: pd.DataFrame,
    kc_problem_dict: Dict,
    test_students: List[str],
    min_history: int = 3,
    max_predictions: Optional[int] = None,
) -> Dict:
    """Evaluate Thinking-Code-KT on test set."""
    
    predictions = []
    ground_truths = []
    reasoning_traces = []
    episode_stats = defaultdict(int)
    unparseable = 0
    total = 0
    
    for student_id in tqdm(test_students, desc="Evaluating students"):
        interactions = build_student_history(df, student_id, kc_problem_dict)
        
        if len(interactions) < min_history + 1:
            continue
        
        for t in range(min_history, len(interactions)):
            if max_predictions and total >= max_predictions:
                break
            
            history = interactions[:t]
            next_inter = interactions[t]
            
            kc_mastery = estimate_kc_mastery(history, kc_problem_dict)
            
            result = model.predict_with_reasoning(
                problem_history=history,
                kc_mastery=kc_mastery,
                next_problem=next_inter['prompt'],
                next_kcs=next_inter['kcs'],
            )
            
            if result['prediction'] == -1:
                unparseable += 1
                continue
            
            predictions.append(result['prediction'])
            ground_truths.append(next_inter['score'])
            reasoning_traces.append(result)
            
            episodes = model.classify_reasoning_episodes(result['reasoning_trace'])
            for ep, count in episodes.items():
                episode_stats[ep] += count
            
            total += 1
        
        if max_predictions and total >= max_predictions:
            break
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    results = {}
    
    if len(predictions) > 0:
        results['accuracy'] = float(np.mean(predictions == ground_truths))
        
        from sklearn.metrics import f1_score, roc_auc_score
        results['f1'] = float(f1_score(ground_truths, predictions))
        
        try:
            results['auc'] = float(roc_auc_score(ground_truths, predictions))
        except ValueError:
            results['auc'] = 0.0
    
    results['total_predictions'] = int(total)
    results['unparseable'] = int(unparseable)
    results['episode_distribution'] = dict(episode_stats)
    results['avg_reasoning_length'] = float(
        np.mean([r['reasoning_length'] for r in reasoning_traces])
    ) if reasoning_traces else 0.0
    
    return results


def run_thinking_budget_ablation(
    model_name: str,
    df: pd.DataFrame,
    kc_problem_dict: Dict,
    test_students: List[str],
    budgets: List[int] = [256, 512, 1024, 2048],
    max_predictions: int = 100,
) -> Dict[int, Dict]:
    """Run ablation study on thinking budget B."""
    
    results = {}
    
    for budget in budgets:
        print(f"\n=== Thinking Budget B={budget} ===")
        model = ThinkingCodeKT(
            model_name=model_name,
            thinking_budget=budget,
        )
        model.load_model()
        
        eval_results = evaluate_thinking_code_kt(
            model=model,
            df=df,
            kc_problem_dict=kc_problem_dict,
            test_students=test_students,
            max_predictions=max_predictions,
        )
        
        results[budget] = eval_results
        print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
        print(f"  F1: {eval_results.get('f1', 0):.4f}")
        print(f"  AUC: {eval_results.get('auc', 0):.4f}")
        print(f"  Avg reasoning length: {eval_results.get('avg_reasoning_length', 0):.1f} tokens")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def compare_prompt_configs(
    model_name: str,
    df: pd.DataFrame,
    kc_problem_dict: Dict,
    test_students: List[str],
    configs: List[str] = ["full", "no_code", "minimal"],
    max_predictions: int = 100,
) -> Dict[str, Dict]:
    """Compare different prompt configurations (code-specific vs generic)."""
    
    results = {}
    
    for config_name in configs:
        print(f"\n=== Prompt Config: {config_name} ===")
        model = ThinkingCodeKT(
            model_name=model_name,
            thinking_budget=1024,
            prompt_config=config_name,
        )
        model.load_model()
        
        eval_results = evaluate_thinking_code_kt(
            model=model,
            df=df,
            kc_problem_dict=kc_problem_dict,
            test_students=test_students,
            max_predictions=max_predictions,
        )
        
        results[config_name] = eval_results
        print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
        print(f"  F1: {eval_results.get('f1', 0):.4f}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-Free Thinking-Code-KT")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--kc_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--thinking_budget", type=int, default=1024)
    parser.add_argument("--prompt_config", type=str, default="full",
                        choices=["full", "no_code", "minimal", "extended"])
    parser.add_argument("--max_predictions", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/thinking_code_kt_results.json")
    parser.add_argument("--ablation", action="store_true", help="Run thinking budget ablation")
    parser.add_argument("--compare_prompts", action="store_true", help="Compare prompt configs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("Loading data...")
    df = load_dataset(args.data_path)
    kc_problem_dict, kc_dict_res = load_kc_mapping(args.kc_path)
    df['knowledge_component'] = df['prompt'].map(kc_problem_dict)
    
    from sklearn.model_selection import train_test_split
    students = df['SubjectID'].unique()
    _, test_students = train_test_split(students, test_size=0.2, random_state=args.seed)
    _, test_students = train_test_split(test_students, test_size=0.5, random_state=args.seed)
    test_students = list(test_students)
    
    print(f"Test students: {len(test_students)}")
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)
    
    if args.ablation:
        results = run_thinking_budget_ablation(
            model_name=args.model,
            df=df,
            kc_problem_dict=kc_problem_dict,
            test_students=test_students,
            max_predictions=args.max_predictions or 100,
        )
        results_serializable = {str(k): v for k, v in results.items()}
    
    elif args.compare_prompts:
        results_serializable = compare_prompt_configs(
            model_name=args.model,
            df=df,
            kc_problem_dict=kc_problem_dict,
            test_students=test_students,
            max_predictions=args.max_predictions or 100,
        )
    
    else:
        print(f"\nRunning Thinking-Code-KT with budget={args.thinking_budget}...")
        model = ThinkingCodeKT(
            model_name=args.model,
            thinking_budget=args.thinking_budget,
            prompt_config=args.prompt_config,
        )
        model.load_model()
        
        results_serializable = evaluate_thinking_code_kt(
            model=model,
            df=df,
            kc_problem_dict=kc_problem_dict,
            test_students=test_students,
            max_predictions=args.max_predictions,
        )
        
        print(f"\n=== Results ===")
        print(f"Accuracy: {results_serializable.get('accuracy', 0):.4f}")
        print(f"F1: {results_serializable.get('f1', 0):.4f}")
        print(f"AUC: {results_serializable.get('auc', 0):.4f}")
        print(f"Total predictions: {results_serializable.get('total_predictions', 0)}")
        print(f"Avg reasoning length: {results_serializable.get('avg_reasoning_length', 0):.1f}")
        print(f"Episode distribution: {results_serializable.get('episode_distribution', {})}")
    
    with open(args.output, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")
