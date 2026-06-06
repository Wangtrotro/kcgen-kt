"""
Execution Verification Module for Reasoning-Enhanced Code-KT

This module implements Method 3: Execution-Grounded Reasoning Verification.
It validates reasoning traces by checking if predicted misconceptions are
consistent with actual code execution against test cases.

The key innovation: programming KT reasoning can be VERIFIED through execution,
unlike math/general education KT where reasoning is not objectively verifiable.

Pipeline:
1. Parse reasoning trace → extract predicted misconception
2. Generate buggy code that embodies the predicted misconception
3. Execute buggy code against test cases
4. Compute Reasoning Verification Score (RVS):
   - Does the predicted error pattern match actual test failures?
"""

import os
import re
import json
import subprocess
import tempfile
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    pass


# ==============================================================================
# Misconception Extraction from Reasoning Traces
# ==============================================================================

COMMON_MISCONCEPTIONS = [
    "off-by-one error",
    "missing base case",
    "incorrect loop bounds", 
    "wrong comparison operator",
    "null/empty check missing",
    "incorrect return type",
    "wrong variable initialization",
    "string comparison with ==",
    "array index out of bounds",
    "infinite loop",
    "wrong recursive call",
    "missing edge case",
    "incorrect operator precedence",
    "wrong data type",
    "scope error",
]


def extract_misconception(reasoning_trace: str) -> Optional[str]:
    """Extract the predicted misconception from a reasoning trace."""
    
    misconception_section = re.search(
        r'\[MisconceptionID\]\s*(.+?)(?:\[Verify\]|Prediction:|$)',
        reasoning_trace, re.DOTALL | re.IGNORECASE
    )
    
    if misconception_section:
        return misconception_section.group(1).strip()
    
    diag_match = re.search(
        r'Diagnosis:\s*(.+?)(?:\n|$)',
        reasoning_trace, re.IGNORECASE
    )
    if diag_match:
        return diag_match.group(1).strip()
    
    for misconception in COMMON_MISCONCEPTIONS:
        if misconception.lower() in reasoning_trace.lower():
            return misconception
    
    return None


def categorize_misconception(misconception_text: str) -> str:
    """Categorize a misconception into a standard category."""
    text = misconception_text.lower()
    
    categories = {
        "boundary_error": ["off-by-one", "bounds", "boundary", "index", "out of range"],
        "missing_case": ["base case", "edge case", "missing", "empty", "null"],
        "logic_error": ["comparison", "operator", "condition", "logic", "precedence"],
        "loop_error": ["loop", "infinite", "iteration", "termination"],
        "type_error": ["type", "cast", "conversion", "string.*=="],
        "recursion_error": ["recursive", "recursion", "stack overflow"],
        "initialization_error": ["initial", "variable", "declaration", "scope"],
        "return_error": ["return", "output", "result"],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(keyword, text):
                return category
    
    return "other"


# ==============================================================================
# Buggy Code Generation
# ==============================================================================

BUGGY_CODE_PROMPT = """Given the following programming problem and a specific misconception a student has, 
generate a Java method that contains EXACTLY the described error and no other errors.

Problem: {problem_description}
Method Signature: {method_signature}

Student's Misconception: {misconception}

Requirements:
1. The code must be syntactically valid Java
2. It should contain ONLY the specific error described in the misconception
3. It should be a plausible student submission (not intentionally obfuscated)
4. Return ONLY the method code, nothing else

Code:"""


def generate_buggy_code_llm(
    problem_description: str,
    method_signature: str,
    misconception: str,
    client: 'OpenAI',
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """Generate buggy code using LLM based on predicted misconception."""
    
    prompt = BUGGY_CODE_PROMPT.format(
        problem_description=problem_description,
        method_signature=method_signature,
        misconception=misconception,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        code = response.choices[0].message.content.strip()
        if code.startswith("```"):
            code = re.sub(r'^```\w*\n?', '', code)
            code = re.sub(r'\n?```$', '', code)
        return code
    except Exception as e:
        print(f"Error generating buggy code: {e}")
        return None


def generate_buggy_code_rule(
    correct_code: str,
    misconception_category: str,
) -> Optional[str]:
    """Generate buggy code using rule-based transformation.
    
    This is a fallback when LLM API is unavailable.
    Applies common error patterns to correct code.
    """
    if not correct_code:
        return None
    
    buggy_code = correct_code
    
    if misconception_category == "boundary_error":
        buggy_code = re.sub(r'<\s*(\w+)\.length', r'<= \1.length', buggy_code, count=1)
        if buggy_code == correct_code:
            buggy_code = re.sub(r'<=\s*(\w+)', r'< \1', buggy_code, count=1)
    
    elif misconception_category == "missing_case":
        buggy_code = re.sub(
            r'if\s*\(\s*\w+\s*==\s*null\s*\)\s*\{[^}]*\}',
            '', buggy_code, count=1
        )
        if buggy_code == correct_code:
            buggy_code = re.sub(
                r'if\s*\(\s*\w+\.length\s*==\s*0\s*\)\s*\{[^}]*\}',
                '', buggy_code, count=1
            )
    
    elif misconception_category == "logic_error":
        buggy_code = re.sub(r'>=', '>', buggy_code, count=1)
        if buggy_code == correct_code:
            buggy_code = re.sub(r'<=', '<', buggy_code, count=1)
    
    elif misconception_category == "loop_error":
        buggy_code = re.sub(r'\+\+', '', buggy_code, count=1)
        if buggy_code == correct_code:
            buggy_code = re.sub(r'i\s*<\s*', 'i <= ', buggy_code, count=1)
    
    elif misconception_category == "type_error":
        buggy_code = re.sub(r'\.equals\(', ' == ', buggy_code, count=1)
    
    elif misconception_category == "initialization_error":
        buggy_code = re.sub(r'=\s*0;', '= 1;', buggy_code, count=1)
    
    if buggy_code == correct_code:
        return None
    
    return buggy_code


# ==============================================================================
# Code Execution Engine
# ==============================================================================

class JavaExecutor:
    """Execute Java code against test cases."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute_method(
        self,
        method_code: str,
        test_cases: List[Dict],
        class_name: str = "Solution",
    ) -> List[Dict]:
        """Execute a Java method against test cases.
        
        Args:
            method_code: The Java method to test
            test_cases: List of dicts with 'input' and 'expected_output'
            class_name: Name for the wrapper class
            
        Returns:
            List of execution results per test case
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected = test_case.get('expected_output', '')
            
            java_code = self._wrap_in_class(method_code, test_input, class_name)
            
            result = self._compile_and_run(java_code, class_name)
            result['test_case_id'] = i
            result['expected'] = expected
            result['passed'] = (
                result.get('output', '').strip() == str(expected).strip() 
                if result.get('success', False) else False
            )
            results.append(result)
        
        return results
    
    def _wrap_in_class(self, method_code: str, test_input: str, class_name: str) -> str:
        """Wrap method code in a compilable Java class with main method."""
        return f"""
public class {class_name} {{
    {method_code}
    
    public static void main(String[] args) {{
        {class_name} sol = new {class_name}();
        System.out.println(sol.{self._extract_method_call(method_code, test_input)});
    }}
}}
"""
    
    def _extract_method_call(self, method_code: str, test_input: str) -> str:
        """Extract method name and format the call with test input."""
        match = re.search(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', method_code)
        if match:
            method_name = match.group(1)
            return f"{method_name}({test_input})"
        return f"solve({test_input})"
    
    def _compile_and_run(self, java_code: str, class_name: str) -> Dict:
        """Compile and run Java code in a temporary directory."""
        result = {'success': False, 'output': '', 'error': ''}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            java_file = os.path.join(tmpdir, f"{class_name}.java")
            with open(java_file, 'w') as f:
                f.write(java_code)
            
            try:
                compile_proc = subprocess.run(
                    ['javac', java_file],
                    capture_output=True, text=True, timeout=self.timeout
                )
                
                if compile_proc.returncode != 0:
                    result['error'] = f"Compilation error: {compile_proc.stderr}"
                    result['error_type'] = 'compilation'
                    return result
                
                run_proc = subprocess.run(
                    ['java', '-cp', tmpdir, class_name],
                    capture_output=True, text=True, timeout=self.timeout
                )
                
                if run_proc.returncode != 0:
                    result['error'] = f"Runtime error: {run_proc.stderr}"
                    result['error_type'] = 'runtime'
                else:
                    result['success'] = True
                    result['output'] = run_proc.stdout.strip()
                    
            except subprocess.TimeoutExpired:
                result['error'] = "Timeout"
                result['error_type'] = 'timeout'
            except FileNotFoundError:
                result['error'] = "Java compiler not found"
                result['error_type'] = 'environment'
        
        return result


# ==============================================================================
# Reasoning Verification Score (RVS) Computation
# ==============================================================================

def compute_rvs(
    reasoning_trace: str,
    actual_student_code: str,
    actual_score: int,
    test_cases: Optional[List[Dict]] = None,
    correct_code: Optional[str] = None,
    client: Optional['OpenAI'] = None,
) -> Dict:
    """Compute the Reasoning Verification Score (RVS) for a single prediction.
    
    RVS measures whether the model's reasoning about student misconceptions
    is consistent with actual execution evidence.
    
    Components:
    1. Prediction Consistency: Did the prediction match the actual outcome?
    2. Misconception Plausibility: Is the identified misconception consistent 
       with the observed failure pattern?
    3. Execution Verification: Does generated buggy code produce similar 
       failure patterns as the actual student code?
    """
    result = {
        'prediction_consistent': False,
        'misconception_extracted': False,
        'misconception_category': 'none',
        'execution_verified': False,
        'rvs_score': 0.0,
    }
    
    misconception = extract_misconception(reasoning_trace)
    
    if misconception:
        result['misconception_extracted'] = True
        result['misconception_text'] = misconception
        result['misconception_category'] = categorize_misconception(misconception)
    
    pred_match = re.search(r'Prediction:\s*(correct|wrong)', reasoning_trace, re.IGNORECASE)
    if pred_match:
        predicted = 1 if pred_match.group(1).lower() == 'correct' else 0
        result['prediction_consistent'] = (predicted == actual_score)
    
    score_components = []
    score_components.append(1.0 if result['prediction_consistent'] else 0.0)
    score_components.append(0.5 if result['misconception_extracted'] else 0.0)
    
    if test_cases and misconception and actual_score == 0:
        if correct_code:
            buggy_code = generate_buggy_code_rule(
                correct_code, result['misconception_category']
            )
            if buggy_code:
                executor = JavaExecutor()
                buggy_results = executor.execute_method(buggy_code, test_cases)
                student_results = executor.execute_method(actual_student_code, test_cases)
                
                buggy_failures = {r['test_case_id'] for r in buggy_results if not r['passed']}
                student_failures = {r['test_case_id'] for r in student_results if not r['passed']}
                
                if buggy_failures and student_failures:
                    overlap = len(buggy_failures & student_failures)
                    union = len(buggy_failures | student_failures)
                    failure_similarity = overlap / union if union > 0 else 0.0
                    result['execution_verified'] = failure_similarity > 0.3
                    result['failure_similarity'] = failure_similarity
                    score_components.append(failure_similarity)
    
    result['rvs_score'] = np.mean(score_components) if score_components else 0.0
    
    return result


def batch_compute_rvs(
    reasoning_results: List[Dict],
    student_codes: List[str],
    actual_scores: List[int],
    test_cases_dict: Optional[Dict] = None,
) -> Dict:
    """Compute RVS across a batch of predictions."""
    
    all_rvs = []
    category_counts = defaultdict(int)
    prediction_correct = 0
    misconception_found = 0
    execution_verified = 0
    total = 0
    
    for i, (result, code, score) in enumerate(
        zip(reasoning_results, student_codes, actual_scores)
    ):
        reasoning_trace = result.get('reasoning_trace', '') or result.get('raw_output', '')
        
        test_cases = None
        if test_cases_dict:
            problem_id = result.get('problem_id')
            if problem_id and problem_id in test_cases_dict:
                test_cases = test_cases_dict[problem_id]
        
        rvs_result = compute_rvs(
            reasoning_trace=reasoning_trace,
            actual_student_code=code,
            actual_score=score,
            test_cases=test_cases,
        )
        
        all_rvs.append(rvs_result['rvs_score'])
        
        if rvs_result['prediction_consistent']:
            prediction_correct += 1
        if rvs_result['misconception_extracted']:
            misconception_found += 1
            category_counts[rvs_result['misconception_category']] += 1
        if rvs_result['execution_verified']:
            execution_verified += 1
        
        total += 1
    
    return {
        'mean_rvs': float(np.mean(all_rvs)) if all_rvs else 0.0,
        'median_rvs': float(np.median(all_rvs)) if all_rvs else 0.0,
        'prediction_accuracy': prediction_correct / total if total > 0 else 0.0,
        'misconception_extraction_rate': misconception_found / total if total > 0 else 0.0,
        'execution_verification_rate': execution_verified / total if total > 0 else 0.0,
        'misconception_categories': dict(category_counts),
        'total_evaluated': total,
    }


# ==============================================================================
# Main Evaluation Pipeline
# ==============================================================================

def run_execution_verification(
    reasoning_results_path: str,
    data_path: str,
    test_cases_path: Optional[str] = None,
    output_path: str = "results/exec_verify_results.json",
):
    """Run the full execution verification pipeline."""
    
    print("Loading reasoning results...")
    with open(reasoning_results_path, 'r') as f:
        reasoning_data = json.load(f)
    
    test_cases_dict = None
    if test_cases_path and os.path.exists(test_cases_path):
        print("Loading test cases...")
        with open(test_cases_path, 'r') as f:
            test_cases_dict = json.load(f)
    
    print("Computing RVS...")
    
    if isinstance(reasoning_data, dict) and 'reasoning_trace' in str(reasoning_data):
        all_results = []
        all_codes = []
        all_scores = []
        
        for student_id, traces in reasoning_data.items():
            if isinstance(traces, list):
                for trace in traces:
                    all_results.append(trace)
                    all_codes.append(trace.get('actual_code', ''))
                    all_scores.append(trace.get('actual_score', 0))
        
        rvs_metrics = batch_compute_rvs(
            reasoning_results=all_results,
            student_codes=all_codes,
            actual_scores=all_scores,
            test_cases_dict=test_cases_dict,
        )
    else:
        rvs_metrics = {'error': 'Unsupported data format'}
    
    print(f"\n=== Execution Verification Results ===")
    print(f"Mean RVS: {rvs_metrics.get('mean_rvs', 0):.4f}")
    print(f"Prediction Accuracy: {rvs_metrics.get('prediction_accuracy', 0):.4f}")
    print(f"Misconception Extraction Rate: {rvs_metrics.get('misconception_extraction_rate', 0):.4f}")
    print(f"Execution Verification Rate: {rvs_metrics.get('execution_verification_rate', 0):.4f}")
    print(f"Misconception Categories: {rvs_metrics.get('misconception_categories', {})}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rvs_metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return rvs_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution Verification for Code-KT Reasoning")
    parser.add_argument("--reasoning_results", type=str, required=True,
                        help="Path to reasoning results JSON")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--test_cases", type=str, default=None,
                        help="Path to test cases JSON (from TIKTOC augmented CodeWorkout)")
    parser.add_argument("--output", type=str, default="results/exec_verify_results.json")
    
    args = parser.parse_args()
    
    run_execution_verification(
        reasoning_results_path=args.reasoning_results,
        data_path=args.data_path,
        test_cases_path=args.test_cases,
        output_path=args.output,
    )
