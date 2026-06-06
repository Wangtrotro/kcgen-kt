"""
Reasoning Trace Generation for Code-KT

This module generates gold reasoning traces for programming knowledge tracing
using GPT-4o. Each reasoning trace explains WHY a student got a problem right/wrong
based on their KC mastery history and code submission patterns.

Reasoning traces follow the Code-Specific Episode categories:
- CodeRead: Understanding the problem requirements
- PatternRecall: Connecting to known programming patterns
- ExecTrace: Mental execution of expected student code
- MisconceptionID: Identifying likely errors
- Verify: Checking reasoning consistency
"""

import os
import json
import pickle
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")

import pandas as pd
import numpy as np
from tqdm import tqdm


REASONING_SYSTEM_PROMPT = """You are an expert programming education researcher specializing in knowledge tracing and student misconception analysis. 

Given a student's programming history (previous problems attempted, their code, and scores) and their Knowledge Component (KC) mastery levels, you will generate a detailed reasoning trace that explains WHY the student will likely succeed or fail on the next problem.

Your reasoning must follow this structured format with these Code-Specific Episodes:

1. [CodeRead] Analyze the next problem's requirements and identify which programming concepts are needed.
2. [PatternRecall] Based on the student's history, identify which relevant patterns they have demonstrated mastery of and which they have not.
3. [ExecTrace] Mentally trace what code a student at this mastery level would likely write. Consider their coding style and common approaches seen in their history.
4. [MisconceptionID] If the student is likely to fail, identify the specific misconception or error pattern (e.g., off-by-one, missing base case, incorrect loop bounds, wrong data type handling).
5. [Verify] Check if your reasoning is consistent — does the predicted misconception align with the student's KC gaps?

After the reasoning, provide:
- Prediction: "correct" or "wrong"
- Diagnosis: A brief explanation of the specific misconception (if wrong) or mastery evidence (if correct)

Be specific about code constructs and programming concepts. Reference actual code patterns from the student's history when relevant."""


REASONING_USER_PROMPT_TEMPLATE = """Student Programming History:

Problems attempted (most recent last):
{problem_history}

Student's KC mastery profile:
{kc_mastery}

Code submissions (most recent):
{code_history}

Score sequence: {score_sequence}

---

Next Problem:
{next_problem}

Required KCs for next problem: {next_kcs}

Student's actual outcome on next problem: {actual_outcome}
Student's actual code for next problem:
{actual_code}

---

Generate a detailed reasoning trace following the [CodeRead] → [PatternRecall] → [ExecTrace] → [MisconceptionID] → [Verify] structure. The reasoning should explain why the student achieved the actual outcome."""


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the CodeWorkout dataset."""
    df = pd.read_pickle(data_path)
    if 'Score_x' in df.columns:
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)
    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    return df


def load_kc_mapping(kc_path: str) -> Tuple[Dict, Dict]:
    """Load KC-to-problem mapping."""
    with open(kc_path, 'r') as f:
        kc_problem_dict = json.load(f)

    uniq_kcs = set()
    for key, val in kc_problem_dict.items():
        for kc in val:
            uniq_kcs.add(kc)
    uniq_kcs = list(uniq_kcs)
    kc_dict_res = {uniq_kcs[i]: i for i in range(len(uniq_kcs))}

    return kc_problem_dict, kc_dict_res


def build_student_history(df: pd.DataFrame, student_id: str, 
                          kc_problem_dict: Dict, max_history: int = 10) -> List[Dict]:
    """Build interaction history for a student."""
    student_df = df[df['SubjectID'] == student_id].sort_values('ServerTimestamp')
    
    interactions = []
    for _, row in student_df.iterrows():
        interaction = {
            'problem_id': row['ProblemID'],
            'prompt': row['prompt'],
            'code': row['Code'],
            'score': row['Score'],
            'kcs': kc_problem_dict.get(row['prompt'], []),
        }
        interactions.append(interaction)
    
    return interactions


def format_problem_history(interactions: List[Dict], max_show: int = 5) -> str:
    """Format problem history for the prompt."""
    recent = interactions[-max_show:] if len(interactions) > max_show else interactions
    lines = []
    for i, inter in enumerate(recent, 1):
        score_str = "PASS" if inter['score'] == 1 else "FAIL"
        kc_str = ", ".join(inter['kcs'][:3]) if inter['kcs'] else "Unknown"
        lines.append(f"{i}. [{score_str}] {inter['prompt'][:100]}... (KCs: {kc_str})")
    return "\n".join(lines)


def format_code_history(interactions: List[Dict], max_show: int = 3) -> str:
    """Format recent code submissions."""
    recent = interactions[-max_show:] if len(interactions) > max_show else interactions
    lines = []
    for i, inter in enumerate(recent, 1):
        code_snippet = inter['code'][:300] if inter['code'] else "No code"
        score_str = "PASS" if inter['score'] == 1 else "FAIL"
        lines.append(f"--- Submission {i} [{score_str}] ---\n{code_snippet}\n")
    return "\n".join(lines)


def estimate_kc_mastery(interactions: List[Dict], kc_problem_dict: Dict) -> Dict[str, float]:
    """Estimate KC mastery based on interaction history."""
    kc_attempts = defaultdict(list)
    
    for inter in interactions:
        for kc in inter['kcs']:
            kc_attempts[kc].append(inter['score'])
    
    kc_mastery = {}
    for kc, scores in kc_attempts.items():
        recent_scores = scores[-5:]
        weights = np.linspace(0.5, 1.0, len(recent_scores))
        kc_mastery[kc] = float(np.average(recent_scores, weights=weights))
    
    return kc_mastery


def generate_reasoning_trace(
    client: 'OpenAI',
    interactions: List[Dict],
    next_interaction: Dict,
    kc_problem_dict: Dict,
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> Optional[str]:
    """Generate a reasoning trace for a single (student, problem) pair."""
    
    if len(interactions) < 2:
        return None
    
    kc_mastery = estimate_kc_mastery(interactions, kc_problem_dict)
    
    kc_mastery_str = "\n".join([
        f"  - {kc}: {mastery:.2f}" for kc, mastery in sorted(kc_mastery.items(), key=lambda x: -x[1])
    ])
    
    problem_history_str = format_problem_history(interactions)
    code_history_str = format_code_history(interactions)
    score_sequence = [str(inter['score']) for inter in interactions]
    
    next_kcs = ", ".join(next_interaction['kcs']) if next_interaction['kcs'] else "Unknown"
    actual_outcome = "correct" if next_interaction['score'] == 1 else "wrong"
    
    user_prompt = REASONING_USER_PROMPT_TEMPLATE.format(
        problem_history=problem_history_str,
        kc_mastery=kc_mastery_str if kc_mastery_str else "  No KC mastery data available yet",
        code_history=code_history_str,
        score_sequence=" → ".join(score_sequence[-10:]),
        next_problem=next_interaction['prompt'][:200],
        next_kcs=next_kcs,
        actual_outcome=actual_outcome,
        actual_code=next_interaction['code'][:400] if next_interaction['code'] else "No code submitted",
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating reasoning trace: {e}")
        return None


def generate_traces_for_dataset(
    df: pd.DataFrame,
    kc_problem_dict: Dict,
    output_path: str,
    model: str = "gpt-4o",
    max_students: Optional[int] = None,
    min_history: int = 3,
):
    """Generate reasoning traces for the entire dataset."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    students = df['SubjectID'].unique()
    if max_students:
        students = students[:max_students]
    
    all_traces = {}
    total_generated = 0
    total_failed = 0
    
    for student_id in tqdm(students, desc="Generating reasoning traces"):
        interactions = build_student_history(df, student_id, kc_problem_dict)
        
        if len(interactions) < min_history + 1:
            continue
        
        student_traces = []
        
        for t in range(min_history, len(interactions)):
            history = interactions[:t]
            next_inter = interactions[t]
            
            trace = generate_reasoning_trace(
                client=client,
                interactions=history,
                next_interaction=next_inter,
                kc_problem_dict=kc_problem_dict,
                model=model,
            )
            
            if trace:
                student_traces.append({
                    'timestep': t,
                    'problem_id': next_inter['problem_id'],
                    'prompt': next_inter['prompt'],
                    'actual_score': next_inter['score'],
                    'reasoning_trace': trace,
                    'kc_mastery': estimate_kc_mastery(history, kc_problem_dict),
                })
                total_generated += 1
            else:
                total_failed += 1
        
        if student_traces:
            all_traces[student_id] = student_traces
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2)
    
    print(f"\nGeneration complete:")
    print(f"  Students processed: {len(all_traces)}")
    print(f"  Traces generated: {total_generated}")
    print(f"  Failed: {total_failed}")
    print(f"  Output saved to: {output_path}")
    
    return all_traces


def generate_condensed_reasoning(
    problem_prompt: str,
    kc_list: List[str],
    kc_mastery: Dict[str, float],
    score_history: List[int],
) -> str:
    """Generate a condensed reasoning template for training data augmentation.
    
    This version doesn't call GPT-4o but creates a rule-based reasoning trace
    that can be used when API access is unavailable.
    """
    weak_kcs = [kc for kc in kc_list if kc_mastery.get(kc, 0.5) < 0.6]
    strong_kcs = [kc for kc in kc_list if kc_mastery.get(kc, 0.5) >= 0.6]
    recent_trend = score_history[-5:] if len(score_history) >= 5 else score_history
    trend_score = sum(recent_trend) / len(recent_trend) if recent_trend else 0.5
    
    reasoning_parts = []
    
    reasoning_parts.append(
        f"[CodeRead] This problem requires: {', '.join(kc_list[:5])}."
    )
    
    if strong_kcs:
        reasoning_parts.append(
            f"[PatternRecall] Student has demonstrated mastery in: {', '.join(strong_kcs[:3])}."
        )
    if weak_kcs:
        reasoning_parts.append(
            f"[PatternRecall] Student struggles with: {', '.join(weak_kcs[:3])}."
        )
    
    if weak_kcs:
        reasoning_parts.append(
            f"[ExecTrace] Given low mastery on {weak_kcs[0]} "
            f"(mastery={kc_mastery.get(weak_kcs[0], 0.0):.2f}), "
            f"student may produce code with errors in this area."
        )
    else:
        reasoning_parts.append(
            f"[ExecTrace] Student has adequate mastery on all required KCs. "
            f"Expected to produce correct implementation."
        )
    
    if weak_kcs and trend_score < 0.7:
        reasoning_parts.append(
            f"[MisconceptionID] Likely misconception related to {weak_kcs[0]}. "
            f"Recent performance trend: {trend_score:.2f}."
        )
    else:
        reasoning_parts.append(
            f"[MisconceptionID] No strong misconception signal. "
            f"Recent performance trend: {trend_score:.2f}."
        )
    
    prediction = "correct" if (not weak_kcs or trend_score >= 0.7) else "wrong"
    reasoning_parts.append(
        f"[Verify] KC gap analysis {'consistent' if weak_kcs else 'shows no gaps'}. "
        f"Prediction: {prediction}."
    )
    
    return " ".join(reasoning_parts)


def batch_generate_condensed_traces(
    df: pd.DataFrame,
    kc_problem_dict: Dict,
    output_path: str,
    min_history: int = 3,
):
    """Generate condensed (rule-based) reasoning traces for all interactions.
    
    This is a fallback when GPT-4o API is unavailable. It creates structured
    reasoning traces based on KC mastery patterns that can be used for training.
    """
    students = df['SubjectID'].unique()
    all_traces = {}
    total = 0
    
    for student_id in tqdm(students, desc="Generating condensed traces"):
        interactions = build_student_history(df, student_id, kc_problem_dict)
        
        if len(interactions) < min_history + 1:
            continue
        
        student_traces = []
        
        for t in range(min_history, len(interactions)):
            history = interactions[:t]
            next_inter = interactions[t]
            
            kc_mastery = estimate_kc_mastery(history, kc_problem_dict)
            score_history = [inter['score'] for inter in history]
            
            trace = generate_condensed_reasoning(
                problem_prompt=next_inter['prompt'],
                kc_list=next_inter['kcs'],
                kc_mastery=kc_mastery,
                score_history=score_history,
            )
            
            student_traces.append({
                'timestep': t,
                'problem_id': next_inter['problem_id'],
                'prompt': next_inter['prompt'],
                'actual_score': next_inter['score'],
                'reasoning_trace': trace,
                'kc_mastery': kc_mastery,
            })
            total += 1
        
        if student_traces:
            all_traces[student_id] = student_traces
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2)
    
    print(f"\nCondensed trace generation complete:")
    print(f"  Students: {len(all_traces)}")
    print(f"  Traces: {total}")
    print(f"  Output: {output_path}")
    
    return all_traces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning traces for Code-KT")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl",
                        help="Path to CodeWorkout dataset pickle")
    parser.add_argument("--kc_path", type=str, required=True,
                        help="Path to KC-problem mapping JSON")
    parser.add_argument("--output", type=str, default="data/reasoning_traces.json",
                        help="Output path for reasoning traces")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use")
    parser.add_argument("--max_students", type=int, default=None,
                        help="Max number of students to process (None for all)")
    parser.add_argument("--min_history", type=int, default=3,
                        help="Minimum interaction history before generating traces")
    parser.add_argument("--mode", type=str, default="condensed",
                        choices=["full", "condensed"],
                        help="'full' uses GPT-4o, 'condensed' uses rule-based generation")
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    df = load_dataset(args.data_path)
    
    print("Loading KC mapping...")
    kc_problem_dict, kc_dict_res = load_kc_mapping(args.kc_path)
    
    df['knowledge_component'] = df['prompt'].map(kc_problem_dict)
    
    if args.mode == "full":
        print("Generating full reasoning traces with GPT-4o...")
        generate_traces_for_dataset(
            df=df,
            kc_problem_dict=kc_problem_dict,
            output_path=args.output,
            model=args.model,
            max_students=args.max_students,
            min_history=args.min_history,
        )
    else:
        print("Generating condensed (rule-based) reasoning traces...")
        batch_generate_condensed_traces(
            df=df,
            kc_problem_dict=kc_problem_dict,
            output_path=args.output,
            min_history=args.min_history,
        )
