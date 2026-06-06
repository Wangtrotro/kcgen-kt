"""
Code-Specific Reasoning Prompt Templates for Programming Knowledge Tracing

This module provides structured prompt templates designed specifically for
programming education KT, incorporating code execution reasoning, misconception
identification, and KC gap analysis.

Templates are categorized into:
1. Training prompts (for Reasoning-KCGen-KT training)
2. Inference prompts (for Training-Free Thinking-Code-KT)
3. Execution verification prompts
"""

from typing import List, Dict, Optional


# ==============================================================================
# Training Prompt Templates (Method 1: Reasoning-Augmented KCGen-KT)
# ==============================================================================

def build_reasoning_prompt_with_special_tokens(prompt: str, kcs: List[str], 
                                                reasoning_trace: Optional[str] = None) -> str:
    """Build prompt with KC mastery tokens AND reasoning section.
    
    Extends the original build_prompt_with_special_tokens() from data_loader.py
    by inserting a reasoning section between KC mastery and code generation.
    """
    if ":" in prompt:
        prompt = prompt.replace(":", "")
    if "?" in prompt:
        prompt = prompt.replace("?", ".")
    
    assert "written" not in prompt
    
    result = "Question: " + prompt
    
    for i in range(len(kcs)):
        kc = kcs[i]
        kc_intro = f" KC {i+1}: {kc}."
        kc_level = f" The student's mastery level on {kc} is ?"
        result += kc_intro + kc_level
    
    if reasoning_trace:
        result += f" Reasoning: {reasoning_trace}"
    else:
        result += " Reasoning:"
    
    result += " Student written code:"
    
    return result


def build_reasoning_input_with_special_tokens(prompt: str, kcs: List[str], 
                                               code: str, tokenizer,
                                               reasoning_trace: Optional[str] = None) -> str:
    """Build full training input with reasoning trace included."""
    prompt_part = build_reasoning_prompt_with_special_tokens(prompt, kcs, reasoning_trace)
    return prompt_part + " " + code.strip() + tokenizer.eos_token


# ==============================================================================
# Inference Prompt Templates (Method 2: Training-Free Thinking-Code-KT)
# ==============================================================================

THINKING_CODE_KT_SYSTEM_PROMPT = """You are a programming education expert performing knowledge tracing. 
Analyze the student's programming history and predict whether they will solve the next problem correctly.

You MUST reason through these steps before making a prediction:
1. [CodeRead] What does the next problem require?
2. [PatternRecall] What relevant patterns has the student shown mastery/difficulty with?
3. [ExecTrace] What code would a student at this level likely write?
4. [MisconceptionID] What specific errors might they make?
5. [Verify] Is your reasoning consistent with the KC gap analysis?

After reasoning, output EXACTLY in this format:
Prediction: [correct/wrong]
Diagnosis: [one sentence explaining why]"""


THINKING_CODE_KT_USER_TEMPLATE = """Student's Programming History:

Problems attempted (recent, with outcomes):
{problem_history}

KC Mastery Profile:
{kc_mastery_profile}

Recent Code Submissions:
{code_submissions}

Score Sequence: {score_sequence}

---

Next Problem to Predict:
{next_problem}

Required KCs: {required_kcs}

Predict whether the student will solve this problem correctly. Think step by step using the [CodeRead] → [PatternRecall] → [ExecTrace] → [MisconceptionID] → [Verify] framework."""


THINKING_CODE_KT_NO_CODE_TEMPLATE = """Student's Programming History:

Problems attempted (recent, with outcomes):
{problem_history}

KC Mastery Profile:
{kc_mastery_profile}

Score Sequence: {score_sequence}

---

Next Problem to Predict:
{next_problem}

Required KCs: {required_kcs}

Predict whether the student will solve this problem correctly."""


def build_thinking_code_kt_prompt(
    problem_history: List[Dict],
    kc_mastery: Dict[str, float],
    next_problem: str,
    next_kcs: List[str],
    include_code: bool = True,
    max_history: int = 8,
) -> str:
    """Build the structured inference prompt for Thinking-Code-KT."""
    
    recent_history = problem_history[-max_history:]
    
    history_lines = []
    for i, item in enumerate(recent_history, 1):
        outcome = "PASS" if item.get('score', 0) == 1 else "FAIL"
        kcs_str = ", ".join(item.get('kcs', [])[:3])
        history_lines.append(f"{i}. [{outcome}] {item['prompt'][:80]}... (KCs: {kcs_str})")
    problem_history_str = "\n".join(history_lines)
    
    mastery_lines = []
    for kc, score in sorted(kc_mastery.items(), key=lambda x: -x[1]):
        level = "High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"
        mastery_lines.append(f"  - {kc}: {score:.2f} ({level})")
    kc_mastery_str = "\n".join(mastery_lines[:15])
    
    score_seq = " → ".join([str(item.get('score', 0)) for item in recent_history])
    required_kcs_str = ", ".join(next_kcs) if next_kcs else "Unknown"
    
    if include_code:
        code_lines = []
        code_items = [item for item in recent_history if item.get('code')][-3:]
        for i, item in enumerate(code_items, 1):
            outcome = "PASS" if item.get('score', 0) == 1 else "FAIL"
            code_snippet = item['code'][:200]
            code_lines.append(f"--- Submission {i} [{outcome}] ---\n{code_snippet}\n")
        code_str = "\n".join(code_lines)
        
        prompt = THINKING_CODE_KT_USER_TEMPLATE.format(
            problem_history=problem_history_str,
            kc_mastery_profile=kc_mastery_str,
            code_submissions=code_str,
            score_sequence=score_seq,
            next_problem=next_problem[:200],
            required_kcs=required_kcs_str,
        )
    else:
        prompt = THINKING_CODE_KT_NO_CODE_TEMPLATE.format(
            problem_history=problem_history_str,
            kc_mastery_profile=kc_mastery_str,
            score_sequence=score_seq,
            next_problem=next_problem[:200],
            required_kcs=required_kcs_str,
        )
    
    return prompt


# ==============================================================================
# Execution Verification Prompts (Method 3)
# ==============================================================================

EXEC_VERIFY_PROMPT = """Based on the following reasoning trace about a student's likely misconception, generate a short Java code snippet that demonstrates EXACTLY the predicted error.

Reasoning Trace:
{reasoning_trace}

Problem Description:
{problem_description}

Method Signature:
{method_signature}

Generate ONLY the buggy code that reflects the identified misconception. The code should:
1. Be syntactically valid Java
2. Contain exactly the predicted error (not other errors)
3. Be a plausible student submission

Output only the code, no explanation."""


def build_exec_verify_prompt(reasoning_trace: str, problem_desc: str, 
                              method_sig: str) -> str:
    """Build prompt for generating verification buggy code."""
    return EXEC_VERIFY_PROMPT.format(
        reasoning_trace=reasoning_trace,
        problem_description=problem_desc,
        method_signature=method_sig,
    )


# ==============================================================================
# Reasoning Episode Categories (for trace analysis)
# ==============================================================================

EPISODE_CATEGORIES = {
    "CodeRead": "Understanding the problem requirements and identifying needed constructs",
    "PatternRecall": "Connecting to programming patterns the student has/hasn't mastered",
    "ExecTrace": "Mentally executing the expected student code path",
    "MisconceptionID": "Identifying the specific error or misconception",
    "Verify": "Checking reasoning consistency with KC mastery data",
}

EPISODE_CLASSIFICATION_PROMPT = """Classify the following reasoning trace segment into one of these Code-Specific Episodes:

- CodeRead: Understanding the problem requirements (e.g., "This problem requires implementing a loop that iterates through an array...")
- PatternRecall: Connecting to programming patterns (e.g., "The student has shown mastery in for-loops but struggles with nested conditions...")
- ExecTrace: Mental code execution (e.g., "A student at this level would likely write a for loop but forget to handle the empty array case...")
- MisconceptionID: Identifying errors (e.g., "The student's likely misconception is using == instead of .equals() for string comparison...")
- Verify: Checking consistency (e.g., "This aligns with their low mastery on String manipulation KC...")

Reasoning Trace Segment:
{segment}

Output: Return only the label name (CodeRead, PatternRecall, ExecTrace, MisconceptionID, or Verify)."""


# ==============================================================================
# Prompt Configuration Presets
# ==============================================================================

PROMPT_CONFIGS = {
    "full": {
        "include_code": True,
        "include_kc_mastery": True,
        "max_history": 8,
        "thinking_budget": 1024,
        "description": "Full prompt with code history, KC mastery, and extended thinking",
    },
    "no_code": {
        "include_code": False,
        "include_kc_mastery": True,
        "max_history": 8,
        "thinking_budget": 512,
        "description": "No code snippets, KC mastery only",
    },
    "minimal": {
        "include_code": False,
        "include_kc_mastery": False,
        "max_history": 5,
        "thinking_budget": 256,
        "description": "Minimal prompt for ablation study",
    },
    "extended": {
        "include_code": True,
        "include_kc_mastery": True,
        "max_history": 15,
        "thinking_budget": 2048,
        "description": "Extended context for maximum reasoning depth",
    },
}
