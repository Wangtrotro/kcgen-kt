"""
Path-level KC Generation
=========================
For each solution path (cluster) per problem, generate path-specific KCs
using GPT-4o. This produces a path-conditioned Q-matrix where different
paths activate different KC subsets.

Usage:
    python path_kc_gen.py [--path_dir path_cache] [--model gpt-4o]
"""

import os
import json
import argparse
import openai
from collections import defaultdict


def generate_path_kcs(problem_text, path_codes, path_id, model='gpt-4o', temperature=0):
    """
    Generate KCs specific to a single solution path of a problem.
    
    Args:
        problem_text: the problem description
        path_codes: list of representative code solutions for this path
        path_id: identifier for this path
        model: OpenAI model to use
    
    Returns:
        list of KC names specific to this path
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    code_section = "\n\n".join(
        [f"## Solution {i+1} (Path {path_id}):\n{code}" for i, code in enumerate(path_codes)]
    )

    system_content = f"""You are an experienced computer science teacher. You are given a programming problem along with {len(path_codes)} student solutions that all follow the SAME algorithmic approach/strategy (they belong to the same solution path cluster).

Your task is to identify the specific knowledge components (KCs) that are DEMONSTRATED by this particular solution approach. Focus on:
1. What specific programming constructs and patterns does THIS approach use?
2. What algorithmic strategy does THIS approach employ?
3. What conceptual understanding is required to come up with THIS specific solution?

Important: Only identify KCs that are ACTUALLY used in these solutions. Different solution approaches to the same problem will activate different KCs.

Return your response as a JSON object:
{{
    "path_strategy": "Brief description of the algorithmic strategy used in this path",
    "kcs": [
        {{"name": "KC name", "reasoning": "Why this KC is needed for THIS approach"}},
        ...
    ]
}}"""

    user_prompt = f"# Problem:\n{problem_text}\n\n# Solutions following this approach:\n{code_section}\n\nIdentify the KCs specific to this solution approach."

    try:
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            n=1
        )
        reply = json.loads(response.choices[0].message.content.strip())
        kc_names = [kc['name'] for kc in reply.get('kcs', [])]
        strategy = reply.get('path_strategy', f'path_{path_id}')
        return kc_names, strategy
    except Exception as e:
        print(f"Error generating KCs for path {path_id}: {e}")
        return [], f"path_{path_id}"


def generate_all_path_kcs(path_representatives, problem_texts, model='gpt-4o', temperature=0):
    """
    Generate path-level KCs for all problems and all paths.
    
    Args:
        path_representatives: dict {problem_id: {path_id: [codes]}}
        problem_texts: dict {problem_id: problem_text}
        model: OpenAI model
    
    Returns:
        path_kc_map: dict {problem_id: {path_id: [kc_names]}}
        path_strategies: dict {problem_id: {path_id: strategy_description}}
        all_kcs: set of all unique KC names
    """
    path_kc_map = {}
    path_strategies = {}
    all_kcs = set()

    for pid, paths in path_representatives.items():
        if pid not in problem_texts:
            continue

        problem_text = problem_texts[pid]
        path_kc_map[pid] = {}
        path_strategies[pid] = {}

        for path_id, codes in paths.items():
            if not codes:
                continue
            kc_names, strategy = generate_path_kcs(
                problem_text, codes, path_id, model=model, temperature=temperature
            )
            path_kc_map[pid][path_id] = kc_names
            path_strategies[pid][path_id] = strategy
            all_kcs.update(kc_names)

            print(f"  Problem {pid}, Path {path_id}: {len(kc_names)} KCs, strategy='{strategy[:50]}...'")

    return path_kc_map, path_strategies, all_kcs


def build_path_kc_activation_matrix(path_kc_map, all_kcs):
    """
    Build the path-conditioned KC activation matrix.
    
    For each (problem, path) pair, produces a binary vector indicating
    which KCs are activated.
    
    Args:
        path_kc_map: dict {problem_id: {path_id: [kc_names]}}
        all_kcs: ordered list of all unique KC names
    
    Returns:
        activation_matrix: dict {problem_id: {path_id: list[0/1] of len |all_kcs|}}
        kc_index: dict {kc_name: index}
    """
    kc_list = sorted(list(all_kcs))
    kc_index = {kc: i for i, kc in enumerate(kc_list)}

    activation_matrix = {}
    for pid, paths in path_kc_map.items():
        activation_matrix[pid] = {}
        for path_id, kc_names in paths.items():
            vec = [0] * len(kc_list)
            for kc in kc_names:
                if kc in kc_index:
                    vec[kc_index[kc]] = 1
            activation_matrix[pid][path_id] = vec

    return activation_matrix, kc_index


def save_path_kc_results(path_kc_map, path_strategies, kc_index, activation_matrix, output_dir="path_cache"):
    """Save path-level KC generation results."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "path_kc_map.json"), "w") as f:
        json.dump(path_kc_map, f, indent=2)

    with open(os.path.join(output_dir, "path_strategies.json"), "w") as f:
        json.dump(path_strategies, f, indent=2)

    with open(os.path.join(output_dir, "kc_index.json"), "w") as f:
        json.dump(kc_index, f, indent=2)

    # Convert int keys for JSON
    act_save = {}
    for pid, paths in activation_matrix.items():
        act_save[str(pid)] = {str(k): v for k, v in paths.items()}
    with open(os.path.join(output_dir, "path_kc_activation.json"), "w") as f:
        json.dump(act_save, f)

    print(f"Path KC results saved to {output_dir}/")
    print(f"  Total unique KCs: {len(kc_index)}")
    print(f"  Problems with path KCs: {len(path_kc_map)}")


def load_path_kc_results(output_dir="path_cache"):
    """Load path KC results from disk."""
    with open(os.path.join(output_dir, "path_kc_map.json"), "r") as f:
        path_kc_map = json.load(f)
    # Convert string keys back to int
    path_kc_map = {int(k): {int(pk): v for pk, v in paths.items()} for k, paths in path_kc_map.items()}

    with open(os.path.join(output_dir, "kc_index.json"), "r") as f:
        kc_index = json.load(f)

    with open(os.path.join(output_dir, "path_kc_activation.json"), "r") as f:
        act_raw = json.load(f)
    activation_matrix = {int(k): {int(pk): v for pk, v in paths.items()} for k, paths in act_raw.items()}

    return path_kc_map, kc_index, activation_matrix


def fallback_path_kc_from_existing(problem_kc_file, path_representatives):
    """
    Fallback: if no OpenAI key available, split existing problem-level KCs
    heuristically across paths (for testing the pipeline without API calls).
    
    Strategy: all paths share the base KCs, but we assign subset emphasis
    using simple round-robin to simulate path-specific activation.
    """
    with open(problem_kc_file, 'r') as f:
        problem_kc = json.load(f)

    # problem_kc: {problem_text: [[kc_desc, category], ...]}
    # We need to map problem_text → problem_id somehow
    # For now, return a structure keyed by problem text

    path_kc_map = {}
    all_kcs = set()

    for problem_text, kc_pairs in problem_kc.items():
        kc_names = [pair[0] if isinstance(pair, list) else pair for pair in kc_pairs]
        all_kcs.update(kc_names)

        # Heuristic split: distribute KCs across paths
        # Path 0 gets all KCs (base case)
        # Path 1 gets first half + some unique
        # Path 2 gets second half + some unique
        n_kcs = len(kc_names)
        path_kc_map[problem_text] = {
            0: kc_names,  # all paths share base KCs
            1: kc_names[:max(1, n_kcs * 2 // 3)],
            2: kc_names[max(0, n_kcs // 3):],
        }

    return path_kc_map, all_kcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate path-level KCs")
    parser.add_argument("--path_dir", type=str, default="path_cache")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--fallback", action="store_true",
                        help="Use fallback heuristic instead of LLM (no API key needed)")
    parser.add_argument("--existing_kc", type=str, default="problem_kc.json",
                        help="Existing problem_kc.json for fallback mode")
    args = parser.parse_args()

    if args.fallback:
        print("Using fallback heuristic (no LLM calls)...")
        from path_discovery import load_path_results
        _, _, path_representatives = load_path_results(args.path_dir)
        path_kc_map, all_kcs = fallback_path_kc_from_existing(args.existing_kc, path_representatives)
        kc_list = sorted(list(all_kcs))
        kc_index = {kc: i for i, kc in enumerate(kc_list)}
        activation_matrix, kc_index = build_path_kc_activation_matrix(
            # Need to adapt since fallback uses text keys
            {i: {0: kc_list, 1: kc_list[:len(kc_list)//2], 2: kc_list[len(kc_list)//2:]}
             for i in range(50)},
            all_kcs
        )
        print(f"Fallback: {len(all_kcs)} unique KCs")
        save_path_kc_results(path_kc_map, {}, kc_index, activation_matrix, args.path_dir)
    else:
        import pandas as pd
        from path_discovery import load_path_results

        print("Loading path discovery results...")
        _, _, path_representatives = load_path_results(args.path_dir)

        print("Loading dataset for problem texts...")
        df = pd.read_pickle(args.data_path)
        problem_texts = df.drop_duplicates('ProblemID').set_index('ProblemID')['prompt'].to_dict()

        print(f"Generating path-level KCs with {args.model}...")
        path_kc_map, path_strategies, all_kcs = generate_all_path_kcs(
            path_representatives, problem_texts, model=args.model
        )

        activation_matrix, kc_index = build_path_kc_activation_matrix(path_kc_map, all_kcs)
        save_path_kc_results(path_kc_map, path_strategies, kc_index, activation_matrix, args.path_dir)
