"""
Path Discovery Module
=====================
For each problem, cluster correct submissions to discover solution paths,
then assign path IDs to all student submissions (correct and incorrect).

Usage:
    python path_discovery.py [--n_paths 3] [--data_path data/dataset_time.pkl]
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


def get_code_embeddings(code_list, model_name="microsoft/graphcodebert-base", batch_size=32):
    """Encode a list of code strings into normalized embeddings using GraphCodeBERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.last_hidden_state
        mask_expand = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expand).sum(dim=1)
        sum_mask = mask_expand.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        all_embeddings.append(mean_embeddings.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return normalize(embeddings)


def cluster_solutions(embeddings, n_clusters):
    """Cluster code embeddings into n solution paths."""
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings))

    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int)

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def assign_path_soft(embedding, prototypes):
    """Soft-assign a submission to path prototypes via cosine similarity."""
    sims = cosine_similarity(embedding.reshape(1, -1), prototypes)[0]
    sims = np.clip(sims, 0, None)
    total = sims.sum()
    if total < 1e-9:
        return np.ones(len(prototypes)) / len(prototypes)
    return sims / total


def discover_paths(df, n_paths=3, model_name="microsoft/graphcodebert-base",
                   cache_dir="path_cache", train_students=None):
    """
    Main path discovery pipeline.
    
    Args:
        df: DataFrame with columns [SubjectID, ProblemID, prompt, Code, Score/Score_x]
        n_paths: number of solution paths per problem
        model_name: code embedding model
        cache_dir: directory to cache embeddings
        train_students: set of student IDs to use for prototype learning (train split only)
    
    Returns:
        path_prototypes: dict {problem_id: np.array of shape (n_paths, embed_dim)}
        path_assignments: dict {(subject_id, problem_id): {'hard': int, 'soft': np.array}}
        path_representatives: dict {problem_id: {path_id: [representative_codes]}}
    """
    os.makedirs(cache_dir, exist_ok=True)

    score_col = 'Score_x' if 'Score_x' in df.columns else 'Score'
    problems = df['ProblemID'].unique()

    path_prototypes = {}
    path_assignments = {}
    path_representatives = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    for pid in problems:
        problem_df = df[df['ProblemID'] == pid]

        # Only use training students for building prototypes
        if train_students is not None:
            train_df = problem_df[problem_df['SubjectID'].isin(train_students)]
        else:
            train_df = problem_df

        correct_train = train_df[train_df[score_col] >= 1.0]

        if len(correct_train) < 2:
            # Too few correct submissions - single path
            path_prototypes[pid] = None
            for _, row in problem_df.iterrows():
                path_assignments[(row['SubjectID'], pid)] = {'hard': 0, 'soft': np.array([1.0])}
            path_representatives[pid] = {0: correct_train['Code'].tolist()[:3]}
            continue

        # Get embeddings for correct training submissions
        cache_file = os.path.join(cache_dir, f"embeddings_p{pid}.npy")
        code_list = correct_train['Code'].tolist()

        if os.path.exists(cache_file) and len(np.load(cache_file)) == len(code_list):
            embeddings = np.load(cache_file)
        else:
            embeddings = _encode_batch(code_list, tokenizer, model)
            np.save(cache_file, embeddings)

        # Cluster into paths
        k = min(n_paths, len(code_list))
        labels = cluster_solutions(embeddings, k)

        # Compute prototypes (cluster centroids)
        actual_k = len(set(labels))
        centroids = np.zeros((actual_k, embeddings.shape[1]))
        for c in range(actual_k):
            mask = labels == c
            centroids[c] = embeddings[mask].mean(axis=0)
        centroids = normalize(centroids)
        path_prototypes[pid] = centroids

        # Store representative codes per path
        path_representatives[pid] = {}
        for c in range(actual_k):
            mask = labels == c
            cluster_codes = [code_list[i] for i in range(len(code_list)) if mask[i]]
            # Pick up to 3 representatives (closest to centroid)
            cluster_embs = embeddings[mask]
            dists = 1 - cosine_similarity(cluster_embs, centroids[c:c+1]).flatten()
            top_idx = np.argsort(dists)[:3]
            path_representatives[pid][c] = [cluster_codes[i] for i in top_idx]

        # Assign paths to training correct submissions
        train_sids = correct_train['SubjectID'].tolist()
        for idx, (sid, label) in enumerate(zip(train_sids, labels)):
            soft = assign_path_soft(embeddings[idx], centroids)
            path_assignments[(sid, pid)] = {'hard': int(label), 'soft': soft}

        # Assign paths to ALL other submissions (including incorrect and test students)
        other_df = problem_df[~((problem_df['SubjectID'].isin(correct_train['SubjectID'])) &
                                (problem_df[score_col] >= 1.0))]

        if len(other_df) > 0:
            other_codes = other_df['Code'].tolist()
            other_embs = _encode_batch(other_codes, tokenizer, model)
            other_sids = other_df['SubjectID'].tolist()

            for idx, sid in enumerate(other_sids):
                soft = assign_path_soft(other_embs[idx], centroids)
                hard = int(np.argmax(soft))
                path_assignments[(sid, pid)] = {'hard': hard, 'soft': soft}

    return path_prototypes, path_assignments, path_representatives


def _encode_batch(code_list, tokenizer, model, batch_size=32):
    """Encode code list with pre-loaded model."""
    all_embeddings = []
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.last_hidden_state
        mask_expand = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expand).sum(dim=1)
        sum_mask = mask_expand.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        all_embeddings.append(mean_embeddings.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return normalize(embeddings)


def save_path_results(path_prototypes, path_assignments, path_representatives, output_dir="path_cache"):
    """Save path discovery results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save prototypes
    proto_save = {}
    for pid, proto in path_prototypes.items():
        proto_save[int(pid)] = proto.tolist() if proto is not None else None
    with open(os.path.join(output_dir, "path_prototypes.json"), "w") as f:
        json.dump(proto_save, f)

    # Save assignments (convert tuple keys to string)
    assign_save = {}
    for (sid, pid), val in path_assignments.items():
        key = f"{sid}||{pid}"
        assign_save[key] = {'hard': val['hard'], 'soft': val['soft'].tolist()}
    with open(os.path.join(output_dir, "path_assignments.json"), "w") as f:
        json.dump(assign_save, f)

    # Save representatives
    repr_save = {}
    for pid, paths in path_representatives.items():
        repr_save[int(pid)] = {int(k): v for k, v in paths.items()}
    with open(os.path.join(output_dir, "path_representatives.json"), "w") as f:
        json.dump(repr_save, f)

    print(f"Path discovery results saved to {output_dir}/")


def load_path_results(output_dir="path_cache"):
    """Load previously saved path discovery results."""
    with open(os.path.join(output_dir, "path_prototypes.json"), "r") as f:
        proto_raw = json.load(f)
    path_prototypes = {}
    for pid, proto in proto_raw.items():
        path_prototypes[int(pid)] = np.array(proto) if proto is not None else None

    with open(os.path.join(output_dir, "path_assignments.json"), "r") as f:
        assign_raw = json.load(f)
    path_assignments = {}
    for key, val in assign_raw.items():
        sid, pid = key.split("||")
        path_assignments[(sid, int(pid))] = {'hard': val['hard'], 'soft': np.array(val['soft'])}

    with open(os.path.join(output_dir, "path_representatives.json"), "r") as f:
        repr_raw = json.load(f)
    path_representatives = {}
    for pid, paths in repr_raw.items():
        path_representatives[int(pid)] = {int(k): v for k, v in paths.items()}

    return path_prototypes, path_assignments, path_representatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover solution paths per problem")
    parser.add_argument("--n_paths", type=int, default=3, help="Number of solution paths per problem")
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--model_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--output_dir", type=str, default="path_cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset...")
    df = pd.read_pickle(args.data_path)

    # Use same train/test split logic as main pipeline
    from sklearn.model_selection import train_test_split

    if 'Score_x' in df.columns:
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)

    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df_dedup = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    students = df_dedup['SubjectID'].unique()
    train_stu, test_stu = train_test_split(students, test_size=0.2, random_state=args.seed)

    print(f"Total students: {len(students)}, Train: {len(train_stu)}, Test: {len(test_stu)}")
    print(f"Discovering {args.n_paths} paths per problem...")

    path_prototypes, path_assignments, path_representatives = discover_paths(
        df_dedup,
        n_paths=args.n_paths,
        model_name=args.model_name,
        cache_dir=args.output_dir,
        train_students=set(train_stu)
    )

    save_path_results(path_prototypes, path_assignments, path_representatives, args.output_dir)

    # Print summary statistics
    n_problems = len(path_prototypes)
    n_assigned = len(path_assignments)
    path_counts = defaultdict(int)
    for pid, proto in path_prototypes.items():
        if proto is not None:
            path_counts[len(proto)] += 1
        else:
            path_counts[1] += 1

    print(f"\n=== Path Discovery Summary ===")
    print(f"Problems processed: {n_problems}")
    print(f"Submissions assigned: {n_assigned}")
    print(f"Path count distribution: {dict(path_counts)}")
