"""
Lightweight Path-aware KT Ablation (CPU-compatible)
====================================================
Tests the core hypothesis WITHOUT the full Llama-3 pipeline.
Uses only LSTM + linear predictor for correctness prediction.

This is a focused test: does path-conditioned KC activation improve
the LSTM's ability to predict student correctness?

Three conditions:
  1. baseline:    LSTM predicts correctness with all KCs activated equally
  2. real_path:   LSTM predicts correctness with path-conditioned KC activation
  3. random_path: Same as real_path but path IDs are randomly shuffled (ablation control)

Usage:
    python run_lightweight_ablation.py [--epochs 10] [--n_paths 3]
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from collections import defaultdict
from tqdm import tqdm
import argparse
import random
from copy import deepcopy


class PathAwareLSTM(nn.Module):
    """
    LSTM-based knowledge tracer with path-conditioned KC activation.
    Input: student history features (prompt embedding + ASTNN = 4296d)
    Output: per-KC mastery (sigmoid), then aggregated to correctness prediction
    """
    def __init__(self, input_dim, n_kcs, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.kc_head = nn.Linear(hidden_dim, n_kcs)
        self.predictor = nn.Linear(n_kcs, 1)
        self.n_kcs = n_kcs

    def forward(self, x, kc_indices, path_weights=None):
        """
        Args:
            x: (B, T, input_dim) - student feature sequence
            kc_indices: (B, T, max_kc) - KC indices for each timestep (-1 for padding)
            path_weights: (B, T, max_kc) - path activation weights (1.0 = active, 0.0 = inactive)
        Returns:
            predictions: (B, T) - correctness predictions
            kc_mastery: (B, T, max_kc) - per-KC mastery levels
        """
        B, T, _ = x.shape
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)
        all_kc_mastery = torch.sigmoid(self.kc_head(lstm_out))  # (B, T, n_kcs)

        # Gather relevant KC mastery for each timestep
        padding_mask = kc_indices == -1
        safe_indices = kc_indices.clone()
        safe_indices[padding_mask] = 0

        # Gather per-problem KC mastery
        kc_mastery = torch.gather(all_kc_mastery, 2, safe_indices.long())  # (B, T, max_kc)
        kc_mastery[padding_mask] = 0.0

        # Apply path conditioning
        if path_weights is not None:
            kc_mastery = kc_mastery * path_weights
            # Re-mask padded positions
            kc_mastery[padding_mask] = 0.0

        # Aggregate to prediction: mean of active KC mastery values
        active_count = (~padding_mask).float().sum(dim=-1).clamp(min=1)  # (B, T)
        if path_weights is not None:
            active_count = (path_weights * (~padding_mask).float()).sum(dim=-1).clamp(min=1)

        mastery_mean = kc_mastery.sum(dim=-1) / active_count  # (B, T)

        return mastery_mean, kc_mastery


def prepare_data(data_path, path_dir, kc_file, seed=0):
    """Load and prepare data for lightweight experiment."""
    df = pd.read_pickle(data_path)

    if 'Score_x' in df.columns:
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)
    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    # Load KCs
    with open(kc_file, 'r') as f:
        problem_kc_raw = json.load(f)

    # Build KC index from problem_kc.json (values are [[desc, category], ...])
    all_kcs = set()
    for kcs in problem_kc_raw.values():
        for kc in kcs:
            name = kc[0] if isinstance(kc, list) else kc
            all_kcs.add(name)
    kc_list = sorted(list(all_kcs))
    kc_index = {kc: i for i, kc in enumerate(kc_list)}

    # Map problem text → KC indices
    prompt_to_kc_indices = {}
    for prompt, kcs in problem_kc_raw.items():
        indices = [kc_index[kc[0] if isinstance(kc, list) else kc] for kc in kcs]
        prompt_to_kc_indices[prompt] = indices

    df['kc_indices'] = df['prompt'].map(prompt_to_kc_indices)

    # Load path assignments
    with open(os.path.join(path_dir, "path_assignments.json")) as f:
        path_raw = json.load(f)
    path_map = {}
    for key, val in path_raw.items():
        sid, pid = key.split("||")
        try:
            pid = int(pid)
        except ValueError:
            pass
        path_map[(sid, pid)] = val['hard']

    df['path_id'] = df.apply(lambda r: path_map.get((r['SubjectID'], r['ProblemID']), 0), axis=1)

    # Split students
    students = df['SubjectID'].unique()
    train_stu, test_stu = train_test_split(students, test_size=0.2, random_state=seed)
    valid_stu, test_stu = train_test_split(test_stu, test_size=0.5, random_state=seed)

    return df, kc_index, kc_list, train_stu, valid_stu, test_stu


def build_sequences(df, students, max_len=20):
    """Build student sequences for training."""
    sequences = []
    for sid in students:
        student_df = df[df['SubjectID'] == sid].sort_values('ServerTimestamp')
        if len(student_df) < 2:
            continue

        # Chunk into subsequences
        for start in range(0, len(student_df), max_len):
            chunk = student_df.iloc[start:start + max_len]
            if len(chunk) < 2:
                continue
            sequences.append({
                'inputs': [row['input'] for _, row in chunk.iterrows()],
                'scores': [row['Score'] for _, row in chunk.iterrows()],
                'kc_indices': [row['kc_indices'] for _, row in chunk.iterrows()],
                'path_ids': [row['path_id'] for _, row in chunk.iterrows()],
                'problem_ids': [row['ProblemID'] for _, row in chunk.iterrows()],
            })
    return sequences


def collate_sequences(batch, n_paths=3, randomize_paths=False, path_kc_weights_map=None):
    """Collate sequences into tensors."""
    B = len(batch)
    max_t = max(len(b['scores']) for b in batch)
    max_kc = max(len(kc) for b in batch for kc in b['kc_indices'])
    input_dim = len(batch[0]['inputs'][0]) if isinstance(batch[0]['inputs'][0], (list, np.ndarray)) else batch[0]['inputs'][0].shape[0]

    inputs = torch.zeros(B, max_t, input_dim)
    scores = torch.full((B, max_t), -1.0)
    kc_indices = torch.full((B, max_t, max_kc), -1, dtype=torch.long)
    path_weights = torch.ones(B, max_t, max_kc)

    for b_idx, b in enumerate(batch):
        T = len(b['scores'])
        for t in range(T):
            inp = b['inputs'][t]
            if isinstance(inp, torch.Tensor):
                inputs[b_idx, t] = inp
            else:
                inputs[b_idx, t] = torch.tensor(inp, dtype=torch.float)

            scores[b_idx, t] = b['scores'][t]

            kcs = b['kc_indices'][t]
            if kcs is not None:
                for k_idx, kc in enumerate(kcs[:max_kc]):
                    kc_indices[b_idx, t, k_idx] = kc

            # Path weights: for now, use uniform (all 1) as baseline
            # In path-aware mode, could weight down non-path KCs
            path_id = b['path_ids'][t]
            if randomize_paths:
                path_id = random.randint(0, n_paths - 1)

            # Simple path weighting: path 0 emphasizes first third of KCs,
            # path 1 middle, path 2 last third
            if kcs is not None and len(kcs) > 0:
                n_kc = len(kcs)
                for k_idx in range(min(n_kc, max_kc)):
                    # Heuristic path-based weight
                    segment = k_idx * n_paths // n_kc
                    if segment == path_id:
                        path_weights[b_idx, t, k_idx] = 1.0
                    else:
                        path_weights[b_idx, t, k_idx] = 0.3  # down-weight non-path KCs

    return inputs, scores, kc_indices, path_weights


def train_epoch(model, sequences, optimizer, loss_fn, batch_size=16, mode='baseline',
                n_paths=3, randomize_paths=False):
    """Train for one epoch."""
    model.train()
    random.shuffle(sequences)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        use_path = mode != 'baseline'

        inputs, scores, kc_indices, path_weights = collate_sequences(
            batch, n_paths=n_paths, randomize_paths=(mode == 'random_path')
        )

        # Use path weights only in path-aware modes
        pw = path_weights if use_path else None

        predictions, _ = model(inputs, kc_indices, path_weights=pw)

        # Mask: only compute loss where scores are valid (not padding)
        # Use t >= 1 (predict next from history)
        valid_mask = scores[:, 1:] >= 0
        pred_shifted = predictions[:, :-1]  # predict t from history up to t-1
        score_shifted = scores[:, 1:]

        if valid_mask.sum() == 0:
            continue

        loss = loss_fn(pred_shifted[valid_mask], score_shifted[valid_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * valid_mask.sum().item()
        total_count += valid_mask.sum().item()

        preds_np = pred_shifted[valid_mask].detach().numpy()
        labels_np = score_shifted[valid_mask].detach().numpy()
        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels_np.tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(np.array(all_labels) > 0.5, np.array(all_preds) > 0.5)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, acc, auc


def evaluate(model, sequences, loss_fn, batch_size=16, mode='baseline', n_paths=3):
    """Evaluate on a set of sequences."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            use_path = mode != 'baseline'

            inputs, scores, kc_indices, path_weights = collate_sequences(
                batch, n_paths=n_paths, randomize_paths=(mode == 'random_path')
            )

            pw = path_weights if use_path else None
            predictions, _ = model(inputs, kc_indices, path_weights=pw)

            valid_mask = scores[:, 1:] >= 0
            pred_shifted = predictions[:, :-1]
            score_shifted = scores[:, 1:]

            if valid_mask.sum() == 0:
                continue

            loss = loss_fn(pred_shifted[valid_mask], score_shifted[valid_mask])
            total_loss += loss.item() * valid_mask.sum().item()
            total_count += valid_mask.sum().item()

            all_preds.extend(pred_shifted[valid_mask].numpy().tolist())
            all_labels.extend(score_shifted[valid_mask].numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(np.array(all_labels) > 0.5, np.array(all_preds) > 0.5)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    f1 = f1_score(np.array(all_labels) > 0.5, np.array(all_preds) > 0.5, zero_division=0)

    return avg_loss, acc, auc, f1


def run_experiment(mode, sequences_train, sequences_valid, sequences_test,
                   input_dim, n_kcs, args):
    """Run a single experiment configuration."""
    print(f"\n{'─'*50}")
    print(f"  Mode: {mode}")
    print(f"{'─'*50}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = PathAwareLSTM(input_dim, n_kcs, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    best_valid_loss = float('inf')
    best_test_metrics = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc = train_epoch(
            model, sequences_train, optimizer, loss_fn,
            batch_size=args.batch_size, mode=mode, n_paths=args.n_paths,
            randomize_paths=(mode == 'random_path')
        )
        valid_loss, valid_acc, valid_auc, valid_f1 = evaluate(
            model, sequences_valid, loss_fn,
            batch_size=args.batch_size, mode=mode, n_paths=args.n_paths
        )
        test_loss, test_acc, test_auc, test_f1 = evaluate(
            model, sequences_test, loss_fn,
            batch_size=args.batch_size, mode=mode, n_paths=args.n_paths
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_test_metrics = {
                'loss': test_loss, 'acc': test_acc,
                'auc': test_auc, 'f1': test_f1
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} ACC: {train_acc:.4f} AUC: {train_auc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} ACC: {valid_acc:.4f} AUC: {valid_auc:.4f}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\n  BEST TEST: Loss={best_test_metrics['loss']:.4f} "
          f"ACC={best_test_metrics['acc']:.4f} "
          f"AUC={best_test_metrics['auc']:.4f} "
          f"F1={best_test_metrics['f1']:.4f}")

    return best_test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dataset_time.pkl")
    parser.add_argument("--path_dir", type=str, default="path_cache")
    parser.add_argument("--kc_file", type=str, default="problem_kc.json")
    parser.add_argument("--n_paths", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs with different seeds")
    args = parser.parse_args()

    print("=" * 60)
    print("  Lightweight Path-aware KT Ablation")
    print("  (CPU-compatible, LSTM + correctness prediction only)")
    print("=" * 60)

    print("\nLoading data...")
    df, kc_index, kc_list, train_stu, valid_stu, test_stu = prepare_data(
        args.data_path, args.path_dir, args.kc_file, args.seed
    )
    n_kcs = len(kc_list)
    print(f"  Students: train={len(train_stu)}, valid={len(valid_stu)}, test={len(test_stu)}")
    print(f"  KCs: {n_kcs}")

    print("\nBuilding sequences...")
    seq_train = build_sequences(df, train_stu)
    seq_valid = build_sequences(df, valid_stu)
    seq_test = build_sequences(df, test_stu)
    print(f"  Sequences: train={len(seq_train)}, valid={len(seq_valid)}, test={len(seq_test)}")

    # Determine input dimension
    sample_input = seq_train[0]['inputs'][0]
    if isinstance(sample_input, torch.Tensor):
        input_dim = sample_input.shape[0]
    else:
        input_dim = len(sample_input)
    print(f"  Input dim: {input_dim}")

    # Run multiple seeds
    all_results = defaultdict(list)
    modes = ['baseline', 'real_path', 'random_path']

    for run in range(args.n_runs):
        seed = args.seed + run
        print(f"\n{'='*60}")
        print(f"  RUN {run+1}/{args.n_runs} (seed={seed})")
        print(f"{'='*60}")

        for mode in modes:
            args_copy = deepcopy(args)
            args_copy.seed = seed
            metrics = run_experiment(mode, seq_train, seq_valid, seq_test,
                                    input_dim, n_kcs, args_copy)
            all_results[mode].append(metrics)

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON (averaged over {} runs)".format(args.n_runs))
    print("=" * 60)
    print(f"  {'Mode':<15} {'ACC':>8} {'AUC':>8} {'F1':>8} {'Loss':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    summary = {}
    for mode in modes:
        results = all_results[mode]
        avg_acc = np.mean([r['acc'] for r in results])
        avg_auc = np.mean([r['auc'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_loss = np.mean([r['loss'] for r in results])
        std_auc = np.std([r['auc'] for r in results])

        print(f"  {mode:<15} {avg_acc:>8.4f} {avg_auc:>7.4f}±{std_auc:.3f} {avg_f1:>8.4f} {avg_loss:>8.4f}")
        summary[mode] = {'acc': avg_acc, 'auc': avg_auc, 'f1': avg_f1, 'loss': avg_loss,
                        'auc_std': std_auc}

    # Interpretation
    print(f"\n  {'─'*50}")
    real_auc = summary['real_path']['auc']
    rand_auc = summary['random_path']['auc']
    base_auc = summary['baseline']['auc']

    if real_auc > rand_auc and real_auc > base_auc:
        print("  ✓ real_path > random_path AND real_path > baseline")
        print("  → PATH IDENTITY CARRIES DIAGNOSTIC SIGNAL")
        print("  → Path-conditioned KC activation is validated!")
    elif real_auc > base_auc:
        print("  ~ real_path > baseline but real_path ≈ random_path")
        print("  → Path conditioning helps, but signal may be from extra structure")
    else:
        print("  × real_path ≤ baseline")
        print("  → Path conditioning does not improve prediction in this setup")

    delta_real_rand = real_auc - rand_auc
    delta_real_base = real_auc - base_auc
    print(f"\n  Δ(real - random) AUC: {delta_real_rand:+.4f}")
    print(f"  Δ(real - baseline) AUC: {delta_real_base:+.4f}")
    print("  " + "─" * 50)

    # Save results
    output_file = os.path.join(args.path_dir, "lightweight_ablation_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()},
            'detailed': {k: v for k, v in all_results.items()},
            'params': vars(args),
        }, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
