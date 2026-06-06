"""
Direct Path Signal Ablation
=============================
Tests the most direct form of the hypothesis:
Does knowing WHICH PATH a student took improve correctness prediction?

Instead of trying to do KC masking with heuristic weights, this test
adds path_id as a direct input feature to the model.

Three conditions:
  1. baseline:    LSTM uses only (prompt_embedding + ASTNN) = 4296d
  2. real_path:   LSTM uses (prompt_embedding + ASTNN + path_onehot) = 4296 + n_paths
  3. random_path: Same input dimension, but path_id is randomly shuffled

If real_path > random_path: path identity carries predictive signal
that cannot be explained by extra input dimensions alone.
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
import argparse
import random
from copy import deepcopy


class DirectPathLSTM(nn.Module):
    """LSTM KT model with optional path embedding input."""
    def __init__(self, input_dim, n_kcs, hidden_dim=128, n_paths=0):
        super().__init__()
        actual_input_dim = input_dim + n_paths  # path one-hot appended
        self.lstm = nn.LSTM(actual_input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.kc_head = nn.Linear(hidden_dim, n_kcs)
        self.n_kcs = n_kcs
        self.n_paths = n_paths

    def forward(self, x, kc_indices, path_onehot=None):
        """
        x: (B, T, input_dim)
        kc_indices: (B, T, max_kc) — KC indices per timestep
        path_onehot: (B, T, n_paths) — one-hot path encoding (or None for baseline)
        """
        if path_onehot is not None and self.n_paths > 0:
            x = torch.cat([x, path_onehot], dim=-1)
        elif self.n_paths > 0:
            # Pad with zeros if no path info (baseline mode with path-dim model)
            zeros = torch.zeros(x.shape[0], x.shape[1], self.n_paths)
            x = torch.cat([x, zeros], dim=-1)

        lstm_out, _ = self.lstm(x)
        all_kc_mastery = torch.sigmoid(self.kc_head(lstm_out))  # (B, T, n_kcs)

        # Gather relevant KC mastery
        padding_mask = kc_indices == -1
        safe_indices = kc_indices.clone()
        safe_indices[padding_mask] = 0
        kc_mastery = torch.gather(all_kc_mastery, 2, safe_indices.long())
        kc_mastery[padding_mask] = 0.0

        # Mean mastery as correctness prediction
        active_count = (~padding_mask).float().sum(dim=-1).clamp(min=1)
        prediction = kc_mastery.sum(dim=-1) / active_count

        return prediction, kc_mastery


def prepare_data(data_path, path_dir, kc_file, seed=0):
    """Load and prepare data."""
    df = pd.read_pickle(data_path)
    if 'Score_x' in df.columns:
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)
    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    # Load KCs
    with open(kc_file, 'r') as f:
        problem_kc_raw = json.load(f)
    all_kcs = set()
    for kcs in problem_kc_raw.values():
        for kc in kcs:
            name = kc[0] if isinstance(kc, list) else kc
            all_kcs.add(name)
    kc_list = sorted(list(all_kcs))
    kc_index = {kc: i for i, kc in enumerate(kc_list)}

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

    students = df['SubjectID'].unique()
    train_stu, test_stu = train_test_split(students, test_size=0.2, random_state=seed)
    valid_stu, test_stu = train_test_split(test_stu, test_size=0.5, random_state=seed)

    return df, kc_index, kc_list, train_stu, valid_stu, test_stu


def build_sequences(df, students, max_len=20):
    """Build student sequences."""
    sequences = []
    for sid in students:
        student_df = df[df['SubjectID'] == sid].sort_values('ServerTimestamp')
        if len(student_df) < 2:
            continue
        for start in range(0, len(student_df), max_len):
            chunk = student_df.iloc[start:start + max_len]
            if len(chunk) < 2:
                continue
            sequences.append({
                'inputs': [row['input'] for _, row in chunk.iterrows()],
                'scores': [row['Score'] for _, row in chunk.iterrows()],
                'kc_indices': [row['kc_indices'] for _, row in chunk.iterrows()],
                'path_ids': [row['path_id'] for _, row in chunk.iterrows()],
            })
    return sequences


def collate_fn(batch, n_paths=3, mode='baseline'):
    """Collate with path one-hot encoding."""
    B = len(batch)
    max_t = max(len(b['scores']) for b in batch)
    max_kc = max(len(kc) for b in batch for kc in b['kc_indices'])
    input_dim = len(batch[0]['inputs'][0]) if isinstance(batch[0]['inputs'][0], (list, np.ndarray)) else batch[0]['inputs'][0].shape[0]

    inputs = torch.zeros(B, max_t, input_dim)
    scores = torch.full((B, max_t), -1.0)
    kc_indices = torch.full((B, max_t, max_kc), -1, dtype=torch.long)
    path_onehot = torch.zeros(B, max_t, n_paths)

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

            # Path one-hot
            pid = b['path_ids'][t]
            if mode == 'random_path':
                pid = random.randint(0, n_paths - 1)
            if 0 <= pid < n_paths:
                path_onehot[b_idx, t, pid] = 1.0

    return inputs, scores, kc_indices, path_onehot


def train_epoch(model, sequences, optimizer, loss_fn, batch_size, mode, n_paths):
    model.train()
    random.shuffle(sequences)
    all_preds, all_labels = [], []
    total_loss, total_count = 0.0, 0

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs, scores, kc_indices, path_onehot = collate_fn(batch, n_paths, mode)

        p_input = path_onehot if mode != 'baseline' else None
        predictions, _ = model(inputs, kc_indices, path_onehot=p_input)

        valid_mask = scores[:, 1:] >= 0
        pred_shifted = predictions[:, :-1]
        score_shifted = scores[:, 1:]

        if valid_mask.sum() == 0:
            continue

        loss = loss_fn(pred_shifted[valid_mask], score_shifted[valid_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * valid_mask.sum().item()
        total_count += valid_mask.sum().item()
        all_preds.extend(pred_shifted[valid_mask].detach().numpy().tolist())
        all_labels.extend(score_shifted[valid_mask].detach().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(np.array(all_labels) > 0.5, np.array(all_preds) > 0.5)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    return avg_loss, acc, auc


def evaluate_model(model, sequences, loss_fn, batch_size, mode, n_paths):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_count = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            inputs, scores, kc_indices, path_onehot = collate_fn(batch, n_paths, mode)
            p_input = path_onehot if mode != 'baseline' else None
            predictions, _ = model(inputs, kc_indices, path_onehot=p_input)

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


def run_single(mode, seq_train, seq_valid, seq_test, input_dim, n_kcs, n_paths, args):
    """Run one experimental condition."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # For baseline, model has n_paths=0 (no extra input dims)
    # For path-aware modes, model has n_paths extra input dims
    model_n_paths = n_paths if mode != 'baseline' else 0
    model = DirectPathLSTM(input_dim, n_kcs, hidden_dim=args.hidden_dim, n_paths=model_n_paths)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    best_valid_loss = float('inf')
    best_test = None
    patience = 0

    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc = train_epoch(
            model, seq_train, optimizer, loss_fn, args.batch_size, mode, n_paths)
        valid_loss, valid_acc, valid_auc, _ = evaluate_model(
            model, seq_valid, loss_fn, args.batch_size, mode, n_paths)
        test_loss, test_acc, test_auc, test_f1 = evaluate_model(
            model, seq_test, loss_fn, args.batch_size, mode, n_paths)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_test = {'loss': test_loss, 'acc': test_acc, 'auc': test_auc, 'f1': test_f1}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:2d} | TrLoss {train_loss:.4f} TrAUC {train_auc:.4f} | "
                  f"VaLoss {valid_loss:.4f} VaAUC {valid_auc:.4f} | TeAUC {test_auc:.4f}")

        if patience >= args.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    return best_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dataset_time.pkl")
    parser.add_argument("--path_dir", default="path_cache")
    parser.add_argument("--kc_file", default="problem_kc.json")
    parser.add_argument("--n_paths", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  Direct Path Signal Ablation")
    print("  Q: Does path identity improve correctness prediction?")
    print("=" * 60)

    df, kc_index, kc_list, train_stu, valid_stu, test_stu = prepare_data(
        args.data_path, args.path_dir, args.kc_file, args.seed)
    n_kcs = len(kc_list)

    seq_train = build_sequences(df, train_stu)
    seq_valid = build_sequences(df, valid_stu)
    seq_test = build_sequences(df, test_stu)

    sample = seq_train[0]['inputs'][0]
    input_dim = sample.shape[0] if isinstance(sample, torch.Tensor) else len(sample)

    print(f"\n  Data: {len(seq_train)} train, {len(seq_valid)} valid, {len(seq_test)} test seqs")
    print(f"  Input dim: {input_dim}, KCs: {n_kcs}, Paths: {args.n_paths}")
    print(f"  Running {args.n_runs} seeds × 3 modes = {args.n_runs * 3} experiments\n")

    all_results = defaultdict(list)

    for run in range(args.n_runs):
        seed = args.seed + run * 7  # different seeds
        args_copy = deepcopy(args)
        args_copy.seed = seed
        print(f"{'─'*60}")
        print(f"  Run {run+1}/{args.n_runs} (seed={seed})")

        for mode in ['baseline', 'real_path', 'random_path']:
            print(f"  [{mode}]")
            result = run_single(mode, seq_train, seq_valid, seq_test,
                              input_dim, n_kcs, args.n_paths, args_copy)
            all_results[mode].append(result)
            print(f"    → ACC={result['acc']:.4f} AUC={result['auc']:.4f} F1={result['f1']:.4f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("  FINAL RESULTS (mean ± std over {} runs)".format(args.n_runs))
    print("=" * 60)
    print(f"  {'Mode':<15} {'ACC':>12} {'AUC':>12} {'F1':>12}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12}")

    summary = {}
    for mode in ['baseline', 'real_path', 'random_path']:
        accs = [r['acc'] for r in all_results[mode]]
        aucs = [r['auc'] for r in all_results[mode]]
        f1s = [r['f1'] for r in all_results[mode]]
        summary[mode] = {
            'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
            'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
            'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        }
        print(f"  {mode:<15} {np.mean(accs):.4f}±{np.std(accs):.3f} "
              f"{np.mean(aucs):.4f}±{np.std(aucs):.3f} "
              f"{np.mean(f1s):.4f}±{np.std(f1s):.3f}")

    # Statistical significance via paired comparison
    from scipy import stats
    real_aucs = [r['auc'] for r in all_results['real_path']]
    rand_aucs = [r['auc'] for r in all_results['random_path']]
    base_aucs = [r['auc'] for r in all_results['baseline']]

    if len(real_aucs) >= 3:
        t_real_rand, p_real_rand = stats.ttest_rel(real_aucs, rand_aucs)
        t_real_base, p_real_base = stats.ttest_rel(real_aucs, base_aucs)
    else:
        t_real_rand, p_real_rand = stats.ttest_ind(real_aucs, rand_aucs)
        t_real_base, p_real_base = stats.ttest_ind(real_aucs, base_aucs)

    print(f"\n  Paired t-tests (AUC):")
    print(f"    real vs random: t={t_real_rand:.3f}, p={p_real_rand:.4f}")
    print(f"    real vs baseline: t={t_real_base:.3f}, p={p_real_base:.4f}")

    delta_rr = np.mean(real_aucs) - np.mean(rand_aucs)
    delta_rb = np.mean(real_aucs) - np.mean(base_aucs)
    print(f"\n  Δ AUC (real - random):   {delta_rr:+.4f}")
    print(f"  Δ AUC (real - baseline): {delta_rb:+.4f}")

    print(f"\n  {'─'*50}")
    if delta_rr > 0 and p_real_rand < 0.1:
        print("  ✓ real_path > random_path (path signal detected)")
    elif delta_rr > 0:
        print("  ~ real_path > random_path (trend, not significant)")
    else:
        print("  × real_path ≤ random_path")

    if delta_rb > 0 and p_real_base < 0.1:
        print("  ✓ real_path > baseline (path input improves prediction)")
    elif delta_rb > 0:
        print("  ~ real_path > baseline (trend, not significant)")
    else:
        print("  × real_path ≤ baseline")
    print(f"  {'─'*50}")

    # Save
    output = {
        'summary': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()},
        'delta_auc_real_random': float(delta_rr),
        'delta_auc_real_baseline': float(delta_rb),
        'p_value_real_random': float(p_real_rand),
        'p_value_real_baseline': float(p_real_base),
        'raw_aucs': {'real': real_aucs, 'random': rand_aucs, 'baseline': base_aucs},
    }
    out_file = os.path.join(args.path_dir, "direct_path_ablation_results.json")
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_file}")


if __name__ == "__main__":
    main()
