"""
Path-aware Data Loader
======================
Extends the existing data_loader.py to incorporate path information
into the student sequence data.
"""

import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from data_loader import (
    read_data, make_pytorch_dataset, get_problem_kc, extract_baseline_kc,
    build_prompt_with_special_tokens, build_input_with_special_tokens
)


def load_path_assignments(path_dir="path_cache"):
    """Load path assignments for each (student, problem) pair."""
    path_file = f"{path_dir}/path_assignments.json"
    with open(path_file, 'r') as f:
        raw = json.load(f)

    assignments = {}
    for key, val in raw.items():
        sid, pid = key.split("||")
        try:
            pid = int(pid)
        except ValueError:
            pass
        assignments[(sid, pid)] = {
            'hard': val['hard'],
            'soft': np.array(val['soft'])
        }
    return assignments


def load_path_kc_activation(path_dir="path_cache"):
    """
    Load path-conditioned KC activation matrix.
    Returns:
        activation: dict {problem_id: {path_id: [0/1 vector]}}
        kc_index: dict {kc_name: index}
    """
    act_file = f"{path_dir}/path_kc_activation.json"
    kc_file = f"{path_dir}/kc_index.json"

    with open(act_file, 'r') as f:
        act_raw = json.load(f)
    activation = {}
    for pid, paths in act_raw.items():
        activation[int(pid)] = {int(k): v for k, v in paths.items()}

    with open(kc_file, 'r') as f:
        kc_index = json.load(f)

    return activation, kc_index


def read_data_with_paths(file, kc_problem_dict, configs, path_assignments):
    """
    Extended read_data that also attaches path_id to each row.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    df = pd.read_pickle(file)

    if configs.label_type == 'binary':
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)
    else:
        df['Score'] = df['Score_x']
    df.drop(columns=['Score_x', 'Score_y'], inplace=True)

    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df['knowledge_component'] = df['prompt'].map(kc_problem_dict)

    # Attach path_id
    def get_path_id(row):
        key = (row['SubjectID'], row['ProblemID'])
        if key in path_assignments:
            return path_assignments[key]['hard']
        return 0  # default path

    def get_path_soft(row):
        key = (row['SubjectID'], row['ProblemID'])
        if key in path_assignments:
            return path_assignments[key]['soft'].tolist()
        return [1.0]  # default: all weight on path 0

    df['path_id'] = df.apply(get_path_id, axis=1)
    df['path_soft'] = df.apply(get_path_soft, axis=1)

    if configs.first_ast_convertible:
        df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    prev_subject_id = 0
    subjectid_appendix = []
    timesteps = []
    for i in tqdm(range(len(df)), desc="splitting students' records ..."):
        if prev_subject_id != df.iloc[i].SubjectID:
            prev_subject_id = df.iloc[i].SubjectID
            accumulated = 0
            id_appendix = 1
        else:
            accumulated += 1
            if accumulated >= configs.max_len:
                id_appendix += 1
                accumulated = 0
        timesteps.append(accumulated)
        subjectid_appendix.append(id_appendix)
    df['timestep'] = timesteps
    df['SubjectID_appendix'] = subjectid_appendix
    df['SubjectID'] = [df.iloc[i].SubjectID + '_{}'.format(df.iloc[i].SubjectID_appendix) for i in range(len(df))]

    students = df['SubjectID'].unique()
    train_stu, test_stu = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
    valid_stu, test_stu = train_test_split(test_stu, test_size=0.5, random_state=configs.seed)

    return train_stu, valid_stu, test_stu, df, students


def make_path_pytorch_dataset(students, dataset):
    """Extended dataset builder that includes path information."""
    lstm_student = []

    for student in students:
        subset = dataset[dataset['SubjectID'] == student]
        if len(subset) > 1:
            subset.loc[:, 'prompt-embedding'] = subset['prompt-embedding'].apply(lambda x: torch.tensor(x))
            data_dict = {
                'SubjectID': student,
                'ProblemID_seq': subset.ProblemID.tolist(),
                'Score': subset.Score.tolist(),
                'prompt-embedding': subset['prompt-embedding'].tolist(),
                'input': subset.input.tolist(),
                'KC': subset['knowledge_component'].tolist(),
                'next_prompt': subset.prompt.tolist(),
                'next_code': subset.Code.tolist(),
                'path_id': subset.path_id.tolist(),
                'path_soft': subset.path_soft.tolist(),
            }
            lstm_student.append(data_dict)

    return lstm_student


def make_path_dataloader(students, dataset, collate_fn, configs, sampler=None, shuffle=False):
    """Make dataloader with path information."""
    lstm_student = make_path_pytorch_dataset(students, dataset)
    data_loader = torch.utils.data.DataLoader(
        lstm_student, collate_fn=collate_fn, shuffle=shuffle,
        sampler=sampler, batch_size=configs.batch_size
    )
    return data_loader


class CollateForPathKC(object):
    """
    Extended collate function that includes path_id information in each batch.
    Supports both path-aware and path-randomized (ablation) modes.
    """

    def __init__(self, tokenizer, configs, device, kc_dict, path_kc_activation=None,
                 n_paths=3, randomize_paths=False, eval=False):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" if eval else "right"
        self.configs = configs
        self.device = device
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids("Ġwritten")
        self.level_token_id = tokenizer.convert_tokens_to_ids('Ġ?')
        self.kc_dict = kc_dict
        self.eval = eval
        self.path_kc_activation = path_kc_activation  # {pid: {path_id: [0/1 vec]}}
        self.n_paths = n_paths
        self.randomize_paths = randomize_paths  # For ablation: shuffle path IDs

    def __call__(self, batch):
        scores = [b['Score'] for b in batch]
        max_len = max([len(i) for i in scores])
        padded_scores = [i + [-100] * (max_len - len(i)) for i in scores]
        padded_scores = torch.Tensor(padded_scores).t().to(self.device)

        question_seqs = [b['ProblemID_seq'] for b in batch]
        padded_question_seqs = [i + [-100] * (max_len - len(i)) for i in question_seqs]
        padded_question_seqs = torch.tensor(padded_question_seqs)

        inputs = [b['input'] for b in batch]
        padded_inputs = [i + [torch.zeros(self.configs.lstm_inp_dim)] * (max_len - len(i)) for i in inputs]
        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float().to(self.device)

        # Path IDs
        path_ids = [b['path_id'] for b in batch]
        problem_ids = [b['ProblemID_seq'] for b in batch]

        if self.randomize_paths:
            # Ablation: randomize path assignments within each problem
            import random
            path_ids_randomized = []
            for pid_seq, path_seq in zip(problem_ids, path_ids):
                randomized = []
                for pid, path in zip(pid_seq, path_seq):
                    # Random path for this problem (same number of possible paths)
                    if self.path_kc_activation and pid in self.path_kc_activation:
                        n_available = len(self.path_kc_activation[pid])
                        randomized.append(random.randint(0, max(0, n_available - 1)))
                    else:
                        randomized.append(random.randint(0, self.n_paths - 1))
                path_ids_randomized.append(randomized)
            path_ids = path_ids_randomized

        padded_path_ids = [i + [0] * (max_len - len(i)) for i in path_ids]
        padded_path_ids = torch.tensor(padded_path_ids).t().to(self.device)  # (T, B)

        # Build path-conditioned KC activation weights
        # Shape: (T, B, max_kc_len) - weight for each KC based on path
        kcs_name = [b['KC'] for b in batch]
        kcs = [[[self.kc_dict[elem] for elem in kc_i] for kc_i in sub_ls] for sub_ls in kcs_name]
        max_kc_inner_len = max([len(inner) for kc_i in kcs for inner in kc_i])
        max_kc_len = max(len(kc_i) for kc_i in kcs)

        def pad_inner(inner, max_len_inner):
            return inner + [-1] * (max_len_inner - len(inner))

        padded_kc = []
        for kc_i in kcs:
            padded_inner = [pad_inner(i, max_kc_inner_len) for i in kc_i]
            padded_inner += [[-1] * max_kc_inner_len] * (max_kc_len - len(kc_i))
            padded_kc.append(padded_inner)
        padded_kc = torch.tensor(padded_kc).float().to(self.device)
        padded_kc = padded_kc.transpose(0, 1)  # (T, B, max_single_kc_len)

        # Build path activation mask: (B, T) → which KCs to weight more/less
        # This is the key differentiator from the original collate
        path_kc_weights = self._build_path_kc_weights(problem_ids, path_ids, kcs_name, max_len)

        # Standard tokenization (same as original CollateForKC)
        codes = [b['next_code'] for b in batch]
        students = []
        for i in range(len(batch)):
            stu_name = batch[i]['SubjectID']
            student_ls = [stu_name] * len(codes[i])
            students.append(student_ls)
        padded_students = [i + [''] * (max_len - len(i)) for i in students]
        stacked_students = list(map(list, zip(*padded_students)))

        padded_codes = [i + [''] * (max_len - len(i)) for i in codes]
        stacked_codes = list(map(list, zip(*padded_codes)))

        prompts = [b['next_prompt'] for b in batch]
        padded_prompts = [i + [''] * (max_len - len(i)) for i in prompts]
        stacked_prompts = list(map(list, zip(*padded_prompts)))

        if self.eval:
            input_texts = [[build_prompt_with_special_tokens(prompt_i, kc_i) for prompt_i, kc_i in
                            zip(entry['next_prompt'], entry['KC'])] for entry in batch]
        else:
            input_texts = [[build_input_with_special_tokens(prompt_i, kc_i, code_i, self.tokenizer) for
                            prompt_i, kc_i, code_i in zip(entry['next_prompt'], entry['KC'], entry['next_code'])] for entry in batch]

        inputs_ids_ls, attention_mask_ls, labels_ls, prompt_id_lens_ls, level_loc_ls = [], [], [], [], []

        for input_sub in input_texts:
            inputs_tok = self.tokenizer(input_sub, return_tensors='pt', padding=True, truncation=True)
            inputs_ids, attention_mask = inputs_tok['input_ids'].to(self.device), inputs_tok['attention_mask'].to(self.device)

            if not self.eval:
                inputs_ids[:, -1] = self.tokenizer.eos_token_id

            delimiter_indices = torch.where(inputs_ids == self.delimiter_token_id, 1, 0)
            prompt_id_lens = torch.argmax(delimiter_indices, dim=-1) + 3

            labels = inputs_ids.detach().clone()
            labels = labels.masked_fill((attention_mask == 0), -100)
            range_tensor = torch.arange(inputs_ids.size(1), device=self.device).unsqueeze(0)
            range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
            mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1))
            labels[mask_tensor] = -100

            inputs_ids_ls.append(inputs_ids)
            attention_mask_ls.append(attention_mask)
            labels_ls.append(labels)
            prompt_id_lens_ls.append(prompt_id_lens)

            col_indices = torch.arange(inputs_ids.shape[1]).unsqueeze(0)
            valid_mask = col_indices.to(self.device) < prompt_id_lens.unsqueeze(1)
            matches = (inputs_ids == self.level_token_id) & valid_mask
            level_sent_ind, level_token_ind = torch.nonzero(matches, as_tuple=True)
            mask = level_sent_ind != 0
            filtered_sent_ind = level_sent_ind[mask] - 1
            filtered_token_ind = level_token_ind[mask]
            level_loc_ls.append((filtered_sent_ind, filtered_token_ind))

        max_length = max([sub.shape[1] for sub in inputs_ids_ls])

        padded_input_ids_ls = [torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]),
                               value=self.tokenizer.eos_token_id) for input_ids in inputs_ids_ls]
        padded_input_ids_ls = pad_sequence(padded_input_ids_ls, batch_first=True,
                              padding_value=self.tokenizer.eos_token_id)

        padded_attention_mask_ls = [torch.nn.functional.pad(attention_mask, (0, max_length - attention_mask.shape[1]),
                                    value=0) for attention_mask in attention_mask_ls]
        padded_attention_mask_ls = pad_sequence(padded_attention_mask_ls, batch_first=True, padding_value=0)
        padded_attention_mask_ls = torch.transpose(padded_attention_mask_ls, 0, 1)

        padded_labels_ls = [torch.nn.functional.pad(labels, (0, max_length - labels.shape[1]), value=-100) for labels in labels_ls]
        padded_labels_ls = pad_sequence(padded_labels_ls, batch_first=True, padding_value=-100)
        padded_labels_ls = torch.transpose(padded_labels_ls, 0, 1)

        padded_prompt_id_lens_ls = [torch.cat((i, torch.zeros(max_len - i.size(0)).to(self.device)), 0) for i in prompt_id_lens_ls]
        padded_prompt_id_lens_ls = torch.stack(padded_prompt_id_lens_ls).t()

        if self.eval:
            return (padded_inputs, padded_input_ids_ls, padded_attention_mask_ls,
                    stacked_codes, stacked_prompts, padded_scores, stacked_students,
                    padded_kc, level_loc_ls, padded_question_seqs, padded_path_ids, path_kc_weights)

        return (padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls,
                padded_labels_ls, padded_prompt_id_lens_ls, padded_kc, level_loc_ls,
                padded_path_ids, path_kc_weights)

    def _build_path_kc_weights(self, problem_ids, path_ids, kcs_name, max_len):
        """
        Build per-KC weight based on path assignment.
        
        Returns tensor of shape (B, T, max_kc_inner_len) with values in [0, 1]
        indicating how strongly each KC should be activated given the path.
        
        If no path_kc_activation is provided, returns all-ones (equivalent to no path conditioning).
        """
        B = len(problem_ids)
        max_kc_inner_len = max([len(inner) for kc_i in kcs_name for inner in kc_i])

        # Default: all KCs equally activated
        if self.path_kc_activation is None:
            return torch.ones(B, max_len, max_kc_inner_len).to(self.device)

        weights = torch.ones(B, max_len, max_kc_inner_len).to(self.device)

        for b in range(B):
            for t in range(min(len(problem_ids[b]), max_len)):
                pid = problem_ids[b][t]
                path = path_ids[b][t] if t < len(path_ids[b]) else 0

                if pid in self.path_kc_activation and path in self.path_kc_activation[pid]:
                    # Get the activation vector for this (problem, path)
                    act_vec = self.path_kc_activation[pid][path]
                    # Map global KC activation to this problem's KC subset
                    kc_names_t = kcs_name[b][t] if t < len(kcs_name[b]) else []
                    for k_idx, kc_name in enumerate(kc_names_t):
                        if k_idx < max_kc_inner_len:
                            kc_global_idx = self.kc_dict.get(kc_name, -1)
                            if kc_global_idx >= 0 and kc_global_idx < len(act_vec):
                                weights[b, t, k_idx] = float(act_vec[kc_global_idx])
                            # If KC not in activation vector, keep weight=1 (always active)

        # Transpose to match (T, B, max_kc_inner_len) convention used in trainer
        return weights.transpose(0, 1)
