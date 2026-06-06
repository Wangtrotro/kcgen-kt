"""
Path-aware Knowledge Tracing: Main Experiment
==============================================
Supports three modes:
  1. path_aware (real paths): Full path-conditioned KC activation
  2. path_random (ablation): Same model, but path IDs randomly shuffled per-problem
  3. no_path (baseline): Original kcgen-kt behavior (all KCs activated equally)

The key experiment is comparing mode 1 vs mode 2:
  If real_path > random_path → path identity carries diagnostic signal
  
Usage:
    # Run with real paths
    python main_path_kt.py path_mode=hard path_dir=path_cache
    
    # Run ablation with randomized paths
    python main_path_kt.py path_mode=hard path_dir=path_cache randomize_paths=true
    
    # Run original baseline (no path conditioning)
    python main_path_kt.py path_mode=none
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from omegaconf import OmegaConf
from datetime import datetime
import hydra
import transformers
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from data_loader import get_problem_kc, extract_baseline_kc
from path_data_loader import (
    load_path_assignments, load_path_kc_activation,
    read_data_with_paths, make_path_dataloader, CollateForPathKC
)
from model import (
    create_model, create_multitask_predictor, create_binary_predictor,
    create_knowledge_linear
)
from path_trainer import path_aware_generator_step
from utils import set_random_seed, aggregate_metrics


@hydra.main(version_base=None, config_path=".", config_name="configs_path_kt")
def main(configs):
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    mode_tag = f"path_{configs.path_mode}"
    if configs.randomize_paths:
        mode_tag += "_RANDOM"
    print(f"\n{'='*60}")
    print(f"  Path-aware KT Experiment: {mode_tag}")
    print(f"  Time: {now}")
    print(f"{'='*60}\n")

    set_random_seed(configs.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if configs.use_cuda:
        assert device.type == 'cuda', 'No GPU found'

    # Load KC mapping (problem-level, as before)
    if configs.baseline:
        kc_problem_dict, kc_no_dict = extract_baseline_kc('data/prompt_concept.csv')
    else:
        kc_problem_dict, kc_no_dict = get_problem_kc(configs.kc_path)

    # Load path information
    path_assignments = {}
    path_kc_activation = None

    if configs.path_mode != 'none':
        print("Loading path assignments...")
        path_assignments = load_path_assignments(configs.path_dir)
        print(f"  Loaded {len(path_assignments)} path assignments")

        if os.path.exists(os.path.join(configs.path_dir, "path_kc_activation.json")):
            path_kc_activation, path_kc_index = load_path_kc_activation(configs.path_dir)
            print(f"  Loaded path KC activation for {len(path_kc_activation)} problems")

    # Load dataset with path info
    train_stu, valid_stu, test_stu, df, students = read_data_with_paths(
        'data/dataset_time.pkl', kc_problem_dict, configs, path_assignments
    )

    # Create model
    if not configs.transition:
        lstm, model, tokenizer = create_model(configs, device, len(kc_no_dict))
        trans_linear = None
    else:
        lstm, model, tokenizer = create_model(configs, device, configs.transition_dim)
        trans_linear = create_knowledge_linear(device, len(kc_no_dict), configs.transition_dim)

    predictor = None
    if configs.multitask:
        if configs.binary_loss_fn == 'BCE':
            predictor = create_multitask_predictor(device, configs.predictor_multilayer)
        else:
            predictor = create_binary_predictor(device)

    # Create path-aware collate function
    collate_fn = CollateForPathKC(
        tokenizer, configs, device, kc_no_dict,
        path_kc_activation=path_kc_activation,
        n_paths=configs.n_paths,
        randomize_paths=configs.randomize_paths,
        eval=False
    )

    if configs.testing:
        train_stu = train_stu[:3]
        valid_stu = valid_stu[:3]
        test_stu = test_stu[:3]
        configs.epochs = 1

    train_loader = make_path_dataloader(train_stu, df, collate_fn, configs)
    valid_loader = make_path_dataloader(valid_stu, df, collate_fn, configs)
    test_loader = make_path_dataloader(test_stu, df, collate_fn, configs)

    # Optimizers
    optimizers_generator = [optim.AdamW(model.parameters(), lr=configs.lr)]
    optimizers_lstm = [optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)]

    optimizers_transition = None
    if configs.transition:
        optimizers_transition = [optim.AdamW(trans_linear.parameters(), lr=configs.trans_linear_lr)]

    kc_loss_fn = nn.BCELoss(reduction='none')

    optimizers_predictor = None
    binary_loss_fn = None
    if configs.multitask:
        optimizers_predictor = [optim.AdamW(predictor.parameters(), lr=configs.pred_linear_lr)]
        if configs.binary_loss_fn == 'BCE':
            binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            binary_loss_fn = nn.CrossEntropyLoss(reduction='none')

    num_training_steps = len(train_loader) * configs.epochs
    num_warmup_steps = configs.warmup_ratio * num_training_steps
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizers_generator[0], num_warmup_steps, num_training_steps
    )

    # Training loop
    best_valid_loss = float('inf')
    best_test_metrics = {}
    train_dl_len = len(train_loader)

    results_log = []

    for ep in tqdm(range(configs.epochs), desc="epochs", mininterval=20.0):
        train_logs, valid_logs, test_logs = [], [], []

        # Training
        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                optimizers=optimizers_generator, optimizers_lstm=optimizers_lstm,
                configs=configs, train_dl_len=train_dl_len, train=True,
                scheduler=scheduler, device=device, group_size=2,
                multitask=configs.multitask, predictor=predictor,
                pred_loss_fn=binary_loss_fn, optimizers_multitask=optimizers_predictor,
                kc_loss_fn=kc_loss_fn, trans_linear=trans_linear,
                optimizers_trans=optimizers_transition, path_mode=configs.path_mode
            )
            train_logs.append(log)

        # Validation
        for idx, batch in enumerate(tqdm(valid_loader, desc="validation", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                configs=configs, train_dl_len=train_dl_len, train=False,
                device=device, group_size=2, multitask=configs.multitask,
                predictor=predictor, pred_loss_fn=binary_loss_fn,
                kc_loss_fn=kc_loss_fn, trans_linear=trans_linear,
                optimizers_trans=optimizers_transition, path_mode=configs.path_mode
            )
            valid_logs.append(log)

        # Testing
        for idx, batch in enumerate(tqdm(test_loader, desc="testing", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                configs=configs, train_dl_len=train_dl_len, train=False,
                device=device, group_size=2, multitask=configs.multitask,
                predictor=predictor, pred_loss_fn=binary_loss_fn,
                kc_loss_fn=kc_loss_fn, trans_linear=trans_linear,
                optimizers_trans=optimizers_transition, path_mode=configs.path_mode
            )
            test_logs.append(log)

        train_metrics = aggregate_metrics(train_logs, configs)
        valid_metrics = aggregate_metrics(valid_logs, configs)
        test_metrics = aggregate_metrics(test_logs, configs)

        # Track best
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            best_test_metrics = test_metrics.copy()
            print(f"\n  [Epoch {ep}] New best valid loss: {best_valid_loss:.4f}")
            print(f"  Test metrics: {test_metrics}")

            # Save model
            if configs.save_model:
                model_dir = os.path.join(configs.model_save_dir, now, 'model')
                os.makedirs(os.path.join(configs.model_save_dir, now), exist_ok=True)
                model.save_pretrained(model_dir)
                torch.save(lstm.state_dict(), os.path.join(configs.model_save_dir, now, 'lstm'))
                if configs.transition:
                    torch.save(trans_linear.state_dict(), os.path.join(configs.model_save_dir, now, 'transition'))
                if configs.multitask:
                    torch.save(predictor.state_dict(), os.path.join(configs.model_save_dir, now, 'predictor'))

        results_log.append({
            'epoch': ep,
            'mode': mode_tag,
            'train': train_metrics,
            'valid': valid_metrics,
            'test': test_metrics,
        })

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — Mode: {mode_tag}")
    print(f"{'='*60}")
    print(f"  Best valid loss: {best_valid_loss:.4f}")
    if best_test_metrics:
        for k, v in best_test_metrics.items():
            if isinstance(v, float):
                print(f"  Test {k}: {v:.4f}")
    print(f"{'='*60}\n")

    # Save results
    import json
    results_file = os.path.join(configs.model_save_dir, f"results_{mode_tag}_{now}.json")
    os.makedirs(configs.model_save_dir, exist_ok=True)

    save_results = {
        'mode': mode_tag,
        'path_mode': configs.path_mode,
        'randomize_paths': configs.randomize_paths,
        'n_paths': configs.n_paths,
        'best_valid_loss': float(best_valid_loss),
        'best_test_metrics': {k: float(v) if isinstance(v, (float, int)) else str(v)
                             for k, v in best_test_metrics.items()},
    }
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to {results_file}")

    return best_test_metrics


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
