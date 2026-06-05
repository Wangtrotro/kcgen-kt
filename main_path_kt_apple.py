"""
Full Path-aware KT Experiment for Apple Silicon (M4 Pro)
=========================================================
Complete experiment pipeline that runs on MPS backend.

This is the "big experiment" version for mid-term evaluation:
- Full Llama-3-8B with LoRA (float16 on MPS)
- LSTM knowledge tracing with path-conditioned KC activation
- Multi-task: code generation + KC mastery + correctness prediction
- Ablation: real_path vs random_path vs no_path

Requirements:
- macOS with Apple Silicon (M4 Pro or better)
- 18GB+ unified memory (36GB recommended)
- PyTorch with MPS support (torch >= 2.0)

Usage:
    # Real paths (main experiment)
    python main_path_kt_apple.py --config configs_path_kt_apple.yaml

    # Random paths (ablation control)
    python main_path_kt_apple.py --config configs_path_kt_apple.yaml --randomize

    # No paths (baseline)
    python main_path_kt_apple.py --config configs_path_kt_apple.yaml --no_path

    # Test mode (verify everything works)
    python main_path_kt_apple.py --config configs_path_kt_apple.yaml --test
"""

import os
import sys
import json
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import torch.nn as nn
import transformers
from tqdm import tqdm

from data_loader import get_problem_kc, extract_baseline_kc
from path_data_loader import (
    load_path_assignments, load_path_kc_activation,
    read_data_with_paths, make_path_dataloader, CollateForPathKC
)
from model_apple import (
    get_device, create_model_apple, create_multitask_predictor,
    create_binary_predictor, create_knowledge_linear
)
from path_trainer import path_aware_generator_step
from utils import set_random_seed, aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs_path_kt_apple.yaml")
    parser.add_argument("--randomize", action="store_true", help="Randomize path assignments (ablation)")
    parser.add_argument("--no_path", action="store_true", help="Disable path conditioning (baseline)")
    parser.add_argument("--test", action="store_true", help="Quick test with minimal data")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    if args.seed is not None:
        configs.seed = args.seed
    if args.epochs is not None:
        configs.epochs = args.epochs
    if args.test:
        configs.testing = True
        configs.epochs = 1
    if args.randomize:
        configs.randomize_paths = True
    if args.no_path:
        configs.path_mode = 'none'

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = f"path_{configs.path_mode}"
    if configs.randomize_paths:
        mode_tag += "_RANDOM"

    print(f"\n{'='*60}")
    print(f"  Path-aware KT (Apple Silicon)")
    print(f"  Mode: {mode_tag}")
    print(f"  Config: {args.config}")
    print(f"{'='*60}\n")

    set_random_seed(configs.seed)

    # Device selection
    device = get_device()
    print(f"  Device: {device}")
    if device.type == 'mps':
        print(f"  Backend: Apple Metal Performance Shaders")
        # MPS-specific optimizations
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # prevent OOM
    elif device.type == 'cuda':
        print(f"  Backend: NVIDIA CUDA")
    else:
        print(f"  WARNING: Running on CPU, will be very slow!")

    # Load KCs
    if configs.baseline:
        kc_problem_dict, kc_no_dict = extract_baseline_kc('data/prompt_concept.csv')
    else:
        kc_problem_dict, kc_no_dict = get_problem_kc(configs.kc_path)

    # Load path info
    path_assignments = {}
    path_kc_activation = None
    if configs.path_mode != 'none':
        print("  Loading path assignments...")
        path_assignments = load_path_assignments(configs.path_dir)
        if os.path.exists(os.path.join(configs.path_dir, "path_kc_activation.json")):
            path_kc_activation, _ = load_path_kc_activation(configs.path_dir)

    # Load dataset
    train_stu, valid_stu, test_stu, df, students = read_data_with_paths(
        'data/dataset_time.pkl', kc_problem_dict, configs, path_assignments
    )

    # Create model (Apple Silicon compatible)
    if not configs.transition:
        lstm, model, tokenizer = create_model_apple(configs, device, len(kc_no_dict))
        trans_linear = None
    else:
        lstm, model, tokenizer = create_model_apple(configs, device, configs.transition_dim)
        trans_linear = create_knowledge_linear(device, len(kc_no_dict), configs.transition_dim)

    # Print model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model: {configs.okt_model}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    predictor = None
    if configs.multitask:
        if configs.binary_loss_fn == 'BCE':
            predictor = create_multitask_predictor(device, getattr(configs, 'predictor_multilayer', False))
        else:
            predictor = create_binary_predictor(device)

    # Collate function
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

    train_loader = make_path_dataloader(train_stu, df, collate_fn, configs)
    valid_loader = make_path_dataloader(valid_stu, df, collate_fn, configs)
    test_loader = make_path_dataloader(test_stu, df, collate_fn, configs)

    print(f"  Data: {len(train_loader)} train / {len(valid_loader)} valid / {len(test_loader)} test batches")

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
        binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none') if configs.binary_loss_fn == 'BCE' else nn.CrossEntropyLoss(reduction='none')

    num_training_steps = len(train_loader) * configs.epochs
    num_warmup_steps = int(configs.warmup_ratio * num_training_steps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizers_generator[0], num_warmup_steps, num_training_steps
    )

    # Training
    best_valid_loss = float('inf')
    best_test_metrics = {}
    train_dl_len = len(train_loader)

    print(f"\n  Starting training: {configs.epochs} epochs")
    print(f"  {'─'*50}")

    for ep in tqdm(range(configs.epochs), desc="Epochs"):
        train_logs, valid_logs, test_logs = [], [], []

        for idx, batch in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                optimizers=optimizers_generator, optimizers_lstm=optimizers_lstm,
                configs=configs, train_dl_len=train_dl_len, train=True,
                scheduler=scheduler, device=device, group_size=configs.batch_size,
                multitask=configs.multitask, predictor=predictor,
                pred_loss_fn=binary_loss_fn, optimizers_multitask=optimizers_predictor,
                kc_loss_fn=kc_loss_fn, trans_linear=trans_linear,
                optimizers_trans=optimizers_transition, path_mode=configs.path_mode
            )
            train_logs.append(log)

            # MPS memory management
            if device.type == 'mps' and idx % 10 == 0:
                torch.mps.empty_cache()

        for idx, batch in enumerate(tqdm(valid_loader, desc="Valid", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                configs=configs, train_dl_len=train_dl_len, train=False,
                device=device, group_size=configs.batch_size,
                multitask=configs.multitask, predictor=predictor,
                pred_loss_fn=binary_loss_fn, kc_loss_fn=kc_loss_fn,
                trans_linear=trans_linear, optimizers_trans=optimizers_transition,
                path_mode=configs.path_mode
            )
            valid_logs.append(log)

        for idx, batch in enumerate(tqdm(test_loader, desc="Test", leave=False)):
            log = path_aware_generator_step(
                idx, batch, model, lstm, tokenizer,
                configs=configs, train_dl_len=train_dl_len, train=False,
                device=device, group_size=configs.batch_size,
                multitask=configs.multitask, predictor=predictor,
                pred_loss_fn=binary_loss_fn, kc_loss_fn=kc_loss_fn,
                trans_linear=trans_linear, optimizers_trans=optimizers_transition,
                path_mode=configs.path_mode
            )
            test_logs.append(log)

        train_m = aggregate_metrics(train_logs, configs)
        valid_m = aggregate_metrics(valid_logs, configs)
        test_m = aggregate_metrics(test_logs, configs)

        if valid_m['loss'] < best_valid_loss:
            best_valid_loss = valid_m['loss']
            best_test_metrics = test_m.copy()
            print(f"\n  [Ep {ep}] Best valid loss: {best_valid_loss:.4f} | Test: {test_m}")

            if configs.save_model:
                save_dir = os.path.join(configs.model_save_dir, now)
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(os.path.join(save_dir, 'model'))
                torch.save(lstm.state_dict(), os.path.join(save_dir, 'lstm'))
                if trans_linear:
                    torch.save(trans_linear.state_dict(), os.path.join(save_dir, 'transition'))
                if predictor:
                    torch.save(predictor.state_dict(), os.path.join(save_dir, 'predictor'))

    # Final results
    print(f"\n{'='*60}")
    print(f"  RESULTS — {mode_tag}")
    print(f"{'='*60}")
    for k, v in best_test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"{'='*60}\n")

    # Save
    results_file = os.path.join(configs.model_save_dir, f"results_{mode_tag}_{now}.json")
    os.makedirs(configs.model_save_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'mode': mode_tag,
            'best_valid_loss': float(best_valid_loss),
            'test_metrics': {k: float(v) if isinstance(v, (float, int)) else str(v)
                           for k, v in best_test_metrics.items()},
            'config': OmegaConf.to_container(configs),
        }, f, indent=2)
    print(f"  Saved to {results_file}")


if __name__ == '__main__':
    main()
