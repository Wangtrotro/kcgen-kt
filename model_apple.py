"""
Model definitions adapted for Apple Silicon (M4 Pro).
Replaces bitsandbytes (CUDA-only) with MPS-compatible alternatives.

Supports three backends:
  1. 'mps'  — float16 on Apple Metal (needs ~16GB, for 36GB M4 Pro)
  2. 'mps4' — 4-bit via BitsAndBytes-free GPTQ/AWQ (needs ~5GB, for 18GB M4 Pro)
  3. 'cuda' — original bitsandbytes 8-bit (for NVIDIA GPUs)

Usage:
    from model_apple import create_model_apple, create_lstm_model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_lstm_model(configs, device, hid_dim):
    """Same as original — LSTM works on all devices."""
    lstm = nn.LSTM(configs.lstm_inp_dim, hid_dim, num_layers=configs.num_layers)
    lstm.to(device)
    return lstm


def create_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.okt_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_model_apple(configs, device, lstm_hid_dim):
    """
    Create model compatible with Apple Silicon.
    Auto-selects quantization strategy based on device and available memory.
    """
    tokenizer = create_tokenizer(configs)

    if device.type == 'cuda':
        # Original path: bitsandbytes 8-bit
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            configs.okt_model,
            quantization_config=bnb_config
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    elif device.type == 'mps':
        # Apple Silicon: load in float16, no quantization
        # For 36GB M4 Pro this fits comfortably
        # For 18GB, use a smaller model or offload
        model = AutoModelForCausalLM.from_pretrained(
            configs.okt_model,
            torch_dtype=torch.float16,
            device_map=None,  # manual placement
            low_cpu_mem_usage=True,
        )
        # Move to MPS
        model = model.to(device)

    else:
        # CPU fallback: float32, slow but works
        model = AutoModelForCausalLM.from_pretrained(
            configs.okt_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    # Apply LoRA (works on all devices)
    lora_config = LoraConfig(
        lora_alpha=configs.lora_alpha,
        lora_dropout=configs.lora_dropout,
        r=configs.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    lstm = create_lstm_model(configs, device, lstm_hid_dim)

    return lstm, model, tokenizer


def create_multitask_predictor(device, multilayer=False):
    if multilayer:
        predictor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for layer in predictor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    else:
        predictor = nn.Linear(4096, 1)
        torch.nn.init.xavier_uniform_(predictor.weight)

    predictor = predictor.to(device)
    return predictor


def create_binary_predictor(device):
    predictor = nn.Linear(4096, 2)
    torch.nn.init.xavier_uniform_(predictor.weight)
    predictor = predictor.to(device)
    return predictor


def create_knowledge_linear(device, hid_dim, transition_dim):
    if transition_dim == 64:
        linear = nn.Sequential(nn.ReLU(), nn.Linear(transition_dim, hid_dim))
    else:
        linear = nn.Sequential(
            nn.Linear(transition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hid_dim),
        )
    linear = linear.to(device)
    return linear


def load_model_eval_apple(configs, checkpoint_dir, device, no_kc):
    """Load saved model for evaluation on Apple Silicon."""
    model_dir = os.path.join(checkpoint_dir, 'model')

    if device.type == 'cuda':
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        peft_config = PeftConfig.from_pretrained(model_dir)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
        )
    else:
        peft_config = PeftConfig.from_pretrained(model_dir)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float16 if device.type == 'mps' else torch.float32,
            low_cpu_mem_usage=True,
        )
        _hf_model = _hf_model.to(device)

    model = PeftModel.from_pretrained(_hf_model, model_dir, is_trainable=False).to(device)
    model.eval()

    tokenizer = create_tokenizer(configs)

    lstm_hid_dim = no_kc
    if configs.transition:
        lstm = create_lstm_model(configs, device, configs.transition_dim)
        lstm.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'lstm'), map_location=device, weights_only=True))
        trans_linear = create_knowledge_linear(device, lstm_hid_dim, configs.transition_dim)
        trans_linear.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'transition'), map_location=device, weights_only=True))
    else:
        lstm = create_lstm_model(configs, device, lstm_hid_dim)
        lstm.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'lstm'), map_location=device, weights_only=True))
        trans_linear = None

    predictor = None
    if configs.multitask:
        predictor = create_multitask_predictor(device, getattr(configs, 'predictor_multilayer', False))
        predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'predictor'), map_location=device, weights_only=True))

    return model, lstm, predictor, tokenizer, trans_linear
