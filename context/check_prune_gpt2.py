# %%
import os
import time
import argparse
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    default_data_collator,
    GPT2Config,
    GPT2LMHeadModel,
)

from model.distilled_mla import Transformer
from utils import set_seed, get_tinystories, get_wikitext_2, get_wikitext_103, evaluate_distill

def get_pruned(
    state_dict: dict,
    alpha: bool,
    mu_pruning_threshold: float = None,
    alpha_pruning_threshold: float = None,
    device: str = "cpu",
) -> tuple[int, int]:
    """
    Calculates the number of pruned mu parameters and total mu parameters in a state dictionary.
    """
    pruned_count = 0
    total_mu_params = 0

    new_state_dict = state_dict.copy()

    if alpha:
        for i in range(12):
            mu_c = state_dict[f"h.{i}.attn.mu_c"].to(device)
            sigma_c = state_dict[f"h.{i}.attn.sigma_c"].to(device)

            total_mu_params += mu_c.numel()
            pruning_mask_c = (mu_c / sigma_c).pow(2) < alpha_pruning_threshold
            pruned_count += pruning_mask_c.sum().item()

            new_state_dict[f"h.{i}.attn.mu_c"][pruning_mask_c] = 0.0

    else:
        for key, tensor in state_dict.items():
            if "mu_c" in key:
                if not isinstance(tensor, torch.Tensor):
                    continue

                tensor_on_device = tensor.to(device)
                total_mu_params += tensor_on_device.numel()
                pruning_mask = torch.abs(tensor_on_device) < mu_pruning_threshold
                pruned_count += pruning_mask.sum().item()

                new_state_dict[key][pruning_mask] = 0.0

    return pruned_count, total_mu_params, new_state_dict

# %%
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_config = GPT2Config.from_pretrained("gpt2")
model = Transformer(gpt2_config).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

_, val_dataset, _ = get_wikitext_103(tokenizer, 512)
collate_fn = default_data_collator
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
val_loader = DataLoader(val_dataset, batch_size=16)

# mu_pruning_thresholds = [0.27, 0.28, 0.29, 0.46, 0.47, 0.48, 0.63, 0.64, 0.65]
mu_pruning_thresholds = [0.00, 0.01]
# mu_pruning_thresholds = [0.46, 0.47, 0.48, 0.63, 0.64, 0.65]
alpha_pruning_thresholds = []

main_dir = "../models"

print(f"\nProcessing checkpoints in directory: {main_dir}")

checkpoint_files = [f"{main_dir}/gpt2-wikitext-2/ckp_750.pth"]

if not checkpoint_files:
    print(f"No checkpoint files found in '{main_dir}'. Skipping.")

for input_path in checkpoint_files:
    print(f"\n--- Loading checkpoint: {input_path} ---")

    try:
        state_dict = torch.load(input_path)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(
            f"Error loading state dictionary from {input_path}: {e}. Skipping this checkpoint."
        )
        continue

    print(f"\n  MU pruning {input_path}")
    for threshold in mu_pruning_thresholds:
        pruned_count, total_mu_params, pruned_state_dict = get_pruned(
            state_dict, alpha=False, mu_pruning_threshold=threshold, device=device
        )

        if total_mu_params == 0:
            print(
                f"        Threshold {threshold:.6f}: No 'mu_qk' or 'mu_v' parameters found. No pruning applicable."
            )
        else:
            pruning_percentage = pruned_count / total_mu_params * 100
            print(
                f"        Threshold {threshold:.6f}: Pruned {pruned_count} / {total_mu_params} mu parameters ({pruning_percentage:.2f}%)"
            )

            model.load_state_dict(pruned_state_dict, strict=False)
            val_loss, val_bpc, val_ppl = evaluate_distill(
                model, val_loader, loss_fn, device, tokenizer.vocab_size
            )
            print(
                f"            Val Loss: {val_loss:.4f} | Val bpc: {val_bpc:.4f} | Val ppl: {val_ppl:.4f}",
                flush=True,
            )

    print(f"\n    ALPHA pruning {input_path}")
    for threshold in alpha_pruning_thresholds:
        pruned_count, total_mu_params, pruned_state_dict = get_pruned(
            state_dict, alpha=True, alpha_pruning_threshold=threshold, device=device
        )

        if total_mu_params == 0:
            print(
                f"        Threshold {threshold:.6f}: No 'mu_qk' or 'mu_v' parameters found. No pruning applicable."
            )
        else:
            pruning_percentage = pruned_count / total_mu_params * 100
            print(
                f"        Threshold {threshold:.6f}: Pruned {pruned_count} / {total_mu_params} mu parameters ({pruning_percentage:.2f}%)"
            )

            model.load_state_dict(pruned_state_dict, strict=False)
            val_loss, val_bpc, val_ppl = evaluate_distill(
                model, val_loader, loss_fn, device, tokenizer.vocab_size
            )
            print(
                f"            Val Loss: {val_loss:.4f} | Val bpc: {val_bpc:.4f} | Val ppl: {val_ppl:.4f}",
                flush=True,
            )



