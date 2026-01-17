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

from context.gpt2 import Transformer
from utils import set_seed, get_tinystories, get_wikitext_2, get_wikitext_103, evaluate_distill2

def get_pruned(
    state_dict: dict,
    mu_pruning_threshold: float = None,
    device: str = "cpu",
) -> tuple[int, int]:
    """
    Calculates the number of pruned mu parameters and total mu parameters in a state dictionary.
    """
    pruned_count = 0
    total_mu_params = 0
    total_kv_params = 0
    sum_mu = 0

    new_state_dict = state_dict.copy()

    for key, tensor in state_dict.items():
        if "mu_c" in key:
            if not isinstance(tensor, torch.Tensor):
                continue

            tensor_on_device = tensor.to(device)
            sum_mu += tensor_on_device.sum().item()
            total_mu_params += tensor_on_device.numel()
            pruning_mask = torch.abs(tensor_on_device) < mu_pruning_threshold
            pruned_count += pruning_mask.sum().item()

            new_state_dict[key][pruning_mask] = 0.0

    return pruned_count, total_mu_params, new_state_dict


# %%
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = f"HuggingFaceTB/SmolLM2-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = "wikitext-2"

if dataset == "wikitext-103":
    #_, val_dataset, _ = get_wikitext_103(tokenizer, 512)
    ckp = torch.load("de_dmla_smollm_360M_wikitext-103_1755530637/checkpoint-120396/pytorch_model.bin")
elif dataset == "wikitext-2":
    #_, val_dataset, _ = get_wikitext_2(tokenizer, 512)
    ckp = torch.load("dmla_smollm_360M_wikitext-2_1755532913/checkpoint-103075/pytorch_model.bin")
elif dataset == "tinyStories":
    #_, val_dataset, _ = get_tinystories(tokenizer, 512)
    ckp = torch.load("de_dmla_smollm_360M_tinyStories_1755530638/checkpoint-110110/pytorch_model.bin")
else:
    raise NotImplementedError(f"Dataset {dataset} not supported.")

from context.partial_smollm import Transformer, SmolLM2Config
config = SmolLM2Config(
        n_embd= 960, 
        n_hidden=2560, 
        bias = False, 
        block_size=8192, 
        n_layer=32, 
        n_head=15, 
        n_kv_heads=5, 
        norm_eps=1e-05, 
        rope_theta=100000, 
        vocab_size=49152,
        dtype=torch.float32
    ) 
model = Transformer(config, tokenizer=tokenizer).to("cuda:0")
model.load_state_dict(ckp, strict=False)

collate_fn = default_data_collator
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#val_loader = DataLoader(val_dataset, batch_size=64)

mu_pruning_thresholds = [0.01 * i for i in range(100)]
alpha_pruning_thresholds = []

for threshold in mu_pruning_thresholds:
    pruned_count, total_mu_params, pruned_state_dict = get_pruned(
        ckp, mu_pruning_threshold=threshold, device=device
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

        #model.load_state_dict(pruned_state_dict, strict=False)
        #val_loss, val_bpc, val_ppl = evaluate_distill(
        #    model, val_loader, loss_fn, device, tokenizer.vocab_size
        #)
        #print(
        #    f"            Val Loss: {val_loss:.4f} | Val bpc: {val_bpc:.4f} | Val ppl: {val_ppl:.4f}",
        #    flush=True,
        #)


# %%


# %%



