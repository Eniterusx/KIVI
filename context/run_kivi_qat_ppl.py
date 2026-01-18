import argparse
import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import math
from transformers import AutoTokenizer, GPT2Config, default_data_collator

from kivi_gpt2 import KIVIQuantConfig, TransformerKIVI as GPT2KIVI
from kivi_smollm import SmolLM2Config, TransformerKIVI as SmolLMKIVI
from utils import set_seed, get_wikitext_2, get_wikitext_103_de, get_tinystories_de


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


WEIGHTS_MAP = {
    "gpt2": {
        "wikitext-2": "weights/gpt2-test/best.pth",
        "wikitext-103": "weights/gpt2-wikitext-103/pytorch_model.pth",
        "tinystories": "weights/gpt2-tinystories/pytorch_model.pth",
    },
    "smollm360": {
        "wikitext-2": "weights/smollm360-wikitext-2/pytorch_model.bin",
        "wikitext-103": "weights/smollm360-wikitext-103/pytorch_model.bin",
        "tinystories": "weights/smollm360-tinystories/pytorch_model.bin",
    },
}


def _resolve_weights_path(model_name: str, dataset: str) -> str:
    rel_path = WEIGHTS_MAP[model_name][dataset]
    path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing weights at {path}")
    return path


def _get_dataset(tokenizer, dataset: str, block_size: int):
    if dataset == "wikitext-2":
        return get_wikitext_2(tokenizer, block_size)
    if dataset == "wikitext-103":
        return get_wikitext_103_de(tokenizer, block_size)
    if dataset == "tinystories":
        return get_tinystories_de(tokenizer, block_size)
    raise ValueError(f"Unknown dataset: {dataset}")


def _infer_gpt2_n_ctx(state_dict: Dict[str, torch.Tensor], fallback: int) -> int:
    for key in ("wpe.weight", "transformer.wpe.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return fallback


def _build_gpt2_model(state_dict: Dict[str, torch.Tensor], vocab_size: int, kivi_cfg: KIVIQuantConfig) -> GPT2KIVI:
    n_ctx = _infer_gpt2_n_ctx(state_dict, fallback=1024)
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.n_ctx = n_ctx
    gpt2_config.n_positions = n_ctx
    gpt2_config.vocab_size = vocab_size
    gpt2_config.k_bits = kivi_cfg.k_bits
    gpt2_config.v_bits = kivi_cfg.v_bits
    gpt2_config.group_size = kivi_cfg.group_size
    gpt2_config.residual_length = kivi_cfg.residual_length
    gpt2_config.use_kivi = kivi_cfg.use_kivi
    return GPT2KIVI(gpt2_config, quant_config=kivi_cfg)


def _build_smollm_model(tokenizer, rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor], args) -> SmolLMKIVI:
    config = SmolLM2Config(
        n_embd=960,
        n_hidden=2560,
        bias=False,
        block_size=max(args.block_size, 8192),
        n_layer=32,
        n_head=15,
        n_kv_heads=5,
        norm_eps=1e-05,
        rope_theta=args.rope_theta,
        vocab_size=tokenizer.vocab_size,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        use_kivi=True,
    )
    return SmolLMKIVI(config, tokenizer=tokenizer, rope_cos=rope_cos, rope_sin=rope_sin)


def _train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, pad_token_id: int, desc: str):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        if "label_ids" in batch:
            labels = batch["label_ids"].to(device)
        else:
            labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        labels = labels[:, 1:].contiguous()

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    return total_loss / max(total_tokens, 1)


def _evaluate_ppl(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, pad_token_id: int, vocab_size: int, desc: str):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            if "label_ids" in batch:
                labels = batch["label_ids"].to(device)
            else:
                labels = input_ids.clone()
            labels[labels == pad_token_id] = -100
            labels = labels[:, 1:].contiguous()

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)
    return avg_loss, bpc, ppl


def _evaluate_ppl_autoregressive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    vocab_size: int,
    desc: str,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            if "label_ids" in batch:
                labels = batch["label_ids"].to(device)
            else:
                labels = input_ids.clone()
            labels[labels == pad_token_id] = -100

            seq_len = input_ids.size(1)
            for t in range(1, seq_len):
                targets = labels[:, t]
                if (targets != -100).any():
                    logits = model(input_ids[:, :t])
                    step_logits = logits[:, -1, :]
                    loss = loss_fn(step_logits, targets)
                    total_loss += loss.item()
                    total_tokens += (targets != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)
    return avg_loss, bpc, ppl


def _evaluate_ppl_streaming(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    vocab_size: int,
    desc: str,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            if "label_ids" in batch:
                labels = batch["label_ids"].to(device)
            else:
                labels = input_ids.clone()
            labels[labels == pad_token_id] = -100

            seq_len = input_ids.size(1)
            past_kv = None
            
            # Streaming evaluation: process token by token, maintaining cache
            for t in range(seq_len - 1):
                input_token = input_ids[:, t:t+1]
                target_token = labels[:, t+1]
                
                # Forward pass with cache updates
                # Note: valid only for models supporting past_kv (e.g. unmodified KIVI models)
                if hasattr(model, 'module'):
                    logits, past_kv = model.module(input_token, past_kv=past_kv, use_cache=True)
                else:
                    logits, past_kv = model(input_token, past_kv=past_kv, use_cache=True)
                
                step_logits = logits[:, -1, :]
                
                if (target_token != -100).any():
                    loss = loss_fn(step_logits, target_token)
                    total_loss += loss.item()
                    total_tokens += (target_token != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)
    return avg_loss, bpc, ppl


def _set_kivi_enabled(model: torch.nn.Module, enabled: bool):
    for module in model.modules():
        if hasattr(module, "quant_config") and module.quant_config is not None:
            module.quant_config.use_kivi = enabled
        if hasattr(module, "config") and hasattr(module.config, "use_kivi"):
            module.config.use_kivi = enabled


def _sanitize_state_dict(state_dict: Dict[str, torch.Tensor], model_name: str) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict
    sanitized = {}
    for key, value in state_dict.items():
        new_key = key
        if model_name == "gpt2" and new_key.startswith("transformer."):
            new_key = new_key[len("transformer."):]
        if model_name == "gpt2" and new_key.startswith("model."):
            new_key = new_key[len("model."):]
        sanitized[new_key] = value
    return sanitized


def run_experiment(model_name: str, dataset: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Running {model_name} on {dataset} ===", flush=True)
    print(f"Device: {device}", flush=True)

    if model_name == "gpt2":
        print("Loading GPT-2 tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif model_name == "smollm360":
        print("Loading SmolLM360 tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Preparing datasets (block_size={args.block_size})...", flush=True)
    train_dataset, val_dataset, _ = _get_dataset(tokenizer, dataset, args.block_size)
    train_bs = args.batch_size
    eval_bs = args.eval_batch_size
    if model_name == "smollm360":
        train_bs = max(1, train_bs // 2)
        eval_bs = max(1, eval_bs // 2)
        print(f"Using half batch size for SmolLM360: train={train_bs}, eval={eval_bs}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, collate_fn=default_data_collator)

    weights_path = _resolve_weights_path(model_name, dataset)
    print(f"Loading weights from {weights_path}", flush=True)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if isinstance(state_dict, dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    kivi_cfg = KIVIQuantConfig(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        use_kivi=True,
    )

    if model_name == "gpt2":
        print("Building GPT-2 KIVI model...", flush=True)
        model = _build_gpt2_model(state_dict, tokenizer.vocab_size, kivi_cfg)
        state_dict = _sanitize_state_dict(state_dict, model_name)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"GPT-2 load_state_dict: {len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected",
                flush=True,
            )
            if incompatible.missing_keys:
                print(f"Missing keys (sample): {incompatible.missing_keys[:5]}", flush=True)
            if incompatible.unexpected_keys:
                print(f"Unexpected keys (sample): {incompatible.unexpected_keys[:5]}", flush=True)
    else:
        print("Building SmolLM360 KIVI model...", flush=True)
        model = _build_smollm_model(tokenizer, None, None, args)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"SmolLM360 load_state_dict: {len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected",
                flush=True,
            )
            if incompatible.missing_keys:
                print(f"Missing keys (sample): {incompatible.missing_keys[:5]}", flush=True)
            if incompatible.unexpected_keys:
                print(f"Unexpected keys (sample): {incompatible.unexpected_keys[:5]}", flush=True)

    model.to(device)

    print("Evaluating full-precision model (no KIVI)...", flush=True)
    _set_kivi_enabled(model, False)
    
    if args.eval_mode == "autoregressive":
        eval_fn = _evaluate_ppl_autoregressive
    elif args.eval_mode == "streaming":
        eval_fn = _evaluate_ppl_streaming
    else:
        eval_fn = _evaluate_ppl
        
    # epoch_eval_fn = _evaluate_ppl if args.eval_mode in ["autoregressive", "streaming"] else eval_fn
    epoch_eval_fn = eval_fn
    # base_eval_fn = _evaluate_ppl if args.eval_mode in ["autoregressive", "streaming"] else eval_fn
    base_eval_fn = _evaluate_ppl

    base_loss, base_bpc, base_ppl = base_eval_fn(
        model,
        val_loader,
        device,
        tokenizer.pad_token_id,
        tokenizer.vocab_size,
        desc=f"Eval FP {model_name}-{dataset}",
    )
    print(
        f"[{model_name}/{dataset}] FP val loss {base_loss:.4f} | val ppl {base_ppl:.4f}",
        flush=True,
    )

    print("Evaluating quantized model (KIVI enabled)...", flush=True)
    _set_kivi_enabled(model, True)
    q_loss, q_bpc, q_ppl = base_eval_fn(
        model,
        val_loader,
        device,
        tokenizer.pad_token_id,
        tokenizer.vocab_size,
        desc=f"Eval KIVI {model_name}-{dataset}",
    )
    print(
        f"[{model_name}/{dataset}] KIVI val loss {q_loss:.4f} | val ppl {q_ppl:.4f}",
        flush=True,
    )

    print(f"Starting QAT (KIVI enabled): epochs={args.epochs}, lr={args.lr}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        _set_kivi_enabled(model, True)
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, tokenizer.pad_token_id, desc=f"Train {model_name}-{dataset} [epoch {epoch + 1}]"
        )
        val_loss, val_bpc, val_ppl = epoch_eval_fn(
            model,
            val_loader,
            device,
            tokenizer.pad_token_id,
            tokenizer.vocab_size,
            desc=f"Eval {model_name}-{dataset} [epoch {epoch + 1}]",
        )
        print(
            f"[{model_name}/{dataset}] epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val ppl {val_ppl:.4f}",
            flush=True,
        )

    if args.eval_mode in ["autoregressive", "streaming"] and args.epochs > 0:
        final_eval_fn = _evaluate_ppl_autoregressive if args.eval_mode == "autoregressive" else _evaluate_ppl_streaming
        final_loss, final_bpc, final_ppl = final_eval_fn(
            model,
            val_loader,
            device,
            tokenizer.pad_token_id,
            tokenizer.vocab_size,
            desc=f"Final {args.eval_mode.capitalize()} Eval {model_name}-{dataset}",
        )
        print(
            f"[{model_name}/{dataset}] final autoregressive | val loss {final_loss:.4f} | val ppl {final_ppl:.4f}",
            flush=True,
        )

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}-{dataset}-qat.pt")
    torch.save({"state_dict": model.state_dict(), "config": asdict(kivi_cfg)}, out_path)
    print(f"Saved QAT model to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="QAT + perplexity evaluation for GPT-2 and SmolLM360 with KIVI quantization.")
    parser.add_argument("--model", choices=["gpt2", "smollm360", "all"], default="all")
    parser.add_argument("--dataset", choices=["wikitext-2", "wikitext-103", "tinystories", "all"], default="all")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--rope-theta", type=int, default=100000)
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--output-dir", type=str, default="outputs/qat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-mode", choices=["batched", "autoregressive", "streaming"], default="batched")
    return parser.parse_args()


def main():
    args = parse_args()
    args.dtype = torch.float32

    set_seed(args.seed)

    models = ["gpt2", "smollm360"] if args.model == "all" else [args.model]
    datasets = ["wikitext-2", "wikitext-103", "tinystories"] if args.dataset == "all" else [args.dataset]

    for model_name in models:
        for dataset in datasets:
            run_experiment(model_name, dataset, args)


if __name__ == "__main__":
    main()