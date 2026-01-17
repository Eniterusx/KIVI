import os
import time
import argparse
import math
import random
import numpy as np

from sklearn.metrics import matthews_corrcoef

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_distill(model, dataloader, loss_fn, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["input_ids"].to(device), batch["label_ids"].to(device)[:, 1:]

            logits, _ = model(x)
            logits = logits[:, :-1, :].reshape(-1, vocab_size)
            targets = y.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)

    return avg_loss, bpc, ppl

def evaluate_LM(model, dataloader, loss_fn, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = x.clone()[:, 1:]

            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            logits = logits[:, :-1, :].reshape(-1, vocab_size)
            targets = y.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)

    return avg_loss, bpc, ppl


def evaluate(model, dataloader, loss_fn, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = x.clone()[:, 1:]

            logits = model(x)

            logits = logits[:, :-1, :].reshape(-1, vocab_size)
            targets = y.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)

    return avg_loss, bpc, ppl


def evaluate_wikitext(model, dataloader, loss_fn, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["input_ids"].to(device), batch["label_ids"].to(device)[:, 1:]

            logits = model(x)
            logits = logits[:, :-1, :].reshape(-1, vocab_size)
            targets = y.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)

    return avg_loss, bpc, ppl


def evaluate_lambada(
    model,
    dataloader,
    tokenizer,
    device,
):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets = batch["input_ids"].to(device), batch["labels"].to(
                device
            )

            outputs = model(input_ids)
            last_token_indices = (input_ids != tokenizer.pad_token_id).sum(dim=1) - 1
            logits = outputs[range(len(input_ids)), last_token_indices]
            predictions = torch.argmax(logits, dim=-1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    return accuracy


def evaluate_GLUE(model, dataloader, loss_fn, metric, device, vocab_size):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)

            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds)
            all_labels.append(y)

    avg_loss = total_loss / len(dataloader.dataset)
    score = metric.compute(
        predictions=torch.cat(all_preds).cpu().numpy(),
        references=torch.cat(all_labels).cpu().numpy(),
    )

    return avg_loss, score


def get_wikitext_103(tokenizer, block_size):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)


    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["label_ids"] = result["input_ids"]
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    val_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    test_dataset.set_format("torch", columns=["input_ids", "label_ids"])

    return train_dataset, val_dataset, test_dataset


def get_wikitext_2(tokenizer, block_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["label_ids"] = result["input_ids"]
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    val_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    test_dataset.set_format("torch", columns=["input_ids", "label_ids"])

    return train_dataset, val_dataset, test_dataset

def get_tinystories(tokenizer, block_size):
    dataset = load_dataset("roneneldan/TinyStories")
    
    validation_test_split = dataset["validation"].train_test_split(test_size=0.5, seed=42)
    
    dedicated_rng = random.Random(42)
    train_len = len(dataset["train"])
    ten_percent_len = int(0.1 * train_len)
    selected_indices = dedicated_rng.sample(range(train_len), ten_percent_len)
    ten_percent_train = dataset["train"].select(selected_indices)

    dataset = DatasetDict({
        "train": ten_percent_train,
        "validation": validation_test_split["train"],
        "test": validation_test_split["test"]
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["label_ids"] = result["input_ids"]
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    val_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    test_dataset.set_format("torch", columns=["input_ids", "label_ids"])

    return train_dataset, val_dataset, test_dataset

def get_tinystories_de(tokenizer, block_size):
    dataset = load_dataset("roneneldan/TinyStories")
    
    validation_test_split = dataset["validation"].train_test_split(test_size=0.5, seed=42)
    
    dedicated_rng = random.Random(42)
    train_len = len(dataset["train"])
    ten_percent_len = int(0.01 * train_len)
    selected_indices = dedicated_rng.sample(range(train_len), ten_percent_len)
    ten_percent_train = dataset["train"].select(selected_indices)

    dataset = DatasetDict({
        "train": ten_percent_train,
        "validation": validation_test_split["train"],
        "test": validation_test_split["test"]
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["label_ids"] = result["input_ids"]
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    val_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    test_dataset.set_format("torch", columns=["input_ids", "label_ids"])

    return train_dataset, val_dataset, test_dataset

def get_wikitext_103_de(tokenizer, block_size):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    dedicated_rng = random.Random(42)
    train_len = len(dataset["train"])
    ten_percent_len = int(0.1 * train_len)
    selected_indices = dedicated_rng.sample(range(train_len), ten_percent_len)
    ten_percent_train = dataset["train"].select(selected_indices)

    dataset = DatasetDict({
        "train": ten_percent_train,
        "validation": dataset["validation"],
        "test": dataset["test"]
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["label_ids"] = result["input_ids"]
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    val_dataset.set_format("torch", columns=["input_ids", "label_ids"])
    test_dataset.set_format("torch", columns=["input_ids", "label_ids"])

    return train_dataset, val_dataset, test_dataset


def get_lambada(tokenizer, block_size):
    dataset = load_dataset("lambada")

    def prepare_lambada_example(examples):

        input_ids = []
        labels = []

        for text_example in examples["text"]:
            tokenized_full = tokenizer(
                text_example, truncation=True, add_special_tokens=True
            )

            if len(tokenized_full["input_ids"]) > 1:
                input_ids.append(tokenized_full["input_ids"][:-1])
                labels.append(tokenized_full["input_ids"][-1])

        return {"input_ids": input_ids, "labels": labels}

    processed_datasets = dataset.map(
        prepare_lambada_example,
        batched=True,
        remove_columns=["text", "domain"],
    ).filter(lambda example: example["input_ids"] and example["labels"])

    train_dataset = processed_datasets["train"]
    val_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    train_dataset.set_format("torch", columns=["input_ids", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "labels"])

    return train_dataset, val_dataset, test_dataset


def get_cola(tokenizer, max_seq_len):
    dataset = load_dataset("glue", "cola")

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"], dataset["test"]


def get_sst2(tokenizer, max_seq_len):
    dataset = load_dataset("glue", "sst2")

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"], dataset["test"]


def get_mrpc(tokenizer, max_seq_len):
    dataset = load_dataset("glue", "mrpc")

    def preprocess(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"], dataset["test"]


def get_rte(tokenizer, max_seq_len):
    dataset = load_dataset("glue", "rte")

    def preprocess(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"], dataset["test"]


def get_wnli(tokenizer, max_seq_len):
    dataset = load_dataset("glue", "wnli")

    def preprocess(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"], dataset["test"]


def get_pubmed(tokenizer, block_size):
    dataset = load_dataset("common-pile/pubmed_filtered", split="train")

    def prepare_example(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = tokenized_batch["input_ids"]
        return {"input_ids": input_ids}

    old_columns = dataset.column_names
    processed_dataset = dataset.map(
        prepare_example,
        batched=True,
        remove_columns=old_columns,
    )

    split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    test_val_split = split["test"].train_test_split(test_size=0.5, seed=42)

    train_dataset = split["train"]
    val_dataset = test_val_split["train"]
    test_dataset = test_val_split["test"]

    train_dataset.set_format("torch", columns=["input_ids"])
    val_dataset.set_format("torch", columns=["input_ids"])
    test_dataset.set_format("torch", columns=["input_ids"])

    return train_dataset, val_dataset, test_dataset


def get_legalpile(tokenizer, block_size):
    config = "en_caselaw"
    dataset = load_dataset(
        "joelniklaus/eurlex_resources", config, split="train", trust_remote_code=True
    )

    def prepare_example(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = tokenized_batch["input_ids"]
        return {"input_ids": input_ids}

    old_columns = dataset.column_names
    processed_dataset = dataset.map(
        prepare_example,
        batched=True,
        remove_columns=old_columns,
    )

    split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    test_val_split = split["test"].train_test_split(test_size=0.5, seed=42)

    train_dataset = split["train"]
    val_dataset = test_val_split["train"]
    test_dataset = test_val_split["test"]

    train_dataset.set_format("torch", columns=["input_ids"])
    val_dataset.set_format("torch", columns=["input_ids"])
    test_dataset.set_format("torch", columns=["input_ids"])

    return train_dataset, val_dataset, test_dataset


def get_fin(tokenizer, block_size):
    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")

    def prepare_example(examples):
        texts = [
            s + "\n" + u + "\n" + a
            for s, u, a in zip(
                examples["system"], examples["user"], examples["assistant"]
            )
        ]
        tokenized_batch = tokenizer(
            texts,
            truncation=True,
            max_length=block_size,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = tokenized_batch["input_ids"]
        return {"input_ids": input_ids}

    old_columns = dataset.column_names
    processed_dataset = dataset.map(
        prepare_example,
        batched=True,
        remove_columns=old_columns,
    )

    split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    test_val_split = split["test"].train_test_split(test_size=0.5, seed=42)

    train_dataset = split["train"]
    val_dataset = test_val_split["train"]
    test_dataset = test_val_split["test"]

    train_dataset.set_format("torch", columns=["input_ids"])
    val_dataset.set_format("torch", columns=["input_ids"])
    test_dataset.set_format("torch", columns=["input_ids"])

    return train_dataset, val_dataset, test_dataset


def compute_class_weights(dataset, num_labels):
    labels = [x["labels"] for x in dataset]
    counts = torch.bincount(torch.tensor(labels), minlength=num_labels)
    weights = 1.0 / (counts.float() + 1e-6)
    weights = weights / weights.sum()
    return weights


def prune_state_dict_from_mu_c(state_dict, mu_pruning_threshold):
    lat_dims = []
    for layer_idx in range(12):
        prefix = f"h.{layer_idx}.attn"
        mu_key = f"{prefix}.mu_c"

        mu_c = state_dict[mu_key]
        keep_mask = mu_c.abs() >= mu_pruning_threshold
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        lat_dims.append(len(keep_idx))

        state_dict[mu_key] = state_dict[mu_key][keep_idx]

        Wc_key = f"{prefix}.Wc.weight"
        state_dict[Wc_key] = state_dict[Wc_key][keep_idx, :]
        state_dict[f"{prefix}.Wc.bias"] = state_dict[f"{prefix}.Wc.bias"][keep_idx]

        Wk_key = f"{prefix}.Wk.weight"
        state_dict[Wk_key] = state_dict[Wk_key][:, keep_idx]

        Wv_key = f"{prefix}.Wv.weight"
        state_dict[Wv_key] = state_dict[Wv_key][:, keep_idx]

    return state_dict, lat_dims

def smollm_prune_state_dict_from_mu_c(state_dict, mu_pruning_threshold):
    lat_dims = []
    for layer_idx in range(24):
        prefix = f"model.layers.{layer_idx}.self_attn"
        mu_key = f"{prefix}.mu_c"

        mu_c = state_dict[mu_key]
        keep_mask = mu_c.abs() >= mu_pruning_threshold
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        lat_dims.append(len(keep_idx))

        state_dict[mu_key] = state_dict[mu_key][keep_idx]

        Wc_key = f"{prefix}.Wc.weight"
        state_dict[Wc_key] = state_dict[Wc_key][keep_idx, :]
        state_dict[f"{prefix}.Wc.bias"] = state_dict[f"{prefix}.Wc.bias"][keep_idx]

        Wk_key = f"{prefix}.k_proj.weight"
        state_dict[Wk_key] = state_dict[Wk_key][:, keep_idx]

        Wv_key = f"{prefix}.v_proj.weight"
        state_dict[Wv_key] = state_dict[Wv_key][:, keep_idx]

    return state_dict, lat_dims

def smollm_prune_state_dict_from_mu_c_360(state_dict, mu_pruning_threshold):
    lat_dims = []
    for layer_idx in range(32):
        prefix = f"model.layers.{layer_idx}.self_attn"
        mu_key = f"{prefix}.mu_c"

        mu_c = state_dict[mu_key]
        keep_mask = mu_c.abs() >= mu_pruning_threshold
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        lat_dims.append(len(keep_idx))

        state_dict[mu_key] = state_dict[mu_key][keep_idx]

        Wc_key = f"{prefix}.Wc.weight"
        state_dict[Wc_key] = state_dict[Wc_key][keep_idx, :]
        state_dict[f"{prefix}.Wc.bias"] = state_dict[f"{prefix}.Wc.bias"][keep_idx]

        Wk_key = f"{prefix}.k_proj.weight"
        state_dict[Wk_key] = state_dict[Wk_key][:, keep_idx]

        Wv_key = f"{prefix}.v_proj.weight"
        state_dict[Wv_key] = state_dict[Wv_key][:, keep_idx]

    return state_dict, lat_dims
