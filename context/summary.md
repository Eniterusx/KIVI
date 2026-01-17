# KVQuant Repository Changes Analysis

This document summarizes the changes made to the KVQuant repository to support custom GPT-2 and SmolLM models, and the addition of evaluation scripts for multiple datasets.

## 1. Core Quantization Script Modifications

The original `llama_simquant.py` was adapted to create `gpt2_simquant.py` and `smollm360_simquant.py`.

### `llama_simquant.py` (Original)
- **Target**: LLaMA models (e.g., LLaMA-2).
- **Loading**: Uses `transformers.AutoModelForCausalLM`.
- **Structure**: Assumes standard LLaMA architecture (`model.model.layers`, `model.lm_head`).
- **Calibration/Eval**: Standard forward pass hook interception.

### `gpt2_simquant.py` (New)
- **Target**: Custom Distilled GPT-2 model.
- **Imports**: `distilled_mla` (Custom model definition).
- **Loading (`get_model`)**:
  - Checks for "gpt2" in model name.
  - Loads `GPT2Config` and `AutoTokenizer` from "gpt2".
  - Instantiates `distilled_mla.Transformer`.
  - Loads weights from `ckp_*.pth` or `pytorch_model.bin`.
- **Evaluation (`gpt2_eval`)**:
  - Adapted for GPT-2 architecture: `model.h` (layers), `model.wte` (token embeddings), `model.wpe` (position embeddings).
  - Manually computes embeddings and positional embeddings before the first layer to simulate the forward pass for calibration/eval.
- **Calibration (`gpt2_calibration`)**:
  - Similar adaptation as `gpt2_eval` for the calibration loop.
  - Intercepts layer inputs after manual embedding computation.

### `smollm360_simquant.py` (New)
- **Target**: Custom SmolLM 360M model.
- **Imports**: `distilled_smollm` (Custom model definition).
- **Loading (`get_model`)**:
  - Checks for "smollm360".
  - Defines `SmolLM2Config` manually with specific parameters (n_embd=960, n_layer=32, etc.).
  - Instantiates `distilled_smollm.Transformer`.
  - Loads weights from `pytorch_model.bin`.
- **Evaluation/Calibration**:
  - Reuses `llama_eval` and `llama_calibration` but with modifications.
  - Checks for `type(layer).__name__ == 'Block'` to handle the custom model's layer class.
  - Handles `attention_mask` and `position_ids` appropriately for the custom model.

## 2. Evaluation Scripts

New scripts were added to evaluate perplexity on 3 datasets: **Wikitext-103**, **Wikitext-2**, and **TinyStories**.

### `evaluate_gpt2_perplexity.py`
- **Model**: Custom GPT-2 (`distilled_mla`).
- **Datasets**: Supports `wikitext-103`, `wikitext-2`, `tinystories`.
- **Quantization**:
  - Loads pre-computed quantizers from a pickle file.
  - Applies quantization using `make_quant_sim`.
  - Maps `model.layers` keys to `h` to match GPT-2 structure.
  - Applies per-channel quantization to `k_proj` (Wk) and per-token to `v_proj` (Wv).
- **Evaluation**:
  - Uses `evaluate_wikitext` from `utils.py`.
  - Wraps the model in `ModelWrapper` to ensure the forward pass returns logits (handling the custom model's output format).

### `evaluate_smollm17_perplexity.py`
- **Model**: Custom SmolLM 1.7B (`distilled_smollm`).
- **Configuration**: Manually defines `SmolLM2Config` for the 1.7B model (n_embd=2048, n_layer=24, etc.).
- **RoPE Patch**:
  - Forces recomputation of RoPE tables (`pre_compute_rope`) with a specific theta (1,000,000) to ensure correctness, as checkpoint buffers might be stale.
- **Quantization**: Similar logic to GPT-2, applying per-channel and per-token quantization.
- **Evaluation**:
  - Uses `evaluate_wikitext`.
  - Wraps the model to handle output format.
  - Explicitly handles dataset loading for the 3 supported datasets.

## 3. Custom Models and Utilities

The scripts rely on new context files:
- **`context/distilled_mla.py`**: Definition of the custom GPT-2 model.
- **`context/distilled_smollm.py`**: Definition of the custom SmolLM model.
- **`context/utils.py`**: Utility functions for dataset loading (`get_wikitext_103`, `get_wikitext_2`, `get_tinystories`) and evaluation (`evaluate_wikitext`).

## Summary

The repository has been extended from supporting only LLaMA models to supporting custom **GPT-2** and **SmolLM** architectures. This involved:
1.  **Custom Model Integration**: Implementing `get_model` logic to load specific architectures and checkpoints.
2.  **Architecture Adaptation**: Modifying the calibration and evaluation loops to handle different layer structures (e.g., `model.h` vs `model.layers`) and embedding mechanisms.
3.  **Comprehensive Evaluation**: Adding dedicated scripts to evaluate these models on multiple datasets (Wikitext-103, Wikitext-2, TinyStories) with quantization support.
