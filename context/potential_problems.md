# KIVI Implementation Issues Analysis

# WARNING: Old file, the problems are mainly fixed in the new solution.

**Status**: Confirmed after code review ✓

## Confirmed Problems

### 1) Autoregressive eval is not streaming-cache eval ✓ CONFIRMED
   - In `context/run_kivi_qat_ppl.py`, `_evaluate_ppl_autoregressive` loops t=1..T-1 and calls `model(input_ids[:, :t])` each step.
   - This recomputes the full prefix every time, so it does not use KV cache nor "prefill + decode" behavior.
   - Complexity is O(T²) and significantly slower than cached decoding.
   - **Root cause**: The models (`kivi_smollm.py`, `kivi_gpt2.py`) don't accept/return `past_key_value` state.

### 2) Short-prefix quantization can produce NaNs ✓ CONFIRMED
   - In `models/utils_quant.py`, `AsymGroupedQuantizer` and `AsymGroupedQuantizerByChannel` compute `scale = (mx - mn) / (2^bits - 1)`.
   - For very short prefixes (e.g., length 1), mx==mn → scale=0 → division by zero → inf/NaN.
   - Autoregressive eval uses short prefixes at early steps, so NaNs are likely when KIVI quantization is enabled.
   - **Code reference**: `utils_quant.py` lines 191-192 and 230-231.

### 3) GPT-2/SmolLM KIVI residual-length logic bug ⚠️ PARTIALLY CORRECT (needs refinement)
   - **Actual behavior on line 83-84 of kivi_gpt2.py**:
     ```python
     if residual_length is None or residual_length <= 0 or residual_length >= seqlen:
         return self._quantize_key(key_states), self._quantize_value(value_states)
     ```
   - When `seqlen <= residual_length`, the condition `residual_length >= seqlen` is TRUE → quantizes ALL tokens.
   - This is **opposite** of KIVI intent: short sequences should be kept full-precision (they fit in residual window).
   - **SmolLM has extra protection** via `quantize_prefill` flag (line 111-112), but GPT-2 does not.

### 4) AR eval vs batched eval gives different signals for cache methods ✓ CONFIRMED
   - Batched eval uses full-sequence forward without streaming cache; it measures NLL but not cache behavior.
   - AR eval here is stepwise but still non-streaming, so it does not test cache quantization dynamics.
   - **Key insight**: Neither mode tests the actual KIVI residual→quantized flush behavior.

### 5) SmolLM has an additional confusing flag: `quantize_prefill` (NEW FINDING)
   - `kivi_smollm.py` line 111-112 returns early without quantization if `quantize_prefill=False`.
   - This means KIVI quantization only applies if explicitly enabled via `quantize_prefill=True`.
   - Default in `run_kivi_qat_ppl.py` requires `--quantize-prefill` flag to enable quantization.
   - **Without this flag, even "KIVI enabled" tests may run unquantized!**

### 6) Quantization during prefill doesn't match KIVI paper (NEW FINDING)
   - Original KIVI paper: prefill uses full precision, quantization happens incrementally during decode.
   - Context implementation: quantizes during prefill if `quantize_prefill=True`, or doesn't quantize at all if False.
   - There's no option for "prefill→decode with incremental quantization" as in the paper.

---

## Recommendations

### Immediate fixes (stability):
1. **Fix quantization stability**:
   - Clamp scale with epsilon: `scale = max(scale, 1e-8)`
   - Or short-circuit to full-precision when `mx==mn` or `seq_len < 2`.

2. **Fix residual-length logic for short sequences**:
   - When `seqlen <= residual_length`, return UNQUANTIZED K/V (they fit in residual window).
   - Only quantize when `seqlen > residual_length`.

### For fair KIVI vs KVIB comparison:
3. **Add streaming-cache evaluation path**:
   - Implement `past_key_value` support in `kivi_smollm.py` / `kivi_gpt2.py`.
   - Prefill once, then decode token-by-token with cached K/V updates.
   - Required if KVIB is a streaming cache method.

4. **Use batched eval for finetuning; streaming eval for cache comparison**:
   - Batched eval is sufficient for QAT training loops and is O(T) not O(T²).
   - Streaming eval should be a separate validation for cache methods.

---

## Worries

1. **PPL results may be skewed today**:
   - NaNs or unstable loss at early AR steps due to scale=0.
   - Over-quantization of short contexts distorts PPL.

2. **Fairness risk in KIVI vs KVIB comparison**:
   - If KVIB is designed for streaming cache, using non-streaming eval may misrepresent its error profile.
   - True streaming eval might show different ranking or gap between methods.

3. **Performance bottleneck**:
   - Current AR eval is O(T²) and does not reflect actual inference cost of cached decoding.
   - Results are slow to obtain and not representative of real deployment speed.

4. **Silent misconfiguration risk**:
   - Without `--quantize-prefill`, KIVI quantization is silently disabled for SmolLM.
   - This could lead to comparing unquantized SmolLM to quantized baselines.

---

## Overall Recommendation

**Priority order**:
1. ✅ Patch quantization stability (epsilon clamp) + residual-length logic now.
2. ✅ Keep batched eval for finetuning speed.
3. ⚠️ Add streaming-cache eval for final KIVI vs KVIB comparison.
4. ⚠️ Verify `quantize_prefill` and `prefill_use_quantized` flags are set correctly for experiments.