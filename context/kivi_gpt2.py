import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.utils_quant import AsymGroupedQuantizer, AsymGroupedQuantizerByChannel


@dataclass
class KIVIQuantConfig:
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 64
    residual_length: int = 32
    use_kivi: bool = True


class GPT2AttentionKIVI(nn.Module):
    def __init__(self, config, quant_config: Optional[KIVIQuantConfig] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.split_size = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                .view(1, 1, config.n_ctx, config.n_ctx)
        )
        self.register_buffer("clip_val", torch.tensor([-1e4, 1e4]))

        self.quant_config = quant_config or KIVIQuantConfig(
            k_bits=getattr(config, "k_bits", 2),
            v_bits=getattr(config, "v_bits", 2),
            group_size=getattr(config, "group_size", self.head_dim),
            residual_length=getattr(config, "residual_length", 32),
            use_kivi=getattr(config, "use_kivi", True),
        )

    def _split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.split_size)

    def _quantize_key(self, key_states: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seqlen, head_dim = key_states.shape
        # Key quantization is per-channel across G tokens.
        # We must reshape (B, H, T, D) -> (B*H * T/G, G, D)
        group_size = self.quant_config.group_size
        if group_size <= 0:
            group_size = 32
            
        assert seqlen % group_size == 0, f"Key sequence length {seqlen} must be divisible by group size {group_size}"

        key_flat = key_states.reshape(bsz * n_head * (seqlen // group_size), group_size, head_dim)
        clip_val = self.clip_val.to(dtype=key_states.dtype, device=key_states.device)
        quantized = AsymGroupedQuantizerByChannel.apply(
            key_flat, clip_val, self.quant_config.k_bits, head_dim
        )
        return quantized.view(bsz, n_head, seqlen, head_dim)

    def _quantize_value(self, value_states: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seqlen, head_dim = value_states.shape
        group_size = self.quant_config.group_size
        if group_size <= 0 or head_dim % group_size != 0:
            group_size = head_dim
        value_flat = value_states.reshape(bsz * n_head, seqlen, head_dim)
        clip_val = self.clip_val.to(dtype=value_states.dtype, device=value_states.device)
        quantized = AsymGroupedQuantizer.apply(
            value_flat, clip_val, self.quant_config.v_bits, group_size
        )
        return quantized.view(bsz, n_head, seqlen, head_dim)

    def _maybe_quantize_kv(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply KIVI quantization: keep last residual_length tokens full precision, quantize earlier tokens."""
        if not self.quant_config.use_kivi:
            return key_states, value_states

        seqlen = key_states.size(2)
        residual_length = self.quant_config.residual_length
        
        if residual_length is None or residual_length <= 0 or seqlen <= residual_length:
            return key_states, value_states

        # Quantize older tokens
        # Align key quantization to multiples of residual_length (assuming R % G == 0)
        key_quant_len = (seqlen // residual_length) * residual_length
        
        # Values: FIFO flush (everything except last R)
        value_quant_len = max(0, seqlen - residual_length)

        if key_quant_len > 0:
            key_quant = self._quantize_key(key_states[:, :, :key_quant_len, :])
            key_full = key_states[:, :, key_quant_len:, :]
            key_out = torch.cat([key_quant, key_full], dim=2)
        else:
            key_out = key_states
            
        if value_quant_len > 0:
            value_quant = self._quantize_value(value_states[:, :, :value_quant_len, :])
            value_full = value_states[:, :, value_quant_len:, :]
            value_out = torch.cat([value_quant, value_full], dim=2)
        else:
            value_out = value_states
            
        return key_out, value_out


    def forward(self, x, past_kv=None, use_cache: bool = False):
        bsz, seqlen, embd = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(embd, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # --- Streaming Logic ---
        if use_cache:
            if past_kv is None:
                # Initialize Cache
                key_quant = None
                key_residual = k
                value_quant = None
                value_residual = v
                
                # Check initial quantization (batched prefill case)
                key_residual, value_residual = self._maybe_quantize_kv(key_residual, value_residual)
                
                # If quantization occurred (seqlen > R), split it back into quant/residual parts
                # But _maybe_quantize_kv returns concatenated tensors.
                # For proper state management, we should keep them separate or rely on re-splitting?
                # Option: Just use them as is for now. But for streaming, we need explicit components.
                # Since we assume "Option B" (no prefill prompt), past_kv is initially None for start of generation.
                # So k, v are just meant for the very first token(s).
                # If seqlen > R (batched input), we effectively treat it as a prefill.
                # If seqlen <= R, everything is residual.
            else:
                # Unpack previous state
                (
                    old_key_quant, old_key_residual, 
                    _, _, # old_key_scale, old_key_zero (unused/implicit in tensor)
                    old_value_quant, old_value_residual, 
                    _, _, # old_value_scale, old_value_zero
                    current_len
                ) = past_kv
                
                # Append new token(s) to residual
                key_residual = torch.cat([old_key_residual, k], dim=2)
                value_residual = torch.cat([old_value_residual, v], dim=2)
                
                key_quant = old_key_quant
                value_quant = old_value_quant
                
            # --- Flushing Logic ---
            residual_length = self.quant_config.residual_length
            
            # 1. Key Flush: Batch flush when residual reaches R
            if key_residual.size(2) == residual_length:
                # Quantize ENTIRE residual and move to key_quant
                key_quant_new = self._quantize_key(key_residual)
                if key_quant is None:
                    key_quant = key_quant_new
                else:
                    key_quant = torch.cat([key_quant, key_quant_new], dim=2)
                key_residual = torch.empty(
                    bsz, self.n_head, 0, self.head_dim, 
                    dtype=k.dtype, device=k.device
                )
            
            # 2. Value Flush: FIFO (keep last R)
            if value_residual.size(2) > residual_length:
                num_to_flush = value_residual.size(2) - residual_length
                val_to_flush = value_residual[:, :, :num_to_flush, :]
                val_quant_new = self._quantize_value(val_to_flush)
                
                if value_quant is None:
                    value_quant = val_quant_new
                else:
                    value_quant = torch.cat([value_quant, val_quant_new], dim=2)
                    
                value_residual = value_residual[:, :, -residual_length:, :]
            
            # Prepare for attention
            k_for_attn = []
            v_for_attn = []
            
            if key_quant is not None:
                k_for_attn.append(key_quant)
                # v_for_attn is handled separately bc asymmetric lengths
            
            if value_quant is not None:
                v_for_attn.append(value_quant)

            k_for_attn.append(key_residual)
            v_for_attn.append(value_residual)
            
            k_combined = torch.cat(k_for_attn, dim=2)
            # v_combined not created mostly; we use split attn if using KIVI logic strictly
            # But checking if we use titled attn or standard attn
            
            # Tiled Attention Decoded
            # If we have quantized keys, we split calculation
            split_idx = key_quant.size(2) if key_quant is not None else 0
            
            if split_idx > 0:
                # Tiled Attention
                q_scale = 1.0 / math.sqrt(self.head_dim)
                
                # 1. Attention to Quantized Keys
                attn_quant = torch.matmul(q, key_quant.transpose(-1, -2)) * q_scale
                
                # 2. Attention to Residual Keys
                attn_resid = torch.matmul(q, key_residual.transpose(-1, -2)) * q_scale
                
                # Combine
                attn = torch.cat([attn_quant, attn_resid], dim=-1)
            else:
                attn = (q @ k_combined.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Update cache state
            new_past_kv = (
                key_quant, key_residual, None, None,
                value_quant, value_residual, None, None,
                attn.size(-1) # approximate total length
            )
        else:
            # Non-streaming / Batched Evaluation with simple KIVI
            k, v = self._maybe_quantize_kv(k, v)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            new_past_kv = None
            k_combined = k # For simplicity variable name matching below

        # Masking
        # For streaming decode: attn shape (B, H, 1, TotalLen)
        # Mask: we usually don't need mask for decode step 1 token vs all past
        # BUT if seqlen > 1 (prefill), we do.
        if seqlen > 1:
             attn = attn.masked_fill(self.bias[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        
        att = torch.softmax(attn, dim=-1)
        att = self.attn_dropout(att)

        # Value Aggregation
        if use_cache and split_idx > 0:
            # We have quantized and residual scores.
            # We need to map them to values.
            # As noted in bug check: Key Split point might NOT match Value Split point!
            # We must use Value Quant Length to split attention scores.
            
            v_quant_len = value_quant.size(2) if value_quant is not None else 0
            
            att_quant_part = att[:, :, :, :v_quant_len]
            att_resid_part = att[:, :, :, v_quant_len:]
            
            y_quant = torch.matmul(att_quant_part, value_quant) if v_quant_len > 0 else 0
            y_resid = torch.matmul(att_resid_part, value_residual)
            
            y = y_quant + y_resid
        else:
            # Fallback (all residual or non-streaming)
            # If non-streaming, v is already quantized/cat combined
            if use_cache:
                # If cached but no quant keys, values are all in residual
                v_all = torch.cat(v_for_attn, dim=2)
                y = att @ v_all
            else:
                y = att @ v

        y = self._merge_heads(y)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, new_past_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config, quant_config: Optional[KIVIQuantConfig] = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionKIVI(config, quant_config=quant_config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_past_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_past_kv


class TransformerKIVI(nn.Module):
    def __init__(self, config, quant_config: Optional[KIVIQuantConfig] = None):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([Block(config, quant_config=quant_config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, past_kv=None, use_cache=False):
        bsz, seqlen = input_ids.size()
        
        # Positional Embedding (needs correct offset if streaming)
        if past_kv is not None:
             # Assume past_kv[0] contains valid length stats
             # But simplified: logic usually needs past_length
             # past_kv structure: (k_q, k_r, ..., total_len)
             past_length = past_kv[0][-1] if past_kv[0] is not None else 0
        else:
             past_length = 0
             
        device = input_ids.device
        pos = torch.arange(past_length, past_length + seqlen, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand(bsz, seqlen)

        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        new_past_kvs = []
        for i, block in enumerate(self.h):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_new_kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_past_kvs.append(layer_new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_past_kvs
        return logits