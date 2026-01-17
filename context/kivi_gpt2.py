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
        key_flat = key_states.reshape(bsz * n_head, seqlen, head_dim)
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
        if not self.quant_config.use_kivi:
            return key_states, value_states

        seqlen = key_states.size(2)
        residual_length = self.quant_config.residual_length
        if residual_length is None or residual_length <= 0 or residual_length >= seqlen:
            return self._quantize_key(key_states), self._quantize_value(value_states)

        quant_len = seqlen - residual_length
        key_quant = self._quantize_key(key_states[:, :, :quant_len, :])
        value_quant = self._quantize_value(value_states[:, :, :quant_len, :])
        key_full = key_states[:, :, quant_len:, :]
        value_full = value_states[:, :, quant_len:, :]
        return (
            torch.cat([key_quant, key_full], dim=2),
            torch.cat([value_quant, value_full], dim=2),
        )

    def forward(self, x):
        bsz, seqlen, embd = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(embd, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        k, v = self._maybe_quantize_kv(k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.bias[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = self._merge_heads(y)
        y = self.resid_dropout(self.c_proj(y))
        return y


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

    def forward(self, input_ids):
        bsz, seqlen = input_ids.size()
        pos = torch.arange(0, seqlen, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(bsz, seqlen)

        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits