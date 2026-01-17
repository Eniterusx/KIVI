from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

from models.utils_quant import AsymGroupedQuantizer, AsymGroupedQuantizerByChannel


@dataclass
class SmolLM2Config:
    n_embd: int = 576
    n_hidden: int = 1536
    bias: bool = False
    block_size: int = 8192
    n_layer: int = 30
    n_head: int = 9
    n_kv_heads: int = 3
    norm_eps: float = 1e-05
    dtype: torch.dtype = torch.bfloat16
    rope_theta: int = 100000
    vocab_size: int = 49152
    attn_implementation: str = "sdpa"
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 64
    residual_length: int = 32
    use_kivi: bool = True


def pre_compute_rope(config: SmolLM2Config) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = config.n_embd // config.n_head
    positions = torch.arange(config.block_size)
    thetas = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos_thetas = torch.outer(positions, thetas)
    pos_thetas = torch.concatenate([pos_thetas, pos_thetas], dim=-1)
    cos = torch.cos(pos_thetas).to(dtype=config.dtype)
    sin = torch.sin(pos_thetas).to(dtype=config.dtype)

    def index_rope_frequencies(sin: torch.Tensor, cos: torch.Tensor, head_dim: int, r: int):
        indices = [int((k * head_dim) / (2 * r)) for k in range(r)]
        rope_idxs = torch.tensor(sorted(sum([[2 * i, 2 * i + 1] for i in indices], [])))
        return cos[..., rope_idxs], sin[..., rope_idxs]

    return index_rope_frequencies(sin=sin, cos=cos, head_dim=head_dim, r=4)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    bsz, n_head, seqlen, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos[:seqlen, :]) + (rotated * sin[:seqlen, :])
    return roped.to(dtype=x.dtype)


class MHA(nn.Module):
    def __init__(self, config: SmolLM2Config, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.group_size = config.n_head // config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.attn_implementation = config.attn_implementation

        self.dr = 8
        self.qr_proj = nn.Linear(self.config.n_embd, self.dr * self.config.n_head, bias=False, dtype=config.dtype)
        self.kr_proj = nn.Linear(self.config.n_embd, self.dr * self.config.n_kv_heads, bias=False, dtype=config.dtype)

        self.q_proj = nn.Linear(self.config.n_embd, (self.config.n_embd - self.dr * self.config.n_head), bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(self.config.n_embd, (self.config.n_kv_heads * self.head_dim - self.dr * self.config.n_kv_heads), bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(self.config.n_embd, self.config.n_kv_heads * self.head_dim, bias=False, dtype=config.dtype)

        self.o_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias=False, dtype=config.dtype)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, self.config.block_size, self.config.block_size, dtype=config.dtype)))

        if rope_cos is None or rope_sin is None:
            cos, sin = pre_compute_rope(config=config)
        else:
            cos, sin = rope_cos, rope_sin
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        self.register_buffer("clip_val", torch.tensor([-1e4, 1e4]))

    def _quantize_key(self, key_states: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seqlen, head_dim = key_states.shape
        key_flat = key_states.reshape(bsz * n_head, seqlen, head_dim)
        clip_val = self.clip_val.to(dtype=key_states.dtype, device=key_states.device)
        quantized = AsymGroupedQuantizerByChannel.apply(
            key_flat, clip_val, self.config.k_bits, head_dim
        )
        return quantized.view(bsz, n_head, seqlen, head_dim)

    def _quantize_value(self, value_states: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seqlen, head_dim = value_states.shape
        group_size = self.config.group_size
        if group_size <= 0 or head_dim % group_size != 0:
            group_size = head_dim
        value_flat = value_states.reshape(bsz * n_head, seqlen, head_dim)
        clip_val = self.clip_val.to(dtype=value_states.dtype, device=value_states.device)
        quantized = AsymGroupedQuantizer.apply(
            value_flat, clip_val, self.config.v_bits, group_size
        )
        return quantized.view(bsz, n_head, seqlen, head_dim)

    def _maybe_quantize_kv(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.use_kivi:
            return key_states, value_states

        seqlen = key_states.size(2)
        residual_length = self.config.residual_length
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, embd = x.shape
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        kv_head = self.config.n_kv_heads

        q = self.q_proj(x).view(bsz, seqlen, n_head, head_dim - self.dr).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, kv_head, head_dim - self.dr).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, kv_head, head_dim).transpose(1, 2)

        qr = self.qr_proj(x).view(bsz, seqlen, n_head, self.dr).transpose(1, 2)
        kr = self.kr_proj(x).view(bsz, seqlen, kv_head, self.dr).transpose(1, 2)

        qr = apply_rope(qr, sin=self.sin, cos=self.cos)
        kr = apply_rope(kr, sin=self.sin, cos=self.cos)

        q = torch.cat((q, qr), dim=-1)
        k = torch.cat((k, kr), dim=-1)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        k, v = self._maybe_quantize_kv(k, v)

        if self.attn_implementation == "eager":
            attn_scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            attn_scores = torch.masked_fill(attn_scores, self.mask[:, :, :seqlen, :seqlen] == 0, float("-inf"))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_out = attn_scores @ v
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, embd)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, embd)

        attn_out = self.o_proj(attn_out)
        return attn_out


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.n_hidden, bias=config.bias, dtype=config.dtype)
        self.down_proj = nn.Linear(config.n_hidden, config.n_embd, bias=config.bias, dtype=config.dtype)
        self.gate_proj = nn.Linear(config.n_embd, config.n_hidden, bias=config.bias, dtype=config.dtype)

        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RMSNorm(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.embd_dim = config.n_embd
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(self.embd_dim, dtype=config.dtype))

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


class Block(nn.Module):
    def __init__(self, config: SmolLM2Config, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None):
        super().__init__()
        self.self_attn = MHA(config=config, rope_cos=rope_cos, rope_sin=rope_sin)
        self.input_layernorm = RMSNorm(config=config)
        self.mlp = FFN(config=config)
        self.post_attention_layernorm = RMSNorm(config=config)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.dtype)
        self.layers = nn.ModuleList([Block(config=config, rope_cos=rope_cos, rope_sin=rope_sin) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config=config)

    def forward(self, inputs: torch.Tensor):
        hidden = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return hidden


class TransformerKIVI(nn.Module):
    def __init__(self, config: SmolLM2Config, tokenizer=None, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = SmolLM2(config, rope_cos=rope_cos, rope_sin=rope_sin)
        self.tokenizer = tokenizer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, dtype=config.dtype, bias=config.bias)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = config

    def forward(self, input_ids: torch.Tensor):
        logits = self.lm_head(self.model(input_ids))
        return logits