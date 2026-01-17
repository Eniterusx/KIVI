from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutput

@dataclass
class SmolLM2Config:
    n_embd: int = 576
    n_hidden: int = 1536
    bias: bool = False
    block_size:int = 8192 
    n_layer: int = 30
    n_head:int = 9
    n_kv_heads:int = 3
    norm_eps: float = 1e-05
    dtype: torch.dtype = torch.bfloat16
    rope_theta:int = 100000
    vocab_size: int = 49152    
    attn_implementation:str = "sdpa"

# Minimal rope implementation
def pre_compute_rope(config:SmolLM2Config):
    head_dim = config.n_embd // config.n_head
    positions = torch.arange(config.block_size) # (1, T)
    thetas = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos_thetas = torch.outer(positions, thetas)
    pos_thetas = torch.concatenate([pos_thetas, pos_thetas], dim = -1 )
    cos = torch.cos(pos_thetas).to(dtype = config.dtype)
    sin = torch.sin(pos_thetas).to(dtype = config.dtype)
    def index_rope_frequencies(sin: torch.Tensor, cos: torch.Tensor, head_dim: int, r: int):
        indices = [int((k * head_dim) / (2 * r)) for k in range(r)]
        rope_idxs = torch.tensor(sorted(sum([[2*i, 2*i+1] for i in indices], [])))
        return cos[..., rope_idxs], sin[..., rope_idxs]
    return index_rope_frequencies(sin=sin, cos=cos, head_dim=head_dim, r=4)

def apply_rope(x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor):
    B,n,T,h = x.shape
    x1 = x[..., :h//2]
    x2 = x[..., h//2:]
    rotated = torch.cat((-x2, x1), dim = -1)
    roped = (x * cos[:T, :]) + (rotated * sin[:T, :])
    return roped.to(dtype=x.dtype)

class MHA(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.group_size = config.n_head // config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.attn_implementation = config.attn_implementation

        self.dr = 8
        self.qr_proj = nn.Linear(self.config.n_embd, self.dr * self.config.n_head, bias = False, dtype = config.dtype)
        self.kr_proj = nn.Linear(self.config.n_embd, self.dr * self.config.n_kv_heads, bias = False, dtype = config.dtype)
        
        self.q_proj = nn.Linear(self.config.n_embd, (self.config.n_embd - self.dr * self.config.n_head), bias = False, dtype = config.dtype)
        self.k_proj = nn.Linear(self.config.n_embd, (self.config.n_kv_heads * self.head_dim - self.dr * self.config.n_kv_heads), bias = False, dtype = config.dtype)
        self.v_proj = nn.Linear(self.config.n_embd, self.config.n_kv_heads * self.head_dim, bias = False, dtype = config.dtype)
        
        self.o_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias = False, dtype = config.dtype)
        self.register_buffer("mask", torch.tril(torch.ones(1,1, self.config.block_size, self.config.block_size, dtype= config.dtype)))
        cos, sin = pre_compute_rope(config = config)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
    
    def forward(self, x:torch.Tensor):
        B,T,C = x.shape
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        kv_head = self.config.n_kv_heads

        # calculate q,k,v
        q = self.q_proj(x).view(B,T,n_head, head_dim - self.dr).transpose(1,2) # (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B,T,kv_head, head_dim - self.dr).transpose(1,2) # (B, n_kv_heads, T, head_dim)
        v = self.v_proj(x).view(B,T,kv_head, head_dim).transpose(1,2) # (B, n_kv_heads, T, head_dim)
        
        # calculate qr,kr
        qr = self.qr_proj(x).view(B,T,n_head,self.dr).transpose(1,2)
        kr = self.kr_proj(x).view(B,T,kv_head,self.dr).transpose(1,2)

        # sin, cos uniform sampling
        qr = apply_rope(qr, sin = self.sin, cos = self.cos) 
        kr = apply_rope(kr, sin = self.sin, cos = self.cos)
        
        # cat q,k with qr,kr
        q = torch.cat((q, qr), dim=-1)
        k = torch.cat((k, kr), dim=-1)

        # repeat the k and v groups 
        k = k.repeat_interleave(self.group_size, dim = 1) # (B, n_kv_heads, T, head_dim) -> (B, n_head, T, head_dim)
        v = v.repeat_interleave(self.group_size, dim = 1) # (B, n_kv_heads, T, head_dim) -> (B, n_head, T, head_dim)

        if self.attn_implementation == "eager":
            attn_scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            attn_scores = torch.masked_fill(attn_scores, self.mask[:,:, :T,:T]== 0, float("-inf"))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim = -1)
            attn_out = attn_scores @ v
            attn_out = attn_out.transpose(1,2).contiguous().view(B,T,C)
        elif self.attn_implementation == "sdpa":
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= 0, is_causal=True)
            attn_out = attn_out.transpose(1,2).contiguous().view(B,T,C)

        attn_out = self.o_proj(attn_out)
        return attn_out

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.n_hidden, bias = config.bias, dtype = config.dtype)
        self.down_proj = nn.Linear(config.n_hidden, config.n_embd, bias = config.bias, dtype = config.dtype)
        self.gate_proj = nn.Linear(config.n_embd, config.n_hidden, bias = config.bias, dtype = config.dtype)

        self.act_fn = torch.nn.SiLU() 
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class RMSNorm(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.embd_dim = config.n_embd
        self.eps = config.norm_eps # (1)
        self.weight = nn.Parameter(torch.ones(self.embd_dim, dtype=config.dtype)) # (C)

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True) # (B, T, 1)
        x_normed = x * torch.rsqrt(means + self.eps) # (B, T, C) / root((B, T, 1) + (1)) -> (B, T, C)
        return (x_normed * self.weight).to(dtype=x.dtype) # (B, T, C) * (C) -> (B, T, C) 
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MHA(config = config)
        self.input_layernorm = RMSNorm(config = config)
        self.mlp = FFN(config = config)
        self.post_attention_layernorm = RMSNorm(config = config)
    
    def forward(self, x:torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class SmolLM2(nn.Module):
    
    def __init__(self, config:SmolLM2Config = SmolLM2Config()):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, dtype = config.dtype)
        self.layers = nn.ModuleList([Block(config = config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config = config)

    def forward(self, inputs:torch.Tensor):
        hidden = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return hidden
    
class Transformer(nn.Module):
    
    def __init__(self, config:SmolLM2Config = SmolLM2Config(), tokenizer=None):
        super().__init__()
        self.model = SmolLM2(config)
        self.tokenizer = tokenizer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, dtype = config.dtype, bias= config.bias) 
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, labels: torch.Tensor = None):
        logits = self.lm_head(self.model(labels))
        loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )