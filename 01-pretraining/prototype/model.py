"""
A mini version of GPT-2 (inspired by nanochat)

Reference: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    mlp_mult: int = 4


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rope(x, cos, sin):
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1)


def precompute_rope(seq_len, head_dim, base=10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_per_kv = config.n_head // config.n_kv_head

        self.w_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.w_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.w_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.w_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = rms_norm(q), rms_norm(k)
        cos, sin = cos_sin
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.q_per_kv > 1)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_mult * config.n_embd
        self.w_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.w_fc(x)
        x = F.relu(x).square() # relu^2
        return self.w_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * config.n_layer) # scaled by depth
        for block in self.blocks:
            block.attn.w_proj.weight.data.mul_(scale)
            block.mlp.w_proj.weight.data.mul_(scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, loss_mask=None):
        B, T = input_ids.shape
        assert T <= self.config.sequence_len
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.embed_tokens(input_ids)
        x = rms_norm(x)
        for block in self.blocks:
            x = block(x, cos_sin)
        x = rms_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1, # cover the -1 labels we will use in SFT
                reduction="none",
            )
            if loss_mask is None:
                loss = per_token.mean()
            else:
                loss = (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            input_ids = input_ids[:, -self.config.sequence_len :]
            logits, _ = self(input_ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-6) # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def config_from_depth(depth, vocab_size, sequence_len, aspect_ratio=64, head_dim=64):
    n_embd = depth * aspect_ratio
    assert n_embd % head_dim == 0
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
    )