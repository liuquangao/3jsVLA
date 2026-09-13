from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Explicit Q/K/V attention; no Transformer library is hidden underneath."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = query if context is None else context
        batch, query_length, dim = query.shape
        key_length = context.shape[1]
        q = self.q_proj(query).view(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch, key_length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch, key_length, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if context_mask is not None:
            scores = scores.masked_fill(~context_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, query_length, dim)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, context_mask=mask)
        return tokens + self.mlp(self.mlp_norm(tokens))


class ActionBlock(nn.Module):
    """Action self-attention followed by cross-attention to multimodal context."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, heads, dropout)
        self.cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = MultiHeadAttention(dim, heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, dropout)

    def forward(self, actions: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(actions)
        actions = actions + self.self_attention(normalized)
        actions = actions + self.cross_attention(
            self.cross_norm(actions), self.context_norm(context), context_mask
        )
        return actions + self.mlp(self.mlp_norm(actions))
