from __future__ import annotations

import torch
from torch import nn

from .attention import TransformerBlock


class TinyLanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, max_tokens: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.layers = nn.ModuleList(TransformerBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.word_embedding(token_ids) + self.position_embedding[:, : token_ids.shape[1]]
        for layer in self.layers:
            tokens = layer(tokens, mask)
        return self.norm(tokens)
