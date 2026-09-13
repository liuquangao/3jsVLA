from __future__ import annotations

import math

import torch
from torch import nn

from .attention import ActionBlock, TransformerBlock
from .config import TinyVLAConfig
from .language import TinyLanguageEncoder
from .vision import TinyVisionEncoder


def time_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype) * (-math.log(10_000) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TinyVLA(nn.Module):
    """Two-camera VLM context plus a continuous flow-matching action expert."""

    def __init__(self, config: TinyVLAConfig, vocab_size: int, state_dim: int, action_dim: int) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.config = config
        self.vision = TinyVisionEncoder(dim)
        self.language = TinyLanguageEncoder(
            vocab_size, dim, config.max_language_tokens, config.language_layers,
            config.attention_heads, config.dropout,
        )
        self.state_projection = nn.Linear(state_dim, dim)
        self.modality_embedding = nn.Parameter(torch.randn(3 + len(config.camera_keys), dim) * 0.02)
        self.context_position = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.multimodal_layers = nn.ModuleList(
            TransformerBlock(dim, config.attention_heads, config.dropout)
            for _ in range(config.multimodal_layers)
        )
        self.context_norm = nn.LayerNorm(dim)
        self.action_projection = nn.Linear(action_dim, dim)
        self.action_position = nn.Parameter(torch.randn(1, config.action_chunk, dim) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_layers = nn.ModuleList(
            ActionBlock(dim, config.attention_heads, config.dropout) for _ in range(config.action_layers)
        )
        self.action_norm = nn.LayerNorm(dim)
        self.action_output = nn.Linear(dim, action_dim)

    def encode_context(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        language_ids: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        language = self.language(language_ids, language_mask) + self.modality_embedding[0]
        chunks = [language]
        masks = [language_mask]
        for index, camera_key in enumerate(self.config.camera_keys):
            visual = self.vision(images[camera_key]) + self.modality_embedding[index + 1]
            chunks.append(visual)
            masks.append(torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device))
        state_token = self.state_projection(state).unsqueeze(1) + self.modality_embedding[-1]
        chunks.append(state_token)
        masks.append(torch.ones(state_token.shape[:2], dtype=torch.bool, device=state.device))
        context = torch.cat(chunks, dim=1)
        context_mask = torch.cat(masks, dim=1)
        context = context + self.context_position[:, : context.shape[1]]
        for layer in self.multimodal_layers:
            context = layer(context, context_mask)
        return self.context_norm(context), context_mask

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        language_ids: torch.Tensor,
        language_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        context, context_mask = self.encode_context(images, state, language_ids, language_mask)
        actions = self.action_projection(noisy_actions)
        actions = actions + self.action_position[:, : actions.shape[1]]
        actions = actions + self.time_mlp(time_embedding(flow_time, actions.shape[-1])).unsqueeze(1)
        for layer in self.action_layers:
            actions = layer(actions, context, context_mask)
        return self.action_output(self.action_norm(actions))
