from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyVLAConfig:
    camera_keys: tuple[str, ...] = (
        "observation.images.top",
        "observation.images.gripper",
    )
    image_height: int = 192
    image_width: int = 256
    hidden_dim: int = 256
    attention_heads: int = 8
    language_layers: int = 2
    multimodal_layers: int = 4
    action_layers: int = 4
    action_chunk: int = 16
    max_language_tokens: int = 32
    flow_steps: int = 8
    dropout: float = 0.1
