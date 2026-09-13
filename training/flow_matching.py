from __future__ import annotations

import torch
import torch.nn.functional as functional

from .model import TinyVLA


def flow_matching_loss(
    model: TinyVLA,
    images: dict[str, torch.Tensor],
    state: torch.Tensor,
    language_ids: torch.Tensor,
    language_mask: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(actions)
    time = torch.rand(actions.shape[0], device=actions.device)
    interpolation = time[:, None, None]
    noisy_actions = (1.0 - interpolation) * noise + interpolation * actions
    target_velocity = actions - noise
    predicted_velocity = model(
        images, state, language_ids, language_mask, noisy_actions, time
    )
    loss = functional.mse_loss(predicted_velocity, target_velocity, reduction="none").mean(dim=-1)
    return (loss * action_mask).sum() / action_mask.sum().clamp_min(1)


@torch.no_grad()
def sample_actions(
    model: TinyVLA,
    images: dict[str, torch.Tensor],
    state: torch.Tensor,
    language_ids: torch.Tensor,
    language_mask: torch.Tensor,
) -> torch.Tensor:
    actions = torch.randn(
        state.shape[0], model.config.action_chunk, model.action_output.out_features, device=state.device
    )
    step_size = 1.0 / model.config.flow_steps
    for step in range(model.config.flow_steps):
        time = torch.full((state.shape[0],), step * step_size, device=state.device)
        velocity = model(images, state, language_ids, language_mask, actions, time)
        actions = actions + step_size * velocity
    return actions
