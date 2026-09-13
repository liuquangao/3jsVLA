from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .config import TinyVLAConfig


class FeatureNormalizer:
    def __init__(self, stats: dict, key: str) -> None:
        self.mean = torch.as_tensor(stats[key]["mean"], dtype=torch.float32)
        self.std = torch.as_tensor(stats[key]["std"], dtype=torch.float32).clamp_min(1e-6)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return (values.float() - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)


class TinyVLADataset(Dataset):
    """Thin, visible adapter from LeRobot v3 samples to TinyVLA tensors."""

    def __init__(self, repo_id: str, root: Path, config: TinyVLAConfig) -> None:
        probe = LeRobotDataset(repo_id=repo_id, root=root)
        self.fps = probe.fps
        del probe
        delta_timestamps = {"action": [step / self.fps for step in range(config.action_chunk)]}
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps)
        self.config = config
        self.state_normalizer = FeatureNormalizer(self.dataset.meta.stats, "observation.state")
        self.action_normalizer = FeatureNormalizer(self.dataset.meta.stats, "action")
        self.state_dim = int(self.dataset.features["observation.state"]["shape"][0])
        self.action_dim = int(self.dataset.features["action"]["shape"][0])

    def __len__(self) -> int:
        return len(self.dataset)

    def _image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.max() > 1:
            image = image / 255.0
        image = functional.interpolate(
            image.unsqueeze(0),
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        return (image - mean) / std

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        actions = self.action_normalizer(sample["action"])
        padding = sample.get("action_is_pad")
        action_mask = ~padding.bool() if padding is not None else torch.ones(actions.shape[0], dtype=torch.bool)
        return {
            "images": {key: self._image(sample[key]) for key in self.config.camera_keys},
            "state": self.state_normalizer(sample["observation.state"]),
            "actions": actions,
            "action_mask": action_mask.float(),
            "task": sample["task"],
        }
