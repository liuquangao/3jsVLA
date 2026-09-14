"""
Serves one trained TinyVLA checkpoint to the browser collector.

Runs as a long-lived child of the Vite dev server and speaks newline-delimited JSON on
stdin/stdout, so the model is loaded once rather than per action chunk:

    in   {"images": {"top": "<base64 jpeg>", "gripper": "<base64 jpeg>"},
          "state": [j1..gripper], "task": "Move the red cube ..."}
    out  {"actions": [[j1..gripper] x action_chunk]}

State arrives and actions leave in the collector's own slider units. Normalization uses the
statistics stored in the checkpoint, so inference does not need the training dataset.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import TinyVLAConfig
from training.flow_matching import sample_actions
from training.model import TinyVLA
from training.tokenizer import WordTokenizer

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def prepare_image(encoded: str, config: TinyVLAConfig) -> torch.Tensor:
    """Match TinyVLADataset._image exactly; a different pipeline here is a silent domain shift."""
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
    tensor = functional.interpolate(
        tensor.unsqueeze(0),
        size=(config.image_height, config.image_width),
        mode="bilinear",
        align_corners=False,
    )
    return (tensor.squeeze(0) - IMAGENET_MEAN) / IMAGENET_STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = dict(checkpoint["config"])
    # Checkpoints trained before deltas existed predict absolute angles and say nothing about it.
    stored.setdefault("action_space", "absolute")
    config = TinyVLAConfig(**stored)
    tokenizer = WordTokenizer()
    tokenizer.words = tuple(checkpoint["vocabulary"])
    tokenizer.token_to_id = {word: index for index, word in enumerate(tokenizer.words)}

    state_mean = torch.as_tensor(checkpoint["state_mean"], dtype=torch.float32)
    state_std = torch.as_tensor(checkpoint["state_std"], dtype=torch.float32).clamp_min(1e-6)
    action_mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32)
    action_std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32).clamp_min(1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVLA(config, len(tokenizer.words), state_mean.numel(), action_mean.numel())
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    # The parent waits for this line before sending observations.
    print(json.dumps({
        "ready": True,
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "action_chunk": config.action_chunk,
        "action_space": config.action_space,
        "cameras": [key.rsplit(".", 1)[-1] for key in config.camera_keys],
    }), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            images = {
                key: prepare_image(request["images"][key.rsplit(".", 1)[-1]], config).unsqueeze(0).to(device)
                for key in config.camera_keys
            }
            measured = torch.as_tensor(request["state"], dtype=torch.float32)
            state = ((measured - state_mean) / state_std).unsqueeze(0).to(device)
            language_ids, language_mask = tokenizer.batch(
                [request["task"]], config.max_language_tokens, device
            )
            actions = sample_actions(model, images, state, language_ids, language_mask)[0].cpu()
            actions = actions * action_std + action_mean
            if config.action_space == "delta":
                # The collector drives absolute targets, so anchor the predicted movement to the
                # pose it was predicted from. Re-observing every chunk stops drift accumulating.
                actions = actions + measured
            print(json.dumps({"actions": actions.tolist()}), flush=True)
        except Exception as error:  # one bad frame must not take the policy down
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
