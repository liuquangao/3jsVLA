from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TinyVLAConfig
from .dataset import TinyVLADataset
from .flow_matching import flow_matching_loss
from .model import TinyVLA
from .tokenizer import WordTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the readable 3jsVLA-Tiny policy")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/3jsvla-a1z")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/3jsvla-tiny.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--action-space",
        choices=("delta", "absolute"),
        default=TinyVLAConfig.action_space,
        help="predict joint movement from the observed pose, or the joint angles themselves",
    )
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "images": {key: value.to(device) for key, value in batch["images"].items()},
        "state": batch["state"].to(device),
        "actions": batch["actions"].to(device),
        "action_mask": batch["action_mask"].to(device),
        "task": list(batch["task"]),
    }


def main() -> None:
    args = parse_args()
    config = TinyVLAConfig(action_space=args.action_space)
    tokenizer = WordTokenizer()
    dataset = TinyVLADataset(args.repo_id, args.dataset_root, config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVLA(config, len(tokenizer.words), dataset.state_dim, dataset.action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            language_ids, language_mask = tokenizer.batch(
                batch["task"], config.max_language_tokens, device
            )
            loss = flow_matching_loss(
                model,
                batch["images"],
                batch["state"],
                language_ids,
                language_mask,
                batch["actions"],
                batch["action_mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        print(f"epoch {epoch:03d}  flow_loss={loss_sum / len(loader):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "vocabulary": tokenizer.words,
            "state_mean": dataset.state_normalizer.mean,
            "state_std": dataset.state_normalizer.std,
            "action_mean": dataset.action_normalizer.mean,
            "action_std": dataset.action_normalizer.std,
        },
        args.output,
    )
    print(f"checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
