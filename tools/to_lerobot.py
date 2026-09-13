"""Convert a 3jsVLA collector dump into a LeRobotDataset v3.0.

The browser writes a deliberately dumb tree — one directory per episode, frames as plain
JPEGs, everything else as JSON. This script turns that into a real LeRobot dataset by
driving `LeRobotDataset` itself rather than writing parquet and MP4 by hand, so the
on-disk format stays correct without this file having to know what it looks like.

    python tools/to_lerobot.py <dump-dir> --repo-id <user>/<name> --root <output-dir>

Input layout (3jsvla.dump.v1):

    <dump-dir>/
    ├── meta.json
    └── episodes/episode_00000/
        ├── episode.json
        └── frames/000000.jpg …

Needs `lerobot >= 0.4.0` for v3.0 support; see the README for the environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_KEYS = {
    "top": "observation.images.top",
    "gripper": "observation.images.gripper",
}

DUMP_FORMAT = "3jsvla.dump.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the folder the collector wrote")
    parser.add_argument("--repo-id", required=True, help="e.g. yourname/3jsvla-a1z-pickplace")
    parser.add_argument("--root", type=Path, default=None, help="where to write (default: the HF cache)")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="skip episodes the collector marked as failed",
    )
    return parser.parse_args()


def load_dump(source: Path) -> tuple[dict, list[Path]]:
    meta_path = source / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found. Point this at the folder you chose in the collector — "
            "the one holding meta.json and episodes/."
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != DUMP_FORMAT:
        raise SystemExit(f"unexpected dump format {meta.get('format')!r}, expected {DUMP_FORMAT!r}")

    episodes = sorted((source / "episodes").iterdir())
    if not episodes:
        raise SystemExit(f"no episodes under {source / 'episodes'}")
    return meta, episodes


def build_features(meta: dict) -> dict:
    """Joint vectors and both physical sensor cameras."""
    joints = meta["joints"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
        "action": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    }
    for camera_key in CAMERA_KEYS.values():
        features[camera_key] = {
            "dtype": "video",
            "shape": (meta["image"]["height"], meta["image"]["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def vector(values: dict, joints: list[str]) -> np.ndarray:
    """Dict of joint readings to a fixed-order vector. Order comes from meta.json, not dict order."""
    return np.array([values[name] for name in joints], dtype=np.float32)


def main() -> None:
    args = parse_args()
    meta, episode_dirs = load_dump(args.source)
    joints = meta["joints"]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=meta["capture_hz"],
        features=build_features(meta),
        root=args.root,
        robot_type=meta["robot"],
        use_videos=True,
    )

    written = skipped = total_frames = 0
    for episode_dir in episode_dirs:
        episode = json.loads((episode_dir / "episode.json").read_text())
        if args.success_only and not episode["success"]:
            skipped += 1
            continue

        for frame in episode["frames"]:
            observation = frame["observation"]
            sample = {
                "observation.state": vector(observation["joint_positions"], joints),
                "action": vector(frame["action"]["joint_targets"], joints),
                "task": episode["instruction"],
            }
            for camera, camera_key in CAMERA_KEYS.items():
                sample[camera_key] = np.asarray(
                    Image.open(episode_dir / observation["image_paths"][camera]).convert("RGB")
                )
            dataset.add_frame(sample)
            total_frames += 1

        dataset.save_episode()
        written += 1
        print(f"  {episode_dir.name}  {len(episode['frames']):>3} frames  {episode['instruction']}")

    # Without this the parquet writers never emit their footers and the dataset will not load.
    dataset.finalize()

    print(f"\n{written} episodes, {total_frames} frames -> {dataset.root}")
    if skipped:
        print(f"{skipped} failed episodes skipped (--success-only)")


if __name__ == "__main__":
    main()
