"""Sanity-check the downloaded Push-T zarr against the schema we train on.

Usage:
    uv run --with zarr --with numcodecs --with numpy scripts/verify_pusht.py
"""

from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = REPO_ROOT / "datasets" / "pusht" / "pusht_cchi_v7_replay.zarr"

EXPECTED_SHAPES = {
    "img": (25650, 96, 96, 3),
    "state": (25650, 5),
    "action": (25650, 2),
    "keypoint": (25650, 9, 2),
    "n_contacts": (25650, 1),
}
EXPECTED_EPISODES = 206


def main() -> None:
    if not ZARR_PATH.exists():
        raise SystemExit(f"missing dataset, run datasets/download_pusht.sh first: {ZARR_PATH}")
    root = zarr.open(str(ZARR_PATH), mode="r")
    episode_ends = root["meta/episode_ends"][:]
    lengths = np.diff(np.concatenate([[0], episode_ends]))

    failures = []

    print(f"zarr: {ZARR_PATH}")
    print(f"episodes: {len(episode_ends)} (expected {EXPECTED_EPISODES})")
    if len(episode_ends) != EXPECTED_EPISODES:
        failures.append("episode count")

    print(
        "episode length: "
        f"min {lengths.min()} median {int(np.median(lengths))} max {lengths.max()} "
        f"total frames {int(lengths.sum())}"
    )

    for name, expected in EXPECTED_SHAPES.items():
        actual = root[f"data/{name}"].shape
        status = "ok" if actual == expected else "MISMATCH"
        print(f"data/{name}: {actual} dtype {root[f'data/{name}'].dtype} [{status}]")
        if actual != expected:
            failures.append(f"{name} shape")

    images = root["data/img"]
    smin, smax = float(images[:].min()), float(images[:].max())
    print(f"image value range: [{smin}, {smax}] (expect [65, 255], i.e. NOT normalized)")

    actions = root["data/action"][:]
    states = root["data/state"][:]
    print(f"action range: {np.round(actions.min(axis=0), 2)} .. {np.round(actions.max(axis=0), 2)}")
    print(f"block angle range: [{states[:, 4].min():.2f}, {states[:, 4].max():.2f}] rad")

    if failures:
        raise SystemExit(f"FAILED: {', '.join(failures)}")
    print("PASS")


if __name__ == "__main__":
    main()
