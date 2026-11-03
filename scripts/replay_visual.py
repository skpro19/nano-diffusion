"""Visually replay Push-T episodes by writing the recorded RGB frames to video.

No simulator involved: this only reads `data/img` from the zarr and pipes the raw
frames to ffmpeg. Episode boundaries come from `meta/episode_ends`.

Usage:
    uv run --with zarr --with numcodecs --with numpy scripts/replay_visual.py
    uv run --with zarr --with numcodecs --with numpy scripts/replay_visual.py \
        --episodes 0 3 7 --scale 6 --out datasets/replay
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = REPO_ROOT / "datasets" / "pusht" / "pusht_cchi_v7_replay.zarr"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "replay"
FPS = 10  # the dataset was recorded at 10 Hz
IMAGE_SIZE = 96


def load_episodes(*, zarr_path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the full image array and the per-episode frame slices."""
    root = zarr.open(str(zarr_path), mode="r")
    images = root["data/img"][:]
    episode_ends = root["meta/episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    episodes = [images[start:end] for start, end in zip(episode_starts, episode_ends)]
    return images, episodes


def write_video(*, frames: np.ndarray, out_path: Path, fps: int, scale: int) -> None:
    """Stream an [T, size, size, 3] uint8 array into an H.264 mp4 via ffmpeg."""
    n_frames, height, width, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process.communicate(frames.tobytes())
    if process.returncode != 0:
        raise SystemExit(f"ffmpeg failed for {out_path}")
    print(f"wrote {out_path} ({n_frames} frames)")


def to_uint8(*, frames: np.ndarray) -> np.ndarray:
    """Recorded frames are float32 in [65, 255]; ffmpeg wants uint8."""
    return np.clip(frames, 0, 255).astype(np.uint8)


def pad_to(*, frames: np.ndarray, length: int) -> np.ndarray:
    """Hold the last frame so several episodes can be tiled side by side."""
    if len(frames) >= length:
        return frames[:length]
    padding = np.repeat(frames[-1:], length - len(frames), axis=0)
    return np.concatenate([frames, padding], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--scale", type=int, default=5, help="integer upscale factor for visibility")
    parser.add_argument("--zarr-path", type=Path, default=ZARR_PATH)
    args = parser.parse_args()

    _, episodes = load_episodes(zarr_path=args.zarr_path)

    selected = []
    for index in args.episodes:
        frames = to_uint8(frames=episodes[index])
        selected.append(frames)
        write_video(
            frames=frames,
            out_path=args.out / f"episode_{index:03d}.mp4",
            fps=args.fps,
            scale=args.scale,
        )
        print(f"episode {index}: {len(frames)} frames ({len(frames) / args.fps:.1f} s)")

    # Side-by-side strip of all requested episodes, padded to the longest one.
    longest = max(len(frames) for frames in selected)
    strip = np.concatenate([pad_to(frames=frames, length=longest) for frames in selected], axis=2)
    write_video(
        frames=strip,
        out_path=args.out / "all_episodes.mp4",
        fps=args.fps,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
