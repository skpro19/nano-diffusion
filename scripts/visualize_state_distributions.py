"""Visualize the distribution of `data/state` across the Push-T dataset.

This is diagnostic 4 of the state visualization set: it answers "what range and
spread do the state features have?", which is the information needed to choose and
sanity-check normalization statistics before training.

Figures are written to `debug/state/distributions/`:
    histograms.png       per-feature histograms with mean/std annotations
    positions.png        2-D density of agent and block positions in the 512 frame
    per_episode.png      per-episode mean and std, to show between-episode spread

All positions are in the 512x512 simulator frame, with y increasing downward.

Usage:
    uv run scripts/visualize_state_distributions.py
    uv run scripts/visualize_state_distributions.py --episodes 0 1 2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = REPO_ROOT / "datasets" / "pusht" / "pusht_cchi_v7_replay.zarr"
DEFAULT_OUT_DIR = REPO_ROOT / "debug" / "state" / "distributions"

# Column order of `data/state`: agent position first, then block pose.
FEATURE_NAMES = ["ee_x", "ee_y", "block_x", "block_y", "block_angle"]
FRAME_SIZE = 512
IMAGE_SIZE = 96


def load_state(*, zarr_path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the [N, 5] state array and one frame-index array per episode."""
    root = zarr.open(str(zarr_path), mode="r")
    state = root["data/state"][:]
    episode_ends = root["meta/episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episodes = [np.arange(start, end) for start, end in zip(episode_starts, episode_ends)]
    return state, episodes


def print_statistics(*, state: np.ndarray) -> None:
    """Print the dataset-wide statistics normalization would be derived from."""
    print(f"frames: {len(state)}")
    print(f"{'feature':12s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for index, name in enumerate(FEATURE_NAMES):
        column = state[:, index]
        print(
            f"{name:12s} {column.mean():9.2f} {column.std():9.2f} "
            f"{column.min():9.2f} {column.max():9.2f}"
        )


def plot_histograms(*, state: np.ndarray, out_path: Path) -> None:
    """One histogram per feature, annotated with the mean and standard deviation."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for index, (name, axis) in enumerate(zip(FEATURE_NAMES, axes.flat)):
        column = state[:, index]
        axis.hist(column, bins=80, color="steelblue", edgecolor="none")
        axis.axvline(column.mean(), color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{name}  mean {column.mean():.1f}  std {column.std():.1f}")
        axis.set_xlabel("value" if index < 4 else "radians")
        axis.set_ylabel("frames")

    # The sixth panel reports the image-frame scale, which is easy to forget.
    scale_axis = axes.flat[5]
    scale_axis.axis("off")
    scale_axis.text(
        0.0,
        0.5,
        f"positions are in the {FRAME_SIZE}x{FRAME_SIZE} simulator frame\n"
        f"divide by {FRAME_SIZE / IMAGE_SIZE:.3f} for {IMAGE_SIZE}x{IMAGE_SIZE} image pixels\n"
        "y increases downward; block_angle wraps at 2*pi",
        family="monospace",
        va="center",
    )

    figure.suptitle("Push-T state feature distributions (all episodes)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_position_density(*, state: np.ndarray, out_path: Path) -> None:
    """2-D density of agent and block positions, drawn in image orientation."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [
        ("agent position", state[:, 0], state[:, 1]),
        ("block center position", state[:, 2], state[:, 3]),
    ]
    for (title, xs, ys), axis in zip(panels, axes):
        density = axis.hexbin(xs, ys, gridsize=60, cmap="magma", bins="log")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_aspect("equal")
        axis.invert_yaxis()  # match the simulator's y-down image convention
        figure.colorbar(density, ax=axis, label="frames (log scale)")

    axes[0].set_ylabel("y")
    figure.suptitle("Where the agent and block appear across the dataset")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_per_episode(*, state: np.ndarray, episodes: list[np.ndarray], out_path: Path) -> None:
    """Per-episode mean and std, revealing how much episodes differ from each other."""
    means = np.stack([state[frames].mean(axis=0) for frames in episodes])
    stds = np.stack([state[frames].std(axis=0) for frames in episodes])
    episode_index = np.arange(len(episodes))

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for index in range(4):
        axes[0].plot(episode_index, means[:, index], marker="o", markersize=2, label=FEATURE_NAMES[index])
        axes[1].plot(episode_index, stds[:, index], marker="o", markersize=2, label=FEATURE_NAMES[index])

    axes[0].set_title("Per-episode mean of position features")
    axes[1].set_title("Per-episode std of position features")
    axes[1].set_xlabel("episode index")
    for axis in axes:
        axis.set_ylabel("value")
        axis.legend(ncol=4, fontsize=8)
        axis.axhline(axis.lines[0].get_ydata().mean(), color="gray", linestyle=":", linewidth=1)

    figure.suptitle("Episode-to-episode variation (block_angle omitted: it wraps)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="restrict the analysis to these episode indices (default: all)",
    )
    parser.add_argument("--zarr-path", type=Path, default=ZARR_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    state, episodes = load_state(zarr_path=args.zarr_path)
    if args.episodes is not None:
        episodes = [episodes[index] for index in args.episodes]
        state = state[np.concatenate(episodes)]

    args.out.mkdir(parents=True, exist_ok=True)
    print_statistics(state=state)
    plot_histograms(state=state, out_path=args.out / "histograms.png")
    plot_position_density(state=state, out_path=args.out / "positions.png")
    plot_per_episode(state=state, episodes=episodes, out_path=args.out / "per_episode.png")


if __name__ == "__main__":
    main()
