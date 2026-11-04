"""Visualize the episode-length distribution of the Push-T dataset.

This is the diagnostic needed before fixing an observation/prediction horizon: it
answers "how long is an episode, and how many frames does a given horizon protect
from truncation?".

The figure is written to `debug/dataset/episode_lengths.png`:
    left   histogram of frames per episode, in 10-frame bins
    right  cumulative share of episodes long enough for a candidate horizon

Usage:
    uv run scripts/visualize_episode_lengths.py
    uv run scripts/visualize_episode_lengths.py --episodes 0 1 2
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
DEFAULT_OUT_DIR = REPO_ROOT / "debug" / "dataset"

# Horizons worth checking: a single observation plus 8/16/32-step action chunks.
CANDIDATE_HORIZONS = [8, 16, 32, 48]


def load_episode_lengths(*, zarr_path: Path) -> np.ndarray:
    """Return the number of frames in each episode, derived from `meta/episode_ends`."""
    root = zarr.open(str(zarr_path), mode="r")
    episode_ends = root["meta/episode_ends"][:]
    return np.diff(np.concatenate([[0], episode_ends]))


def print_statistics(*, lengths: np.ndarray) -> None:
    """Print the summary statistics and how many frames each horizon would drop."""
    print(f"episodes: {len(lengths)}")
    print(f"frames: {int(lengths.sum())}")
    print(f"mean: {lengths.mean():.1f}  median: {np.median(lengths):.1f}  std: {lengths.std():.1f}")
    print(f"min: {lengths.min()}  max: {lengths.max()}")
    print("percentiles [0, 5, 25, 50, 75, 95, 100]:", np.percentile(lengths, [0, 5, 25, 50, 75, 95, 100]))

    print(f"\n{'horizon':>8s} {'episodes too short':>20s} {'frames usable':>14s}")
    for horizon in CANDIDATE_HORIZONS:
        # A window of `horizon` frames needs an episode of at least that many frames,
        # and yields one window per starting frame in the episode.
        too_short = int((lengths < horizon).sum())
        usable = int(np.maximum(lengths - horizon + 1, 0).sum())
        print(f"{horizon:8d} {too_short:20d} {usable:14d}")


def plot_episode_lengths(*, lengths: np.ndarray, out_path: Path) -> None:
    """Histogram of episode lengths plus the cumulative fraction of long-enough episodes."""
    figure, (histogram_axis, coverage_axis) = plt.subplots(1, 2, figsize=(13, 5))

    bin_width = 10
    bins = np.arange(lengths.min() // bin_width * bin_width, lengths.max() + bin_width, bin_width)
    histogram_axis.hist(lengths, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    histogram_axis.axvline(lengths.mean(), color="black", linestyle="--", linewidth=1, label=f"mean {lengths.mean():.1f}")
    histogram_axis.axvline(np.median(lengths), color="darkorange", linestyle=":", linewidth=1.5, label=f"median {np.median(lengths):.0f}")
    histogram_axis.set_title(f"frames per episode (n={len(lengths)}, {bin_width}-frame bins)")
    histogram_axis.set_xlabel("frames")
    histogram_axis.set_ylabel("episodes")
    histogram_axis.legend(fontsize=8)

    # Cumulative view: which horizons keep (nearly) every episode trainable.
    sorted_lengths = np.sort(lengths)
    cumulative_share = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    coverage_axis.step(sorted_lengths, cumulative_share, where="post", color="steelblue")
    for index, horizon in enumerate(CANDIDATE_HORIZONS):
        share = float((lengths >= horizon).mean())
        coverage_axis.plot([horizon], [share], marker="o", color="darkorange", markersize=5)
        # Stagger the labels: consecutive horizons often sit at the same share.
        coverage_axis.annotate(
            f"h={horizon}: {share:.1%}",
            xy=(horizon, share),
            xytext=(8, -12 - 11 * index),
            textcoords="offset points",
            fontsize=8,
        )
    coverage_axis.set_title("share of episodes at least this long")
    coverage_axis.set_xlabel("horizon (frames)")
    coverage_axis.set_ylabel("fraction of episodes")
    coverage_axis.set_ylim(0, 1.05)
    coverage_axis.grid(alpha=0.3)

    figure.suptitle("Push-T episode lengths")
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

    lengths = load_episode_lengths(zarr_path=args.zarr_path)
    if args.episodes is not None:
        lengths = lengths[args.episodes]

    args.out.mkdir(parents=True, exist_ok=True)
    print_statistics(lengths=lengths)
    plot_episode_lengths(lengths=lengths, out_path=args.out / "episode_lengths.png")


if __name__ == "__main__":
    main()
