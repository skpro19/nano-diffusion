"""Overlay recorded state trajectories on the Push-T image frames.

This is diagnostic 1 of the state visualization set: it draws the agent path and the
block center path for whole episodes on top of the episode's first RGB frame. It is the
cheapest way to confirm that `data/state` coordinates really line up with `data/img`,
that y increases downward, and that the block center sits inside the T shape.

Figures are written to `debug/state/trajectories/`:
    episode_XXX.png      one episode per figure (path, start/end, block orientation)
    overview.png         the first few selected episodes as small multiples

Usage:
    uv run scripts/visualize_state_trajectories.py
    uv run scripts/visualize_state_trajectories.py --episodes 0 3 7
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = REPO_ROOT / "datasets" / "pusht" / "pusht_cchi_v7_replay.zarr"
DEFAULT_OUT_DIR = REPO_ROOT / "debug" / "state" / "trajectories"

FRAME_SIZE = 512
IMAGE_SIZE = 96
DEFAULT_EPISODES = [0, 1, 2, 3]
ORIENTATION_SAMPLES = 6
ORIENTATION_TICK = 25.0


def load_episode(*, zarr_path: Path, index: int) -> dict:
    """Return one episode's first frame, state array, and frame range."""
    root = zarr.open(str(zarr_path), mode="r")
    episode_ends = root["meta/episode_ends"][:]
    start = 0 if index == 0 else int(episode_ends[index - 1])
    end = int(episode_ends[index])

    return {
        "index": index,
        "start": start,
        "end": end,
        "image": root["data/img"][start],
        "state": root["data/state"][start:end],
    }


def draw_path(*, axis, points: np.ndarray, cmap: str, label: str):
    """Draw a polyline whose color encodes time, and return the collection."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap=cmap, linewidths=2.0, label=label)
    collection.set_array(np.arange(len(segments)))
    axis.add_collection(collection)
    return collection


# Shared by the plot markers and legend: each body has its own color.
# The block uses squares at both endpoints, with a hollow square for the start.
ENDPOINT_STYLES = [
    ("Agent · start", "o", "#0072B2", 170),
    ("Agent · finish", "*", "#0072B2", 320),
    ("Block origin · start", "s", "#D55E00", 130),
    ("Block origin · finish", "s", "#D55E00", 150),
]


def draw_endpoints(*, axis, agent: np.ndarray, block: np.ndarray) -> None:
    """Draw high-contrast endpoints with a white halo and dark border."""
    points = [agent[0], agent[-1], block[0], block[-1]]
    for point, (label, marker, color, size) in zip(points, ENDPOINT_STYLES):
        facecolor = "white" if label == "Block origin · start" else color
        axis.scatter(
            *point, s=size, marker=marker, facecolors=facecolor, edgecolors="white",
            linewidths=3.5, zorder=6,
        )
        axis.scatter(
            *point, s=size, marker=marker, facecolors=facecolor, edgecolors=color,
            linewidths=1.5, zorder=7,
        )


def legend_handles() -> list[Line2D]:
    """Use exactly the same marker shapes and colors as the plotted endpoints."""
    handles = [
        Line2D([], [], color=plt.cm.viridis(0.3), linewidth=2,
               label="Agent trajectory (purple → yellow)"),
        Line2D([], [], color="orange", linewidth=2,
               label="Block origin trajectory (red → yellow)"),
    ]
    for label, marker, color, size in ENDPOINT_STYLES:
        handles.append(Line2D(
            [], [], marker=marker, linestyle="none", markersize=np.sqrt(size),
            markerfacecolor="white" if label == "Block origin · start" else color,
            markeredgecolor=color, label=label,
        ))
    handles.extend([
        Line2D([], [], color="#D55E00", linestyle="--", label="Starting T outline"),
        Line2D([], [], color="#D55E00", linestyle="-", label="Finishing T outline"),
        Line2D([], [], color="black", linewidth=2, label="Block orientation (sampled)"),
        Line2D([], [], color="#C62828", linewidth=2, label="Local +x · along crossbar"),
        Line2D([], [], color="#1565C0", linewidth=2, label="Local +y · toward stem"),
    ])
    return handles


def block_outline(*, pose: np.ndarray) -> np.ndarray:
    """Transform the upstream PushTEnv.add_tee outline about its local (0, 0).

    Source: real-stanford/diffusion_policy, env/pusht/pusht_env.py.
    The crossbar spans x=-60..60, y=0..30; the stem spans x=-15..15,
    y=30..120. Thus (0, 0) lies on the outer edge of the crossbar, not at CoM.
    """
    vertices = np.array([
        [-60, 0], [60, 0], [60, 30], [15, 30],
        [15, 120], [-15, 120], [-15, 30], [-60, 30],
    ])
    x, y, angle = pose
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return vertices @ rotation.T + [x, y]


def draw_block_origins(*, axis, state: np.ndarray) -> None:
    """Show where the reference origin sits within the start and finish shapes."""
    for frame, name, linestyle, offset in [
        (0, "Start origin", "--", (20, 20)),
        (-1, "Finish origin", "-", (20, -28)),
    ]:
        pose = state[frame, 2:5]
        axis.add_patch(Polygon(
            block_outline(pose=pose), closed=True, fill=False,
            edgecolor="#D55E00", linewidth=1.6, linestyle=linestyle, zorder=3,
        ))
        # Body-local +x runs along the crossbar; +y runs inward toward the stem.
        # Rotate both axes using the recorded angle, then translate to the origin.
        angle = pose[2]
        for direction, label, color in [
            (np.array([np.cos(angle), np.sin(angle)]), "+x", "#C62828"),
            (np.array([-np.sin(angle), np.cos(angle)]), "+y", "#1565C0"),
        ]:
            tip = pose[:2] + 45 * direction
            axis.annotate(
                "", xy=tip, xytext=pose[:2], zorder=9,
                arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
            )
            axis.text(
                *(tip + 8 * direction), label, color=color, fontsize=9,
                weight="bold", ha="center", va="center", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            )
        axis.annotate(
            name, xy=pose[:2], xytext=offset, textcoords="offset points",
            fontsize=8, color="#883600", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#883600"),
        )


def draw_world_frame(*, axis) -> None:
    """Draw the fixed world origin at the image's top-left, with y downward."""
    axis.scatter(0, 0, s=45, color="black", zorder=11, clip_on=False)
    for tip, label, color, offset in [
        ((80, 0), "+X", "#C62828", (6, -3)),
        ((0, 80), "+Y", "#1565C0", (-24, 0)),
    ]:
        axis.annotate(
            "", xy=tip, xytext=(0, 0), zorder=11, annotation_clip=False,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2.5),
        )
        axis.annotate(
            label, xy=tip, xytext=offset, textcoords="offset points",
            color=color, fontsize=9, weight="bold", annotation_clip=False,
        )
    axis.annotate(
        "World (0, 0)", xy=(0, 0), xytext=(12, -18),
        textcoords="offset points", fontsize=8, weight="bold",
        annotation_clip=False,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
    )


def draw_block_orientation(*, axis, state: np.ndarray) -> None:
    """Draw short ticks showing the block's recorded orientation."""
    samples = np.linspace(0, len(state) - 1, ORIENTATION_SAMPLES).astype(int)
    for frame in samples:
        x, y, angle = state[frame, 2], state[frame, 3], state[frame, 4]
        dx = ORIENTATION_TICK * np.cos(angle)
        dy = ORIENTATION_TICK * np.sin(angle)
        axis.plot([x - dx, x + dx], [y - dy, y + dy], color="white", linewidth=3.0, zorder=4)
        axis.plot([x, x + dx], [y, y + dy], color="black", linewidth=1.6, zorder=5)


def plot_episode(*, episode: dict, out_path: Path) -> None:
    """Plot one episode's agent and block paths over its first frame."""
    state = episode["state"]
    image = episode["image"] / 255.0
    agent = state[:, :2]
    block = state[:, 2:4]

    figure, axis = plt.subplots(figsize=(7, 7))
    # extent maps simulator coordinates (0..512, y downward) onto the 96x96 image.
    axis.imshow(image, extent=(0, FRAME_SIZE, FRAME_SIZE, 0), interpolation="nearest")

    draw_path(axis=axis, points=agent, cmap="viridis", label="agent path (color = time)")
    draw_path(axis=axis, points=block, cmap="autumn", label="block center path")

    draw_block_origins(axis=axis, state=state)
    draw_endpoints(axis=axis, agent=agent, block=block)
    draw_block_orientation(axis=axis, state=state)

    axis.set_xlim(0, FRAME_SIZE)
    axis.set_ylim(FRAME_SIZE, 0)
    axis.set_aspect("equal")
    draw_world_frame(axis=axis)
    axis.set_title(
        f"episode {episode['index']}  "
        f"frames {episode['start']}-{episode['end'] - 1}  "
        f"({len(state)} steps)"
    )
    axis.set_xlabel("x (simulator frame)")
    axis.set_ylabel("y (simulator frame, downward)")
    figure.legend(handles=legend_handles(), loc="lower center", ncol=2,
                  fontsize=8, title="Blue = agent · Orange = block · Color along paths = time")

    figure.tight_layout(rect=(0, 0.29, 1, 1))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_overview(*, episodes: list[dict], out_path: Path) -> None:
    """Small multiples: one panel per selected episode."""
    columns = 2
    rows = int(np.ceil(len(episodes) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows), squeeze=False)

    for episode, axis in zip(episodes, axes.flat):
        state = episode["state"]
        agent = state[:, :2]
        block = state[:, 2:4]
        axis.imshow(
            episode["image"] / 255.0,
            extent=(0, FRAME_SIZE, FRAME_SIZE, 0),
            interpolation="nearest",
        )
        draw_path(axis=axis, points=agent, cmap="viridis", label="agent")
        draw_path(axis=axis, points=block, cmap="autumn", label="block center")
        draw_block_origins(axis=axis, state=state)
        draw_endpoints(axis=axis, agent=agent, block=block)
        draw_block_orientation(axis=axis, state=state)
        axis.set_xlim(0, FRAME_SIZE)
        axis.set_ylim(FRAME_SIZE, 0)
        axis.set_aspect("equal")
        draw_world_frame(axis=axis)
        axis.set_title(f"episode {episode['index']} ({len(state)} steps)", fontsize=10)
        axis.tick_params(labelsize=7)

    for axis in axes.flat[len(episodes):]:
        axis.axis("off")

    figure.suptitle("State trajectories drawn on the first recorded frame")
    figure.legend(handles=legend_handles(), loc="lower center", ncol=3,
                  fontsize=9, title="Blue = agent · Orange = block · Color along paths = time")
    figure.tight_layout(rect=(0, 0.17, 1, 1))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument("--zarr-path", type=Path, default=ZARR_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = [load_episode(zarr_path=args.zarr_path, index=index) for index in args.episodes]

    for episode in episodes:
        plot_episode(episode=episode, out_path=args.out / f"episode_{episode['index']:03d}.png")
    plot_overview(episodes=episodes, out_path=args.out / "overview.png")


if __name__ == "__main__":
    main()
