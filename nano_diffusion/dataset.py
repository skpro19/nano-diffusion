"""Push-T state dataset: episode splits, normalization, and sequence indexing."""

from __future__ import annotations

from pathlib import Path
from token import STAR
from typing import Literal, TypedDict

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

STATE_DIM = 5
ACTION_DIM = 2
DEFAULT_ZARR_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "pusht"
    / "pusht_cchi_v7_replay.zarr"
)


class NormalizationStatistics(TypedDict):
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def episode_starts_from_ends(*, episode_ends: np.ndarray) -> np.ndarray:
    """Return the first global frame index for each episode."""
    return np.concatenate([[0], episode_ends[:-1]])


def get_validation_episode_mask(
    *, n_episodes: int, val_ratio: float, seed: int,) -> np.ndarray:
    """Mark a reproducible random subset of episodes for validation."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_indices = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_indices] = True
    return val_mask


def downsample_train_episode_mask(
    *, train_mask: np.ndarray, max_train_episodes: int | None, seed: int,) -> np.ndarray:
    """Subsample marked training episodes, matching the reference Push-T recipe."""
    if max_train_episodes is None:
        return train_mask.copy()
    n_train = int(np.sum(train_mask))
    if n_train <= max_train_episodes:
        return train_mask.copy()
    train_indices = np.nonzero(train_mask)[0]
    rng = np.random.default_rng(seed=seed)
    chosen = rng.choice(len(train_indices), size=max_train_episodes, replace=False)
    downsampled = np.zeros_like(train_mask)
    downsampled[train_indices[chosen]] = True
    return downsampled


def create_sequence_indices(
    *,
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int,
    pad_after: int,) -> np.ndarray:
    """Build one row per training window: buffer start/end and padding offsets."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    rows: list[list[int]] = []
    episode_starts = episode_starts_from_ends(episode_ends=episode_ends)
    for episode_index, end_idx in enumerate(episode_ends):
        if not episode_mask[episode_index]:
            continue
        start_idx = int(episode_starts[episode_index])
        end_idx = int(end_idx)
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for relative_start in range(min_start, max_start + 1):
            buffer_start = max(relative_start, 0) + start_idx
            buffer_end = min(relative_start + sequence_length, episode_length) + start_idx
            start_offset = buffer_start - (relative_start + start_idx)
            end_offset = (relative_start + sequence_length + start_idx) - buffer_end
            sample_start = start_offset
            sample_end = sequence_length - end_offset
            rows.append([buffer_start, buffer_end, sample_start, sample_end])

    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def fit_normalization_statistics(
    *,
    state: zarr.Array,
    action: zarr.Array,
    episode_mask: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,) -> NormalizationStatistics:
    """Compute mean and std on all frames belonging to the selected episodes."""
    episode_indices = np.nonzero(episode_mask)[0]
    if len(episode_indices) == 0:
        raise ValueError("cannot fit normalization with zero selected episodes")

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for episode_index in episode_indices:
        start = int(episode_starts[episode_index])
        end = int(episode_ends[episode_index])
        state_chunks.append(np.asarray(state[start:end]))
        action_chunks.append(np.asarray(action[start:end]))

    state_values = np.concatenate(state_chunks, axis=0)
    action_values = np.concatenate(action_chunks, axis=0)
    state_std = state_values.std(axis=0)
    action_std = action_values.std(axis=0)
    return NormalizationStatistics(
        state_mean=state_values.mean(axis=0),
        state_std=np.where(state_std < 1e-8, 1.0, state_std),
        action_mean=action_values.mean(axis=0),
        action_std=np.where(action_std < 1e-8, 1.0, action_std),
    )


class PushTStateDataset(Dataset):
    """State-conditioned Push-T dataset backed by the official replay Zarr store."""

    def __init__(
        self,
        *,
        zarr_path: Path | str = DEFAULT_ZARR_PATH,
        split: Literal["train", "val"],
        observation_horizon: int = 2,
        prediction_horizon: int = 16,
        normalization_statistics: NormalizationStatistics | None = None,
        seed: int = 42,
        val_ratio: float = 0.2,
        max_train_episodes: int | None = None,) -> None:
        self._validate_configuration(
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            val_ratio=val_ratio,
            split=split,
            normalization_statistics=normalization_statistics,
        )

        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Push-T zarr not found: {zarr_path}")

        root = zarr.open(str(zarr_path), mode="r")
        state = root["data/state"]
        action = root["data/action"]
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        self._validate_zarr_arrays(
            state=state,
            action=action,
            episode_ends=episode_ends,
        )

        n_episodes = len(episode_ends)
        episode_starts = episode_starts_from_ends(episode_ends=episode_ends)
        val_mask = get_validation_episode_mask(
            n_episodes=n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_train_episode_mask(
            train_mask=~val_mask,
            max_train_episodes=max_train_episodes,
            seed=seed,
        )
        if split == "train":
            episode_mask = train_mask
        else:
            episode_mask = val_mask

        if not np.any(episode_mask):
            raise ValueError(f"split {split!r} selected zero episodes")

        pad_before = observation_horizon - 1
        pad_after = prediction_horizon - 1
        sequence_indices = create_sequence_indices(
            episode_ends=episode_ends,
            sequence_length=prediction_horizon,
            episode_mask=episode_mask,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        if split == "train":
            normalization = fit_normalization_statistics(
                state=state,
                action=action,
                episode_mask=episode_mask,
                episode_starts=episode_starts,
                episode_ends=episode_ends,
            )
        else:
            assert normalization_statistics is not None
            normalization = normalization_statistics

        self.zarr_path = zarr_path
        self.split = split
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.pad_before = pad_before
        self.pad_after = pad_after

        self._root = root
        self.state = state
        self.action = action
        self.episode_ends = episode_ends
        self.episode_starts = episode_starts
        self.episode_mask = episode_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.sequence_indices = sequence_indices
        self.normalization = normalization

    @staticmethod
    def _validate_configuration(
        *,
        observation_horizon: int,
        prediction_horizon: int,
        val_ratio: float,
        split: str,
        normalization_statistics: NormalizationStatistics | None,) -> None:
        if observation_horizon < 1:
            raise ValueError("observation_horizon must be at least 1")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        if observation_horizon > prediction_horizon:
            raise ValueError("observation_horizon must not exceed prediction_horizon")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        if split == "val" and normalization_statistics is None:
            raise ValueError(
                "validation split requires normalization_statistics from training"
            )

    @staticmethod
    def _validate_zarr_arrays(
        *,
        state: zarr.Array,
        action: zarr.Array,
        episode_ends: np.ndarray,) -> None:
        n_frames = int(state.shape[0])
        if action.shape[0] != n_frames:
            raise ValueError("state and action must have the same number of frames")
        if state.shape[1] != STATE_DIM:
            raise ValueError(f"expected state dimension {STATE_DIM}, got {state.shape[1]}")
        if action.shape[1] != ACTION_DIM:
            raise ValueError(
                f"expected action dimension {ACTION_DIM}, got {action.shape[1]}"
            )
        if len(episode_ends) == 0:
            raise ValueError("episode_ends must not be empty")
        if int(episode_ends[-1]) != n_frames:
            raise ValueError("last episode end must match the number of frames")

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        buffer_start, buffer_end, sample_start, sample_end = self.sequence_indices[index]

        sequences = {}
        for name, source in (("state", self.state), ("action", self.action)):
            values = torch.as_tensor(
                source[buffer_start:buffer_end], dtype=torch.float32
            )
            sequence = torch.empty((self.prediction_horizon, values.shape[1]))
            sequence[sample_start:sample_end] = values
            # Repeat boundary frames rather than borrowing from adjacent episodes.
            sequence[:sample_start] = values[0]
            sequence[sample_end:] = values[-1]

            mean = torch.as_tensor(
                self.normalization[f"{name}_mean"], dtype=torch.float32
            )
            std = torch.as_tensor(
                self.normalization[f"{name}_std"], dtype=torch.float32
            )
            sequences[name] = (sequence - mean) / std

        return {
            "obs": sequences["state"][:self.observation_horizon],
            "actions": sequences["action"],
        }
