"""Dataset-specific HDF5 conversion shared by the public preparation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np


def sorted_h5_keys(handle: h5py.File) -> list[str]:
    def order(key: str) -> tuple[int, str]:
        prefix = key.split("-", 1)[0]
        return (int(prefix) if prefix.isdigit() else 10**9, key)

    return sorted(handle.keys(), key=order)


def to_windows(array: np.ndarray, num_nodes: int, length: int) -> np.ndarray:
    """Convert a condition array to non-overlapping [sample, node, time] windows."""
    values = np.asarray(array)
    if values.ndim == 3:
        if values.shape[1] == num_nodes:
            values = values[:, :, :length]
        elif values.shape[2] == num_nodes:
            values = values.transpose(0, 2, 1)[:, :, :length]
        else:
            raise ValueError(f"Cannot locate {num_nodes} sensor channels in shape {values.shape}")
        if values.shape[-1] < length:
            raise ValueError(f"Window has {values.shape[-1]} points; expected at least {length}")
        return values.astype(np.float32, copy=False)

    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D continuous signal or 3-D windows, got {values.shape}")
    if values.shape[0] == num_nodes:
        continuous = values
    elif values.shape[1] == num_nodes:
        continuous = values.T
    else:
        raise ValueError(f"Cannot locate {num_nodes} sensor channels in shape {values.shape}")
    count = continuous.shape[1] // length
    if count == 0:
        raise ValueError(f"Signal is shorter than one {length}-point window")
    continuous = continuous[:, : count * length]
    return continuous.reshape(num_nodes, count, length).transpose(1, 0, 2).astype(np.float32)


def convert_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str,
    num_nodes: int,
    window_length: int,
    label_for_key: Callable[[str], int | None],
    class_names: list[str],
    max_windows_per_condition: int = 1000,
    start_window: int = 0,
    expected_conditions_per_class: int | None = None,
) -> None:
    """Convert clean condition-keyed HDF5 data into the public NPZ schema."""
    if start_window < 0:
        raise ValueError("start_window must be non-negative")
    if max_windows_per_condition < 1:
        raise ValueError("max_windows_per_condition must be positive")
    signal_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    condition_counts = np.zeros(len(class_names), dtype=np.int64)
    with h5py.File(input_path, "r") as handle:
        for key in sorted_h5_keys(handle):
            label = label_for_key(key)
            if label is None:
                continue
            windows = to_windows(handle[key], num_nodes, window_length)
            stop_window = start_window + max_windows_per_condition
            if len(windows) < stop_window:
                raise ValueError(
                    f"{key} has {len(windows)} windows; indices "
                    f"{start_window}:{stop_window} are required"
                )
            windows = windows[start_window:stop_window]
            condition_counts[label] += 1
            signal_parts.append(windows)
            label_parts.append(np.full(len(windows), label, dtype=np.int64))
            source_parts.append(np.full(len(windows), key))

    if not signal_parts:
        raise ValueError("No recognized conditions were found in the input HDF5 file")
    if expected_conditions_per_class is not None and not np.all(
        condition_counts == expected_conditions_per_class
    ):
        raise ValueError(
            f"Expected {expected_conditions_per_class} conditions per class; "
            f"found {condition_counts.tolist()}"
        )
    signals = np.concatenate(signal_parts)
    labels = np.concatenate(label_parts)
    sources = np.concatenate(source_parts)
    present = np.unique(labels)
    expected = np.arange(len(class_names))
    if not np.array_equal(present, expected):
        raise ValueError(f"Expected labels {expected.tolist()}, found {present.tolist()}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        signals=signals,
        labels=labels,
        source_key=sources,
        class_names=np.asarray(class_names),
        dataset=np.asarray(dataset_name),
        window_length=np.asarray(window_length, dtype=np.int64),
        start_window=np.asarray(start_window, dtype=np.int64),
        max_windows_per_condition=np.asarray(max_windows_per_condition, dtype=np.int64),
        preprocessing=np.asarray("clean"),
    )
    np.savez_compressed(output, **payload)
    counts = np.bincount(labels, minlength=len(class_names))
    print(f"saved {output}: signals={signals.shape}, labels={labels.shape}")
    print("class counts:", {name: int(counts[i]) for i, name in enumerate(class_names)})
