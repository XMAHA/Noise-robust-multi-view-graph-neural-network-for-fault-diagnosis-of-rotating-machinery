#!/usr/bin/env python3
"""Prepare the nine-class SEU mechanical task from condition-keyed HDF5 data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvgnn.preprocessing import convert_hdf5


CLASSES = ["Health", "BW", "IW", "OW", "IOW", "TC", "TM", "TRC", "SW"]


def label_for_key(key: str) -> int | None:
    lower = key.lower()
    if "bearing_health" in lower:
        return None  # The paper uses gearbox health as the single healthy class.
    patterns = [
        ("gearbox_health", 0), ("bearing_ball", 1), ("bearing_inner", 2),
        ("bearing_outer", 3), ("bearing_comb", 4), ("gearbox_chipped", 5),
        ("gearbox_miss", 6), ("gearbox_root", 7), ("gearbox_surface", 8),
    ]
    matches = [label for pattern, label in patterns if pattern in lower]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely map SEU condition key: {key}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original/windowed SEU HDF5 file")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--max-windows-per-condition", type=int, default=1000)
    parser.add_argument(
        "--start-window", type=int, default=0,
        help="First window retained from each condition (default: 0)",
    )
    args = parser.parse_args()
    convert_hdf5(
        input_path=args.input,
        output_path=args.output,
        dataset_name="SEU mechanical dataset",
        num_nodes=8,
        window_length=1024,
        label_for_key=label_for_key,
        class_names=CLASSES,
        max_windows_per_condition=args.max_windows_per_condition,
        start_window=args.start_window,
        expected_conditions_per_class=4,
    )


if __name__ == "__main__":
    main()
