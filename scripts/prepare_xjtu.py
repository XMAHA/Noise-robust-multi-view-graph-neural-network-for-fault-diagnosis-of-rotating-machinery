#!/usr/bin/env python3
"""Prepare the five-class XJTU Spurgear task from condition-keyed HDF5 data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvgnn.preprocessing import convert_hdf5


CLASSES = ["Health", "TRC02", "TRC06", "TRC10", "TRC14"]
PATTERNS = {"gear00": 0, "gear02": 1, "gear06": 2, "gear10": 3, "gear14": 4}


def label_for_key(key: str) -> int | None:
    lower = key.lower()
    matches = [label for pattern, label in PATTERNS.items() if pattern in lower]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely map XJTU condition key: {key}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original/windowed XJTU HDF5 file")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--max-windows-per-condition", type=int, default=1000)
    parser.add_argument(
        "--start-window", type=int, default=100,
        help="First window retained from each condition (default: 100)",
    )
    args = parser.parse_args()
    convert_hdf5(
        input_path=args.input,
        output_path=args.output,
        dataset_name="XJTU Spurgear dataset",
        num_nodes=12,
        window_length=1024,
        label_for_key=label_for_key,
        class_names=CLASSES,
        max_windows_per_condition=args.max_windows_per_condition,
        start_window=args.start_window,
        expected_conditions_per_class=2,
    )


if __name__ == "__main__":
    main()

