#!/usr/bin/env python3
"""Validate a clean prepared NPZ and optionally compare it with source HDF5."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def scalar(payload: np.lib.npyio.NpzFile, name: str):
    return payload[name].item() if name in payload.files else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared clean NPZ file")
    parser.add_argument("--source", help="Source condition-keyed HDF5 file")
    parser.add_argument("--num-nodes", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--conditions-per-class", type=int, default=2)
    parser.add_argument("--windows-per-condition", type=int, default=1000)
    parser.add_argument("--start-window", type=int, default=100)
    parser.add_argument("--time-length", type=int, default=1024)
    args = parser.parse_args()

    with np.load(args.data) as payload:
        required = {"signals", "labels", "source_key", "preprocessing"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        if scalar(payload, "preprocessing") != "clean":
            raise ValueError("This release accepts only preprocessing='clean'")

        signals = payload["signals"]
        labels = payload["labels"].astype(np.int64)
        source_keys = payload["source_key"].astype(str)
        expected_samples = args.num_classes * args.conditions_per_class * args.windows_per_condition
        expected_shape = (expected_samples, args.num_nodes, args.time_length)
        if signals.shape != expected_shape:
            raise ValueError(f"Expected signals {expected_shape}, found {signals.shape}")
        if signals.dtype != np.float32:
            raise ValueError(f"Expected float32 signals, found {signals.dtype}")
        if labels.shape != (expected_samples,) or source_keys.shape != (expected_samples,):
            raise ValueError("labels/source_key do not match the expected sample count")
        if not np.isfinite(signals).all():
            raise ValueError("signals contain NaN or infinity")

        counts = np.bincount(labels, minlength=args.num_classes)
        expected_per_class = args.conditions_per_class * args.windows_per_condition
        if not np.array_equal(counts, np.full(args.num_classes, expected_per_class)):
            raise ValueError(f"Expected {expected_per_class} samples per class, found {counts.tolist()}")
        unique_sources, source_counts = np.unique(source_keys, return_counts=True)
        if len(unique_sources) != args.num_classes * args.conditions_per_class:
            raise ValueError(f"Unexpected number of conditions: {len(unique_sources)}")
        if not np.all(source_counts == args.windows_per_condition):
            raise ValueError(f"Unexpected per-condition counts: {source_counts.tolist()}")

        stored_start = scalar(payload, "start_window")
        if stored_start != args.start_window:
            raise ValueError(f"NPZ start_window={stored_start}, expected {args.start_window}")

        print(f"data={Path(args.data)}")
        print(f"signals={signals.shape} dtype={signals.dtype}")
        print(f"class_counts={counts.tolist()}")
        print(f"conditions={len(unique_sources)} windows_each={source_counts.tolist()}")
        print(f"start_window={stored_start} preprocessing=clean")

        if args.source:
            max_clean_error = 0.0
            with h5py.File(args.source, "r") as handle:
                for key in unique_sources:
                    if key not in handle:
                        raise ValueError(f"Source HDF5 is missing key: {key}")
                    mask = source_keys == key
                    stop = args.start_window + args.windows_per_condition
                    clean = np.asarray(
                        handle[key][args.start_window:stop, :args.num_nodes, :args.time_length],
                        dtype=np.float32,
                    )
                    prepared = signals[mask].astype(np.float32, copy=False)
                    if clean.shape != prepared.shape:
                        raise ValueError(f"Shape mismatch for {key}: {clean.shape} vs {prepared.shape}")
                    max_clean_error = max(max_clean_error, float(np.max(np.abs(prepared - clean))))
            print(f"source_max_abs_error={max_clean_error:.3e}")
            if max_clean_error != 0.0:
                raise ValueError("Clean NPZ does not exactly match the selected source windows")

    print("validation=PASS")


if __name__ == "__main__":
    main()
