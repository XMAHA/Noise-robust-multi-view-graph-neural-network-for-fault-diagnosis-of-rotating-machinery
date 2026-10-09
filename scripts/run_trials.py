#!/usr/bin/env python3
"""Run independent MvGNN trials and summarize test accuracy."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--unseeded", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.unseeded and args.seeds is not None:
        parser.error("--unseeded cannot be combined with --seeds")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    seeds = args.seeds if args.seeds is not None else list(range(args.trials))

    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial_index, seed in enumerate(seeds, start=1):
        checkpoint = output_dir / (
            f"trial_{trial_index}.pt" if args.unseeded else f"seed_{seed}.pt"
        )
        command = [
            sys.executable, str(repo / "train.py"),
            "--config", args.config,
            "--data", args.data,
            "--output", str(checkpoint),
            "--device", args.device,
        ]
        if args.unseeded:
            command.append("--unseeded")
        else:
            command.extend(["--seed", str(seed)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rows.append({
            "trial": trial_index,
            "seed": None if args.unseeded else seed,
            "best_epoch": int(saved["epoch"]),
            "validation_accuracy": float(saved["val_accuracy"]),
            "test_accuracy": float(saved["test_accuracy"]),
            "checkpoint": str(checkpoint),
        })

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    test_scores = [row["test_accuracy"] for row in rows]
    summary = {
        "num_trials": len(rows),
        "unseeded": args.unseeded,
        "seeds": None if args.unseeded else seeds,
        "mean_test_accuracy": statistics.fmean(test_scores),
        "std_test_accuracy_population": statistics.pstdev(test_scores),
        "std_test_accuracy_sample": statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0,
        "results_csv": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
