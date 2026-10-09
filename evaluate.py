#!/usr/bin/env python3
"""Evaluate an MvGNN checkpoint on a complete labeled NPZ file."""

import argparse

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from mvgnn.data import MultiViewDataset
from train import accuracy, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test", "all"],
        help="Checkpoint split to evaluate; use all for the complete NPZ",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)

    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = saved["config"]
    model = build_model(config).to(device)
    model.load_state_dict(saved["model_state"])
    dataset = MultiViewDataset(
        args.data,
        frequency_length=config["frequency_length"],
        graph_k=config.get("graph_k", 1),
        expected_num_nodes=config["num_nodes"],
        expected_time_length=config["time_length"],
        expected_num_classes=config["num_classes"],
    )
    selected = dataset
    if args.split != "all":
        split_indices = saved.get("split_indices")
        if split_indices is None or args.split not in split_indices:
            raise ValueError(f"Checkpoint does not contain the {args.split!r} split indices")
        indices = split_indices[args.split]
        if indices and max(indices) >= len(dataset):
            raise ValueError("Checkpoint split indices do not match this dataset")
        selected = Subset(dataset, indices)
    loader = DataLoader(selected, batch_size=args.batch_size or config["batch_size"], shuffle=False)
    print(f"split={args.split} accuracy={accuracy(model, loader, device):.4f} samples={len(selected)}")


if __name__ == "__main__":
    main()

