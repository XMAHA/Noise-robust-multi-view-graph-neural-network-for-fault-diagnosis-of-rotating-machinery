#!/usr/bin/env python3
"""Train MvGNN and save the validation-selected checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from mvgnn.data import MultiViewDataset
from mvgnn.model import MvGNN
from mvgnn.utils import checkpoint_payload, load_config, set_seed


def accuracy(model: MvGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.time_x, batch.freq_x, batch.edge_index, batch.batch)
            correct += int((logits.argmax(1) == batch.y).sum())
            total += batch.num_graphs
    return correct / total


def stratified_indices(labels: np.ndarray, config: dict) -> tuple[np.ndarray, ...]:
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices,
        test_size=config["test_ratio"],
        stratify=labels,
        random_state=config["seed"],
    )
    relative_val = config["val_ratio"] / (config["train_ratio"] + config["val_ratio"])
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=labels[train_val],
        random_state=config["seed"],
    )
    return train, val, test


def build_model(config: dict) -> MvGNN:
    return MvGNN(
        num_nodes=config["num_nodes"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        num_gnn_layers=config["num_gnn_layers"],
        chebyshev_order=config["chebyshev_order"],
        wide_kernel=config["wide_kernel"],
        frequency_length=config["frequency_length"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="checkpoints/mvgnn.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None, help="Override config for a quick check")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this trial")
    parser.add_argument(
        "--unseeded", action="store_true",
        help="Do not seed NumPy, PyTorch, CUDA, or the data split",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.unseeded and args.seed is not None:
        parser.error("--unseeded cannot be combined with --seed")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.unseeded:
        config["seed"] = None
    else:
        set_seed(config["seed"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    dataset = MultiViewDataset(
        args.data,
        frequency_length=config["frequency_length"],
        graph_k=config.get("graph_k", 1),
        expected_num_nodes=config["num_nodes"],
        expected_time_length=config["time_length"],
        expected_num_classes=config["num_classes"],
    )
    labels = np.load(args.data)["labels"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_indices(labels, config)
    loader_args = {"batch_size": config["batch_size"], "num_workers": 0}
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_args)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_args)

    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = StepLR(optimizer, config["lr_step_size"], gamma=config["lr_gamma"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_val, stale = -1.0, 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum = correct = count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.time_x, batch.freq_x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * batch.num_graphs
            correct += int((logits.argmax(1) == batch.y).sum())
            count += batch.num_graphs
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch:03d} loss={loss_sum/count:.5f} train_acc={correct/count:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, stale = val_acc, 0
            checkpoint = checkpoint_payload(model, config, epoch, val_acc)
            checkpoint["split_indices"] = {
                "train": train_idx.tolist(),
                "validation": val_idx.tolist(),
                "test": test_idx.tolist(),
            }
            torch.save(checkpoint, output)
        else:
            stale += 1
            if stale >= config["patience"]:
                print(f"early stopping after {epoch} epochs")
                break
        scheduler.step()

    saved = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    test_acc = accuracy(model, test_loader, device)
    saved["test_accuracy"] = test_acc
    torch.save(saved, output)
    print(f"best_val_acc={saved['val_accuracy']:.4f} test_acc={test_acc:.4f}")
    print(f"checkpoint={output}")


if __name__ == "__main__":
    main()
