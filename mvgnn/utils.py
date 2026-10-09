from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def checkpoint_payload(model: torch.nn.Module, config: dict, epoch: int, val_accuracy: float) -> dict:
    return {
        "model_state": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
    }

