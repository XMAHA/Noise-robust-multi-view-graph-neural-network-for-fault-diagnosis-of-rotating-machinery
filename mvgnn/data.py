"""Dataset IO and paper-consistent graph construction for MvGNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset


def _edge_index(adjacency: np.ndarray) -> torch.Tensor:
    """Return the bidirectional edges of an undirected adjacency matrix."""
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    sources, targets = np.nonzero(adjacency)
    if sources.size == 0:
        raise ValueError("Each adjacency matrix must contain at least one edge")
    return torch.tensor(np.stack([sources, targets]), dtype=torch.long)


def _knn_edge_index(node_features: np.ndarray, k: int) -> torch.Tensor:
    """Construct the paper graph with torch-cluster kNN and symmetrize it."""
    num_nodes = node_features.shape[0]
    if not 1 <= k < num_nodes:
        raise ValueError("graph_k must be in [1, num_sensors - 1]")
    directed = knn_graph(
        torch.from_numpy(np.ascontiguousarray(node_features)),
        k=k,
        loop=False,
        flow="target_to_source",
    )
    return torch.unique(torch.cat([directed, directed.flip(0)], dim=1), dim=1)


class MultiViewDataset(InMemoryDataset):
    """Load NPZ windows and construct the multi-view graphs from the paper."""

    def __init__(
        self,
        path: str | Path,
        frequency_length: int = 512,
        graph_k: int = 1,
        expected_num_nodes: int | None = None,
        expected_time_length: int = 1024,
        expected_num_classes: int | None = None,
    ) -> None:
        payload = np.load(path)
        missing = {"signals", "labels", "preprocessing"}.difference(payload.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        if payload["preprocessing"].item() != "clean":
            raise ValueError("This release accepts only clean prepared datasets")
        signals = payload["signals"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        if signals.ndim != 3 or signals.shape[0] != labels.shape[0]:
            raise ValueError("signals must be [samples, sensors, time] and match labels")
        if signals.shape[-1] != expected_time_length:
            raise ValueError(
                f"Expected {expected_time_length} time points, found {signals.shape[-1]}"
            )
        if expected_num_nodes is not None and signals.shape[1] != expected_num_nodes:
            raise ValueError(f"Expected {expected_num_nodes} sensors, found {signals.shape[1]}")
        unique_labels = np.unique(labels)
        if expected_num_classes is not None and not np.array_equal(
            unique_labels, np.arange(expected_num_classes)
        ):
            raise ValueError(
                f"Labels must be contiguous 0..{expected_num_classes - 1}; "
                f"found {unique_labels.tolist()}"
            )

        # The paper models normalized TD windows as multivariate temporal graphs.
        mean = signals.mean(axis=-1, keepdims=True)
        std = np.maximum(signals.std(axis=-1, keepdims=True), 1e-12)
        signals = (signals - mean) / std

        # TD and FD are two views of the same normalized sensor window.
        # Match FFT.py: |FFT(x)| / L, remove DC, keep bins 1..512.
        fft_size = signals.shape[-1]
        if frequency_length > fft_size // 2:
            raise ValueError("frequency_length cannot exceed half the signal length")
        full_magnitude = np.abs(np.fft.fft(signals, axis=-1)) / fft_size
        spectra = full_magnitude[..., 1 : frequency_length + 1].astype(np.float32)

        adjacency = None
        if "adjacency" in payload.files:
            adjacency = payload["adjacency"].astype(np.float32)
            if adjacency.ndim == 2:
                adjacency = np.broadcast_to(adjacency, (len(labels), *adjacency.shape))
            expected_shape = (len(labels), signals.shape[1], signals.shape[1])
            if adjacency.shape != expected_shape:
                raise ValueError(f"Expected adjacency shape {expected_shape}, found {adjacency.shape}")

        graphs = []
        for i in range(len(labels)):
            edge_index = (
                _edge_index(adjacency[i].copy())
                if adjacency is not None
                else _knn_edge_index(signals[i], graph_k)
            )
            graphs.append(
                Data(
                    time_x=torch.from_numpy(signals[i]),
                    freq_x=torch.from_numpy(spectra[i]),
                    edge_index=edge_index,
                    y=torch.tensor(labels[i]),
                    num_nodes=signals.shape[1],
                )
            )
        super().__init__(root=None)
        self.data, self.slices = self.collate(graphs)
