"""MvGNN: time/frequency graph encoders with view-attention fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, ChebConv, global_max_pool


class WideCNNEncoder(nn.Module):
    """Three-layer WDCNN node encoder used for the time-domain view."""

    def __init__(self, wide_kernel: int) -> None:
        super().__init__()
        padding = (wide_kernel - 16) // 2
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=wide_kernel, stride=16, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.output_dim = 256

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.network(signal.unsqueeze(1)).flatten(1)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, order: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")
        dims = [input_dim] + [hidden_dim] * layers
        self.convs = nn.ModuleList(
            ChebConv(dims[i], dims[i + 1], K=order) for i in range(layers)
        )
        self.norms = nn.ModuleList(BatchNorm(hidden_dim) for _ in range(layers))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
        return x


class MvGNN(nn.Module):
    """Multi-view GNN from the paper, without fixed batch-size assumptions.

    Input nodes are ordered graph-by-graph as produced by PyG DataLoader.
    ``time_x`` has shape [total_nodes, 1024] and ``freq_x`` has shape
    [total_nodes, 512].
    """

    def __init__(
        self,
        num_nodes: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_gnn_layers: int = 1,
        chebyshev_order: int = 2,
        wide_kernel: int = 88,
        frequency_length: int = 512,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.time_encoders = nn.ModuleList(
            WideCNNEncoder(wide_kernel) for _ in range(num_nodes)
        )
        self.time_graph = GraphEncoder(256, hidden_dim, num_gnn_layers, chebyshev_order)
        self.frequency_graph = GraphEncoder(
            frequency_length, hidden_dim, num_gnn_layers, chebyshev_order
        )

        attention_dim = 2 * num_nodes
        self.attention_reduce = nn.Linear(num_nodes, attention_dim)
        self.view_heads = nn.ModuleList(
            [nn.Linear(attention_dim, num_nodes), nn.Linear(attention_dim, num_nodes)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        time_x: torch.Tensor,
        freq_x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(batch.max().item()) + 1
        expected_nodes = batch_size * self.num_nodes
        if time_x.size(0) != expected_nodes:
            raise ValueError("Every graph must contain exactly num_nodes nodes")

        time_by_graph = time_x.reshape(batch_size, self.num_nodes, -1)
        encoded = [
            encoder(time_by_graph[:, node, :])
            for node, encoder in enumerate(self.time_encoders)
        ]
        time_nodes = torch.stack(encoded, dim=1).reshape(expected_nodes, -1)

        time_nodes = self.time_graph(time_nodes, edge_index)
        freq_nodes = self.frequency_graph(freq_x, edge_index)
        time_graph = time_nodes.reshape(batch_size, self.num_nodes, -1)
        freq_graph = freq_nodes.reshape(batch_size, self.num_nodes, -1)

        summary = (time_graph + freq_graph).mean(dim=-1)
        latent = F.relu(self.attention_reduce(summary))
        logits = torch.stack([head(latent) for head in self.view_heads], dim=0)
        weights = F.softmax(F.softmax(logits, dim=2), dim=0).unsqueeze(-1)
        views = torch.stack([time_graph, freq_graph], dim=0)
        fused_nodes = (weights * views).sum(dim=0).reshape(expected_nodes, -1)
        graph_embedding = global_max_pool(fused_nodes, batch)
        return self.classifier(graph_embedding)

