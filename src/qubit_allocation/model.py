"""Mapping-aware two-graph GCN used to score qubit layouts."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """A small GCN that returns *per-node* embeddings (not pooled embeddings)."""
    def __init__(self, in_channels: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = float(dropout)
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_weight=edge_weight if edge_weight.numel() else None)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MappingAwareLayoutScorer(nn.Module):
    """Predict actual fixed-layout SABRE SWAP cost.

    For each logical qubit, the network combines its circuit embedding with the
    embedding of its assigned physical qubit. It then scores each interacting
    logical pair using the embeddings of both assigned physical qubits plus the
    physical shortest-path distance and interaction multiplicity.
    """
    def __init__(self, circuit_in_dim: int = 3, hardware_in_dim: int = 1,
                 hidden_dim: int = 64, num_gcn_layers: int = 2,
                 mlp_hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.circuit_encoder = GCNEncoder(circuit_in_dim, hidden_dim, num_gcn_layers, dropout)
        self.hardware_encoder = GCNEncoder(hardware_in_dim, hidden_dim, num_gcn_layers, dropout)
        self.assignment_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # c_i, c_j, p_i, p_j, p_i*p_j, |p_i-p_j|, weight, distance, adjacency
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 6 + 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # pooled assignment, pooled pair interactions, pooled logical nodes,
        # pooled physical nodes, and [mean distance, max distance, adjacency rate].
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def encode_graphs(self, circuit_x: torch.Tensor, circuit_edge_index: torch.Tensor,
                      circuit_edge_weight: torch.Tensor, hardware_x: torch.Tensor,
                      hardware_edge_index: torch.Tensor, hardware_edge_weight: torch.Tensor):
        circuit_nodes = self.circuit_encoder(circuit_x, circuit_edge_index, circuit_edge_weight)
        hardware_nodes = self.hardware_encoder(hardware_x, hardware_edge_index, hardware_edge_weight)
        return circuit_nodes, hardware_nodes

    def forward(self, circuit_x: torch.Tensor, circuit_edge_index: torch.Tensor,
                circuit_edge_weight: torch.Tensor, hardware_x: torch.Tensor,
                hardware_edge_index: torch.Tensor, hardware_edge_weight: torch.Tensor,
                layouts: torch.Tensor, pair_index: torch.Tensor,
                pair_weight: torch.Tensor, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Return one standardized SWAP-cost prediction per candidate layout.

        `layouts` has shape [batch, logical_qubits] and contains physical indices.
        """
        if layouts.ndim == 1:
            layouts = layouts.unsqueeze(0)
        circuit_nodes, hardware_nodes = self.encode_graphs(
            circuit_x, circuit_edge_index, circuit_edge_weight,
            hardware_x, hardware_edge_index, hardware_edge_weight,
        )
        batch_size, logical_n = layouts.shape
        if logical_n != circuit_nodes.shape[0]:
            raise ValueError(f"Layout maps {logical_n} logical qubits, graph has {circuit_nodes.shape[0]} nodes.")
        assigned = hardware_nodes[layouts]  # [B, logical_n, hidden]
        logical = circuit_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        assignment_features = torch.cat(
            [logical, assigned, logical * assigned, torch.abs(logical - assigned)], dim=-1
        )
        assignment_embedding = self.assignment_encoder(assignment_features).mean(dim=1)

        pair_count = int(pair_index.shape[1])
        if pair_count:
            src, dst = pair_index[0], pair_index[1]
            c_src = circuit_nodes[src].unsqueeze(0).expand(batch_size, -1, -1)
            c_dst = circuit_nodes[dst].unsqueeze(0).expand(batch_size, -1, -1)
            layout_src = layouts[:, src]
            layout_dst = layouts[:, dst]
            p_src = hardware_nodes[layout_src]
            p_dst = hardware_nodes[layout_dst]
            physical_distances = distance_matrix[layout_src, layout_dst]
            adjacency = (physical_distances <= 1.0).to(circuit_nodes.dtype)
            max_dist = max(1.0, float(distance_matrix.max().detach().item()))
            normalized_distances = physical_distances / max_dist
            if pair_weight.numel():
                max_weight = torch.clamp(pair_weight.max(), min=1.0)
                weights = torch.log1p(pair_weight) / torch.log1p(max_weight)
            else:
                weights = torch.ones(pair_count, dtype=circuit_nodes.dtype, device=circuit_nodes.device)
            weights_b = weights.unsqueeze(0).expand(batch_size, -1)
            pair_features = torch.cat([
                c_src, c_dst, p_src, p_dst, p_src * p_dst, torch.abs(p_src - p_dst),
                weights_b.unsqueeze(-1), normalized_distances.unsqueeze(-1), adjacency.unsqueeze(-1),
            ], dim=-1)
            pair_embeddings = self.pair_encoder(pair_features)
            weighted = weights_b.unsqueeze(-1) * pair_embeddings
            pair_embedding = weighted.sum(dim=1) / torch.clamp(weights_b.sum(dim=1, keepdim=True), min=1e-6)
            mean_distance = (normalized_distances * weights_b).sum(dim=1) / torch.clamp(weights_b.sum(dim=1), min=1e-6)
            max_distance = normalized_distances.max(dim=1).values
            adjacency_rate = (adjacency * weights_b).sum(dim=1) / torch.clamp(weights_b.sum(dim=1), min=1e-6)
        else:
            pair_embedding = torch.zeros(batch_size, self.hidden_dim, device=circuit_nodes.device, dtype=circuit_nodes.dtype)
            mean_distance = torch.zeros(batch_size, device=circuit_nodes.device, dtype=circuit_nodes.dtype)
            max_distance = torch.zeros_like(mean_distance)
            adjacency_rate = torch.zeros_like(mean_distance)

        global_circuit = circuit_nodes.mean(dim=0, keepdim=True).expand(batch_size, -1)
        global_hardware = hardware_nodes.mean(dim=0, keepdim=True).expand(batch_size, -1)
        global_scalars = torch.stack([mean_distance, max_distance, adjacency_rate], dim=1)
        combined = torch.cat([assignment_embedding, pair_embedding, global_circuit, global_hardware, global_scalars], dim=1)
        return self.regressor(combined).squeeze(-1)
