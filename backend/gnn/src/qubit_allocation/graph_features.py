"""Deterministic interaction-graph and hardware-graph feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


@dataclass(frozen=True)
class GraphFeatures:
    """Numerical graph representation used by the GNN.

    `edge_index` is bidirectional and `edge_weight` aligns with its columns.
    `pair_index` contains each undirected interaction once for layout scoring.
    """
    x: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    pair_index: np.ndarray
    pair_weight: np.ndarray


def build_circuit_graph(circuit: QuantumCircuit) -> GraphFeatures:
    n = int(circuit.num_qubits)
    two_q_count = np.zeros(n, dtype=np.float32)
    one_q_count = np.zeros(n, dtype=np.float32)
    partner_sets: list[set[int]] = [set() for _ in range(n)]
    pair_counts: dict[tuple[int, int], int] = {}
    for item in circuit.data:
        qids = [int(circuit.find_bit(q).index) for q in item.qubits]
        if len(qids) == 1:
            one_q_count[qids[0]] += 1.0
        elif len(qids) == 2:
            a, b = qids
            if a == b:
                continue
            two_q_count[a] += 1.0
            two_q_count[b] += 1.0
            partner_sets[a].add(b)
            partner_sets[b].add(a)
            key = (min(a, b), max(a, b))
            pair_counts[key] = pair_counts.get(key, 0) + 1
        elif len(qids) > 2:
            raise ValueError(
                f"Circuit graph extraction does not accept operation {item.operation.name!r} "
                f"with {len(qids)} qubits. Preprocess the circuit first."
            )

    pair_degree = np.asarray([len(s) for s in partner_sets], dtype=np.float32)
    max_partners = max(1, n - 1)
    total_ops = max(1, int(circuit.size()))
    x = np.stack(
        [
            np.log1p(two_q_count) / np.log1p(total_ops),
            pair_degree / float(max_partners),
            np.log1p(one_q_count) / np.log1p(total_ops),
        ],
        axis=1,
    ).astype(np.float32)

    if pair_counts:
        pairs = sorted(pair_counts)
        src: list[int] = []
        dst: list[int] = []
        weights: list[float] = []
        pair_weights: list[float] = []
        for a, b in pairs:
            w = float(np.log1p(pair_counts[(a, b)]))
            src.extend([a, b])
            dst.extend([b, a])
            weights.extend([w, w])
            pair_weights.append(float(pair_counts[(a, b)]))
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_weight = np.asarray(weights, dtype=np.float32)
        pair_index = np.asarray(pairs, dtype=np.int64).T
        pair_weight = np.asarray(pair_weights, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
        pair_index = np.empty((2, 0), dtype=np.int64)
        pair_weight = np.empty((0,), dtype=np.float32)
    return GraphFeatures(x, edge_index, edge_weight, pair_index, pair_weight)


def _edges_from_coupling_map(coupling_map: Any) -> list[tuple[int, int]]:
    # Supports Qiskit's CouplingMap and backend-provided coupling-map objects.
    raw_edges = coupling_map.get_edges()
    unique = {tuple(sorted((int(a), int(b)))) for a, b in raw_edges if int(a) != int(b)}
    return sorted(unique)


def build_hardware_graph(coupling_map: CouplingMap) -> tuple[GraphFeatures, np.ndarray]:
    n = int(coupling_map.size())
    undirected_edges = _edges_from_coupling_map(coupling_map)
    if n <= 0:
        raise ValueError("Hardware graph must have at least one physical qubit.")
    degree = np.zeros(n, dtype=np.float32)
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    adjacency = np.zeros((n, n), dtype=np.float32)
    for a, b in undirected_edges:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Coupling edge {(a, b)} is outside [0, {n}).")
        degree[a] += 1.0
        degree[b] += 1.0
        adjacency[a, b] = adjacency[b, a] = 1.0
        src.extend([a, b])
        dst.extend([b, a])
        weights.extend([1.0, 1.0])
    x = (degree / max(1.0, float(degree.max()))).reshape(-1, 1).astype(np.float32)
    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.empty((2, 0), dtype=np.int64)
    edge_weight = np.asarray(weights, dtype=np.float32)
    pair_index = np.asarray(undirected_edges, dtype=np.int64).T if undirected_edges else np.empty((2, 0), dtype=np.int64)
    pair_weight = np.ones(len(undirected_edges), dtype=np.float32)
    graph = GraphFeatures(x, edge_index, edge_weight, pair_index, pair_weight)

    # Floyd-Warshall over seven nodes is cheap and avoids an extra dependency.
    distances = np.full((n, n), float(n + 1), dtype=np.float32)
    np.fill_diagonal(distances, 0.0)
    distances[adjacency > 0] = 1.0
    for k in range(n):
        distances = np.minimum(distances, distances[:, k:k + 1] + distances[k:k + 1, :])
    return graph, distances
