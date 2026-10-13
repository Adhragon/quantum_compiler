import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qubit_allocation.graph_features import build_circuit_graph, build_hardware_graph


def test_circuit_graph_has_weighted_interaction_pair():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(0, 1)
    qc.cx(1, 2)
    graph = build_circuit_graph(qc)
    assert graph.x.shape == (3, 3)
    assert graph.edge_index.shape == (2, 4)
    assert graph.pair_index.shape == (2, 2)
    assert np.all(graph.edge_weight > 0)
    assert graph.pair_weight.tolist() == [2.0, 1.0]


def test_hardware_edges_are_undirected_unique_and_distance_correct():
    cm = CouplingMap([[0, 1], [1, 0], [1, 2]])
    graph, distances = build_hardware_graph(cm)
    assert graph.x.shape == (3, 1)
    assert graph.pair_index.shape == (2, 2)
    assert np.allclose(distances, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
