import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qubit_allocation.graph_features import build_circuit_graph, build_hardware_graph
from qubit_allocation.model import MappingAwareLayoutScorer


def _tensor_graph(graph):
    return (torch.tensor(graph.x, dtype=torch.float32),
            torch.tensor(graph.edge_index, dtype=torch.long),
            torch.tensor(graph.edge_weight, dtype=torch.float32))


def test_forward_and_gradient_for_candidate_mappings():
    torch.manual_seed(7)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    cg = build_circuit_graph(qc)
    hg, dist = build_hardware_graph(CouplingMap([[0, 1], [1, 2], [2, 3]]))
    c_x, c_e, c_w = _tensor_graph(cg)
    h_x, h_e, h_w = _tensor_graph(hg)
    model = MappingAwareLayoutScorer(hidden_dim=16, mlp_hidden_dim=32, dropout=0.0)
    layouts = torch.tensor([[0, 1, 2], [3, 2, 1], [0, 2, 3]], dtype=torch.long)
    pred = model(c_x, c_e, c_w, h_x, h_e, h_w, layouts,
                 torch.tensor(cg.pair_index, dtype=torch.long),
                 torch.tensor(cg.pair_weight, dtype=torch.float32),
                 torch.tensor(dist, dtype=torch.float32))
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()
    loss = pred.square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
