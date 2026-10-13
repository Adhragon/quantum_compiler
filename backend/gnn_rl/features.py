"""
Phase 4A: GNN embeddings concatenated into the RL observation.

For the current episode's circuit and the agent's *current* qubit placement,
the trained Phase 2 scorer (gnn/) produces:
  * its mapping-aware assignment embedding (hidden_dim floats): how well each
    logical qubit's circuit role matches the physical qubit it sits on, pooled;
  * its predicted SWAP cost of routing the circuit from this placement (1 float).
Both are squashed with tanh into [-1, 1] and appended to the RL observation via
RoutingEnv(extra_obs_dim=..., extra_obs_fn=GNNObservation(...)).

Because the placement changes after every SWAP, the features change every
step; a static per-circuit embedding would only tell the agent which of the
5 circuits it is routing.

Limitation: the circuit graph is built from the whole circuit, not only the
gates still to route, so the predicted cost does not shrink as routing
progresses (the env's own progress/lookahead features cover that).
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit

from gnn.gnn_layout import GNNLayoutSelector
from qubit_allocation.circuits import prepare_for_routing
from qubit_allocation.graph_features import build_circuit_graph
from qubit_allocation.train import _to_model_batch

# Tiny per-step forward passes: thread start-up costs more than the math.
torch.set_num_threads(1)


class GNNObservation:
    def __init__(self, backend=None, seed: int = 42):
        self.selector = GNNLayoutSelector(backend, seed=seed)
        self.model = self.selector.model
        self.seed = seed
        self.dim = self.model.hidden_dim + 1
        self._captured = None
        self.model.assignment_encoder.register_forward_hook(self._capture)
        self._named_graphs = {}           # benchmark name -> graph (reused across episodes)
        self._last_task, self._last_graph = None, None

    def _capture(self, _module, _inp, out):
        self._captured = out

    def _graph(self, task):
        if task is self._last_task:
            return self._last_graph
        if task.name != "random" and task.name in self._named_graphs:
            graph = self._named_graphs[task.name]
        else:
            qc = QuantumCircuit(task.num_logical)
            for op, qs in task.ops:
                qc.append(op, list(qs))
            # Same preprocessing the GNN's training circuits went through.
            graph = build_circuit_graph(prepare_for_routing(qc, self.seed))
            if task.name != "random":
                self._named_graphs[task.name] = graph
        self._last_task, self._last_graph = task, graph
        return graph

    def __call__(self, env) -> np.ndarray:
        graph = self._graph(env.task)
        layout = [int(p) for p in env.l2p[: env.task.num_logical]]
        with torch.no_grad():
            norm = _to_model_batch(self.model, graph, self.selector.hw_graph,
                                   self.selector.distances, [layout], self.selector.device)
        emb = self._captured.mean(dim=1)[0].numpy()     # same pooling as the scorer
        pred = float(norm[0]) * self.selector.target_std + self.selector.target_mean
        return np.concatenate([np.tanh(emb), [np.tanh(max(pred, 0.0) / 5.0)]])
