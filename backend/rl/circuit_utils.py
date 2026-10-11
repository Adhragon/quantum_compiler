"""
Circuit / topology helpers for the Phase 3 RL router.

Everything here is plain Python + numpy so the Gymnasium env can step fast:
Qiskit is only touched when converting a QuantumCircuit into a RoutingTask
(once per circuit) and when rebuilding the routed circuit (once per episode).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import CXGate

# Fallback basis for unrolling logical circuits into <=2-qubit gates when no
# backend is given (see circuit_to_task).
UNROLL_BASIS = ["u", "cx"]


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def coupling_edges(backend) -> list[tuple[int, int]]:
    """Undirected, sorted coupler list. Index in this list == RL action id."""
    return sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})


def distance_matrix(num_qubits: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """All-pairs shortest-path hop distance on the coupling graph."""
    g = nx.Graph()
    g.add_nodes_from(range(num_qubits))
    g.add_edges_from(edges)
    dist = np.zeros((num_qubits, num_qubits), dtype=np.int64)
    for src, lengths in nx.all_pairs_shortest_path_length(g):
        for dst, d in lengths.items():
            dist[src, dst] = d
    return dist


def shortest_path(edges: list[tuple[int, int]], src: int, dst: int) -> list[int]:
    g = nx.Graph()
    g.add_edges_from(edges)
    return nx.shortest_path(g, src, dst)


# ---------------------------------------------------------------------------
# Routing task = unrolled logical circuit as a flat instruction list
# ---------------------------------------------------------------------------

@dataclass
class RoutingTask:
    name: str
    num_logical: int
    # (operation, logical qubit indices) in program order; every op is 1q or 2q.
    ops: list[tuple[Instruction, tuple[int, ...]]]

    @property
    def num_2q(self) -> int:
        return sum(1 for _, qs in self.ops if len(qs) == 2)


def circuit_to_task(qc: QuantumCircuit, name: str | None = None, backend=None,
                    opt_level: int = 3) -> RoutingTask:
    """Flatten a logical circuit into 1q/2q ops for routing.

    With a backend, the circuit goes through the same pre-routing init stage
    SABRE gets at this optimization level (synthesis + ElidePermutations,
    which removes QFT's trailing logical SWAPs), so RL and SABRE route the
    identical circuit. Without one (tests only), it is unrolled to u/cx.
    Barriers / measurements are dropped."""
    if backend is not None:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
        unrolled = pm.init.run(qc)
    else:
        unrolled = transpile(qc, basis_gates=UNROLL_BASIS, optimization_level=0)
    ops = []
    for instr in unrolled.data:
        op = instr.operation
        if op.name in ("barrier", "measure", "delay"):
            continue
        qs = tuple(unrolled.find_bit(q).index for q in instr.qubits)
        if len(qs) > 2:
            raise ValueError(f"{op.name} on {len(qs)} qubits survived unrolling")
        ops.append((op, qs))
    return RoutingTask(name=name or qc.name, num_logical=qc.num_qubits, ops=ops)


def random_task(num_logical: int, num_2q: int, rng: np.random.Generator,
                name: str = "random") -> RoutingTask:
    """Random CX-only circuit used as training data (routing only cares about
    the 2q interaction pattern, so 1q gates are omitted)."""
    ops = []
    for _ in range(num_2q):
        a, b = rng.choice(num_logical, size=2, replace=False)
        ops.append((CXGate(), (int(a), int(b))))
    return RoutingTask(name=name, num_logical=num_logical, ops=ops)


# ---------------------------------------------------------------------------
# SABRE reference (shared initial layout + true SWAP count)
# ---------------------------------------------------------------------------

def sabre_layout_and_swaps(qc: QuantumCircuit, backend, seed: int) -> tuple[list[int], int]:
    """SABRE's initial layout (physical qubit per logical qubit) and the number
    of SWAPs it inserts. Stops after the routing stage: once basis translation
    runs, SWAPs are decomposed into CXs and count_ops() reports 0."""
    from qiskit.transpiler import StagedPassManager
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend,
        layout_method="sabre", routing_method="sabre", seed_transpiler=seed,
    )
    routing_only = StagedPassManager(
        stages=["init", "layout", "routing"],
        init=pm.init, layout=pm.layout, routing=pm.routing,
    )
    routed = routing_only.run(qc)
    layout = list(routed.layout.initial_index_layout(filter_ancillas=True))
    return layout, int(routed.count_ops().get("swap", 0))
