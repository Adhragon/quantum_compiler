"""
Shared config for the quantum-compiler-project.

Every phase (baselines, GNN, RL, GNN+RL, evaluation) imports from here so
the backend, benchmark set, and qubit range stay identical across all
comparisons. Do not fork these constants per-phase — if a phase genuinely
needs a different setting, add a phase-specific override in that phase's
own module and note why in its README, don't edit this file's defaults.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scope freeze (see README.md for the full rationale)
# ---------------------------------------------------------------------------

BACKEND_NAME = "FakeJakartaV2"          # single fixed topology, 7 qubits
BACKEND_NUM_QUBITS = 7

# Phase 1 (SABRE / MQT QMAP baselines) + Phase 2/4 (GNN) circuit set.
CIRCUIT_SIZE = 5                         # qubits per benchmark circuit
BENCHMARK_NAMES = ["ghz", "dj", "graphstate", "qft", "wstate"]
BENCHMARK_LEVEL = "alg"                  # algorithm-level MQT Bench circuits

# Phase 3 (RL routing) qubit range — fixed single topology, no generalization.
RL_QUBIT_RANGE = (5, 8)

SEED = 42
SABRE_OPT_LEVEL = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = "evaluation/results"
BASELINE_CSV = f"{RESULTS_DIR}/sabre_vs_qmap_results.csv"
GNN_CHECKPOINT = "gnn/checkpoints/gcn.pt"
RL_CHECKPOINT = "rl/checkpoints/sb3_agent.zip"
GNN_RL_CHECKPOINT = "gnn_rl/checkpoints/sb3_agent_gnn.zip"


# ---------------------------------------------------------------------------
# Shared backend / circuit loaders
# ---------------------------------------------------------------------------

def get_backend():
    """Return the fixed FakeJakartaV2 backend. Handles both the current
    qiskit-ibm-runtime location and the legacy qiskit.providers path."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
        return FakeJakartaV2()
    except ImportError:
        pass
    try:
        from qiskit.providers.fake_provider import FakeJakartaV2
        return FakeJakartaV2()
    except ImportError as exc:
        raise ImportError(
            "Could not import FakeJakartaV2. Install qiskit-ibm-runtime: "
            "pip install qiskit-ibm-runtime"
        ) from exc


def get_bench_circuit(name: str, size: int = CIRCUIT_SIZE):
    """Load a single MQT Bench algorithm-level circuit, handling both the
    current enum-based API and the older string-level API."""
    from mqt.bench import get_benchmark
    try:
        from mqt.bench import BenchmarkLevel
        return get_benchmark(name, BenchmarkLevel.ALG, size)
    except ImportError:
        return get_benchmark(benchmark_name=name, level=BENCHMARK_LEVEL, circuit_size=size)


def load_benchmark_circuits(names: list[str] | None = None, size: int = CIRCUIT_SIZE) -> dict:
    """Load the full fixed benchmark set (or a subset) as {name: QuantumCircuit},
    with measurements stripped so circuits stay unitary for statevector fidelity."""
    names = names or BENCHMARK_NAMES
    circuits = {}
    for name in names:
        qc = get_bench_circuit(name, size)
        qc.name = name
        qc.remove_final_measurements(inplace=True)
        circuits[name] = qc
    return circuits