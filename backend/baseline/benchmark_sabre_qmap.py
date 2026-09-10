"""
Benchmark: Qiskit SABRE vs MQT QMAP on FakeJakartaV2 (7 qubits)
================================================================

Compiles 5 algorithm-level MQT Bench circuits with:
  (1) Qiskit's built-in SABRE layout+routing (transpile optimization_level=3)
  (2) MQT QMAP heuristic mapper

...and logs: circuit depth, SWAP count, total gate count, compilation
time, an estimated fidelity (product of 1-gate_error over all gates,
from the backend's noise calibration), and a state fidelity (overlap
between the ideal unitary output state and the compiled circuit's
output state, after undoing the qubit permutation introduced by
layout/routing).

RUN :
from backend folder execute :
python -m baseline.benchmark_sabre_qmap 

Running with -m makes Python treat backend as the root package, so config and evaluation.metrics resolve correctly. 
Don't run it with python baseline/benchmark_sabre_qmap.py directly — that breaks the package context.

"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import pandas as pd

from qiskit import transpile

from config import get_backend , get_bench_circuit , load_benchmark_circuits
from evaluation.metrics import count_swaps, total_gate_count, estimated_fidelity, _reorder_statevector , state_fidelity_vs_ideal 

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Circuit size (qubits) requested from MQT Bench. Must be <= backend qubit
# count (7 for Jakarta). Using 5 leaves headroom for routing.
CIRCUIT_SIZE = 5

# Algorithm-level MQT Bench benchmarks to pull (3-5 required; 5 chosen here).
# All of these are available at small sizes in mqt.bench >= 1.x / 2.x.
BENCHMARK_NAMES = ["ghz", "dj", "graphstate", "qft", "wstate"]

SABRE_OPT_LEVEL = 3
SEED = 42
N_TRIALS = 1  # increase to average timing/metrics over multiple runs

# ---------------------------------------------------------------------------
# Compilers under test
# ---------------------------------------------------------------------------

def run_sabre(qc, backend):
    t0 = time.perf_counter()
    compiled = transpile(
        qc,
        backend=backend,
        optimization_level=SABRE_OPT_LEVEL,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=SEED,
    )
    t1 = time.perf_counter()
    return compiled, (t1 - t0)

def run_qmap(qc, backend):
    from mqt.qmap.plugins.qiskit.sc import compile_ as qmap_compile
    from mqt import qmap
    from mqt.qmap.sc import Method
    from qiskit import transpile

    arch = qmap.sc.Architecture(backend.num_qubits, set(backend.coupling_map.get_edges()))

    # QMAP expects an already-decomposed circuit (1-/2-qubit native gates only).
    # No coupling_map passed here → this only does basis translation, not routing.
    basis_gates = list(backend.target.operation_names)
    decomposed = transpile(qc, basis_gates=basis_gates, optimization_level=1)

    t0 = time.perf_counter()
    compiled, res = qmap_compile(decomposed, arch, method=Method.heuristic)

    '''print("compiled.layout:", compiled.layout)
    print("res.output.qubits:", res.output.qubits)
    print("res.input_ attrs:", [a for a in dir(res.input_) if not a.startswith('_')])
    print("res.configuration attrs:", [a for a in dir(res.configuration) if not a.startswith('_')])'''
    t1 = time.perf_counter()

    wall_time = t1 - t0
    reported_time = getattr(res, "time", None)
    swaps_reported = None
    try:
        swaps_reported = res.output.swaps
    except Exception:
        pass

    return compiled, wall_time, reported_time, swaps_reported


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

@dataclass
class Row:
    circuit: str
    method: str
    depth: int = 0
    swaps: int = 0
    total_gates: int = 0
    compile_time_s: float = 0.0
    estimated_fidelity: float = 0.0
    state_fidelity: float = 0.0
    leakage: float = 0.0
    notes: str = ""


def main():
    backend = get_backend()
    print(f"Backend: {backend.name}, num_qubits={backend.num_qubits}")

    circuits = load_benchmark_circuits()
    if not circuits:
        raise RuntimeError("No MQT Bench circuits could be loaded.")
    print(f"Loaded circuits: {list(circuits.keys())}")

    rows: list[Row] = []

    for name, qc in circuits.items():
        print(f"\n=== {name} (n={qc.num_qubits}, depth={qc.depth()}) ===")

        # ---- SABRE ----
        try:
            sabre_qc, sabre_time = run_sabre(qc, backend)
            sf, leak = state_fidelity_vs_ideal(qc, sabre_qc)
            rows.append(Row(
                circuit=name,
                method="SABRE",
                depth=sabre_qc.depth(),
                swaps=count_swaps(sabre_qc),
                total_gates=total_gate_count(sabre_qc),
                compile_time_s=sabre_time,
                estimated_fidelity=estimated_fidelity(sabre_qc, backend),
                state_fidelity=sf,
                leakage=leak,
            ))
            print(f"  SABRE:  depth={rows[-1].depth}, swaps={rows[-1].swaps}, "
                  f"gates={rows[-1].total_gates}, t={sabre_time:.4f}s, "
                  f"est_fid={rows[-1].estimated_fidelity:.4f}, state_fid={sf:.4f}")
        except Exception:
            print(f"[ERROR] SABRE failed on {name}:\n{traceback.format_exc()}")
            rows.append(Row(circuit=name, method="SABRE", notes="FAILED"))

        # ---- MQT QMAP ----
        try:
            qmap_qc, qmap_wall_time, qmap_reported_time, qmap_swaps = run_qmap(qc, backend)
            sf, leak = state_fidelity_vs_ideal(qc, qmap_qc)
            swaps = qmap_swaps if qmap_swaps is not None else count_swaps(qmap_qc)
            rows.append(Row(
                circuit=name,
                method="MQT_QMAP",
                depth=qmap_qc.depth(),
                swaps=swaps,
                total_gates=total_gate_count(qmap_qc),
                compile_time_s=qmap_wall_time,
                estimated_fidelity=estimated_fidelity(qmap_qc, backend),
                state_fidelity=sf,
                leakage=leak,
                notes=f"internal_time={qmap_reported_time}" if qmap_reported_time else "",
            ))
            print(f"  QMAP:   depth={rows[-1].depth}, swaps={rows[-1].swaps}, "
                  f"gates={rows[-1].total_gates}, t={qmap_wall_time:.4f}s, "
                  f"est_fid={rows[-1].estimated_fidelity:.4f}, state_fid={sf:.4f}")
        except Exception:
            print(f"[ERROR] MQT QMAP failed on {name}:\n{traceback.format_exc()}")
            rows.append(Row(circuit=name, method="MQT_QMAP", notes="FAILED"))

    df = pd.DataFrame([r.__dict__ for r in rows])
    pd.set_option("display.width", 140)
    print("\n\n===================== RESULTS =====================")
    print(df.to_string(index=False))

    out_csv = "sabre_vs_qmap_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    return df


if __name__ == "__main__":
    main()