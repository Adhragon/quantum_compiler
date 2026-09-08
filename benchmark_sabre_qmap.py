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

Install (see bottom of this file / chat message for full instructions):
    pip install qiskit qiskit-ibm-runtime "mqt.bench>=2.0" "mqt.qmap>=3.0" pandas numpy

Run:
    python benchmark_sabre_qmap.py
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.quantum_info import Statevector

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
# Backend loading (handles both qiskit-ibm-runtime and legacy qiskit paths)
# ---------------------------------------------------------------------------

def get_backend():
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
            "Could not import FakeJakartaV2 from either qiskit_ibm_runtime.fake_provider "
            "or qiskit.providers.fake_provider. Install qiskit-ibm-runtime: "
            "pip install qiskit-ibm-runtime"
        ) from exc


# ---------------------------------------------------------------------------
# MQT Bench circuit loading (handles both old str-level and new enum-level API)
# ---------------------------------------------------------------------------

def get_bench_circuit(name: str, size: int):
    from mqt.bench import get_benchmark
    try:
        from mqt.bench import BenchmarkLevel
        return get_benchmark(name, BenchmarkLevel.ALG, size)
    except ImportError:
        # older mqt.bench API used string levels
        return get_benchmark(benchmark_name=name, level="alg", circuit_size=size)


def load_benchmark_circuits():
    circuits = {}
    for name in BENCHMARK_NAMES:
        try:
            qc = get_bench_circuit(name, CIRCUIT_SIZE)
            qc.name = name
            # algorithm-level circuits should be unitary (no measurements);
            # strip any measurements/barriers that would break statevector fidelity
            qc.remove_final_measurements(inplace=True)
            circuits[name] = qc
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not build benchmark '{name}' at size {size if False else CIRCUIT_SIZE}: {exc}")
    return circuits


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def count_swaps(qc) -> int:
    ops = qc.count_ops()
    return int(ops.get("swap", 0))


def total_gate_count(qc) -> int:
    ops = qc.count_ops()
    return int(sum(v for k, v in ops.items() if k not in ("barrier",)))


def estimated_fidelity(qc, backend) -> float:
    """Product of (1 - gate_error) over all gates, using backend calibration
    data from backend.target. Falls back to generic error rates for gates
    not present in the target (e.g. if the compiled circuit was not fully
    translated to the native basis)."""
    target = backend.target
    fid = 1.0
    for instr in qc.data:
        name = instr.operation.name
        if name in ("barrier", "measure", "delay", "reset"):
            continue
        qubit_idxs = tuple(qc.find_bit(q).index for q in instr.qubits)
        err = None
        try:
            props = target[name][qubit_idxs]
            if props is not None:
                err = props.error
        except (KeyError, IndexError):
            err = None
        if err is None:
            # generic fallback if the gate/qubit pair isn't calibrated
            err = 0.001 if len(qubit_idxs) == 1 else 0.01
        fid *= (1.0 - err)
    return float(fid)


def _reorder_statevector(data: np.ndarray, num_phys_qubits: int,
                          final_layout: list[int], num_logical: int):
    """Undo the qubit permutation introduced by layout/routing and project
    out idle ancilla qubits (assumed to remain |0>), returning the resulting
    logical statevector (normalized) plus the leakage (probability lost to
    the ancilla-not-|0> assumption failing, should be ~0)."""
    n = num_phys_qubits
    tensor = data.reshape([2] * n)  # numpy axis a <-> qiskit qubit (n-1-a)
    ancillas = [q for q in range(n) if q not in final_layout]

    idx = [slice(None)] * n
    for anc in ancillas:
        idx[n - 1 - anc] = 0  # project ancilla onto |0>
    projected = tensor[tuple(idx)]

    remaining_axes_qubits = [q for q in range(n - 1, -1, -1) if q not in ancillas]
    want_axis_qubits = [final_layout[L] for L in range(num_logical - 1, -1, -1)]
    perm = [remaining_axes_qubits.index(q) for q in want_axis_qubits]
    projected = np.transpose(projected, perm)

    vec = projected.reshape(2 ** num_logical)
    leakage = 1.0 - float(np.vdot(vec, vec).real)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec, leakage


def state_fidelity_vs_ideal(ideal_qc, compiled_qc) -> tuple[float, float]:
    """Returns (state_fidelity, leakage). state_fidelity is |<ideal|compiled>|^2
    after undoing the final layout permutation (best-effort; falls back to
    an identity-permutation assumption with a warning if the compiled
    circuit does not carry layout metadata)."""
    ideal_sv = Statevector.from_instruction(ideal_qc)
    num_logical = ideal_qc.num_qubits

    work = compiled_qc.copy()
    work.remove_final_measurements(inplace=True)
    full_sv = Statevector.from_instruction(work)
    n_phys = work.num_qubits

    layout = getattr(compiled_qc, "layout", None)
    final_layout = None
    if layout is not None:
        try:
            final_layout = list(layout.final_index_layout(filter_ancillas=True))
        except Exception:
            final_layout = None

    if final_layout is None or len(final_layout) != num_logical:
        print("[WARN] no usable final layout found on compiled circuit; "
              "assuming identity permutation on qubits 0.."
              f"{num_logical - 1} (state fidelity may be inaccurate).")
        final_layout = list(range(num_logical))

    logical_vec, leakage = _reorder_statevector(full_sv.data, n_phys, final_layout, num_logical)
    fid = float(np.abs(np.vdot(ideal_sv.data, logical_vec)) ** 2)
    return fid, leakage


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
    from mqt.qmap.plugins.qiskit.sc import compile as qmap_compile

    t0 = time.perf_counter()
    compiled, res = qmap_compile(qc, backend, method="heuristic")
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