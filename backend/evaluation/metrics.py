
import numpy as np
import pandas as pd

import itertools

from qiskit.quantum_info import Statevector

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

def _extract_final_layout(compiled_qc, num_logical):
    """Get the circuit's final physical-qubit layout. Tries Qiskit's normal
    ancilla-filtering path first (correct for SABRE's output). Falls back to
    reading final_layout directly for QMAP, whose plugin sets
    _input_qubit_count to the full architecture width rather than the
    circuit's true qubit count, which breaks filter_ancillas."""
    layout = getattr(compiled_qc, "layout", None)
    if layout is None:
        return None
    try:
        fl = list(layout.final_index_layout(filter_ancillas=True))
        if len(fl) == num_logical:
            return fl
    except Exception:
        pass
    try:
        fl_obj = layout.final_layout
        return [fl_obj[i].index for i in range(num_logical)]
    except Exception:
        return None


def state_fidelity_vs_ideal(ideal_qc, compiled_qc):
    """Brute-force permutation search: finds which physical qubits the
    compiled circuit's output state maps to which logical qubits, by
    trying all assignments and keeping the one that maximizes fidelity
    against the ideal statevector. Tool-agnostic — doesn't rely on any
    compiler's layout metadata, which has proven inconsistent between
    SABRE (Qiskit TranspileLayout) and QMAP (different Layout semantics)."""
    ideal_sv = Statevector.from_instruction(ideal_qc)
    num_logical = ideal_qc.num_qubits

    work = compiled_qc.copy()
    work.remove_final_measurements(inplace=True)
    full_sv = Statevector.from_instruction(work)
    n_phys = work.num_qubits
    tensor = full_sv.data.reshape([2] * n_phys)

    # Physical qubits ever touched by an operation are "active"; the rest
    # are idle ancillas assumed to remain |0>.
    active = set()
    for instr in work.data:
        if instr.operation.name == "barrier":
            continue
        for q in instr.qubits:
            active.add(work.find_bit(q).index)
    active = sorted(active) if active else list(range(min(num_logical, n_phys)))

    ancillas = [q for q in range(n_phys) if q not in active]
    idx = [slice(None)] * n_phys
    for anc in ancillas:
        idx[n_phys - 1 - anc] = 0  # project ancilla onto |0>
    projected = tensor[tuple(idx)]

    flat = projected.reshape(-1)
    leakage = 1.0 - float(np.vdot(flat, flat).real)

    remaining_axes_qubits = [q for q in range(n_phys - 1, -1, -1) if q not in ancillas]

    best_fid = 0.0
    for perm in itertools.permutations(active, num_logical):
        want_axis_qubits = [perm[L] for L in range(num_logical - 1, -1, -1)]
        axis_perm = [remaining_axes_qubits.index(q) for q in want_axis_qubits]
        vec = np.transpose(projected, axis_perm).reshape(2 ** num_logical)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec = vec / norm
        fid = float(np.abs(np.vdot(ideal_sv.data, vec)) ** 2)
        if fid > best_fid:
            best_fid = fid

    return best_fid, leakage