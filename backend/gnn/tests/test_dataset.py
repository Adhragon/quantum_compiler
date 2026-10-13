import json
import pandas as pd
from qubit_allocation.dataset import DATA_COLUMNS, route_fixed_layout, validate_dataset
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


def test_dataset_validator_detects_valid_partial_file(tmp_path):
    rows = []
    layouts = [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [1, 0, 2, 3, 4], [2, 0, 1, 3, 4], [3, 0, 1, 2, 4]]
    names = ["dj_5q", "ghz_5q", "graphstate_5q", "qft_5q", "wstate_5q"]
    for name, layout in zip(names, layouts):
        rows.append({
            "circuit": name, "logical_qubits": 5, "physical_qubits": 7,
            "layout": json.dumps(layout, separators=(",", ":")), "swap_count": 0,
            "source_swap_count": 0, "cx_count": 0, "two_qubit_gate_count": 0,
            "depth": 1, "total_gates": 1, "routing_seconds": 0.01, "seed": 42,
            "benchmark_source": "MQT Bench",
        })
    csv_path = tmp_path / "phase2_data.csv"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame(rows, columns=DATA_COLUMNS).to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps({"rows_written": 5, "status": "complete", "exhaustive": False}), encoding="utf-8")
    report = validate_dataset(csv_path, manifest_path, strict=False)
    assert report["valid"]
    assert report["rows"] == 5


def test_fixed_layout_routing_returns_real_metrics():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(1, 2)
    metrics, elapsed = route_fixed_layout(
        qc, CouplingMap([[0, 1], [1, 2]]), [0, 1, 2], 42,
        ["cx", "id", "rz", "sx", "x", "swap"],
    )
    assert metrics["swap_count"] >= 0
    assert metrics["total_gates"] >= 1
    assert metrics["depth"] >= 1
    assert elapsed >= 0
