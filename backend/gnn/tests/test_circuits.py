from qubit_allocation.circuits import load_benchmarks
from qubit_allocation.config import load_config


def test_mqt_bench_primary_circuits_load_and_are_preprocessed():
    circuits = load_benchmarks(load_config())
    assert set(circuits) == {"dj_5q", "ghz_5q", "graphstate_5q", "qft_5q", "wstate_5q"}
    for circuit in circuits.values():
        assert circuit.num_qubits == 5
        assert all(len(inst.qubits) <= 2 for inst in circuit.data)
        assert all(inst.operation.name not in {"measure", "barrier", "delay"} for inst in circuit.data)
