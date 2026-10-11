import torch
from qiskit import QuantumCircuit, transpile
from qubit_allocation.hardware import load_fake_jakarta


def test_required_torch_release_is_installed():
    assert torch.__version__.split("+", 1)[0] == "2.14.0", (
        f"Required torch==2.14.0, found {torch.__version__}"
    )


def test_fake_jakarta_is_seven_qubits_and_routes_smoke_circuit():
    backend = load_fake_jakarta()
    assert backend.num_qubits == 7
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    routed = transpile(
        qc,
        coupling_map=backend.coupling_map,
        initial_layout=[0, 1, 3],
        layout_method="trivial",
        routing_method="sabre",
        basis_gates=["cx", "id", "rz", "sx", "x", "swap"],
        optimization_level=0,
        seed_transpiler=42,
    )
    assert routed.num_qubits == 7
    assert routed.depth() >= 1
