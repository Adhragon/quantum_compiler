"""Load and deterministically preprocess MQT Bench circuits."""
from __future__ import annotations

from collections import Counter
from typing import Any

from qiskit import QuantumCircuit, transpile


# Keep SWAP in the output basis so routing-added SWAP operations remain visible.
ROUTING_BASIS_GATES = ["cx", "id", "rz", "sx", "x", "swap"]


def _get_benchmark_level() -> Any:
    try:
        from mqt.bench import BenchmarkLevel
    except ImportError:
        try:
            from mqt.bench.benchmark_generation import BenchmarkLevel
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import MQT Bench BenchmarkLevel. "
                "Install the pinned mqt.bench dependency."
            ) from exc

    for candidate in ("ALG", "ALGORITHMIC"):
        if hasattr(BenchmarkLevel, candidate):
            return getattr(BenchmarkLevel, candidate)

    raise RuntimeError(
        f"No algorithmic level found in BenchmarkLevel: {list(BenchmarkLevel)}"
    )


def strip_nonunitary(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove measurements/barriers and preserve quantum-register width."""
    clean = QuantumCircuit(circuit.num_qubits, name=circuit.name)

    for item in circuit.data:
        op = item.operation

        if op.name in {"measure", "barrier", "delay"}:
            continue

        qargs = [circuit.find_bit(q).index for q in item.qubits]
        clean.append(op, qargs)

    return clean


def prepare_for_routing(
    circuit: QuantumCircuit,
    seed: int = 42,
) -> QuantumCircuit:
    """Normalize circuit operations into the documented routing basis."""
    clean = strip_nonunitary(circuit)

    prepared = transpile(
        clean,
        basis_gates=ROUTING_BASIS_GATES,
        optimization_level=0,
        seed_transpiler=seed,
    )

    if prepared.num_qubits != circuit.num_qubits:
        raise RuntimeError(
            "Circuit width changed during preprocessing: "
            f"{circuit.num_qubits} -> {prepared.num_qubits}"
        )

    return prepared


def load_benchmarks(
    config: dict[str, Any],
) -> dict[str, QuantumCircuit]:
    """Load configured MQT Bench circuits with deterministic generation."""
    try:
        from mqt.bench import get_benchmark
    except ImportError as exc:
        raise RuntimeError(
            "MQT Bench is required; hand-built fallback circuits are disabled."
        ) from exc

    level = _get_benchmark_level()
    circuits: dict[str, QuantumCircuit] = {}

    # Use the same configured seed for benchmark generation and preprocessing.
    seed = int(config["seed"])

    for spec in config["benchmarks"]:
        expected_n = int(spec["logical_qubits"])
        benchmark_name = spec["name"]

        # Graph-state circuits may be generated randomly.
        # Pass the seed to MQT Bench itself, not only to Qiskit transpilation.
        benchmark_kwargs: dict[str, Any] = {}

        if benchmark_name.lower() == "graphstate":
            benchmark_kwargs["seed"] = seed

        try:
            original = get_benchmark(
                benchmark=benchmark_name,
                level=level,
                circuit_size=expected_n,
                **benchmark_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MQT Bench benchmark {benchmark_name!r} "
                f"at {expected_n} qubits. "
                "No substitute circuit will be used. "
                "Check whether this MQT Bench version supports the "
                "configured benchmark seed."
            ) from exc

        if original.num_qubits != expected_n:
            raise ValueError(
                f"MQT Bench returned {original.num_qubits} qubits "
                f"for {benchmark_name}; expected {expected_n}."
            )

        prepared = prepare_for_routing(original, seed)
        prepared.name = spec["label"]

        high_arity = [
            (item.operation.name, len(item.qubits))
            for item in prepared.data
            if len(item.qubits) > 2
        ]

        if high_arity:
            raise ValueError(
                f"Preprocessing left multi-qubit operations in "
                f"{spec['label']}: {high_arity[:5]}"
            )

        circuits[spec["label"]] = prepared

    return circuits


def circuit_summary(circuit: QuantumCircuit) -> dict[str, Any]:
    counts = Counter(
        {str(k): int(v) for k, v in circuit.count_ops().items()}
    )

    return {
        "name": circuit.name,
        "num_qubits": int(circuit.num_qubits),
        "circuit_size": int(circuit.size()),
        "depth": int(circuit.depth() or 0),
        "operation_counts": dict(counts),
    }