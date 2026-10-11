"""Hardware backend loading and environment checks."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def load_fake_jakarta() -> Any:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
    except ImportError:
        try:
            from qiskit.providers.fake_provider import FakeJakartaV2
        except ImportError as exc:
            raise RuntimeError(
                "FakeJakartaV2 could not be imported. Check qiskit and qiskit-ibm-runtime versions."
            ) from exc
    return FakeJakartaV2()


def package_version(module_name: str, distribution_name: str | None = None) -> str | None:
    try:
        module = __import__(module_name, fromlist=["__version__"])
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        return importlib.metadata.version(distribution_name or module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report(config: dict[str, Any]) -> dict[str, Any]:
    versions = {
        "torch": package_version("torch"),
        "torch_geometric": package_version("torch_geometric", "torch-geometric"),
        "qiskit": package_version("qiskit"),
        "qiskit_aer": package_version("qiskit_aer"),
        "qiskit_ibm_runtime": package_version("qiskit_ibm_runtime"),
        "mqt.bench": package_version("mqt.bench", "mqt.bench"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("sklearn", "scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "networkx": package_version("networkx"),
    }
    torch_version = versions["torch"]
    torch_base_version = str(torch_version).split("+", 1)[0] if torch_version else None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda = {
            "available": cuda_available,
            "version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except Exception as exc:
        cuda = {"available": False, "error": str(exc)}
    report: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "requested_torch_version": "2.14.0",
        "torch_version_matches_request": torch_base_version == "2.14.0",
        "cuda": cuda,
        "backend": {},
        "benchmark_load": {},
    }
    try:
        backend = load_fake_jakarta()
        coupling = backend.coupling_map
        backend_report = {
            "name": "FakeJakartaV2",
            "num_qubits": int(backend.num_qubits),
            "expected_qubits": int(config["hardware"]["expected_physical_qubits"]),
            "qubit_count_matches": int(backend.num_qubits) == int(config["hardware"]["expected_physical_qubits"]),
            "directed_coupling_edges": [[int(a), int(b)] for a, b in coupling.get_edges()],
        }
        try:
            from qiskit import QuantumCircuit, transpile
            smoke = QuantumCircuit(3)
            smoke.h(0)
            smoke.cx(0, 1)
            smoke.cx(1, 2)
            routed = transpile(
                smoke, coupling_map=coupling, initial_layout=[0, 1, 3],
                layout_method="trivial", routing_method="sabre",
                basis_gates=["cx", "id", "rz", "sx", "x", "swap"],
                optimization_level=0, seed_transpiler=int(config["routing"]["seed_transpiler"]),
            )
            backend_report["smoke_transpile"] = {
                "ok": True, "output_qubits": int(routed.num_qubits),
                "depth": int(routed.depth() or 0), "operations": {str(k): int(v) for k, v in routed.count_ops().items()},
            }
        except Exception as exc:
            backend_report["smoke_transpile"] = {"ok": False, "error": repr(exc)}
        report["backend"] = backend_report
    except Exception as exc:
        report["backend"] = {"ok": False, "error": repr(exc)}
    try:
        from .circuits import load_benchmarks, circuit_summary
        circuits = load_benchmarks(config)
        report["benchmark_load"] = {name: circuit_summary(qc) for name, qc in circuits.items()}
    except Exception as exc:
        report["benchmark_load"] = {"ok": False, "error": repr(exc)}
    report["environment_valid"] = bool(
        report["backend"].get("qubit_count_matches", False)
        and isinstance(report["benchmark_load"], dict)
        and "ok" not in report["benchmark_load"]
        and report["torch_version_matches_request"]
        and all(versions.get(k) for k in ("torch", "torch_geometric", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "mqt.bench"))
    )
    return report


def main_environment_check() -> None:
    from .config import PROJECT_ROOT, load_config
    report = environment_report(load_config())
    out = PROJECT_ROOT / "environment_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["environment_valid"]:
        raise SystemExit(2)
