"""Generate, validate and describe the labelled routing-cost dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from qiskit import transpile

from .circuits import ROUTING_BASIS_GATES, circuit_summary, load_benchmarks
from .config import PROJECT_ROOT, load_config
from .hardware import load_fake_jakarta, package_version
from .layout_generator import all_layouts, number_of_layouts, sample_layouts, validate_layout

LOGGER = logging.getLogger(__name__)
DATA_COLUMNS = [
    "circuit", "logical_qubits", "physical_qubits", "layout", "swap_count",
    "source_swap_count", "cx_count", "two_qubit_gate_count", "depth", "total_gates", "routing_seconds", "seed", "benchmark_source",
]


def route_fixed_layout(circuit, coupling_map, layout: list[int], seed: int, basis_gates: list[str]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    routed = transpile(
        circuit,
        coupling_map=coupling_map,
        initial_layout=layout,
        layout_method="trivial",
        routing_method="sabre",
        basis_gates=basis_gates,
        optimization_level=0,
        seed_transpiler=seed,
    )
    elapsed = time.perf_counter() - t0
    ops = routed.count_ops()
    total_swaps = int(ops.get("swap", 0))
    source_swaps = int(circuit.count_ops().get("swap", 0))
    inserted_swaps = total_swaps - source_swaps
    if inserted_swaps < 0:
        raise RuntimeError(
            f"Routed SWAP count ({total_swaps}) lower than source SWAP count ({source_swaps})."
        )
    return {
        "swap_count": inserted_swaps,
        "total_swap_operations": total_swaps,
        "source_swap_count": source_swaps,
        "cx_count": int(ops.get("cx", 0)),
        "two_qubit_gate_count": int(sum(1 for instruction in routed.data if len(instruction.qubits) == 2)),
        "depth": int(routed.depth() or 0),
        "total_gates": int(sum(ops.values())),
    }, elapsed


def _environment_versions() -> dict[str, str | None]:
    names = {
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "torch_geometric": package_version("torch_geometric", "torch-geometric"),
        "qiskit": package_version("qiskit"),
        "qiskit_aer": package_version("qiskit_aer"),
        "qiskit_ibm_runtime": package_version("qiskit_ibm_runtime"),
        "mqt.bench": package_version("mqt.bench", "mqt.bench"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
    }
    return names


def generate_dataset(
    config_path: str | Path | None = None,
    limit_per_circuit: int | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    project_root = PROJECT_ROOT
    data_path = project_root / config["dataset"]["output_csv"]
    manifest_path = project_root / config["dataset"]["manifest"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if (data_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(f"Dataset or manifest already exists; pass --force to replace them. Dataset: {data_path}")
    if force:
        data_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    backend = load_fake_jakarta()
    expected_physical = int(config["hardware"]["expected_physical_qubits"])
    if int(backend.num_qubits) != expected_physical:
        raise RuntimeError(f"FakeJakartaV2 has {backend.num_qubits} qubits; expected {expected_physical}.")
    coupling_map = backend.coupling_map
    circuits = load_benchmarks(config)
    seed = int(config["seed"])
    routing_seed = int(config["routing"]["seed_transpiler"])
    rng_sample_n = limit_per_circuit
    basis_gates = list(config["routing"]["basis_gates"])
    all_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    circuit_manifest: dict[str, Any] = {}
    started = time.perf_counter()

    for benchmark_spec in config["benchmarks"]:
        name = benchmark_spec["label"]
        circuit = circuits[name]
        logical_n = int(circuit.num_qubits)
        physical_n = int(backend.num_qubits)
        total = number_of_layouts(logical_n, physical_n)
        requested = total if rng_sample_n is None else min(int(rng_sample_n), total)
        if requested == total:
            layouts = list(all_layouts(logical_n, physical_n))
        else:
            layouts = sample_layouts(logical_n, physical_n, requested, seed + len(all_rows))
        LOGGER.info("Generating labels for %s: %d/%d layouts", name, len(layouts), total)
        succeeded = 0
        for idx, layout in enumerate(layouts, start=1):
            if not validate_layout(layout, logical_n, physical_n):
                raise AssertionError(f"Generated invalid layout: {layout}")
            try:
                metrics, elapsed = route_fixed_layout(circuit, coupling_map, list(layout), routing_seed, basis_gates)
                row = {
                    "circuit": name,
                    "logical_qubits": logical_n,
                    "physical_qubits": physical_n,
                    "layout": json.dumps(list(layout), separators=(",", ":")),
                    "swap_count": metrics["swap_count"],
                    "source_swap_count": metrics["source_swap_count"],
                    "cx_count": metrics["cx_count"],
                    "two_qubit_gate_count": metrics["two_qubit_gate_count"],
                    "depth": metrics["depth"],
                    "total_gates": metrics["total_gates"],
                    "routing_seconds": round(elapsed, 6),
                    "seed": routing_seed,
                    "benchmark_source": "MQT Bench",
                }
                all_rows.append(row)
                succeeded += 1
            except Exception as exc:
                failed.append({"circuit": name, "layout": list(layout), "error": repr(exc)})
            if idx % 100 == 0 or idx == len(layouts):
                LOGGER.info("%s: %d/%d attempted; %d successful", name, idx, len(layouts), succeeded)
        circuit_manifest[name] = {
            "benchmark": benchmark_spec["name"],
            "source": "MQT Bench",
            "summary": circuit_summary(circuit),
            "logical_qubits": logical_n,
            "total_possible_layouts": total,
            "attempted_layouts": len(layouts),
            "successful_layouts": succeeded,
        }
        # Persist safely after each circuit so progress survives an interruption.
        pd.DataFrame(all_rows, columns=DATA_COLUMNS).to_csv(data_path, index=False)
        current_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
        manifest_partial = {
            "status": "in_progress",
            "dataset_sha256": current_sha256,
            "configuration": config,
            "package_versions": _environment_versions(),
            "hardware": {
                "name": "FakeJakartaV2",
                "num_qubits": physical_n,
                "directed_coupling_edges": [[int(a), int(b)] for a, b in coupling_map.get_edges()],
            },
            "circuits": circuit_manifest,
            "failed_routes": failed,
            "rows_written": len(all_rows),
            "exhaustive": limit_per_circuit is None,
        }
        manifest_path.write_text(json.dumps(manifest_partial, indent=2), encoding="utf-8")

    elapsed_total = time.perf_counter() - started
    df = pd.DataFrame(all_rows, columns=DATA_COLUMNS)
    if not df.empty:
        dist: dict[str, Any] = {}
        for name, group in df.groupby("circuit"):
            dist[name] = {
                "count": int(len(group)),
                "swap_min": int(group["swap_count"].min()),
                "swap_max": int(group["swap_count"].max()),
                "swap_mean": float(group["swap_count"].mean()),
                "swap_std": float(group["swap_count"].std(ddof=0)),
                "routing_seconds_mean": float(group["routing_seconds"].mean()),
                "unique_layouts": int(group["layout"].nunique()),
            }
    else:
        dist = {}
    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.is_file() else None
    manifest = {
        "status": "complete" if not failed and len(df) == sum(v["attempted_layouts"] for v in circuit_manifest.values()) else "incomplete",
        "dataset_sha256": dataset_sha256,
        "configuration": config,
        "package_versions": _environment_versions(),
        "python": sys.version,
        "hardware": {
            "name": "FakeJakartaV2",
            "num_qubits": int(backend.num_qubits),
            "directed_coupling_edges": [[int(a), int(b)] for a, b in coupling_map.get_edges()],
        },
        "circuits": circuit_manifest,
        "failed_routes": failed,
        "rows_written": int(len(df)),
        "distribution": dist,
        "generation_seconds": round(elapsed_total, 3),
        "exhaustive": limit_per_circuit is None,
        "expected_rows_if_exhaustive": int(sum(v["total_possible_layouts"] for v in circuit_manifest.values())),
        "dataset_columns": DATA_COLUMNS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d rows to %s", len(df), data_path)
    LOGGER.info("Wrote manifest to %s; status=%s", manifest_path, manifest["status"])
    if manifest["status"] != "complete":
        LOGGER.warning("Dataset generation has failed routes or missing rows; see manifest.")
    return data_path, manifest_path


def validate_dataset(data_path: str | Path | None = None, manifest_path: str | Path | None = None, strict: bool = True) -> dict[str, Any]:
    config = load_config()
    data_path = Path(data_path) if data_path else PROJECT_ROOT / config["dataset"]["output_csv"]
    manifest_path = Path(manifest_path) if manifest_path else PROJECT_ROOT / config["dataset"]["manifest"]
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    df = pd.read_csv(data_path)
    missing_columns = sorted(set(DATA_COLUMNS) - set(df.columns))
    errors: list[str] = []
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    if df.empty:
        errors.append("Dataset has no rows.")
    for col in ("swap_count", "logical_qubits", "physical_qubits", "routing_seconds", "seed"):
        if col in df and df[col].isna().any():
            errors.append(f"Column {col} contains missing values.")
    for col in ("swap_count", "source_swap_count", "cx_count", "two_qubit_gate_count", "depth", "total_gates"):
        if col in df:
            if (df[col] < 0).any() or not ((df[col] % 1) == 0).all():
                errors.append(f"Column {col} must contain non-negative integers.")
    if "routing_seconds" in df and (df["routing_seconds"] < 0).any():
        errors.append("Routing times cannot be negative.")
    if "benchmark_source" in df and not (df["benchmark_source"].astype(str) == "MQT Bench").all():
        errors.append("Benchmark provenance must be explicitly recorded as MQT Bench for every row.")
    if "physical_qubits" in df and not (df["physical_qubits"] == int(config["hardware"]["expected_physical_qubits"])).all():
        errors.append("Dataset physical-qubit count does not match experiment configuration.")
    expected_logical = {b["label"]: int(b["logical_qubits"]) for b in config["benchmarks"]}
    if "logical_qubits" in df and "circuit" in df:
        for circuit_name, group in df.groupby("circuit"):
            if circuit_name in expected_logical and not (group["logical_qubits"] == expected_logical[circuit_name]).all():
                errors.append(f"Logical-qubit count does not match configuration for {circuit_name}.")
    if "layout" in df:
        parsed_layouts: list[tuple[int, ...]] = []
        for i, value in enumerate(df["layout"].tolist()):
            try:
                layout = tuple(int(q) for q in json.loads(value))
                logical_n = int(df.iloc[i]["logical_qubits"])
                physical_n = int(df.iloc[i]["physical_qubits"])
                if not validate_layout(layout, logical_n, physical_n):
                    errors.append(f"Invalid layout on row {i}: {layout}")
                parsed_layouts.append(layout)
            except Exception as exc:
                errors.append(f"Cannot parse layout on row {i}: {exc}")
                parsed_layouts.append(tuple())
        if "circuit" in df:
            duplicate_mask = pd.DataFrame({"circuit": df["circuit"].astype(str), "layout": parsed_layouts}).duplicated()
            if duplicate_mask.any():
                errors.append(f"Found {int(duplicate_mask.sum())} duplicate circuit-layout records.")
    expected_labels = {b["label"] for b in config["benchmarks"]}
    observed_labels = set(df["circuit"].astype(str)) if "circuit" in df else set()
    if observed_labels != expected_labels:
        errors.append(f"Expected circuits {sorted(expected_labels)}, observed {sorted(observed_labels)}.")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("rows_written", len(df))) != len(df):
            errors.append("Manifest row count does not match dataset CSV.")
        expected_hash = manifest.get("dataset_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append("Dataset SHA-256 does not match the manifest.")
        if strict:
            if manifest.get("status") != "complete":
                errors.append(f"Manifest status is not complete: {manifest.get('status')!r}.")
            if not manifest.get("exhaustive", False):
                errors.append("Manifest is non-exhaustive; full validation requires all candidate layouts.")
            if manifest.get("failed_routes"):
                errors.append(f"Manifest records {len(manifest['failed_routes'])} failed routes.")
            if manifest.get("exhaustive"):
                expected = int(manifest.get("expected_rows_if_exhaustive", -1))
                if len(df) != expected:
                    errors.append(f"Exhaustive dataset expected {expected} rows but found {len(df)}.")
                for circuit_name, group in df.groupby("circuit"):
                    logical_n = int(group["logical_qubits"].iloc[0])
                    physical_n = int(group["physical_qubits"].iloc[0])
                    expected_per_circuit = number_of_layouts(logical_n, physical_n)
                    if len(group) != expected_per_circuit:
                        errors.append(f"{circuit_name}: expected {expected_per_circuit} layouts, got {len(group)}.")
    elif strict:
        errors.append(f"Manifest file is missing: {manifest_path}")
    report = {
        "valid": not errors,
        "errors": errors,
        "rows": int(len(df)),
        "circuits": sorted(observed_labels),
        "swap_summary": df.groupby("circuit")["swap_count"].agg(["count", "min", "max", "mean", "std"]).fillna(0).to_dict(orient="index") if "swap_count" in df and "circuit" in df else {},
        "strict": strict,
    }
    if errors:
        raise ValueError(json.dumps(report, indent=2))
    return report


def main_generate() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-circuit", type=int, default=None, help="Smoke-test mode; default is exhaustive.")
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_dataset(args.config, args.limit_per_circuit, args.force)


def main_validate() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.data, args.manifest, strict=not args.allow_partial), indent=2))


if __name__ == "__main__":
    main_generate()
