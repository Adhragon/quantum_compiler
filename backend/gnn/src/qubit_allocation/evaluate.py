"""Evaluate the trained GNN, fixed-layout and default SABRE baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from qiskit import transpile

from .circuits import load_benchmarks
from .config import PROJECT_ROOT, config_sha256, load_config
from .dataset import route_fixed_layout, validate_dataset
from .graph_features import GraphFeatures, build_circuit_graph, build_hardware_graph
from .hardware import load_fake_jakarta
from .layout_generator import all_layouts, number_of_layouts
from .metrics import percentage_reduction, regression_metrics, ranking_metrics
from .model import MappingAwareLayoutScorer
from .train import _graph_tensors, _to_model_batch
from .split_utils import layout_id, parse_layout

LOGGER = logging.getLogger(__name__)


def _model_predict(model: MappingAwareLayoutScorer, circuit_graph: GraphFeatures,
                   hardware_graph: GraphFeatures, distances: np.ndarray, layouts: list[tuple[int, ...]],
                   device: torch.device, batch_size: int, target_mean: float, target_std: float) -> np.ndarray:
    model.eval()
    result: list[float] = []
    for start in range(0, len(layouts), batch_size):
        batch = [list(x) for x in layouts[start:start + batch_size]]
        with torch.no_grad():
            pred_norm = _to_model_batch(model, circuit_graph, hardware_graph, distances, batch, device)
            pred = pred_norm.detach().cpu().numpy() * target_std + target_mean
        result.extend(float(max(0.0, x)) for x in pred)
    return np.asarray(result, dtype=np.float64)


def _fixed_layout_metrics(circuit, coupling_map, layout, config) -> dict[str, Any]:
    metrics, elapsed = route_fixed_layout(
        circuit, coupling_map, list(layout), int(config["routing"]["seed_transpiler"]),
        list(config["routing"]["basis_gates"]),
    )
    metrics["compile_seconds"] = float(elapsed)
    metrics["layout"] = list(layout)
    metrics["layout_scoring_seconds"] = 0.0
    metrics["end_to_end_seconds"] = float(elapsed)
    return metrics


def _initial_layout_repr(transpiled, original_circuit) -> str:
    try:
        layout_obj = transpiled.layout.initial_layout
        virtual_to_physical = layout_obj.get_virtual_bits()
        mapping = [int(virtual_to_physical[q]) for q in original_circuit.qubits]
        return json.dumps(mapping, separators=(",", ":"))
    except Exception:
        return "unknown"


def _default_sabre(circuit, coupling_map, config) -> dict[str, Any]:
    t0 = time.perf_counter()
    routed = transpile(
        circuit,
        coupling_map=coupling_map,
        routing_method="sabre",
        layout_method="sabre",
        basis_gates=list(config["routing"]["basis_gates"]),
        optimization_level=int(config["routing"]["optimization_level"]),
        seed_transpiler=int(config["routing"]["seed_transpiler"]),
    )
    elapsed = time.perf_counter() - t0
    ops = routed.count_ops()
    source_swaps = int(circuit.count_ops().get("swap", 0))
    inserted_swaps = int(ops.get("swap", 0)) - source_swaps
    if inserted_swaps < 0:
        raise RuntimeError(
            f"Default SABRE output SWAP count ({ops.get('swap', 0)}) is below source SWAP count ({source_swaps})."
        )
    result = {
        "swap_count": inserted_swaps,
        "total_swap_operations": int(ops.get("swap", 0)),
        "source_swap_count": source_swaps,
        "cx_count": int(ops.get("cx", 0)),
        "two_qubit_gate_count": int(sum(1 for inst in routed.data if len(inst.qubits) == 2)),
        "depth": int(routed.depth() or 0),
        "total_gates": int(sum(ops.values())),
        "compile_seconds": float(elapsed),
        "layout_scoring_seconds": 0.0,
        "end_to_end_seconds": float(elapsed),
        "layout": _initial_layout_repr(routed, circuit),
    }
    return result


def _row_for_method(circuit: str, method: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "circuit": circuit,
        "method": method,
        "layout": json.dumps(result["layout"], separators=(",", ":")) if isinstance(result.get("layout"), list) else str(result.get("layout", "unknown")),
        "swap_count": result.get("swap_count"),
        "depth": result.get("depth"),
        "total_gates": result.get("total_gates"),
        "cx_count": result.get("cx_count"),
        "two_qubit_gate_count": result.get("two_qubit_gate_count"),
        "compile_seconds": result.get("compile_seconds"),
        "layout_scoring_seconds": result.get("layout_scoring_seconds", 0.0),
        "end_to_end_seconds": result.get("end_to_end_seconds"),
    }


def evaluate_model(config_path: str | Path | None = None, data_path: str | Path | None = None,
                   batch_size: int = 256, require_exhaustive: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    root = PROJECT_ROOT
    data_path = Path(data_path) if data_path else root / config["dataset"]["output_csv"]
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    manifest_path = root / config["dataset"]["manifest"]
    validate_dataset(data_path, manifest_path, strict=require_exhaustive)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_exhaustive and not manifest.get("exhaustive", False):
        raise RuntimeError("Evaluation requires an exhaustive dataset manifest. Regenerate without --limit-per-circuit.")
    df = pd.read_csv(data_path)
    split_path = root / "results" / "split_manifest.json"
    checkpoint_path = root / "models" / "best_gnn_scorer.pt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}; train the model first.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing {checkpoint_path}; train the model first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split["test_ids"])
    df = df.copy()
    df["layout_id"] = [layout_id(str(r.circuit), str(r.layout)) for r in df.itertuples(index=False)]
    test_df = df[df["layout_id"].isin(test_ids)].copy()
    if len(test_df) != len(test_ids):
        raise ValueError("Test split identities do not match dataset rows; check split manifest and CSV.")
    if set(test_df["layout_id"]) & set(split["train_ids"]):
        raise AssertionError("Test/train data leakage detected.")

    backend = load_fake_jakarta()
    circuits = load_benchmarks(config)
    circuit_graphs = {name: build_circuit_graph(circuit) for name, circuit in circuits.items()}
    hardware_graph, distances = build_hardware_graph(backend.coupling_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actual_dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if checkpoint.get("dataset_sha256") != actual_dataset_hash:
        raise RuntimeError("Checkpoint was trained on a different dataset. Regenerate splits and retrain the model.")
    if checkpoint.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Experiment configuration differs from the training configuration. Retrain the model.")
    model_cfg = checkpoint["model_config"]
    model = MappingAwareLayoutScorer(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    # Final regression/ranking metrics are computed only on the untouched test split.
    test_prediction_by_id: dict[str, float] = {}
    for circuit_name, group in test_df.groupby("circuit", sort=False):
        group = group.sort_values("layout_id")
        layouts = [tuple(parse_layout(str(v))) for v in group["layout"].tolist()]
        preds = _model_predict(model, circuit_graphs[str(circuit_name)], hardware_graph, distances,
                               layouts, device, batch_size, target_mean, target_std)
        for identity, pred in zip(group["layout_id"], preds):
            test_prediction_by_id[str(identity)] = float(pred)
    test_pred = np.asarray([test_prediction_by_id[str(x)] for x in test_df["layout_id"]], dtype=np.float64)
    test_actual = test_df["swap_count"].to_numpy(dtype=np.float64)
    test_metrics = regression_metrics(test_actual, test_pred)
    train_ids = set(split["train_ids"])
    train_df = df[df["layout_id"].isin(train_ids)]
    training_circuit_means = train_df.groupby("circuit")["swap_count"].mean().to_dict()
    constant_test_pred = np.asarray([float(training_circuit_means[str(name)]) for name in test_df["circuit"]], dtype=np.float64)
    constant_baseline_metrics = regression_metrics(test_actual, constant_test_pred)
    per_circuit_test_metrics: dict[str, Any] = {}
    spearman_values: list[float] = []
    top_overlap_values: list[float] = []
    for circuit_name, group in test_df.groupby("circuit", sort=True):
        actual_values = group["swap_count"].to_numpy(dtype=np.float64)
        predicted_values = np.asarray([test_prediction_by_id[str(x)] for x in group["layout_id"]], dtype=np.float64)
        circuit_regression = regression_metrics(actual_values, predicted_values)
        circuit_ranking = ranking_metrics(actual_values, predicted_values)
        per_circuit_test_metrics[str(circuit_name)] = {
            "regression": circuit_regression,
            "ranking": circuit_ranking,
        }
        if circuit_ranking["spearman"] is not None:
            spearman_values.append(float(circuit_ranking["spearman"]))
        top_overlap_values.append(float(circuit_ranking["top_5_percent_overlap"]))
    # Ranking is evaluated within circuit, because layout rankings across
    # different circuit families do not have a common operational meaning.
    test_ranking = {
        "spearman": float(np.mean(spearman_values)) if spearman_values else None,
        "top_5_percent_overlap": float(np.mean(top_overlap_values)),
        "aggregation": "macro average over circuits",
        "per_circuit": {k: v["ranking"] for k, v in per_circuit_test_metrics.items()},
    }

    result_dir = root / "results"
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "circuit": test_df["circuit"].to_numpy(),
        "layout": test_df["layout"].to_numpy(),
        "actual_swap_count": test_actual,
        "predicted_swap_count": test_pred,
    }).to_csv(result_dir / "test_predictions.csv", index=False)

    # Generate predicted scores for every valid mapping and compare with recorded actual labels.
    rows: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    by_circuit_results: dict[str, Any] = {}
    for benchmark_spec in config["benchmarks"]:
        name = benchmark_spec["label"]
        circuit = circuits[name]
        total_layouts = number_of_layouts(circuit.num_qubits, backend.num_qubits)
        enumeration_start = time.perf_counter()
        candidate_layouts = list(all_layouts(circuit.num_qubits, backend.num_qubits))
        enumeration_seconds = time.perf_counter() - enumeration_start
        if len(candidate_layouts) != total_layouts:
            raise AssertionError("Layout enumeration count mismatch.")
        score_start = time.perf_counter()
        predictions = _model_predict(model, circuit_graphs[name], hardware_graph, distances,
                                     candidate_layouts, device, batch_size, target_mean, target_std)
        scoring_seconds = time.perf_counter() - score_start
        sorted_indices = np.argsort(predictions, kind="stable")
        candidate_strings = [json.dumps(list(layout), separators=(",", ":")) for layout in candidate_layouts]
        actual_lookup = {
            str(row.layout): row._asdict()
            for row in df[df["circuit"] == name].itertuples(index=False)
        }
        for idx, (layout, pred) in enumerate(zip(candidate_layouts, predictions)):
            actual_row = actual_lookup.get(candidate_strings[idx])
            rows.append({
                "circuit": name,
                "layout": candidate_strings[idx],
                "predicted_swap_count": float(pred),
                "actual_swap_count": int(actual_row["swap_count"]) if actual_row else None,
                "prediction_rank": int(np.where(sorted_indices == idx)[0][0]) + 1,
            })
        best_idx = int(sorted_indices[0])
        best_layout = list(candidate_layouts[best_idx])
        best_pred = float(predictions[best_idx])

        gnn_result = _fixed_layout_metrics(circuit, backend.coupling_map, best_layout, config)
        gnn_result["layout_scoring_seconds"] = float(scoring_seconds)
        gnn_result["layout_enumeration_seconds"] = float(enumeration_seconds)
        gnn_result["end_to_end_seconds"] = float(enumeration_seconds + scoring_seconds + gnn_result["compile_seconds"])
        gnn_result["predicted_swap_count"] = best_pred
        default_result = _default_sabre(circuit, backend.coupling_map, config)
        fixed_layout = list(range(circuit.num_qubits))
        fixed_result = _fixed_layout_metrics(circuit, backend.coupling_map, fixed_layout, config)

        group = df[df["circuit"] == name].copy()
        if len(group) != total_layouts:
            if require_exhaustive:
                raise RuntimeError(f"{name}: expected {total_layouts} labels, got {len(group)}.")
        # Sample from the already measured fixed-layout labels (no cherry-picking).
        random_seed = int(config["seed"]) + len(name)
        random_sample = group.sample(n=min(int(config["evaluation"]["random_layout_samples"]), len(group)),
                                     random_state=random_seed)
        random_result = {
            "swap_count": float(random_sample["swap_count"].mean()),
            "swap_std": float(random_sample["swap_count"].std(ddof=0)),
            "swap_min": int(random_sample["swap_count"].min()),
            "swap_max": int(random_sample["swap_count"].max()),
            "depth": float(random_sample["depth"].mean()),
            "total_gates": float(random_sample["total_gates"].mean()),
            "cx_count": float(random_sample["cx_count"].mean()),
            "two_qubit_gate_count": float(random_sample["two_qubit_gate_count"].mean()),
            "compile_seconds": float(random_sample["routing_seconds"].mean()),
            "layout_scoring_seconds": 0.0,
            "end_to_end_seconds": float(random_sample["routing_seconds"].mean()),
            "layout": "random sample summary",
            "sample_count": int(len(random_sample)),
        }
        oracle_row = group.sort_values(["swap_count", "routing_seconds"], ascending=[True, True]).iloc[0]
        oracle_result = {
            "swap_count": int(oracle_row["swap_count"]),
            "depth": int(oracle_row["depth"]),
            "total_gates": int(oracle_row["total_gates"]),
            "cx_count": int(oracle_row["cx_count"]),
            "two_qubit_gate_count": int(oracle_row["two_qubit_gate_count"]),
            "compile_seconds": float(group["routing_seconds"].sum()),
            "single_candidate_route_seconds": float(oracle_row["routing_seconds"]),
            "layout_scoring_seconds": 0.0,
            "end_to_end_seconds": float(group["routing_seconds"].sum()),
            "layout": parse_layout(str(oracle_row["layout"])),
        }
        # Cross-check selected-layout score against the independently generated true label.
        matching_label = actual_lookup.get(candidate_strings[best_idx])
        selected_label_swap = int(matching_label["swap_count"]) if matching_label else None
        if selected_label_swap is not None and int(gnn_result["swap_count"]) != selected_label_swap:
            LOGGER.warning("%s selected mapping reroute mismatch: dataset=%d evaluation=%d", name,
                           selected_label_swap, gnn_result["swap_count"])

        oracle_gap = int(gnn_result["swap_count"]) - int(oracle_result["swap_count"])
        reductions = {
            "vs_default_sabre_pct": percentage_reduction(default_result["swap_count"], gnn_result["swap_count"]),
            "vs_fixed_layout_pct": percentage_reduction(fixed_result["swap_count"], gnn_result["swap_count"]),
        }
        by_circuit_results[name] = {
            "num_layouts_scored": int(len(candidate_layouts)),
            "layout_enumeration_seconds": float(enumeration_seconds),
            "gnn_selected_layout": best_layout,
            "gnn_selected_predicted_swaps": best_pred,
            "gnn_selected_actual": gnn_result,
            "default_sabre": default_result,
            "fixed_layout": fixed_result,
            "random_layout_sample": random_result,
            "exhaustive_oracle": oracle_result,
            "oracle_gap_swaps": oracle_gap,
            "swap_reduction_percent": reductions,
            "selected_label_swap_count": selected_label_swap,
        }
        for method, result in (
            ("Default SABRE", default_result),
            ("GNN-selected + fixed-layout SABRE", gnn_result),
            ("Fixed initial layout + SABRE", fixed_result),
            ("Random layout sample mean", random_result),
            ("Exhaustive oracle (not deployable)", oracle_result),
        ):
            comparison.append(_row_for_method(name, method, result))

    all_scores = pd.DataFrame(rows)
    all_scores.to_csv(result_dir / "layout_scores.csv", index=False)
    pd.DataFrame(comparison).to_csv(result_dir / "comparison_table.csv", index=False)

    # Plot true vs predicted for held-out test set.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_actual, test_pred, alpha=0.55)
    upper = max(float(np.max(test_actual)), float(np.max(test_pred)), 1.0)
    ax.plot([0, upper], [0, upper], linestyle="--")
    ax.set_xlabel("Actual SWAP count")
    ax.set_ylabel("Predicted SWAP count")
    ax.set_title("Held-out layout prediction")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    # Plot measured SWAPs for the three primary methods.
    plot_rows = []
    for name, r in by_circuit_results.items():
        plot_rows.append((name, "Default SABRE", r["default_sabre"]["swap_count"]))
        plot_rows.append((name, "GNN-selected", r["gnn_selected_actual"]["swap_count"]))
        plot_rows.append((name, "Fixed layout", r["fixed_layout"]["swap_count"]))
        plot_rows.append((name, "Random mean", r["random_layout_sample"]["swap_count"]))
        plot_rows.append((name, "Oracle", r["exhaustive_oracle"]["swap_count"]))
    plot_df = pd.DataFrame(plot_rows, columns=["circuit", "method", "swap_count"])
    pivot = plot_df.pivot(index="circuit", columns="method", values="swap_count")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("SWAP count (random shows sample mean)")
    ax.set_title("Routing SWAP count by circuit and method")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "routing_comparison.png", dpi=160)
    plt.close()

    results = {
        "status": "completed",
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "test_split_counts": {str(k): int(v) for k, v in test_df.groupby("circuit").size().items()},
        "test_regression_metrics": test_metrics,
        "test_regression_baseline_per_circuit_train_mean": constant_baseline_metrics,
        "train_set_circuit_mean_labels": {str(k): float(v) for k, v in training_circuit_means.items()},
        "test_ranking_metrics": test_ranking,
        "test_metrics_by_circuit": per_circuit_test_metrics,
        "dataset_sha256": actual_dataset_hash,
        "config_sha256": config_sha256(config),
        "test_set_only": True,
        "exhaustive_layout_scoring": True,
        "layouts_scored_per_circuit": {name: item["num_layouts_scored"] for name, item in by_circuit_results.items()},
        "by_circuit": by_circuit_results,
        "caveats": [
            "Default SABRE chooses its own layout; GNN-selected routing fixes the initial layout.",
            "The exhaustive oracle is a reference requiring all layouts to be routed; it is not a deployable baseline.",
            "Labels predict fixed-layout SABRE SWAP cost and do not by themselves prove fidelity or wall-clock speedup.",
            "Exact state fidelity and noisy fidelity were not computed in this Phase 2 routing experiment.",
        ],
    }
    results_path = result_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction_metrics = {
        "test_regression": test_metrics,
        "test_regression_baseline_per_circuit_train_mean": constant_baseline_metrics,
        "test_ranking": test_ranking,
        "per_circuit": per_circuit_test_metrics,
        "test_count": int(len(test_df)),
        "validation_used_for_checkpoint_selection": True,
        "test_used_for_checkpoint_selection": False,
    }
    (result_dir / "prediction_metrics.json").write_text(json.dumps(prediction_metrics, indent=2), encoding="utf-8")
    _generate_report(results, pd.DataFrame(comparison), prediction_metrics, config)
    LOGGER.info("Evaluation complete. Results: %s", results_path)
    return results


def _generate_report(results: dict[str, Any], comparison: pd.DataFrame,
                     prediction_metrics: dict[str, Any], config: dict[str, Any]) -> None:
    root = PROJECT_ROOT
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Report: Mapping-Aware GNN for Qubit Allocation",
        "",
        "## Objective",
        "Predict fixed-layout SABRE routing SWAP cost for an initial logical-to-physical qubit assignment, then compare GNN-selected layouts with conventional baselines.",
        "",
        "## Configuration",
        f"- Hardware: {config['hardware']['backend']} ({config['hardware']['expected_physical_qubits']} physical qubits).",
        f"- Benchmarks: {', '.join(b['label'] for b in config['benchmarks'])}.",
        "- Logical width: five qubits per primary circuit; the hardware has seven physical qubits.",
        "- Model: two GCN encoders, mapping-aware assignment/pair embeddings, and an MLP regressor.",
        "- Checkpoint selected using validation MSE only; final prediction metrics use the untouched test split.",
        "",
        "## Held-out Test Prediction Metrics",
        f"- MAE: {prediction_metrics['test_regression']['mae']:.4f} SWAPs.",
        f"- RMSE: {prediction_metrics['test_regression']['rmse']:.4f} SWAPs.",
        f"- MSE: {prediction_metrics['test_regression']['mse']:.4f} SWAPs².",
        f"- Spearman rank correlation: {prediction_metrics['test_ranking']['spearman']!s}.",
        f"- Top-5% layout overlap: {prediction_metrics['test_ranking']['top_5_percent_overlap']:.4f}.",
        f"- Per-circuit training-mean baseline MAE: {prediction_metrics['test_regression_baseline_per_circuit_train_mean']['mae']:.4f} SWAPs; RMSE: {prediction_metrics['test_regression_baseline_per_circuit_train_mean']['rmse']:.4f}.",
        "",
        "## Routing Comparison",
        comparison.to_markdown(index=False),
        "",
        "## Per-Circuit Findings",
    ]
    for name, item in results["by_circuit"].items():
        baseline = item["default_sabre"]["swap_count"]
        gnn = item["gnn_selected_actual"]["swap_count"]
        reduction = item["swap_reduction_percent"]["vs_default_sabre_pct"]
        oracle = item["exhaustive_oracle"]["swap_count"]
        reduction_text = "not applicable (baseline has zero SWAPs)" if reduction is None else f"{reduction:.2f}%"
        lines += [
            f"### {name}",
            f"- Default SABRE SWAPs: {baseline}.",
            f"- GNN-selected layout SWAPs: {gnn} (reduction vs default SABRE: {reduction_text}).",
            f"- Exhaustive fixed-layout oracle SWAPs: {oracle}; GNN-to-oracle gap: {item['oracle_gap_swaps']}.",
            f"- Layout enumeration: {item['layout_enumeration_seconds']:.4f}s; GNN scoring: {item['gnn_selected_actual']['layout_scoring_seconds']:.4f}s; fixed-layout route: {item['gnn_selected_actual']['compile_seconds']:.4f}s; total: {item['gnn_selected_actual']['end_to_end_seconds']:.4f}s.",
            "",
        ]
    lines += [
        "## Interpretation and Limitations",
        "The GNN is a learned predictor and layout-selection policy; SABRE still performs the actual routing. The exhaustive oracle is not deployable without paying for every candidate route. Comparison metrics are tied to this hardware topology and benchmark suite. Routing SWAP count, circuit depth, gate count, compilation time, noisy fidelity and exact state fidelity are distinct metrics. Fidelity was not computed in this Phase 2 experiment.",
        "",
        "## Reproduction",
        "```bash",
        "python scripts/check_environment.py",
        "python scripts/generate_dataset.py",
        "python scripts/validate_dataset.py",
        "python scripts/train_gnn.py",
        "python scripts/evaluate_gnn.py",
        "pytest -q",
        "```",
        "",
    ]
    (report_dir / "phase2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-partial", action="store_true", help="Development mode only; does not satisfy final exhaustive acceptance.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    evaluate_model(args.config, args.data, args.batch_size, require_exhaustive=not args.allow_partial)


if __name__ == "__main__":
    main()
