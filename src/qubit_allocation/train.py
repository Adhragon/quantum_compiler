"""Train the mapping-aware GNN using split-safe model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from .circuits import load_benchmarks
from .config import PROJECT_ROOT, config_sha256, load_config
from .dataset import validate_dataset
from .graph_features import GraphFeatures, build_circuit_graph, build_hardware_graph
from .hardware import load_fake_jakarta
from .metrics import regression_metrics
from .model import MappingAwareLayoutScorer
from .split_utils import layout_id, make_split, parse_layout

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _graph_tensors(graph: GraphFeatures, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(graph.x, dtype=torch.float32, device=device),
        torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(graph.edge_weight, dtype=torch.float32, device=device),
    )


def _to_model_batch(model: MappingAwareLayoutScorer, circ: GraphFeatures, hw: GraphFeatures,
                    distance: np.ndarray, layouts: list[list[int]], device: torch.device):
    cx, ce, cw = _graph_tensors(circ, device)
    hx, he, hwgt = _graph_tensors(hw, device)
    pair_idx = torch.as_tensor(circ.pair_index, dtype=torch.long, device=device)
    pair_weight = torch.as_tensor(circ.pair_weight, dtype=torch.float32, device=device)
    dist = torch.as_tensor(distance, dtype=torch.float32, device=device)
    layout_tensor = torch.as_tensor(layouts, dtype=torch.long, device=device)
    return model(cx, ce, cw, hx, he, hwgt, layout_tensor, pair_idx, pair_weight, dist)


def _ids_for_rows(df: pd.DataFrame) -> list[str]:
    return [layout_id(str(r.circuit), str(r.layout)) for r in df.itertuples(index=False)]


def _predict_rows(model: MappingAwareLayoutScorer, df: pd.DataFrame,
                  circuits_graphs: dict[str, GraphFeatures], hardware_graph: GraphFeatures,
                  distances: np.ndarray, device: torch.device, batch_size: int,
                  target_mean: float, target_std: float) -> np.ndarray:
    predictions_by_id: dict[str, float] = {}
    for circuit_name, group in df.groupby("circuit", sort=False):
        rows = list(group.itertuples(index=False))
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            layouts = [parse_layout(str(row.layout)) for row in batch]
            with torch.no_grad():
                normalized = _to_model_batch(model, circuits_graphs[str(circuit_name)], hardware_graph,
                                             distances, layouts, device)
                predicted = np.maximum(normalized.detach().cpu().numpy() * target_std + target_mean, 0.0)
            for row, value in zip(batch, predicted):
                predictions_by_id[layout_id(str(row.circuit), str(row.layout))] = float(value)
    return np.asarray([predictions_by_id[i] for i in _ids_for_rows(df)], dtype=np.float64)


def train_model(config_path: str | Path | None = None, data_path: str | Path | None = None,
                epochs: int | None = None, batch_size: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    project_root = PROJECT_ROOT
    data_path = Path(data_path) if data_path else project_root / config["dataset"]["output_csv"]
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    manifest_path = project_root / config["dataset"]["manifest"]
    # Allow a partial dataset for tests/dev runs, but the final evaluation refuses
    # a non-exhaustive manifest. The strict full-data validator can be run separately.
    validate_dataset(data_path, manifest_path, strict=False)
    df = pd.read_csv(data_path)
    if df["circuit"].nunique() != len(config["benchmarks"]):
        raise ValueError("Training dataset must contain all configured benchmark circuits.")

    ratios = config["split"]
    splits = make_split(df, int(config["seed"]), float(ratios["train"]), float(ratios["validation"]))
    split_path = project_root / "results" / "split_manifest.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    current_config_sha256 = config_sha256(config)
    split_payload = {
        "seed": int(config["seed"]), "ratios": ratios,
        "dataset_sha256": dataset_sha256,
        "config_sha256": current_config_sha256,
        "train_ids": splits["train"], "validation_ids": splits["validation"], "test_ids": splits["test"],
        "counts": {k: len(v) for k, v in splits.items()},
    }
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    all_ids = _ids_for_rows(df)
    ids_column = pd.Series(all_ids, index=df.index)
    split_frames = {name: df[ids_column.isin(set(splits[name]))].copy() for name in splits}
    if sum(len(d) for d in split_frames.values()) != len(df):
        raise AssertionError("A data row was lost while applying the split manifest.")
    y_train = split_frames["train"]["swap_count"].to_numpy(dtype=np.float32)
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if target_std < 1e-8:
        target_std = 1.0

    backend = load_fake_jakarta()
    circuits = load_benchmarks(config)
    circuit_graphs = {name: build_circuit_graph(qc) for name, qc in circuits.items()}
    hardware_graph, distances = build_hardware_graph(backend.coupling_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = MappingAwareLayoutScorer(
        circuit_in_dim=3,
        hardware_in_dim=1,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_gcn_layers=int(model_cfg["gcn_layers"]),
        mlp_hidden_dim=int(model_cfg["mlp_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"]))
    criterion = nn.MSELoss()
    epoch_limit = int(epochs if epochs is not None else model_cfg["epochs"])
    batch_size = int(batch_size if batch_size is not None else model_cfg["batch_size"])
    patience = int(model_cfg["patience"])
    if epoch_limit < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    models_dir = project_root / "models"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    for directory in (models_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "best_gnn_scorer.pt"
    history: list[dict[str, float | int]] = []
    best_val_mse = math.inf
    best_epoch = 0
    patience_used = 0
    training_started = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        # Keep each minibatch within a circuit so it shares a graph pair, while
        # shuffling both rows and batches with the fixed global seed.
        tasks: list[tuple[str, list[Any]]] = []
        for circuit_name, group in split_frames["train"].groupby("circuit", sort=True):
            rows = list(group.itertuples(index=False))
            random.shuffle(rows)
            for start in range(0, len(rows), batch_size):
                tasks.append((str(circuit_name), rows[start:start + batch_size]))
        random.shuffle(tasks)
        for circuit_name, rows in tasks:
            layouts = [parse_layout(str(row.layout)) for row in rows]
            targets = np.asarray([float(row.swap_count) for row in rows], dtype=np.float32)
            y = torch.as_tensor((targets - target_mean) / target_std, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _to_model_batch(model, circuit_graphs[circuit_name], hardware_graph,
                                          distances, layouts, device)
            loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        pred_train = _predict_rows(model, split_frames["train"], circuit_graphs, hardware_graph, distances,
                                   device, batch_size, target_mean, target_std)
        pred_val = _predict_rows(model, split_frames["validation"], circuit_graphs, hardware_graph, distances,
                                 device, batch_size, target_mean, target_std)
        train_metrics = regression_metrics(split_frames["train"]["swap_count"].to_numpy(), pred_train)
        val_metrics = regression_metrics(split_frames["validation"]["swap_count"].to_numpy(), pred_val)
        epoch_row = {
            "epoch": epoch,
            "train_loss_normalized_mse": float(np.mean(train_losses)) if train_losses else 0.0,
            "train_mse": float(train_metrics["mse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_rmse": float(train_metrics["rmse"]),
            "validation_mse": float(val_metrics["mse"]),
            "validation_mae": float(val_metrics["mae"]),
            "validation_rmse": float(val_metrics["rmse"]),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
        }
        history.append(epoch_row)
        LOGGER.info("epoch %03d/%03d train MAE=%.3f val MAE=%.3f val MSE=%.3f",
                    epoch, epoch_limit, train_metrics["mae"], val_metrics["mae"], val_metrics["mse"])

        if float(val_metrics["mse"]) < best_val_mse - 1e-8:
            best_val_mse = float(val_metrics["mse"])
            best_epoch = epoch
            patience_used = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "circuit_in_dim": 3, "hardware_in_dim": 1,
                    "hidden_dim": int(model_cfg["hidden_dim"]),
                    "num_gcn_layers": int(model_cfg["gcn_layers"]),
                    "mlp_hidden_dim": int(model_cfg["mlp_hidden_dim"]),
                    "dropout": float(model_cfg["dropout"]),
                },
                "target_mean": target_mean,
                "target_std": target_std,
                "best_epoch": best_epoch,
                "best_validation_mse": best_val_mse,
                "seed": int(config["seed"]),
                "dataset_sha256": dataset_sha256,
                "config_sha256": current_config_sha256,
                "architecture": "MappingAwareLayoutScorer",
            }, checkpoint_path)
        else:
            patience_used += 1
        if patience_used >= patience:
            LOGGER.info("Early stopping at epoch %d; best validation epoch %d", epoch, best_epoch)
            break

    total_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    final_train_prediction = _predict_rows(model, split_frames["train"], circuit_graphs, hardware_graph, distances,
                                           device, batch_size, target_mean, target_std)
    final_val_prediction = _predict_rows(model, split_frames["validation"], circuit_graphs, hardware_graph, distances,
                                         device, batch_size, target_mean, target_std)
    metrics = {
        "status": "completed",
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset_rows": len(df),
        "split_counts": {k: len(v) for k, v in split_frames.items()},
        "dataset_sha256": dataset_sha256,
        "config_sha256": current_config_sha256,
        "target_standardization": {"mean_from_train": target_mean, "std_from_train": target_std},
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": round(total_seconds, 3),
        "final_train": regression_metrics(split_frames["train"]["swap_count"].to_numpy(), final_train_prediction),
        "final_validation": regression_metrics(split_frames["validation"]["swap_count"].to_numpy(), final_val_prediction),
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "split_manifest": str(split_path.relative_to(project_root)),
        "note": "The test split is intentionally not evaluated during training or checkpoint selection.",
    }
    metrics_path = results_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Plot train/validation MSE and MAE from actual epoch measurements.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["epoch"] for x in history], [x["train_mse"] for x in history], label="Train MSE")
    ax.plot([x["epoch"] for x in history], [x["validation_mse"] for x in history], label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (SWAP count²)")
    ax.set_title("GNN training and validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    # The test plot is generated only by evaluate.py after final model selection.
    LOGGER.info("Training complete: best validation MSE %.4f at epoch %d", best_val_mse, best_epoch)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    train_model(args.config, args.data, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
