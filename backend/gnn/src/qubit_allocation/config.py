"""Load and validate the project configuration."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiment_config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != 2:
        raise ValueError("This project expects phase=2.")
    if config["hardware"]["expected_physical_qubits"] < 2:
        raise ValueError("At least two physical qubits are required.")
    splits = config["split"]
    if abs(sum(float(splits[k]) for k in ("train", "validation", "test")) - 1.0) > 1e-8:
        raise ValueError("Train/validation/test split proportions must sum to 1.")
    if any(float(splits[k]) <= 0 for k in ("train", "validation", "test")):
        raise ValueError("Every data split must be positive.")
    if not config["benchmarks"]:
        raise ValueError("At least one benchmark must be configured.")


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
