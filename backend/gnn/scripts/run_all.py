"""Run the full Phase 2 pipeline in a safe, fail-fast order."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT
from qubit_allocation.config import load_config
from qubit_allocation.hardware import environment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Generate a small non-exhaustive dataset to test code only.")
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    env = environment_report(config)
    (ROOT / "environment_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    if not env.get("environment_valid", False):
        raise SystemExit(
            "Environment verification failed. See environment_report.json. "
            "The pipeline requires stable torch==2.14.0 and compatible Qiskit/MQT Bench dependencies."
        )

    # Import quantum/ML modules only after environment verification so that the
    # verifier can explain missing packages instead of failing during import.
    from qubit_allocation.dataset import generate_dataset, validate_dataset
    from qubit_allocation.train import train_model
    from qubit_allocation.evaluate import evaluate_model

    data_path = ROOT / config["dataset"]["output_csv"]
    if data_path.exists() and args.force:
        data_path.unlink()
    if not data_path.exists():
        generate_dataset(limit_per_circuit=20 if args.smoke else None, force=args.force)
    else:
        logging.info("Using existing dataset: %s", data_path)
    validate_dataset(strict=not args.smoke)
    train_model(epochs=args.epochs or (5 if args.smoke else None), batch_size=32 if args.smoke else None)
    if not args.smoke:
        evaluate_model(require_exhaustive=True)
    else:
        logging.warning("Smoke mode omits final exhaustive evaluation; regenerate full data before claiming Phase 2 complete.")
    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + test_env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=test_env, check=True)
    logging.info("Pipeline stages completed.")


if __name__ == "__main__":
    main()
