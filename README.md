# GNN-Based Qubit Allocation — Phase 2

This repository implements a reproducible experiment to predict the SWAP cost of an initial logical-to-physical qubit mapping on IBM's seven-qubit `FakeJakartaV2` topology. A mapping-aware, two-encoder GCN scores candidate layouts; Qiskit SABRE still performs the actual routing.

## Important environment requirement

The requested version is **PyTorch 2.14.0**. This project pins `torch==2.14.0` and its environment check explicitly fails when another version is installed. Use the official PyTorch installation page for the correct wheel/index for your operating system and accelerator: <https://pytorch.org/get-started/locally/>. For a CPU-only Linux environment, the typical command is:

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

If that wheel is not available for your platform, consult the official installation selector and do not silently substitute another version. PyTorch Geometric and Qiskit versions are constrained in `pyproject.toml` / `requirements.txt`.

## Setup (Ubuntu / WSL)

Use Python 3.11–3.13. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python scripts/check_environment.py
```

The environment check records dependency versions, verifies the requested PyTorch version, initializes `FakeJakartaV2`, and attempts to load the configured MQT Bench circuits. A nonzero exit means the environment is not ready.

## Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python scripts/check_environment.py
```

For a GPU build, install the corresponding official PyTorch 2.14.0 wheel before `pip install -e .`, and check `torch.cuda.is_available()` in `environment_report.json`.

## Full pipeline

Run from the repository root with the virtual environment activated:

```bash
python scripts/check_environment.py
python scripts/generate_dataset.py
python scripts/validate_dataset.py
python scripts/train_gnn.py
python scripts/evaluate_gnn.py
pytest -q
```

Or run the workflow with fail-fast orchestration:

```bash
python scripts/run_all.py
```

The default dataset generator is exhaustive. For five logical qubits on seven physical qubits, there are `7P5 = 2,520` one-to-one layouts per circuit. Across three circuits, the full dataset contains 7,560 circuit-layout samples if every routing operation succeeds.

For a quick code smoke test only, a small dataset can be generated with:

```bash
python scripts/generate_dataset.py --limit-per-circuit 20 --force
```

**A limited dataset is not a final experiment.** It does not support the exhaustive oracle comparison and the normal evaluator refuses it. Delete/regenerate it with `python scripts/generate_dataset.py --force` to produce the full set before final evaluation.

## Model design

- Circuit graph: one node per logical qubit. Node features summarize two-qubit interaction count, distinct partners, and single-qubit gate count. Undirected edge weights encode interaction multiplicity.
- Hardware graph: one node per physical qubit, with edges taken from the backend coupling map and degree-based node features.
- GNN: separate two-layer GCN encoders preserve node-level embeddings. A mapping-aware assignment network compares each logical node with its selected physical node; an interaction scorer also uses the mapped physical embeddings and shortest-path distances for each interacting logical pair.
- Target: actual inserted SWAP count measured after routing the circuit with a fixed initial mapping and SABRE.
- Model selection: split layouts per circuit into training/validation/test. Checkpoint selection and early stopping use validation data only. Final prediction metrics use test data after training.

## Artifacts

- `data/phase2_data.csv`: routing-cost samples.
- `data/dataset_manifest.json`: circuit, topology, dependency, routing and failure provenance.
- `models/best_gnn_scorer.pt`: selected checkpoint.
- `results/training_metrics.json`: training and validation curves/statistics (test is not used here).
- `results/split_manifest.json`: fixed train/validation/test identities.
- `results/test_predictions.csv`, `results/prediction_metrics.json`: held-out test predictions and metrics.
- `results/layout_scores.csv`: model scores for all candidate layouts.
- `results/comparison_table.csv`, `results/evaluation_results.json`: routing comparisons.
- `results/figures/`: learning curve, prediction scatter and routing comparison.
- `reports/phase2_report.md`: generated report based on measured outputs.

## Comparison methods

The evaluation compares default SABRE layout and routing, a fixed initial layout followed by SABRE, a random-layout sample summary, and an exhaustive fixed-layout oracle. The oracle routes every candidate and is a reference, not a practically cheap algorithm. End-to-end time for GNN includes scoring all candidate layouts plus routing the selected layout.

SWAP count, depth, total gates, CX gates and time are reported separately. This experiment does not compute exact state fidelity or noisy estimated fidelity. A smaller SWAP count does not automatically imply lower depth, faster hardware execution, or higher fidelity.

## Notes and troubleshooting

- Circuit or benchmark loading errors are fatal by design; the code does not silently replace MQT Bench circuits with hand-written substitutes.
- The data generator saves progress after each circuit and records failed routes in the manifest. The final strict validator fails if the exhaustive data is incomplete.
- If the trained model underperforms SABRE, that is a valid result. Review the prediction/ranking scores, random baseline distribution, and oracle gap rather than changing the test set.
- Run `python scripts/validate_dataset.py` before training. Run `python scripts/evaluate_gnn.py` only after a complete dataset and checkpoint exist.
