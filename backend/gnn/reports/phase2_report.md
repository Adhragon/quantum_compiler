# Phase 2 Report: Mapping-Aware GNN for Qubit Allocation

## Objective
Predict fixed-layout SABRE routing SWAP cost for an initial logical-to-physical qubit assignment, then compare GNN-selected layouts with conventional baselines.

## Configuration
- Hardware: FakeJakartaV2 (7 physical qubits).
- Benchmarks: dj_5q, ghz_5q, graphstate_5q.
- Logical width: five qubits per primary circuit; the hardware has seven physical qubits.
- Model: two GCN encoders, mapping-aware assignment/pair embeddings, and an MLP regressor.
- Checkpoint selected using validation MSE only; final prediction metrics use the untouched test split.

## Held-out Test Prediction Metrics
- MAE: 0.6838 SWAPs.
- RMSE: 0.8184 SWAPs.
- MSE: 0.6698 SWAPs².
- Spearman rank correlation: 0.7829438382523911.
- Top-5% layout overlap: 0.4561.
- Per-circuit training-mean baseline MAE: 1.1135 SWAPs; RMSE: 1.3944.

## Routing Comparison
| circuit       | method                             | layout                |   swap_count |   depth |   total_gates |   cx_count |   two_qubit_gate_count |   compile_seconds |   layout_scoring_seconds |   end_to_end_seconds |
|:--------------|:-----------------------------------|:----------------------|-------------:|--------:|--------------:|-----------:|-----------------------:|------------------:|-------------------------:|---------------------:|
| dj_5q         | Default SABRE                      | [1,4,5,6,3]           |         1    |   12    |         37    |          4 |                   5    |        0.00582583 |                 0        |           0.00582583 |
| dj_5q         | GNN-selected + fixed-layout SABRE  | [0,2,3,5,1]           |         1    |   15    |         37    |          4 |                   5    |        0.00315717 |                 0.165438 |           0.168992   |
| dj_5q         | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         5    |   18    |         41    |          4 |                   9    |        0.00293758 |                 0        |           0.00293758 |
| dj_5q         | Random layout sample mean          | random sample summary |         3.97 |   18.38 |         39.97 |          4 |                   7.97 |        0.0030088  |                 0        |           0.0030088  |
| dj_5q         | Exhaustive oracle (not deployable) | [3,6,4,1,5]           |         1    |   14    |         37    |          4 |                   5    |        7.58172    |                 0        |           7.58172    |
| ghz_5q        | Default SABRE                      | [1,0,3,5,6]           |         1    |    7    |          8    |          4 |                   5    |        0.00329811 |                 0        |           0.00329811 |
| ghz_5q        | GNN-selected + fixed-layout SABRE  | [0,1,3,5,4]           |         0    |    7    |          7    |          4 |                   4    |        0.00238995 |                 0.173772 |           0.17643    |
| ghz_5q        | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         3    |   10    |         10    |          4 |                   7    |        0.00232784 |                 0        |           0.00232784 |
| ghz_5q        | Random layout sample mean          | random sample summary |         4.37 |    9.89 |         11.37 |          4 |                   8.37 |        0.00286953 |                 0        |           0.00286953 |
| ghz_5q        | Exhaustive oracle (not deployable) | [2,1,3,5,6]           |         0    |    7    |          7    |          4 |                   4    |        7.18699    |                 0        |           7.18699    |
| graphstate_5q | Default SABRE                      | [3,1,5,4,6]           |         2    |   22    |         52    |          5 |                   7    |        0.00347812 |                 0        |           0.00347812 |
| graphstate_5q | GNN-selected + fixed-layout SABRE  | [0,1,5,3,2]           |         3    |   32    |         53    |          5 |                   8    |        0.00299993 |                 0.184334 |           0.187428   |
| graphstate_5q | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         4    |   31    |         54    |          5 |                   9    |        0.00297978 |                 0        |           0.00297978 |
| graphstate_5q | Random layout sample mean          | random sample summary |         5.3  |   25.97 |         55.3  |          5 |                  10.3  |        0.00291606 |                 0        |           0.00291606 |
| graphstate_5q | Exhaustive oracle (not deployable) | [3,5,1,2,0]           |         2    |   22    |         52    |          5 |                   7    |        7.28777    |                 0        |           7.28777    |

## Per-Circuit Findings
### dj_5q
- Default SABRE SWAPs: 1.
- GNN-selected layout SWAPs: 1 (reduction vs default SABRE: 0.00%).
- Exhaustive fixed-layout oracle SWAPs: 1; GNN-to-oracle gap: 0.
- Layout enumeration: 0.0004s; GNN scoring: 0.1654s; fixed-layout route: 0.0032s; total: 0.1690s.

### ghz_5q
- Default SABRE SWAPs: 1.
- GNN-selected layout SWAPs: 0 (reduction vs default SABRE: 100.00%).
- Exhaustive fixed-layout oracle SWAPs: 0; GNN-to-oracle gap: 0.
- Layout enumeration: 0.0003s; GNN scoring: 0.1738s; fixed-layout route: 0.0024s; total: 0.1764s.

### graphstate_5q
- Default SABRE SWAPs: 2.
- GNN-selected layout SWAPs: 3 (reduction vs default SABRE: -50.00%).
- Exhaustive fixed-layout oracle SWAPs: 2; GNN-to-oracle gap: 1.
- Layout enumeration: 0.0001s; GNN scoring: 0.1843s; fixed-layout route: 0.0030s; total: 0.1874s.

## Interpretation and Limitations
The GNN is a learned predictor and layout-selection policy; SABRE still performs the actual routing. The exhaustive oracle is not deployable without paying for every candidate route. Comparison metrics are tied to this hardware topology and benchmark suite. Routing SWAP count, circuit depth, gate count, compilation time, noisy fidelity and exact state fidelity are distinct metrics. Fidelity was not computed in this Phase 2 experiment.

## Reproduction
```bash
python scripts/check_environment.py
python scripts/generate_dataset.py
python scripts/validate_dataset.py
python scripts/train_gnn.py
python scripts/evaluate_gnn.py
pytest -q
```
