# Phase 2 Report: Mapping-Aware GNN for Qubit Allocation

## Objective
Predict fixed-layout SABRE routing SWAP cost for an initial logical-to-physical qubit assignment, then compare GNN-selected layouts with conventional baselines.

## Configuration
- Hardware: FakeJakartaV2 (7 physical qubits).
- Benchmarks: dj_5q, ghz_5q, graphstate_5q, qft_5q, wstate_5q.
- Logical width: five qubits per primary circuit; the hardware has seven physical qubits.
- Model: two GCN encoders, mapping-aware assignment/pair embeddings, and an MLP regressor.
- Checkpoint selected using validation MSE only; final prediction metrics use the untouched test split.

## Held-out Test Prediction Metrics
- MAE: 0.8681 SWAPs.
- RMSE: 1.1001 SWAPs.
- MSE: 1.2103 SWAPs².
- Spearman rank correlation: 0.6812129727293057.
- Top-5% layout overlap: 0.4632.
- Per-circuit training-mean baseline MAE: 1.2592 SWAPs; RMSE: 1.5902.

## Routing Comparison
| circuit       | method                             | layout                |   swap_count |   depth |   total_gates |   cx_count |   two_qubit_gate_count |   compile_seconds |   layout_scoring_seconds |   end_to_end_seconds |
|:--------------|:-----------------------------------|:----------------------|-------------:|--------:|--------------:|-----------:|-----------------------:|------------------:|-------------------------:|---------------------:|
| dj_5q         | Default SABRE                      | [1,4,5,6,3]           |         1    |   12    |         37    |          4 |                   5    |        0.0067399  |                  0       |           0.0067399  |
| dj_5q         | GNN-selected + fixed-layout SABRE  | [0,2,3,5,1]           |         1    |   15    |         37    |          4 |                   5    |        0.0057756  |                  2.1297  |           2.13655    |
| dj_5q         | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         5    |   18    |         41    |          4 |                   9    |        0.0051257  |                  0       |           0.0051257  |
| dj_5q         | Random layout sample mean          | random sample summary |         3.97 |   18.38 |         39.97 |          4 |                   7.97 |        0.00599735 |                  0       |           0.00599735 |
| dj_5q         | Exhaustive oracle (not deployable) | [0,3,2,5,1]           |         1    |   15    |         37    |          4 |                   5    |       15.2444     |                  0       |          15.2444     |
| ghz_5q        | Default SABRE                      | [1,0,3,5,6]           |         1    |    7    |          8    |          4 |                   5    |        0.0058613  |                  0       |           0.0058613  |
| ghz_5q        | GNN-selected + fixed-layout SABRE  | [0,1,3,5,4]           |         0    |    7    |          7    |          4 |                   4    |        0.006111   |                  1.77878 |           1.78536    |
| ghz_5q        | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         3    |   10    |         10    |          4 |                   7    |        0.0047429  |                  0       |           0.0047429  |
| ghz_5q        | Random layout sample mean          | random sample summary |         4.37 |    9.89 |         11.37 |          4 |                   8.37 |        0.00496281 |                  0       |           0.00496281 |
| ghz_5q        | Exhaustive oracle (not deployable) | [6,5,3,1,2]           |         0    |    7    |          7    |          4 |                   4    |       12.5719     |                  0       |          12.5719     |
| graphstate_5q | Default SABRE                      | [3,1,5,4,6]           |         2    |   22    |         52    |          5 |                   7    |        0.0047926  |                  0       |           0.0047926  |
| graphstate_5q | GNN-selected + fixed-layout SABRE  | [0,1,5,3,2]           |         3    |   32    |         53    |          5 |                   8    |        0.005927   |                  2.08071 |           2.08701    |
| graphstate_5q | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         4    |   31    |         54    |          5 |                   9    |        0.0058013  |                  0       |           0.0058013  |
| graphstate_5q | Random layout sample mean          | random sample summary |         5.3  |   25.97 |         55.3  |          5 |                  10.3  |        0.00695711 |                  0       |           0.00695711 |
| graphstate_5q | Exhaustive oracle (not deployable) | [5,3,0,2,1]           |         2    |   23    |         52    |          5 |                   7    |       16.5308     |                  0       |          16.5308     |
| qft_5q        | Default SABRE                      | [5,2,0,3,1]           |         4    |   46    |         71    |         20 |                  26    |        0.0072806  |                  0       |           0.0072806  |
| qft_5q        | GNN-selected + fixed-layout SABRE  | [3,0,2,1,5]           |         9    |   52    |         76    |         20 |                  31    |        0.0084883  |                  2.23936 |           2.24812    |
| qft_5q        | Fixed initial layout + SABRE       | [0,1,2,3,4]           |        11    |   57    |         78    |         20 |                  33    |        0.0049001  |                  0       |           0.0049001  |
| qft_5q        | Random layout sample mean          | random sample summary |        10.55 |   52.57 |         77.55 |         20 |                  32.55 |        0.00630243 |                  0       |           0.00630243 |
| qft_5q        | Exhaustive oracle (not deployable) | [5,1,2,3,0]           |         5    |   47    |         72    |         20 |                  27    |       15.8679     |                  0       |          15.8679     |
| wstate_5q     | Default SABRE                      | [5,3,2,1,0]           |         1    |   46    |         74    |          8 |                   9    |        0.0057055  |                  0       |           0.0057055  |
| wstate_5q     | GNN-selected + fixed-layout SABRE  | [0,1,3,5,4]           |         0    |   45    |         73    |          8 |                   8    |        0.0064375  |                  1.53554 |           1.54232    |
| wstate_5q     | Fixed initial layout + SABRE       | [0,1,2,3,4]           |         4    |   47    |         77    |          8 |                  12    |        0.0051077  |                  0       |           0.0051077  |
| wstate_5q     | Random layout sample mean          | random sample summary |         5.24 |   47.82 |         78.24 |          8 |                  13.24 |        0.00436962 |                  0       |           0.00436962 |
| wstate_5q     | Exhaustive oracle (not deployable) | [2,1,3,5,6]           |         0    |   45    |         73    |          8 |                   8    |       11.5701     |                  0       |          11.5701     |

## Per-Circuit Findings
### dj_5q
- Default SABRE SWAPs: 1.
- GNN-selected layout SWAPs: 1 (reduction vs default SABRE: 0.00%).
- Exhaustive fixed-layout oracle SWAPs: 1; GNN-to-oracle gap: 0.
- Layout enumeration: 0.0011s; GNN scoring: 2.1297s; fixed-layout route: 0.0058s; total: 2.1366s.

### ghz_5q
- Default SABRE SWAPs: 1.
- GNN-selected layout SWAPs: 0 (reduction vs default SABRE: 100.00%).
- Exhaustive fixed-layout oracle SWAPs: 0; GNN-to-oracle gap: 0.
- Layout enumeration: 0.0005s; GNN scoring: 1.7788s; fixed-layout route: 0.0061s; total: 1.7854s.

### graphstate_5q
- Default SABRE SWAPs: 2.
- GNN-selected layout SWAPs: 3 (reduction vs default SABRE: -50.00%).
- Exhaustive fixed-layout oracle SWAPs: 2; GNN-to-oracle gap: 1.
- Layout enumeration: 0.0004s; GNN scoring: 2.0807s; fixed-layout route: 0.0059s; total: 2.0870s.

### qft_5q
- Default SABRE SWAPs: 4.
- GNN-selected layout SWAPs: 9 (reduction vs default SABRE: -125.00%).
- Exhaustive fixed-layout oracle SWAPs: 5; GNN-to-oracle gap: 4.
- Layout enumeration: 0.0003s; GNN scoring: 2.2394s; fixed-layout route: 0.0085s; total: 2.2481s.

### wstate_5q
- Default SABRE SWAPs: 1.
- GNN-selected layout SWAPs: 0 (reduction vs default SABRE: 100.00%).
- Exhaustive fixed-layout oracle SWAPs: 0; GNN-to-oracle gap: 0.
- Layout enumeration: 0.0003s; GNN scoring: 1.5355s; fixed-layout route: 0.0064s; total: 1.5423s.

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
