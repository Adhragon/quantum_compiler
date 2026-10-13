# Phase 4 — GNN + RL

Combines the Phase 2 GNN (`gnn/`, picks initial layouts) with the Phase 3 RL
router (`rl/`, inserts SWAPs) in two ways:

| | What | Code |
|---|---|---|
| **4A** (frozen-scope item) | GNN embeddings concatenated into the RL agent's observation | `gnn_rl/features.py`, `python -m rl.train --gnn-obs` |
| **4B** | GNN chooses the initial layout, RL routes from it | `gnn/gnn_layout.py`, used by `rl/evaluate.py` |

```bash
# from backend/, venv active, GNN trained (gnn/models/best_gnn_scorer.pt)
python -m rl.train --gnn-obs               # seed 42 -> gnn_rl/checkpoints/sb3_agent_gnn.zip (~13 min)
python -m rl.train --gnn-obs --seed 1
python -m rl.train --gnn-obs --seed 2
python -m gnn.evaluate_shared              # GNN layout + SABRE   -> evaluation/results/gnn_results.csv
python -m rl.evaluate                      # all agents x both layouts -> evaluation/results/rl_results.csv
```

## 4A: what the GNN adds to the observation

At every step, for the current circuit and the agent's **current** qubit placement:

- **GNN embedding (64 floats):** the scorer's mapping-aware assignment embedding, i.e. how well each logical qubit's role in the circuit matches the physical qubit it sits on, averaged over qubits.
- **Predicted SWAP cost (1 float):** the GNN's estimate of SWAPs needed to route the circuit from this placement.

Both are squashed into [−1, 1] and appended to the 34 routing features, giving 99 in total. Everything else (reward, actions, MaskablePPO, 200k steps, training data) is identical to Phase 3, so any difference comes from the GNN features.

**Why per-step, not per-circuit:** a fixed per-circuit embedding over 5 circuits only tells the agent *which* circuit it is routing. The placement-dependent embedding changes after every SWAP, so it carries routing-relevant information.

**Cost:** one GNN forward pass per step (about 5 ms on CPU), so training takes about 13 min instead of about 10.

## Results

All circuits routed correctly (state fidelity 1.0), no shortest-path fallbacks.
RL = mean over 3 seeds (42, 1, 2).

**Total SWAPs over the 5 circuits: layout × routing**

| Initial layout | SABRE routing | RL routing (Phase 3) | RL + GNN obs (4A) |
|---|---|---|---|
| SABRE's layout | 7 | 7 ± 0 | 7 ± 0 |
| GNN's layout (4B) | 10 | 11 ± 1 | **9.3 ± 0.6** |
| GNN's top-20 layouts, fewest SWAPs kept | 7 | 8 ± 0 | 7.3 ± 0.6 |

**Per circuit (SWAPs)**

| Method | dj | ghz | graphstate | qft | wstate |
|---|---|---|---|---|---|
| SABRE (layout + routing) | 1 | 0 | 2 | 4 | 0 |
| MQT QMAP | 1 | 1 | 3 | 7 | 3 |
| RL / RL+GNN obs, SABRE layout | 1 | 0 | 2 | 4 | 0 |
| GNN layout + SABRE routing | 1 | 0 | 3 | 6 | 0 |
| GNN layout + RL | 1 | 0 | 3.67 | 6.33 | 0 |
| GNN layout + RL + GNN obs | 1 | 0 | 3 | 5.33 | 0 |

**Totals over the 5 circuits (other metrics)**

| Method | Depth | Total gates | Est. fidelity (mean) | Compile time (s, total) |
|---|---|---|---|---|
| SABRE | 127 | 203 | 0.906 | 0.04 |
| RL, SABRE layout | 128.3 ± 1.2 | 202.3 ± 0.6 | 0.905 | 0.02 |
| RL + GNN obs, SABRE layout | 129 ± 0 | 203.7 ± 2.9 | 0.905 | 0.06 |
| GNN layout + SABRE | 139 | 230 | 0.890 | 8.2 |
| GNN layout + RL + GNN obs | 145.3 ± 3.2 | 222.3 ± 4.9 | 0.889 | 7.4 |

GNN-layout compile times are dominated by scoring all 2,520 candidate layouts per
circuit (about 1 s per circuit on the i3 laptop with PyTorch pinned to 1 thread;
PyTorch's default 8 threads oversubscribe the 4-thread CPU and run ~3x slower).
Full, current numbers: `evaluation/results/gnn_results.csv` and `rl_results.csv`.

## Improvement: GNN shortlist + noise-aware selection

**Why:** SWAP count is already at its floor on this device. Routing all 2,520
layouts per circuit with SABRE (O3) gives at best 8 SWAPs in total; SABRE gets 7.
The remaining headroom is in **estimated fidelity**, because Jakarta's qubits
differ: CX error 0.61%–1.34% per coupler, readout error 1.6%–5.5% per qubit.

**Headroom check** (`python -m evaluation.fidelity_headroom`): estimated fidelity
of every layout routed with SABRE at O3.

| circuit | SABRE | best layout | headroom | layouts beating SABRE |
|---|---|---|---|---|
| ghz | 0.9695 | 0.9695 | 0% | 0 / 2520 |
| dj | 0.9438 | 0.9566 | 1.4% | 22 |
| graphstate | 0.9053 | 0.9249 | 2.2% | 57 |
| qft | 0.7776 | 0.8052 | 3.6% | 20 |
| wstate | 0.9319 | 0.9367 | 0.5% | 3 |
| **mean** | **0.9056** | **0.9186** | **1.5%** | |

**Method** (`gnn/evaluate_shared.py`, no retraining): the GNN shortlists its 20
best-predicted layouts; each is routed with SABRE; keep either the one with the
fewest SWAPs (ties: lower depth) or the one with the highest estimated fidelity
(computed from the backend's calibration data, as a noise-aware compiler would).

| Method (totals over 5 circuits) | SWAPs | Depth | Total gates | Est. fidelity (mean) | Compile time (s) |
|---|---|---|---|---|---|
| SABRE | 7 | 127 | 203 | 0.9056 | 0.04 |
| MQT QMAP | 15 | 124 | 192 | 0.8954 | 0.01 |
| GNN top-1 + SABRE | 10 | 139 | 230 | 0.8902 | ~4 |
| GNN top-20 + SABRE, fewest SWAPs | **7** | **124** | **192** | 0.9028 | ~5 |
| GNN top-20 + SABRE, highest fidelity | **7** | 134 | 195 | **0.9097** | ~5 |

- **Fewest-SWAPs selection** ties SABRE on SWAPs and beats it on depth and gate count, but picks noisier qubits (lower fidelity).
- **Fidelity selection** ties or beats SABRE on every circuit (dj 0.9527 vs 0.9438, graphstate 0.9120 vs 0.9053, wstate 0.9367 vs 0.9319, ghz and qft equal) with the same 7 SWAPs, at the cost of depth (134 vs 127). It recovers about a third of the 1.5% headroom. qft gains nothing: the GNN ranks qft layouts worst (Spearman 0.34), so its top 20 miss the high-fidelity ones.
- **Shortlist size matters** (fewest-SWAPs selection, SABRE routing): k = 1 → 10 SWAPs, k = 5 → 9, k = 10 → 8, k = 20 → 7.

## What the results say

1. **From SABRE's layout, every router reaches 7 SWAPs.** At 5 qubits on a 7-qubit device there is no headroom left: greedy, RL and RL + GNN obs all match SABRE exactly.
2. **The GNN-selected layouts are worse starting points than SABRE's** (graphstate and qft). Every router does worse from them, so 4B does not improve on SABRE.
3. **The GNN features (4A) help where there is something to fix.** From the GNN's layouts, RL + GNN obs needs 9.3 ± 0.6 SWAPs against 11 ± 1 for plain RL and 10 for SABRE routing, the gain coming from qft and graphstate. This is the clearest evidence that the embeddings carry useful routing information.
4. **The GNN works best as a shortlist, not a single pick.** Its top choice is often wrong, but good layouts are in its top 20. Routing that shortlist and choosing by estimated fidelity is the only configuration that beats SABRE on a reported metric (+0.45% mean estimated fidelity, same SWAPs).
5. **qft is the hard case for every learned component**: lowest GNN ranking quality (Spearman 0.34) and the largest spread between methods.

## Caveats for the report

- **Checkpoint selection used the evaluation circuits.** `rl/train.py` keeps the checkpoint with the fewest SWAPs on the 5 benchmarks from SABRE's layout, the same setting reported in the SABRE-layout rows; this applies to Phase 3 agents too. Ties keep the earliest checkpoint, and every run reached 7 at its first evaluation (10k steps), so the saved agents are 10k-step agents. **The GNN-layout rows were never used for selection** and are the cleaner comparison.
- The 5 benchmarks are also in the RL training data (with random layouts), and the GNN was trained on layouts of the same 5 circuits. No generalization claims, consistent with the frozen scope.
- The GNN's predicted cost is computed on the whole circuit, so it does not decrease as routing progresses.
- 3 seeds, so the standard deviations are rough.
- **k = 20 was chosen after seeing that the best layouts sit inside the GNN's top 20** on this data; the k = 1/5/10 numbers above show the dependence and should be reported alongside.
- **Fidelity selection optimizes the reported metric directly** (estimated fidelity from the same calibration data). This is what noise-aware compilers do, but say so explicitly; the gain is in an *estimate*, not a measured hardware fidelity.
- The fidelity gain is small (+0.0041 mean, +0.45% relative) and comes with higher depth.
