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
circuit (about 1.5 s per circuit on the i3 laptop).

## What the results say

1. **From SABRE's layout, every router reaches 7 SWAPs.** At 5 qubits on a 7-qubit device there is no headroom left: greedy, RL and RL + GNN obs all match SABRE exactly.
2. **The GNN-selected layouts are worse starting points than SABRE's** (graphstate and qft). Every router does worse from them, so 4B does not improve on SABRE.
3. **The GNN features (4A) help where there is something to fix.** From the GNN's layouts, RL + GNN obs needs 9.3 ± 0.6 SWAPs against 11 ± 1 for plain RL and 10 for SABRE routing, the gain coming from qft and graphstate. This is the clearest evidence that the embeddings carry useful routing information.
4. **qft is the hard case for every learned component**: lowest GNN ranking quality (Spearman 0.34) and the largest spread between methods.

## Caveats for the report

- **Checkpoint selection used the evaluation circuits.** `rl/train.py` keeps the checkpoint with the fewest SWAPs on the 5 benchmarks from SABRE's layout, the same setting reported in the SABRE-layout rows; this applies to Phase 3 agents too. Ties keep the earliest checkpoint, and every run reached 7 at its first evaluation (10k steps), so the saved agents are 10k-step agents. **The GNN-layout rows were never used for selection** and are the cleaner comparison.
- The 5 benchmarks are also in the RL training data (with random layouts), and the GNN was trained on layouts of the same 5 circuits. No generalization claims, consistent with the frozen scope.
- The GNN's predicted cost is computed on the whole circuit, so it does not decrease as routing progresses.
- 3 seeds, so the standard deviations are rough.
