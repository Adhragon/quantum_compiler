# Phase 3 — RL qubit routing

A Stable-Baselines3 agent that decides **where to insert SWAP gates** so a 5-qubit
circuit can run on the 7-qubit FakeJakartaV2. It is compared against SABRE and
MQT QMAP from Phase 1.

## Files

| File | Role | Run it? |
|---|---|---|
| `circuit_utils.py` | Coupling map, distance matrix, circuit → gate list, random circuits, SABRE reference | no (imported) |
| `env.py` | `RoutingEnv`, the Gymnasium environment | no (imported) |
| `smoke_test.py` | Correctness checks for the env | after changing the env |
| `train.py` | Trains the agent, saves best checkpoint | yes |
| `evaluate.py` | Scores checkpoints on the 5 benchmarks → `evaluation/results/rl_results.csv` | yes, after training |

```bash
# from backend/, venv active
python -m rl.smoke_test
python -m rl.train                 # seed 42 -> rl/checkpoints/sb3_agent.zip  (~15 min on CPU)
python -m rl.train --seed 1        # extra seeds for mean +/- std
python -m rl.train --seed 2
python -m rl.evaluate              # evaluates every checkpoint in rl/checkpoints/
```

## What the agent does

| | |
|---|---|
| **Decides** | Routing only: which coupler to SWAP next. |
| **Does not decide** | Initial layout (given), gate synthesis / optimization (Qiskit O3 afterwards). |
| **Action** | One of Jakarta's 6 couplers: (0,1) (1,2) (1,3) (3,5) (4,5) (5,6). |
| **After each SWAP** | Every gate whose dependencies are met and whose qubits are now adjacent runs automatically. |
| **Observation (34 floats)** | Current qubit placement (7) · how far apart the next 8 two-qubit gates are (8) · for each coupler, how much swapping it would shrink the distance of the current front layer (6) and of the next 8 gates (6) · last action (6) · fraction of gates done (1). |
| **Reward** | −1 per SWAP, +1 per two-qubit gate executed, +0.5 × front-layer distance reduction, +5 on finishing. |
| **Episode end** | All gates executed, or 50 SWAPs (then finished by shortest-path SWAPs, flagged in the results). |
| **Algorithm** | MaskablePPO (sb3-contrib), MLP 64-64, 200k steps. |

## What it is trained on

Same device (FakeJakartaV2) and same circuit size (5 qubits) as everything else. Each episode is:

- **50%:** a random 5-qubit circuit of 5–30 CX gates.
- **50%:** one of the 5 benchmarks (ghz, dj, graphstate, qft, wstate).
- **Initial layout:** a random placement on the 7 physical qubits, every episode.

The 5 benchmark circuits **are in the training set**; only their evaluation layout
(SABRE's) is unseen. This is acceptable under the frozen scope (no generalization
claims) but must be stated in the report. Setting `random_circuit_prob=1.0` in
`RoutingEnv` trains on random circuits only, if a held-out evaluation is wanted.

## What it is evaluated on

| | |
|---|---|
| **Circuits** | The 5 MQT Bench algorithm-level benchmarks at 5 qubits (graphstate seeded, see below). |
| **Initial layout** | SABRE's own initial layout for each circuit, so RL and SABRE route from the same starting point. |
| **After routing** | Qiskit's O3 translation + optimization with the layout pinned and routing disabled: exactly what SABRE's output goes through after its own routing. |
| **Metrics** | Same as Phase 1 (`evaluation/metrics.py`): depth, SWAPs, total gates, compile time, estimated fidelity, state fidelity. |
| **Compile time** | RL inference + routing + O3 post-processing. SABRE's covers layout search + routing + O3. |
| **Comparisons** | SABRE, MQT QMAP (from Phase 1 CSV), a greedy router (same env, best one-step SWAP), and RL (mean +/- std over seeds 42, 1, 2). |

## Why each decision

| Decision | Why |
|---|---|
| RL does routing only, layout fixed | Keeps the problem small enough to train on a laptop CPU. Follows Pozzi et al. (2022). Makes the comparison routing-vs-routing instead of mixing in layout quality. |
| Evaluate from SABRE's layout | Both routers start from the same placement, so SWAP differences come from routing alone. |
| Same pre-routing prep as SABRE (Qiskit O3 `init` stage) | That stage removes QFT's 2 built-in logical SWAPs. Without it, RL routed a bigger QFT (109 gates vs SABRE's 72) for reasons unrelated to routing. |
| Same O3 post-processing as SABRE | Gate count and depth then reflect routing choices, not a different optimizer. |
| Random circuits + random layouts in training | 5 circuits alone would be memorized; random data teaches the agent to route generally on this topology. |
| Per-coupler "distance reduction" features | They line up one-to-one with the actions, so the agent learns quickly (matches SABRE after about 10k steps). |
| Action masking (MaskablePPO) | Blocks pointless SWAPs and immediately undoing the last one, which plain PPO wastes many steps on. `--no-mask` trains plain PPO for comparison. |
| 50-SWAP cap + shortest-path fallback | Every episode produces a valid circuit; any use of the fallback is reported in `notes`. |
| Greedy baseline | Shows how much of RL's result a simple one-step rule already achieves. |
| Flat fixed-length observation + `extra_obs_fn` hook | Phase 4 appends the GNN embedding to this vector without changing anything else. |
| CPU only | The policy is tiny (~13k parameters) and the env is plain Python; a GPU would not speed it up. |

## Results

Totals over the 5 circuits (RL = mean ± std over seeds 42, 1, 2; full per-circuit
numbers in `evaluation/results/rl_results.csv`):

| Method | SWAPs | Total gates | Depth | Est. fidelity (mean) |
|---|---|---|---|---|
| SABRE | 7 | 203 | 127 | 0.906 |
| MQT QMAP | 15 | 192 | 124 | 0.895 |
| Greedy | 7 | 207 | 129 | 0.907 |
| **RL (MaskablePPO)** | **7 ± 0** | **202.3 ± 0.5** | **128.3 ± 0.9** | **0.905 ± 0.001** |

RL matches SABRE's SWAP count on every circuit across all 3 seeds
(ghz 0, dj 1, graphstate 2, qft 4, wstate 0). The only circuit where the methods
differ is qft: RL needs 71–72 gates, SABRE 72, greedy 76. State fidelity is 1.0
everywhere, so every routed circuit is correct.

## Fixes made to shared code during this phase

- **SABRE SWAP count was always 0** (`baseline/benchmark_sabre_qmap.py`). Qiskit decomposes SWAPs into CXs after routing, so counting `swap` gates in the final circuit found none. It now counts the SWAPs the routing pass inserts. Corrected SABRE counts: ghz 0, dj 1, graphstate 2, qft 4, wstate 0. QMAP's counts were correct.
- **graphstate was a different random circuit on every load** (`config.py`). MQT Bench builds a random graph unless seeded; it is now seeded with `SEED`, so all phases score the same circuit.

## Caveats for the report

- Checkpoint selection (`train.py` keeps the run's fewest-SWAP checkpoint on the 5 benchmarks from SABRE's layout) uses the evaluation setting itself. Every seed reached SABRE's 7 SWAPs at its first evaluation (10k steps), and ties keep the earliest checkpoint. Phase 4's GNN-layout rows (see `gnn_rl/README.md`) were not used for selection.
- Evaluation circuits are in the training set (see above).
- With 5-qubit circuits, the greedy router already matches SABRE's SWAP count. The realistic claim is that RL matches SABRE, not that it beats it.
- State fidelity is computed for all 5 circuits because it is cheap at 7 qubits. It checks correctness and is 1.0 for every routed circuit; noise is reflected only in the estimated fidelity.
