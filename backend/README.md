# Quantum Compiler Project — Scope Freeze
## Fixed constraints

- **Topology:** one backend only — `FakeJakartaV2` (7 qubits). No multi-topology generalization.
- **Benchmark circuits:** 3–5 fixed MQT Bench algorithm-level circuits — `ghz`, `dj`, `graphstate`, `qft`, `wstate` at 5 qubits. No expansion to other circuit families.
- **Qubit range:** 5–8 qubits (RL phase). No claims outside this range.
- **Metrics (all phases report the same six):** circuit depth, SWAP count, total gate count, compilation time, estimated fidelity (gate-error product), state fidelity (statevector overlap, exact — only computed on 1–2 small circuits as a sanity check, not the full set).

## Phases

| Phase | What | Method |
|---|---|---|
| 1 | Baselines | Run SABRE (Qiskit) and MQT QMAP as-is. No implementation. |
| 2 | GNN | Small fixed 2–3 layer GCN, supervised. |
| 3 | RL | Stable-Baselines3 out of the box, single fixed topology, 5–8 qubit circuits. |
| 4 | GNN+RL | Concatenate GNN embeddings into the RL observation vector. |

