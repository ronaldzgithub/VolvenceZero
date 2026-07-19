# Probe → VZ R-ID 双向索引

本文档把 `volvence-research/probe/11_vz_implications.md` 的 38 P0 + 52 P1 行动按 labs probe 反查。

## Probe → R-ID 映射

| Probe ID | Primitive | R-IDs | 对应 P0 行动 |
|---|---|---|---|
| `pe-baseline-v0` | P5 | R-PE | P0-PE.4 (BPC SHADOW evidence) |
| `pe-curiosity-critic-v1` | P5 | R-PE | P0-PE.4 |
| `cma-probes-v1` | P4 | R1, R5, R6 | P0-R5.1 (Titans/Miras 写入 memory spec) |
| `cpd-option-critic-v1` | P3 | R3 | P0-R3.1 (CPD + Option-Critic 写入 dual-track) |
| `refusal-direction-v1` | P7 | R12 | P0-R12.1 (Persona Vectors 只读 readout) |
| `bounded-self-mod-v1` | P6 | R10 | P0-R10.1 (Two-Gate VC capacity), P0-R10.2 (Sleeper motivation) |
| `latent-controller-v1` | P2 | R3, R4 | P0-R3.2 (ETA latent control) |
| `frozen-substrate-v1` | P1 | R2 | P0-R2.1 (V-JEPA frozen + controller) |
| `r15-rollback-v0` | F5 | R8, R15, R12 | P0-R15.1 (rollback formalization) |
| `r15-rollback-v1` | F5 | R8, R15, R12 | P0-R15.1 |

## R-ID → Probe 反查

| R-ID | 覆盖 Probes | 未覆盖 P0 行动（阶段 2+） |
|---|---|---|
| R-PE | pe-baseline-v0, pe-curiosity-critic-v1 | F1 (LLM scale PE) |
| R1 | cma-probes-v1 | — |
| R2 | frozen-substrate-v1 | — |
| R3 | cpd-option-critic-v1, latent-controller-v1 | — |
| R4 | latent-controller-v1 | F2 (cross-modal z_t) |
| R5 | cma-probes-v1 | — |
| R6 | cma-probes-v1 | — |
| R7 | (covered by P3 dual-track) | P0-R7.1 (Sophia), P0-R7.2 (Alignment Faking) |
| R8 | r15-rollback-v0/v1 | — |
| R9 | bounded-self-mod-v1 (partial) | P0-R9.1 (COCOA), P0-R9.2 (Math-Shepherd) |
| R10 | bounded-self-mod-v1 | P0-R10.3 (Persona Vectors monitoring), P0-R10.4 (AlphaEvolve) |
| R11 | refusal-direction-v1 (partial) | P0-R11.1 (Sparse Feature Circuits) |
| R12 | refusal-direction-v1, r15-rollback-v0/v1 | — |
| R14 | (covered by P7 monitoring) | P0-R14.1 (regime drift) |
| R15 | r15-rollback-v0/v1 | — |

## 未覆盖的 P0 行动（需阶段 2 或新 probe）

- P0-R7.1: Sophia 作为 dual_track 工程实例化
- P0-R7.2: Alignment Faking 实证
- P0-R9.1: COCOA counterfactual credit
- P0-R9.2: Math-Shepherd MC rollout
- P0-R10.3: Persona Vectors 集成为 regime 漂移监控
- P0-R10.4: AlphaEvolve evaluator 完备性
- P0-R11.1: Sparse Feature Circuits 作为 owner 工具学起点
