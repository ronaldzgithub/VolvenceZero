# vz-temporal

> R3 (Temporal Abstraction) · R4 (Internal Control Above Token Space)

The system's **only** decision layer above token space: ETA-style metacontroller (encoder + switch unit β_t + decoder) plus Internal RL on the controller-code latent `z_t`.

## What it owns

| Sub-package | Role |
|---|---|
| `temporal` | `MetacontrollerRuntimeState`, `TemporalAbstractionSnapshot`, dual-track owners (`world_temporal`, `self_temporal`), SSL trainer, M3 optimizer |
| `planning` | Imagination/rollout in latent space |
| `internal_rl` | RL sandbox on `z_t` (proof environments, sparse-reward taxonomies) |
| `joint_loop` | SSL → RL alternation, rare-heavy artifact pipeline, substrate-aware adapter-delta |

## Hard limits

- **No token-level RL.** RL acts on `z_t` (low-dimensional latent), not on raw tokens.
- Two-stage training is mandatory: SSL discovers `z_t / β_t`, then Internal RL refines them with the substrate frozen.
- Switch sparsity (β_t binarisation) is published as a health metric for `vz-cognition.evaluation` to track.

## Install

```bash
pip install vz-temporal               # forward-only inference path
pip install "vz-temporal[torch]"      # + torch for SSL/RL training
```
