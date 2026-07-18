# companion-encoder

Training + evaluation scaffold for the **open-weights relationship
encoder**: a small model that reads a canonical
[`companion-standard`](../companion-standard/README.md) interaction
trajectory and predicts structured relationship state.

## Contract

- **Input**: canonical `InteractionTrajectory` JSON (the standard's only
  exchange format). At each label anchor the model sees the trajectory
  *prefix up to that anchor*.
- **Output 1 (structured prediction)**: relationship phase (8-way
  classification over the standard's closed `RelationshipPhase`
  vocabulary) + `trust_level` / `continuity_level` / `repair_pressure`
  regressions in [0, 1], with confidence.
- **Output 2 (embedding)**: an L2-normalized trajectory embedding; the
  trained checkpoint wraps into the standard's
  `SemanticEmbeddingBackend` protocol (`companion_encoder.backend`).

## What ships here vs what does not

This package ships **code only**: dataset loading, deterministic
serialization, the two-head model, the training loop, the G2 evaluation
harness, and the baselines it must beat. **No trained weights ship**
until release gates G1 (data audit) → G2 (beats baselines) → G3 (leakage
audit) → G4 (claims audit) all pass.

## Backbones

- `tiny` — a from-scratch byte-level transformer (pure torch, CPU/MPS
  friendly). Exists so the full train → eval → baseline chain dry-runs
  on deterministic FSM trajectories without GPUs or model downloads.
  Explicitly NOT the release candidate.
- `hf:<model_id>` — any Hugging Face causal/masked LM as a frozen or
  fine-tuned backbone (requires the `[hf]` extra). The intended M2 path
  (e.g. a Qwen-family small model).

## G2 evaluation harness

`companion-encoder evaluate` produces a JSON report with, side by side:

| Metric | Encoder | Baselines |
|---|---|---|
| Phase accuracy / macro-F1 | ✓ | majority-class, LLM zero-shot |
| trust/continuity/repair MAE | ✓ | global-mean, LLM zero-shot |
| Embedding retrieval (family top-1) | ✓ | standard stub embedding |
| Calibration (ECE) | ✓ | — |

G2 passes only if structured prediction significantly beats both the
majority-class and LLM zero-shot columns on the val split.

## Usage

```bash
# 1. generate data (see companion-trajgen)
companion-trajgen generate --mode fsm --out-dir data/traj-v0

# 2. train (tiny backbone dry-run)
companion-encoder train --data-dir data/traj-v0 --out-dir runs/tiny-v0 \
  --backbone tiny --epochs 4

# 3. evaluate against baselines
companion-encoder evaluate --checkpoint runs/tiny-v0/encoder.pt \
  --data-dir data/traj-v0 --report runs/tiny-v0/g2-report.json

# 4. baselines only (majority; add --llm-base-url for zero-shot column)
companion-encoder baseline --data-dir data/traj-v0
```

## Boundaries

Depends on `companion-standard` only (plus optional torch/transformers).
Never imports internal runtime wheels, never touches held-out scenarios,
and labels are read from trajectory documents as-is — this package never
manufactures labels (enforced by
`tests/contracts/test_companion_encoder_boundaries.py`).
