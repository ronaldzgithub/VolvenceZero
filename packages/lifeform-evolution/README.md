# lifeform-evolution

Product-layer skeleton: long-horizon **evidence and gating**.

Future home of:
- Scripted dialogue / perturbation / replay-selection benchmark drivers (currently in `volvence_zero.agent.dialogue` etc.)
- Artifact acceptance gates for rare-heavy
- Human-review dashboards
- KPI export pipelines

Reads `vz-cognition.evaluation` snapshots; emits proposals into `ModificationGate`. Never bypasses the kernel.
