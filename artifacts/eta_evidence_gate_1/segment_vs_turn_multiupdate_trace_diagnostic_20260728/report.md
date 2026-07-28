# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- SSL updates per cycle: `25`
- Controller dim: `8`
- SSL alpha: `0.1`
- Switch prior: `0.35`
- Switch rate weight: `0.01`
- Switch binary weight: `0.01`
- Switch group weight: `0.001`
- Proposal prediction / gate choice weight: `0.5` / `1.0`
- Gate choice temperature: `0.02`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.0000 [0.0000, 0.0000]
- False-credit reduction: 0.0000 [0.0000, 0.0000]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0833 [0.0833, 0.0833]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 42
- Ground-truth subgoal boundaries: 2
- Held-out delayed events: 0
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.108354
- Mean SSL KL loss: 0.370852
- Mean SSL switch frequency: 0.142222
- Final SSL switch frequency: 0.200000
- Mean SSL switch probability: 0.504718
- Final SSL switch probability: 0.533008
- Mean SSL switch-rate loss: 0.052588
- Mean SSL gate-choice loss: 0.685018
- Mean keep/switch counterfactual loss: 0.115594 / 0.109903
- Mean SSL target variance: 0.124544
- Final SSL expert-action boundary F1: 0.200000
- Final SSL boundary/continuation switch probability: 0.514080 / 0.549571
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 75

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
