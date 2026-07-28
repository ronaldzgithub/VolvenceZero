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

- Credit F1 delta: 0.4301 [0.4301, 0.4301]
- False-credit reduction: 0.2795 [0.2795, 0.2795]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0000 [0.0000, 0.0000]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 0
- Ground-truth subgoal boundaries: 24
- Held-out delayed events: 8
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.118594
- Mean SSL KL loss: 0.557701
- Mean SSL switch frequency: 0.000000
- Final SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.487860
- Final SSL switch probability: 0.518004
- Mean SSL switch-rate loss: 0.042543
- Mean SSL gate-choice loss: 0.693057
- Mean keep/switch counterfactual loss: 0.120776 / 0.118964
- Mean SSL target variance: 0.124544
- Final SSL expert-action boundary F1: 0.000000
- Final SSL boundary/continuation switch probability: 0.517791 / 0.518190
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 75

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
