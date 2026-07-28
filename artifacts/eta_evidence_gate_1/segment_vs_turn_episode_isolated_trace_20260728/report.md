# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- SSL updates per cycle: `25`
- Controller dim: `16`
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
- Temporal fast prior enabled: `False`

## Matched-control results

- Credit F1 delta: 0.6928 [0.5853, 0.7497]
- False-credit reduction: 0.5612 [0.4394, 0.6310]
- Family-assignment delta: 0.0286 [0.0000, 0.0857]
- Held-out PE reduction delta: -0.0262 [-0.0786, 0.0000]
- Segment-boundary F1: 0.4990 [0.2969, 0.6054]

## Mechanism diagnosis

- Mean active family count per run: 1.20
- Beta boundaries observed: 100
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 34
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.342170
- Mean SSL KL loss: 0.191070
- Mean SSL switch frequency: 0.388800
- Final SSL switch frequency: 0.346667
- Mean SSL switch probability: 0.524521
- Final SSL switch probability: 0.608191
- Mean SSL switch-rate loss: 0.071987
- Mean SSL gate-choice loss: 0.670403
- Mean keep/switch counterfactual loss: 0.377767 / 0.342395
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.392308
- Final SSL boundary/continuation switch probability: 0.596178 / 0.618702
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
