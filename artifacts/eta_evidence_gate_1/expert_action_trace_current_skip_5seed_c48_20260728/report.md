# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `48`
- SSL updates per cycle: `1`
- Controller dim: `16`
- SSL alpha: `0.01`
- Switch prior: `0.25`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Proposal prediction / gate choice weight: `1.0` / `5.0`
- Gate choice temperature: `0.001`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`
- Temporal fast prior enabled: `False`

## Matched-control results

- Credit F1 delta: 0.1720 [0.0000, 0.3440]
- False-credit reduction: 0.1118 [0.0000, 0.2236]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.3657 [0.1219, 0.6095]

## Mechanism diagnosis

- Mean active family count per run: 1.20
- Beta boundaries observed: 126
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 16
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.357359
- Mean SSL KL loss: 0.703903
- Mean SSL switch frequency: 0.314444
- Final SSL switch frequency: 0.266667
- Mean SSL switch probability: 0.516141
- Final SSL switch probability: 0.590433
- Mean SSL switch-rate loss: 0.194707
- Mean SSL gate-choice loss: 0.678893
- Mean keep/switch counterfactual loss: 0.374643 / 0.350721
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.218182
- Final SSL boundary/continuation switch probability: 0.577390 / 0.601846
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 240

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
