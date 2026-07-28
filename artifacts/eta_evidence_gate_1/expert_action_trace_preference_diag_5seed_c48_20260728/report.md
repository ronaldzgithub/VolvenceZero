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
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.5758 [0.4702, 0.6785]
- False-credit reduction: 0.4273 [0.3186, 0.5361]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.3315 [0.0875, 0.5756]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 51
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 39
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.352562
- Mean SSL KL loss: 0.723138
- Mean SSL switch frequency: 0.148056
- Final SSL switch frequency: 0.240000
- Mean SSL switch probability: 0.542886
- Final SSL switch probability: 0.579370
- Mean SSL switch-rate loss: 0.258117
- Mean SSL gate-choice loss: 0.663683
- Mean keep/switch counterfactual loss: 0.380313 / 0.347900
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.206993
- Final SSL boundary/continuation switch probability: 0.566905 / 0.590276
- Final calibrated switch threshold: 0.697957
- Final causal-runtime switch threshold: 0.641332
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 240

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
