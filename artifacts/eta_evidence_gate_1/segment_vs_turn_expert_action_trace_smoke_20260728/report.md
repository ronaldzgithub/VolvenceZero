# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- Controller dim: `8`
- SSL alpha: `0.1`
- Switch prior: `0.1`
- Switch rate weight: `0.001`
- Switch binary weight: `0.001`
- Switch group weight: `0.001`
- Proposal prediction / gate choice weight: `0.5` / `1.0`
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
- Mean SSL trained steps per run: 45.00
- Mean SSL prediction loss: 0.140044
- Mean SSL KL loss: 2.340092
- Mean SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.514251
- Mean SSL switch-rate loss: 0.542728
- Mean SSL gate-choice loss: 0.137917
- Mean keep/switch counterfactual loss: 0.142234 / 0.140332
- Mean SSL target variance: 0.124544
- Persistent optimizer final step/reuse count: 3 / 2
- SSL ACTIVE writebacks: 3

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
