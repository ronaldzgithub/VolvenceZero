# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.1`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Prediction horizon: `3`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.4114 [0.4114, 0.4114]
- False-credit reduction: 0.2639 [0.2639, 0.2639]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0000 [0.0000, 0.0000]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 0
- Ground-truth subgoal boundaries: 120
- Held-out delayed events: 40
- Mean SSL trained steps per run: 60.00
- Mean SSL prediction loss: 0.313955
- Mean SSL KL loss: 24.500462
- Mean SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.492392
- Mean SSL switch-rate loss: 0.494380
- Mean keep/switch counterfactual loss: 0.325212 / 0.287179
- Persistent optimizer final step/reuse count: 3 / 2
- SSL ACTIVE writebacks: 15

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
