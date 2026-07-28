# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `6`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.2`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Prediction horizon: `3`
- Distortion target: `innovation`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.2468 [0.0823, 0.4114]
- False-credit reduction: 0.1583 [0.0528, 0.2639]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0286 [0.0000, 0.0571]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 98
- Ground-truth subgoal boundaries: 76
- Held-out delayed events: 24
- Mean SSL trained steps per run: 120.00
- Mean SSL prediction loss: 0.281620
- Mean SSL KL loss: 14.551316
- Mean SSL switch frequency: 0.354902
- Mean SSL switch probability: 0.534492
- Mean SSL switch-rate loss: 0.275744
- Mean keep/switch counterfactual loss: 0.305722 / 0.265040
- Mean SSL target variance: 0.070530
- Persistent optimizer final step/reuse count: 6 / 5
- SSL ACTIVE writebacks: 30

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
