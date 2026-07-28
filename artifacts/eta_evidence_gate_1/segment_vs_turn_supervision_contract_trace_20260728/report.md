# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `1`
- Controller dim: `8`
- SSL alpha: `0.1`
- Switch prior: `0.1`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Proposal prediction / gate choice weight: `0.5` / `1.0`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `next-residual-summary-innovation-proxy`
- Expert action supervision: `False`
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
- Ground-truth subgoal boundaries: 24
- Held-out delayed events: 8
- Mean SSL trained steps per run: 20.00
- Mean SSL prediction loss: 0.463644
- Mean SSL KL loss: 2.367586
- Mean SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.502032
- Mean SSL switch-rate loss: 0.515298
- Mean SSL gate-choice loss: 0.458147
- Mean keep/switch counterfactual loss: 0.469612 / 0.458860
- Mean SSL target variance: 0.096940
- Persistent optimizer final step/reuse count: 1 / 0
- SSL ACTIVE writebacks: 1

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
