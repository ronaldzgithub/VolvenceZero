# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
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

- Credit F1 delta: 0.4505 [0.4505, 0.4505]
- False-credit reduction: 0.2969 [0.2969, 0.2969]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0000 [0.0000, 0.0000]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 0
- Ground-truth subgoal boundaries: 120
- Held-out delayed events: 40
- Mean SSL trained steps per run: 48.00
- Mean SSL prediction loss: 0.010435
- Mean SSL KL loss: 36.829257
- Mean SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.485929
- Mean SSL switch-rate loss: 0.480531
- Mean keep/switch counterfactual loss: 0.010675 / 0.009382
- Persistent optimizer final step/reuse count: 3 / 2
- SSL ACTIVE writebacks: 15

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
