# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `12`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.1`
- Switch rate weight: `0.001`
- Switch binary weight: `0.001`
- Switch group weight: `0.001`
- Prediction horizon: `3`
- Distortion target: `innovation`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.5399 [0.4505, 0.7188]
- False-credit reduction: 0.4042 [0.2969, 0.6188]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.1061 [0.0000, 0.3182]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 28
- Ground-truth subgoal boundaries: 120
- Held-out delayed events: 40
- Mean SSL trained steps per run: 192.00
- Mean SSL prediction loss: 0.153583
- Mean SSL KL loss: 14.418868
- Mean SSL switch frequency: 0.303846
- Mean SSL switch probability: 0.524375
- Mean SSL switch-rate loss: 0.568677
- Mean keep/switch counterfactual loss: 0.152864 / 0.153170
- Mean SSL target variance: 0.065013
- Persistent optimizer final step/reuse count: 12 / 11
- SSL ACTIVE writebacks: 60

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
