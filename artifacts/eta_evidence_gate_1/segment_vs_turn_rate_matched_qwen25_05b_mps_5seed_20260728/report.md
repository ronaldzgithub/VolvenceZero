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
- Switch rate weight: `0.001`
- Switch binary weight: `0.001`
- Switch group weight: `0.001`
- Prediction horizon: `3`
- Distortion target: `innovation`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.5277 [0.4505, 0.6403]
- False-credit reduction: 0.3860 [0.2969, 0.4984]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.1967 [0.0250, 0.4250]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 19
- Ground-truth subgoal boundaries: 120
- Held-out delayed events: 40
- Mean SSL trained steps per run: 48.00
- Mean SSL prediction loss: 0.251024
- Mean SSL KL loss: 35.122192
- Mean SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.503994
- Mean SSL switch-rate loss: 0.520327
- Mean keep/switch counterfactual loss: 0.250875 / 0.255883
- Mean SSL target variance: 0.065013
- Persistent optimizer final step/reuse count: 3 / 2
- SSL ACTIVE writebacks: 15

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
