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
- Proposal prediction / gate choice weight: `0.5` / `1.0`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.5720 [0.2000, 0.8860]
- False-credit reduction: 0.5118 [0.1677, 0.8559]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.3038 [0.0167, 0.5909]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 126
- Ground-truth subgoal boundaries: 98
- Held-out delayed events: 32
- Mean SSL trained steps per run: 180.00
- Mean SSL prediction loss: 0.139381
- Mean SSL KL loss: 1.657465
- Mean SSL switch frequency: 0.565556
- Final SSL switch frequency: 0.440000
- Mean SSL switch probability: 0.552656
- Final SSL switch probability: 0.551982
- Mean SSL switch-rate loss: 0.633495
- Mean SSL gate-choice loss: 0.138690
- Mean keep/switch counterfactual loss: 0.140501 / 0.138844
- Mean SSL target variance: 0.136301
- Persistent optimizer final step/reuse count: 12 / 11
- SSL ACTIVE writebacks: 60

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
