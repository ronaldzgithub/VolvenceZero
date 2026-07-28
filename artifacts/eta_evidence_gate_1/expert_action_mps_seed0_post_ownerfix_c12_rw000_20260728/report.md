# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `12`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.1`
- Switch rate weight: `0.0`
- Switch binary weight: `0.001`
- Switch group weight: `0.001`
- Proposal prediction / gate choice weight: `0.5` / `1.0`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.0000 [0.0000, 0.0000]
- False-credit reduction: 0.0000 [0.0000, 0.0000]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0833 [0.0833, 0.0833]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 42
- Ground-truth subgoal boundaries: 2
- Held-out delayed events: 0
- Mean SSL trained steps per run: 180.00
- Mean SSL prediction loss: 0.137770
- Mean SSL KL loss: 1.762921
- Mean SSL switch frequency: 0.750000
- Final SSL switch frequency: 1.000000
- Mean SSL switch probability: 0.578616
- Final SSL switch probability: 0.581445
- Mean SSL switch-rate loss: 0.699135
- Mean SSL gate-choice loss: 0.136752
- Mean keep/switch counterfactual loss: 0.138527 / 0.137281
- Mean SSL target variance: 0.136301
- Persistent optimizer final step/reuse count: 12 / 11
- SSL ACTIVE writebacks: 12

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
