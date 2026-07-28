# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- SSL updates per cycle: `25`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.35`
- Switch rate weight: `0.01`
- Switch binary weight: `0.01`
- Switch group weight: `0.001`
- Proposal prediction / gate choice weight: `0.5` / `1.0`
- Gate choice temperature: `0.02`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`

## Matched-control results

- Credit F1 delta: 0.0860 [0.0000, 0.2580]
- False-credit reduction: 0.0559 [0.0000, 0.1677]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0667 [0.0333, 0.0833]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 168
- Ground-truth subgoal boundaries: 32
- Held-out delayed events: 8
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.115464
- Mean SSL KL loss: 0.416477
- Mean SSL switch frequency: 0.186489
- Final SSL switch frequency: 0.373333
- Mean SSL switch probability: 0.512814
- Final SSL switch probability: 0.547539
- Mean SSL switch-rate loss: 0.059381
- Mean SSL gate-choice loss: 0.683673
- Mean keep/switch counterfactual loss: 0.121510 / 0.116314
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.365462
- Final SSL boundary/continuation switch probability: 0.549569 / 0.545762
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
