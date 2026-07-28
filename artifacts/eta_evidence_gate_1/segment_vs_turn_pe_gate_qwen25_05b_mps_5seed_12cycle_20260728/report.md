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

- Credit F1 delta: 0.4301 [0.4301, 0.4301]
- False-credit reduction: 0.2795 [0.2795, 0.2795]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0000 [0.0000, 0.0000]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 0
- Ground-truth subgoal boundaries: 120
- Held-out delayed events: 40
- Mean SSL trained steps per run: 180.00
- Mean SSL prediction loss: 0.139328
- Mean SSL KL loss: 1.682479
- Mean SSL switch frequency: 0.145556
- Final SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.512427
- Final SSL switch probability: 0.477786
- Mean SSL switch-rate loss: 0.056855
- Mean SSL gate-choice loss: 0.694452
- Mean keep/switch counterfactual loss: 0.140506 / 0.139095
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.000000
- Final SSL boundary/continuation switch probability: 0.477812 / 0.477763
- Persistent optimizer final step/reuse count: 12 / 11
- SSL ACTIVE writebacks: 60

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
