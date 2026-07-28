# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `24`
- Controller dim: `16`
- SSL alpha: `0.1`
- Switch prior: `0.25`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Proposal prediction / gate choice weight: `1.0` / `5.0`
- Gate choice temperature: `0.001`
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
- Ground-truth subgoal boundaries: 24
- Held-out delayed events: 8
- Mean SSL trained steps per run: 360.00
- Mean SSL prediction loss: 0.133748
- Mean SSL KL loss: 1.911171
- Mean SSL switch frequency: 0.561111
- Final SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.551687
- Final SSL switch probability: 0.524133
- Mean SSL switch-rate loss: 0.206804
- Mean SSL gate-choice loss: 0.692894
- Mean keep/switch counterfactual loss: 0.134915 / 0.133074
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.000000
- Final SSL boundary/continuation switch probability: 0.523946 / 0.524297
- Persistent optimizer final step/reuse count: 24 / 23
- SSL ACTIVE writebacks: 24

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
