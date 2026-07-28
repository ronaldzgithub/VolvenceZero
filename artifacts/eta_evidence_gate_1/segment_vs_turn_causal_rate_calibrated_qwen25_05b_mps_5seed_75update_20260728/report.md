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
- Temporal fast prior enabled: `False`
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.6433 [0.5421, 0.7238]
- False-credit reduction: 0.5019 [0.3844, 0.5901]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.4815 [0.2249, 0.6485]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 63
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 39
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.314595
- Mean SSL KL loss: 0.553177
- Mean SSL switch frequency: 0.375111
- Final SSL switch frequency: 0.320000
- Mean SSL switch probability: 0.556394
- Final SSL switch probability: 0.589781
- Mean SSL switch-rate loss: 0.110312
- Mean SSL gate-choice loss: 0.653137
- Mean keep/switch counterfactual loss: 0.355956 / 0.308066
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.406061
- Final SSL boundary/continuation switch probability: 0.609081 / 0.572894
- Final calibrated switch threshold: 0.717397
- Final causal-runtime switch threshold: 0.844024
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
