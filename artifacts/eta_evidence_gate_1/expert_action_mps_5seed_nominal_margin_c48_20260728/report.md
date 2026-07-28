# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `48`
- SSL updates per cycle: `1`
- Controller dim: `16`
- SSL alpha: `0.01`
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
- Temporal fast prior enabled: `False`
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.7099 [0.6808, 0.7389]
- False-credit reduction: 0.5607 [0.5231, 0.5983]
- Family-assignment delta: 0.0667 [0.0000, 0.2000]
- Held-out PE reduction delta: 0.0077 [0.0000, 0.0232]
- Segment-boundary F1: 0.5006 [0.4256, 0.5756]

## Mechanism diagnosis

- Mean active family count per run: 1.20
- Beta boundaries observed: 85
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 32
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.344486
- Mean SSL KL loss: 1.570405
- Mean SSL switch frequency: 0.284167
- Final SSL switch frequency: 0.280000
- Mean SSL switch probability: 0.536023
- Final SSL switch probability: 0.578023
- Mean SSL switch-rate loss: 0.203672
- Mean SSL gate-choice loss: 0.649397
- Mean keep/switch counterfactual loss: 0.375885 / 0.340524
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.321212
- Final SSL boundary/continuation switch probability: 0.549684 / 0.602819
- Final calibrated switch threshold: 0.742565
- Final causal-runtime switch threshold: 0.780396
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 240

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
