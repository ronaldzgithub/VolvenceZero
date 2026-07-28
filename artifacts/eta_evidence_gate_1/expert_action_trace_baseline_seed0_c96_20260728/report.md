# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `96`
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

- Credit F1 delta: 0.6735 [0.6735, 0.6735]
- False-credit reduction: 0.5143 [0.5143, 0.5143]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.4375 [0.4375, 0.4375]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 14
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 7
- Mean SSL trained steps per run: 1440.00
- Mean SSL prediction loss: 0.318248
- Mean SSL KL loss: 0.614226
- Mean SSL switch frequency: 0.161806
- Final SSL switch frequency: 0.266667
- Mean SSL switch probability: 0.675307
- Final SSL switch probability: 0.817604
- Mean SSL switch-rate loss: 0.426029
- Mean SSL gate-choice loss: 0.598384
- Mean keep/switch counterfactual loss: 0.381225 / 0.294972
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.181818
- Final SSL boundary/continuation switch probability: 0.800325 / 0.832724
- Final calibrated switch threshold: 0.875703
- Final causal-runtime switch threshold: 0.774497
- Persistent optimizer final step/reuse count: 96 / 95
- SSL ACTIVE writebacks: 96

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
