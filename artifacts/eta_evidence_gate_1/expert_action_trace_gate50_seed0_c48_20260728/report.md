# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `48`
- SSL updates per cycle: `1`
- Controller dim: `16`
- SSL alpha: `0.01`
- Switch prior: `0.25`
- Switch rate weight: `0.05`
- Switch binary weight: `0.01`
- Switch group weight: `0.01`
- Proposal prediction / gate choice weight: `1.0` / `50.0`
- Gate choice temperature: `0.001`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`
- Temporal fast prior enabled: `False`
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.0000 [0.0000, 0.0000]
- False-credit reduction: 0.0000 [0.0000, 0.0000]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.6095 [0.6095, 0.6095]

## Mechanism diagnosis

- Mean active family count per run: 2.00
- Beta boundaries observed: 42
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 0
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.403520
- Mean SSL KL loss: 1.094534
- Mean SSL switch frequency: 0.145833
- Final SSL switch frequency: 0.000000
- Mean SSL switch probability: 0.456180
- Final SSL switch probability: 0.333616
- Mean SSL switch-rate loss: 0.142475
- Mean SSL gate-choice loss: 0.706261
- Mean keep/switch counterfactual loss: 0.430471 / 0.399111
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.000000
- Final SSL boundary/continuation switch probability: 0.333634 / 0.333600
- Final calibrated switch threshold: 0.334473
- Final causal-runtime switch threshold: 0.317921
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 48

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
