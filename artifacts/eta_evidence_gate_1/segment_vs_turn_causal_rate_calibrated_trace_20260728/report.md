# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
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

- Credit F1 delta: 0.6881 [0.6608, 0.7340]
- False-credit reduction: 0.5454 [0.5120, 0.6025]
- Family-assignment delta: 0.1000 [0.0000, 0.3000]
- Held-out PE reduction delta: -0.0953 [-0.2859, 0.0000]
- Segment-boundary F1: 0.5429 [0.4387, 0.6470]

## Mechanism diagnosis

- Mean active family count per run: 1.20
- Beta boundaries observed: 96
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 35
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.340320
- Mean SSL KL loss: 0.202932
- Mean SSL switch frequency: 0.377956
- Final SSL switch frequency: 0.266667
- Mean SSL switch probability: 0.518789
- Final SSL switch probability: 0.599794
- Mean SSL switch-rate loss: 0.068466
- Mean SSL gate-choice loss: 0.675626
- Mean keep/switch counterfactual loss: 0.373726 / 0.341247
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.266667
- Final SSL boundary/continuation switch probability: 0.587449 / 0.610596
- Final calibrated switch threshold: 0.661295
- Final causal-runtime switch threshold: 0.675589
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
