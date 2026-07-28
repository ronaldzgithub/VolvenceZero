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
- Switch prior: `0.4`
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

- Credit F1 delta: 0.6099 [0.6099, 0.6099]
- False-credit reduction: 0.4688 [0.4688, 0.4688]
- Family-assignment delta: 0.5000 [0.5000, 0.5000]
- Held-out PE reduction delta: -0.0013 [-0.0013, -0.0013]
- Segment-boundary F1: 0.2833 [0.2833, 0.2833]

## Mechanism diagnosis

- Mean active family count per run: 2.00
- Beta boundaries observed: 8
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 8
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.340801
- Mean SSL KL loss: 0.741631
- Mean SSL switch frequency: 0.175000
- Final SSL switch frequency: 0.400000
- Mean SSL switch probability: 0.564617
- Final SSL switch probability: 0.608563
- Mean SSL switch-rate loss: 0.081198
- Mean SSL gate-choice loss: 0.672667
- Mean keep/switch counterfactual loss: 0.370882 / 0.333729
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.461538
- Final SSL boundary/continuation switch probability: 0.613231 / 0.604478
- Final calibrated switch threshold: 0.642246
- Final causal-runtime switch threshold: 0.589886
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 48

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
