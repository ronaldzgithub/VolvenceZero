# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
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
- Proposal prediction / gate choice weight: `1.0` / `20.0`
- Gate choice temperature: `0.001`
- Prediction horizon: `3`
- Distortion target: `innovation`
- SSL supervision target: `expert-action-vector`
- Expert action supervision: `True`
- Rollout replacement mode: `causal`
- Temporal fast prior enabled: `False`
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.6953 [0.6953, 0.6953]
- False-credit reduction: 0.5594 [0.5594, 0.5594]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.5167 [0.5167, 0.5167]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 6
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 8
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.386896
- Mean SSL KL loss: 1.226879
- Mean SSL switch frequency: 0.223611
- Final SSL switch frequency: 0.333333
- Mean SSL switch probability: 0.455347
- Final SSL switch probability: 0.670673
- Mean SSL switch-rate loss: 0.127526
- Mean SSL gate-choice loss: 0.690426
- Mean keep/switch counterfactual loss: 0.396355 / 0.381598
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.166667
- Final SSL boundary/continuation switch probability: 0.667928 / 0.673075
- Final calibrated switch threshold: 0.713221
- Final causal-runtime switch threshold: 0.493526
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 48

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
