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

- Credit F1 delta: 0.7103 [0.7103, 0.7103]
- False-credit reduction: 0.5583 [0.5583, 0.5583]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.4554 [0.4554, 0.4554]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 14
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 6
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.345923
- Mean SSL KL loss: 1.639582
- Mean SSL switch frequency: 0.325000
- Final SSL switch frequency: 0.266667
- Mean SSL switch probability: 0.532403
- Final SSL switch probability: 0.602548
- Mean SSL switch-rate loss: 0.197831
- Mean SSL gate-choice loss: 0.666677
- Mean keep/switch counterfactual loss: 0.367669 / 0.344466
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.181818
- Final SSL boundary/continuation switch probability: 0.583113 / 0.619553
- Final calibrated switch threshold: 0.698121
- Final causal-runtime switch threshold: 0.671697
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 48

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
