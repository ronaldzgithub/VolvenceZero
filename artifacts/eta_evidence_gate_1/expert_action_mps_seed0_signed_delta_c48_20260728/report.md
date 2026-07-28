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

- Credit F1 delta: 0.7102 [0.7102, 0.7102]
- False-credit reduction: 0.5714 [0.5714, 0.5714]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.5774 [0.5774, 0.5774]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 20
- Ground-truth subgoal boundaries: 19
- Held-out delayed events: 7
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.348409
- Mean SSL KL loss: 1.520446
- Mean SSL switch frequency: 0.400000
- Final SSL switch frequency: 0.400000
- Mean SSL switch probability: 0.516283
- Final SSL switch probability: 0.636354
- Mean SSL switch-rate loss: 0.176943
- Mean SSL gate-choice loss: 0.697953
- Mean keep/switch counterfactual loss: 0.380009 / 0.327390
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.307692
- Final SSL boundary/continuation switch probability: 0.633213 / 0.639102
- Final calibrated switch threshold: 0.649362
- Final causal-runtime switch threshold: 0.650351
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 48

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
