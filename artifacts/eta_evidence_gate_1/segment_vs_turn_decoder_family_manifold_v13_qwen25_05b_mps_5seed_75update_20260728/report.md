# ETA Segment Credit Evidence

- Verdict: `weak`
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
- Family truth source: `environment-expert-action-target`
- Family mapping fit split: `train-only`
- Causal family manifold projection: `True`
- Rollout replacement mode: `causal`
- Temporal fast prior enabled: `False`
- Episode recurrent state isolated: `True`

## Matched-control results

- Credit F1 delta: 0.9000 [0.9000, 0.9000]
- False-credit reduction: 0.8333 [0.8333, 0.8333]
- Family-assignment delta: 0.5000 [0.5000, 0.5000]
- Held-out PE reduction delta: 0.1589 [0.0000, 0.3177]
- Segment-boundary F1: 0.7652 [0.7652, 0.7652]

## Mechanism diagnosis

- Mean active family count per run: 4.00
- Beta boundaries observed: 125
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 10
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.313692
- Mean SSL KL loss: 0.717926
- Mean SSL switch frequency: 0.374222
- Final SSL switch frequency: 0.333333
- Mean SSL switch probability: 0.569876
- Final SSL switch probability: 0.648771
- Mean SSL switch-rate loss: 0.119383
- Mean SSL gate-choice loss: 0.634166
- Mean keep/switch counterfactual loss: 0.358131 / 0.308856
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.500000
- Final SSL boundary/continuation switch probability: 0.726604 / 0.580666
- Final calibrated switch threshold: 0.853325
- Final causal-runtime switch threshold: 0.808220
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
