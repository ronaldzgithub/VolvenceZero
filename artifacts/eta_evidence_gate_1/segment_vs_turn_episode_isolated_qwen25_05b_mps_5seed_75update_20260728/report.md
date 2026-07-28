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

- Credit F1 delta: 0.4806 [0.1499, 0.8114]
- False-credit reduction: 0.4197 [0.1254, 0.7130]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.6315 [0.5988, 0.6763]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 163
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 20
- Mean SSL trained steps per run: 1125.00
- Mean SSL prediction loss: 0.312327
- Mean SSL KL loss: 0.555289
- Mean SSL switch frequency: 0.383644
- Final SSL switch frequency: 0.333333
- Mean SSL switch probability: 0.547934
- Final SSL switch probability: 0.607663
- Mean SSL switch-rate loss: 0.102191
- Mean SSL gate-choice loss: 0.651715
- Mean keep/switch counterfactual loss: 0.355986 / 0.306084
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.533333
- Final SSL boundary/continuation switch probability: 0.650170 / 0.570469
- Final calibrated switch threshold: 0.724751
- Persistent optimizer final step/reuse count: 75 / 74
- SSL ACTIVE writebacks: 375

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
