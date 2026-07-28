# ETA Segment Credit Evidence

- Verdict: `retain`
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

- Credit F1 delta: 0.7317 [0.7175, 0.7460]
- False-credit reduction: 0.5883 [0.5683, 0.6083]
- Family-assignment delta: 0.4667 [0.4000, 0.5000]
- Held-out PE reduction delta: 0.0231 [0.0012, 0.0417]
- Segment-boundary F1: 0.6095 [0.6012, 0.6179]

## Mechanism diagnosis

- Mean active family count per run: 2.00
- Beta boundaries observed: 88
- Ground-truth subgoal boundaries: 95
- Held-out delayed events: 30
- Mean SSL trained steps per run: 720.00
- Mean SSL prediction loss: 0.345826
- Mean SSL KL loss: 1.658612
- Mean SSL switch frequency: 0.316389
- Final SSL switch frequency: 0.266667
- Mean SSL switch probability: 0.558216
- Final SSL switch probability: 0.611609
- Mean SSL switch-rate loss: 0.231967
- Mean SSL gate-choice loss: 0.663613
- Mean keep/switch counterfactual loss: 0.376252 / 0.344808
- Mean SSL target variance: 0.136301
- Final SSL expert-action boundary F1: 0.181818
- Final SSL boundary/continuation switch probability: 0.584468 / 0.635357
- Final calibrated switch threshold: 0.719507
- Final causal-runtime switch threshold: 0.801675
- Persistent optimizer final step/reuse count: 48 / 47
- SSL ACTIVE writebacks: 240

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
