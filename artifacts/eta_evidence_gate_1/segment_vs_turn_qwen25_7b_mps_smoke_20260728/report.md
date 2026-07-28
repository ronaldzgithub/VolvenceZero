# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-7B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- Controller dim: `16`
- SSL alpha: `0.1`

## Matched-control results

- Credit F1 delta: 0.0000 [0.0000, 0.0000]
- False-credit reduction: 0.0000 [0.0000, 0.0000]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0833 [0.0833, 0.0833]

## Mechanism diagnosis

- Mean active family count per run: 2.00
- Beta boundaries observed: 42
- Ground-truth subgoal boundaries: 2
- Held-out delayed events: 0
- Mean SSL trained steps per run: 48.00
- Mean SSL prediction loss: 0.144151
- Mean SSL KL loss: 1.475982
- Mean SSL switch frequency: 0.000000
- SSL ACTIVE writebacks: 9

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
