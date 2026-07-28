# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `trace`
- Model: `trace` on `cpu`
- Runtime origin: `trace`
- Seeds: `(0,)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- Controller dim: `16`
- SSL alpha: `0.1`

## Matched-control results

- Credit F1 delta: 1.0000 [1.0000, 1.0000]
- False-credit reduction: 1.0000 [1.0000, 1.0000]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0714 [0.0714, 0.0714]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 49
- Ground-truth subgoal boundaries: 2
- Mean SSL trained steps per run: 60.00
- Mean SSL prediction loss: 0.254984
- Mean SSL KL loss: 0.654128
- Mean SSL switch frequency: 0.343922
- SSL ACTIVE writebacks: 9

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
