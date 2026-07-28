# ETA Segment Credit Evidence

- Verdict: `fail`
- Backend: `transformers-open-weight`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps`
- Runtime origin: `hf-local`
- Seeds: `(0, 1, 2, 3, 4)`
- Training mode: `ssl-rl-alternating`
- Training cycles: `3`
- Controller dim: `16`
- SSL alpha: `0.1`

## Matched-control results

- Credit F1 delta: 0.2703 [0.0901, 0.4505]
- False-credit reduction: 0.1781 [0.0594, 0.2969]
- Family-assignment delta: 0.0000 [0.0000, 0.0000]
- Held-out PE reduction delta: 0.0000 [0.0000, 0.0000]
- Segment-boundary F1: 0.0333 [0.0000, 0.0667]

## Mechanism diagnosis

- Mean active family count per run: 1.00
- Beta boundaries observed: 84
- Ground-truth subgoal boundaries: 76
- Held-out delayed events: 24
- Mean SSL trained steps per run: 48.00
- Mean SSL prediction loss: 0.195460
- Mean SSL KL loss: 1.265731
- Mean SSL switch frequency: 0.028149
- SSL ACTIVE writebacks: 45

This report is an evaluation readout. It does not write back to prediction-error, credit, or temporal owners.
