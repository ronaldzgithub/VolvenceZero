# Gate 4 segment-aware active-learning evidence

- status: `not-supported`
- mechanism passed: `True`
- causal passed: `False`
- claim: typed active feedback is runnable but causal efficiency is unsupported
- segment vs shuffled aggregate labels-needed gain: `0`
- source segment length range: `2–2`; unique action families: `1`

## Segment-aware vs controls

- `turn-level-active`: aggregate label gain `0`, per-seed `[0, 0, 0]`, final accuracy gain `0.000000`, primary `False`
- `random-feedback`: aggregate label gain `0`, per-seed `[0, 0, 0]`, final accuracy gain `0.000000`, primary `False`

## Claim boundary

- Outcome labels are typed task/action measurements. PE participates only in acquisition and is excluded from the learned predictor and label definition.
- Failure of the shuffled-boundary kill control contracts the claim to PE-driven requests at most; locked evidence is not retuned or rerun.
