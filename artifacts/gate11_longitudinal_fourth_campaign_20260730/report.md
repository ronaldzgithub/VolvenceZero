# Gate 11 + longitudinal fourth campaign

## Outcome

The campaign remains `mechanism-supported` overall. Gate 11 is the first gate
to receive a `longitudinal-supported` verdict, but Gate 5 remains
`not-supported`; the system thesis is not retained and live/production
promotion remains blocked.

## Fresh source

`gate11-longitudinal-settled-trace.v2` generated 510 settled transitions for
each of seeds `1201 / 1213 / 1223` on the same frozen strict-local
Qwen2.5-0.5B substrate. The aggregate contains 1530 transitions with complete
lineage, distinct trace digests, one shared substrate fingerprint, and zero
fallback, empty residual, duplicate settlement, lineage mismatch, or substrate
mutation.

The v1 development attempt coupled source capture to an accumulating runtime
state and was stopped after seven transitions because retrieval context growth
contaminated the substrate variable. Formal v2 separates fresh capture from
consumer-side persistence. Both locked consumers reconstruct their owner every
ten transitions.

## Gate 11

The original v1 evaluator returned `not-supported` only because it added an
unregistered `correct_state_consistency_perfect` kill gate. That artifact is
immutable and is recorded as `invalid-superseded`. The v2 reconciliation
copied the raw rows, removed only the unregistered evaluator gate, and reran no
locked arm.

The authoritative verdict is `longitudinal-supported`:

- correct minus stateless continuity composite: `+0.759259`
- correct minus swapped continuity composite: `+0.759259`
- correct minus shuffled continuity composite: `+0.666667`
- all three paired-seed 95% confidence lower bounds are positive
- cross-user read/write leakage and key collisions: `0`
- persistence round-trip, deletion and checkpoint rollback: exact

The claim remains narrow. Correct-state callback absolute hit rate was only
`0.277778`; commitment and boundary consistency were `1.0`. No blind human
relationship-quality ground truth was collected, so this supports isolated
owner continuity, not external relational quality.

## Gate 5

The five arms replayed 7650 arm-transitions. Every arm/seed crossed 50
filesystem persistence/constructor-restart boundaries. Cadence, parameter
budget, lineage, frozen substrate, persistence round-trip and rollback gates
passed, and full was Pareto non-worse than every control.

The minimum-effect gate failed:

- full minus single-timescale absorption: `+0.000000201`
- full minus single-timescale retention: `+0.000001187`
- preregistered minimum effect: `0.02`

Gate 5 therefore remains `not-supported`. The supported statement is only that
the multi-frequency CMS mechanism is cross-session runnable, auditable and
rollback-capable.

## System claim boundary

Gate 1-11 mechanism coverage is complete (debt #92 defines no Gate 3).
Causal-supported gates are `2 / 8 / 11`; only Gate 11 is
longitudinal-supported. Gate `1 / 4 / 5 / 6 / 7 / 9 / 10` remain
causal-not-supported. The full-chain rollback drill from Gate 10 remains valid,
but reversibility does not establish gain.

Debt #92 remains open. Production/live promotion is not authorized.
