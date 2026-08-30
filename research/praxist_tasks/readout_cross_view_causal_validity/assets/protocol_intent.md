# Protocol Intent

| Stage | Launch | Rank | Praxist parent | Formal claim | Runtime authority |
|---|---:|---:|---:|---:|---:|
| `preliminary` | yes | no | no | no | no |
| `complete` | yes | yes | development only | no | no |
| loop-external sealed formal | outside Praxist | release review only | n/a | yes, if authorized | no automatic authority |

The natural evaluation unit is one held-out semantic group with all selected
views. `preliminary` evaluates two balanced groups and three views after using
the complete frozen training split; its effort and coverage ratios are `0.25`.
It exists only for wiring, schema, and obvious-failure triage.

`complete` evaluates all eight held-out groups and all six views. It is the only
ranking stage and reports effort and coverage ratios of `1.0`. The primary
scalar is the minimum normalized margin across every preregistered gate, so a
candidate cannot hide a weak view or random-control failure behind a mean.

The three development exits are:

- `PASS`: every frozen discrimination, calibration, retrieval, coherence,
  causal-effect, and random-separation gate clears;
- `DOMAIN_LOCAL`: the same-view instrument and causal check remain valid but a
  cross-view requirement fails;
- `INSTRUMENT_INVALID`: same-view discrimination, causal attribution, or the
  matched-control separation is not adequate.

`DOMAIN_LOCAL` complete evidence may remain in the incubator for repair, but it
does not authorize a cross-format snapshot. `INSTRUMENT_INVALID` and
`random_control` results are diagnostic. A complete `PASS` may be retained as a
development parent only. Formal qualification, owner publication,
`ModificationGate`, SHADOW, canary, ACTIVE, and rollback remain separate,
named-human-controlled stages.
