# Protocol Intent

| Mode | Launch | Rank | Mature in Praxist | Durable parent | Normal close |
|---|---:|---:|---:|---:|---:|
| `preliminary` | yes | no | no | no | no |
| `complete` | yes | yes | yes | yes, development only | yes |
| loop-external sealed formal | no, outside Praxist | release gate only | n/a | n/a | n/a |

`preliminary` runs one full ten-episode chain through the same replay and
serialization path. Its coverage and effort ratios are `0.125`; it is only a
wiring or diagnostic signal.

`complete` runs all eight frozen public chains. Ratios are `1.0`. It can rank
and retain policies inside Praxist because it completely executes the declared
development protocol. This maturity does not mean coding quality was remeasured.
The public replay cannot authorize import, formal validation, SHADOW, or ACTIVE.

The parent contract is Pareto plus constraints. A policy must select at least
75% of owner-returned task-scoped entries and at least 95% of owner-returned
failed experiences. Every selected entry must remain as a complete owner-published
line after rendering; the evaluator does not synthesize replacement prose.
Within that floor, maximize scaling margin and selection coverage. Confirmed
candidates must also clear the frozen 0.10 token-ratio gate and the configured
hard byte budget. Complete candidates that preserve selected evidence but miss
an efficiency gate may remain in the incubator for repair.

This selection contract replaces an unreachable draft requirement that demanded
100% of every recalled full-text entry inside a 3500-character envelope. The
real baseline probe demonstrated that the full owner-returned set itself exceeds
that envelope in some contexts. The replacement keeps omission visible as a
metric and prevents partial-line truncation from masquerading as retention.
