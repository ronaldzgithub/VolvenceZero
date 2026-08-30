# Forge research portfolios

This directory contains create-only, content-addressed
`forge-research-portfolio.v1` artifacts. Author a draft outside this directory,
omit `portfolio_id`, and seal it with:

```bash
forge research-portfolio-seal research/portfolio_drafts/<name>.json --json
```

Portfolio registration does not approve TopicBinding, A0, a Praxist start,
formal validation, candidate import, ModificationGate, SHADOW, ACTIVE, or
production wiring. Downstream dependency eligibility requires an exact
`RUN_COMPLETED` Request plus a named-human `StudyOutcome(PROCEED)`.

## Current registered program

The current 4ables improvement program is the immutable Portfolio
[`baaf616c...`](./baaf616c923bc77b3eb38a0fb68ce7a3d8b48bb3c6f9129cd592d67fcbde1f6b.json):

- `readout_cross_view_causal_validity` (P0, runnable mapping)
- `substrate_control_authority` (after P0)
- `relationship_memory_write_eligibility` (after P0, separate memory lane)
- `per_instance_layer_dose_headroom` (after substrate authority)
- `steering_side_effect_matrix` (after per-instance headroom)

The first study is registered but remains behind named-human TopicBinding and
A0. The other four studies remain behind exact predecessor outcomes as well as
their own task-design, TopicBinding, and A0 gates.
