# Gate 7 locked invalid postmortem

The preregistered `gate78-shared-trace.v2` locked partition was consumed once
on 2026-07-30.  The twelve-file bundle is retained unchanged with
`verdict=invalid`.

## Observed result

- source admission, prefix-only future-leakage, token-space mutation and
  whole-cycle rollback gates passed;
- causal takeover passed for all three seeds;
- seed 701 retained the frozen action-family topology;
- seeds 709 and 719 each changed the full-arm structure fingerprint once;
- terminal-return and composition gains against both primary controls were
  exactly zero.

## Root cause

The temporal owner correctly guarded ordinary create/split/merge/prune
maintenance with `structure_frozen`, but called
`_anti_collapse_topology_maintenance()` unconditionally after classification.
The higher-support family learned on seeds 709 and 719 therefore crossed the
anti-collapse condition during the first causal prefix and was split despite
`structure_frozen=True`.

The owner fix gates anti-collapse maintenance behind both
`allow_topology_maintenance` and `not structure_frozen`, with an owner-local
regression test.  This is an implementation correction, not a threshold or
arm change.

## Evidence discipline

- The consumed v2 locked bundle was not regenerated or overwritten.
- No post-locked threshold, arm, metric or minimum effect was changed.
- Gate 7 receives no causal promotion from this campaign.
- A future confirmation requires a new preregistration and fresh locked
  source version; the current v2 rows may be used only as development data.
