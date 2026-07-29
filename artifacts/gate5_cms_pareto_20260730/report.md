# Gate 5 CMS Pareto evidence

- status: `not-supported`
- mechanism passed: `True`
- causal passed: `False`
- settled transitions replayed per arm: `1530`
- locked partition consumed once: `True`

## Full vs controls

- `single-timescale`: absorption gain `0.000000`, retention gain `0.000001`, Pareto non-worse `True`
- `no-ATLAS-replay`: absorption gain `-0.000000`, retention gain `0.000000`, Pareto non-worse `True`
- `no-PE-write-gate`: absorption gain `0.000000`, retention gain `0.000000`, Pareto non-worse `True`
- `memory-only`: absorption gain `0.000003`, retention gain `0.000000`, Pareto non-worse `True`

## Claim boundary

- The primary metrics are owner-published band-drift proxies. Retrieval and payoff are diagnostics; this packet does not establish deployment-time behavioral memory.
- A failed Pareto/minimum-effect gate contracts the claim without retuning or rerunning the locked partition.
