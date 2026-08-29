# Resource Plan

## Hardware and unchanged baseline observation

- Host: Apple arm64 laptop, 10 logical CPUs, 24 GiB memory.
- Storage at observation: 16 GiB available on the data volume; disk is at 97%
  capacity, so task outputs stay compact and concurrency is capped at two.
- Accelerator: not used; this evaluator is CPU, filesystem, git, and Python
  replay work. No CUDA handoff applies.
- Unchanged command: `python scripts/run_coding_lab_packet2.py smoke` with one
  chain, four episodes, digest budget 3500, seed 20260812, hand seed 11.
- Observed 2026-08-29: 12 arm-episodes completed in 8.21 s; user 6.61 s; sys
  1.29 s; maximum resident set size 88,276,992 bytes; all three smoke
  instrumentation verdicts passed.
- Interpretation: the command proves the real worktree, pytest, Brain, and
  MemoryStore path. It is not a performance baseline or memory-value result.
- Bottleneck: mixed CPU/filesystem process churn; memory and accelerator are not
  limiting at the observed scale.

## Evaluation cost and ranking convention

- Preliminary: one full 10-episode replay chain, coverage/effort `0.125`,
  observed 59.01 s on the repaired canary; non-ranking.
- Complete: eight full replay chains, 80 trajectory units, coverage/effort
  `1.0`. The canonical v2 legacy baseline completed in 928.65 s wall time with
  539,869,184-byte maximum RSS (`evaluator_wall_seconds=927.27`). Two complete
  lane fixtures run concurrently completed in 778.28 s and 779.01 s, at roughly
  0.5 GiB RSS per process. The 18-minute launch estimate is a conservative
  planning bound above the 15.48-minute worst observation; three observations
  are not enough to claim a statistical p90.
- Natural independent unit: one ordered ten-episode chain. Episodes within a
  chain are serial because each next recall depends on earlier owner state.
- Ranking: constrained Pareto. Maximize scaling margin and owner-entry selection
  coverage. Confirm only when context ratio is at most 0.10, selected owner
  entries are at least 75% of the returned set, selected failed entries are at
  least 95% of returned failures, every selected line is retained whole, and
  every rendered context respects its declared byte budget. Worst-chain margin
  is a robustness signal.
- Public replay is intentionally not formal quality evidence. Early-vs-formal
  calibration must compare public replay metrics with a later sealed rerun and
  must not mix preliminary rows into ranking statistics.

## Central scheduling

- Mode/profile: `central` / `context_replay`, CPU only.
- Initial/min/max total experiment concurrency: `2 / 1 / 2`.
- Default profile matches the only public evaluator.
- One peer may hold one experiment at a time. No core reservation or
  accelerator fallback is declared.
- Supply feedback: enabled after three low-pressure samples; lease response
  window 600 s; mature fraction 0.25; redundancy 3.0; minimum completion
  probability 0.25; one exploration slot reserved.
- Repeated identical infrastructure failures stop affected retries; valid weak
  scores and heterogeneous policy failures remain evidence.

## Run settings and close safety

- Cohort: 4; generations: 3; per-generation bound: 2.0 h.
- DIG: enabled only at absolute generation zero; 8 proposals, at least four
  mechanism families and three intervention surfaces.
- QD: independently enabled for generation zero and later PI synthesis.
- Constructive target: 0.75; forward slots: enabled; with cohort four, one
  diagnostic slot still leaves three constructive slots.
- Gems reset: disabled for continuous evolution.
- Mature quorum fraction: 0.25, therefore at least one of four peers must
  publish complete evidence for normal close.
- Launch guard: conservative complete estimate 18 min × 1.5 safety factor is
  27 min, below the 70 min adaptive ceiling minus a 30 min drain margin.
- Closing freezes all new evaluator or shell launches. Admitted process groups
  may drain; peers may only inspect, publish, and update notes afterward.

## Known limits

- Only the public 8-chain replay is runnable without a coding-hand provider.
- Formal sealed quality validation, handoff export, target adapter, SHADOW, and
  ACTIVE are later convergence packages and remain blocked by separate gates.
- The Praxist model connection is Codex-native saved-login auth; it is unrelated
  to the coding-hand API used by historical Packet 2 formal runs.
