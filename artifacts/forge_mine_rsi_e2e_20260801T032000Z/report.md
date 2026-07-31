# Forge Failure Mining Report

- Created: 2026-07-31T18:52:08.984062Z
- Analysis backend: `replay` / `rsi-e2e-reviewed-replay-v1`
- Embedding model: `moka-ai/m3e-base`
- Inputs: 93 transcripts, 1 verdicts, 29 plans
- Explicit failed sources: 11
- Failure patterns: 2

## Failure patterns

### fp_84eabb42585228d0

- Occurrences: 10
- Surface: `in-surface` → `.cursor/rules/cursor-convergence-workflow.mdc` (similarity=0.810)
- Verifier cause: The transcript ended with an explicit structured tool or runtime error.
- Agent behavior: The development agent did not recover to a successful terminal state after the failed tool sequence.
- Mechanism: Long-horizon workflow recovery lacks a bounded evidence-preserving retry and handoff mechanism.

### fp_9678e251a9209250

- Occurrences: 1
- Surface: `in-surface` → `forge/prompts/failure_mining.system.md` (similarity=0.807)
- Verifier cause: A preregistered Gate 2 promotion criterion remained false and promotion_allowed was false.
- Agent behavior: The evidence run completed but did not establish the required causal identity effect against controls.
- Mechanism: The development evidence workflow needs to preserve negative gate results and route them into a bounded next hypothesis without weakening the verifier.

## Prediction checks

- No previously applied proposal is awaiting longitudinal comparison.
