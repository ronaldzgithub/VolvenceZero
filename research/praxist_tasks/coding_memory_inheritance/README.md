# Coding Memory Inheritance Praxist Task

This task is the first Volvence Research Control Plane pilot. It researches one
owner-scoped artifact: a declarative policy for composing coding-memory context.

## Public evaluator

From the task root:

```bash
../../../.venv/bin/python evaluations/context_replay/run.py \
  --variant-dir assets/baseline \
  --output-dir experiments/baseline_preflight \
  --mode preliminary
```

Use `--mode complete` only for all eight public development chains. A valid low
score exits successfully and remains useful negative evidence. Candidate schema,
frozen corpus, evaluator, audit rules, and task prompts are protected. A variant
directory contains `policy.json` only. Parent-eligible development evidence must
select at least 75% of recalled entries and 95% of failed entries, retain every
selected line whole, and publish the exact selection/retention ratios.

Substantial complete evaluations should be submitted through Praxist's central
`protected_pids launch` facade with profile `context_replay`, a stable semantic
tag, and a result-specific output directory. Runtime task notification and exit
status are completion facts; `tasks/<task-id>.output` byte count is not.

## Evidence boundary

The source trajectories are public to peers. A complete result is mature only
for Praxist development retention. It does not rerun the coding hand and cannot
prove coding quality, authorize formal validation, or change runtime wiring.
Praxist owns frontier/incubator/Gems/memory/generation state; task code publishes
only evaluator summaries and findings.

Current facts are the canonical evaluator summary, structured findings,
`frontier/frontier_manifest.json`, committed `gems/gems_state.json`, and
`gen_N/generation_boundary.json`. Reports, leaderboards, PI packs, and rendered
prompts are derived views. An empty incubator despite complete, clean,
evidence-preserving policies is a task-lane defect worth diagnosing.

## Codex-native operator profile

The pilot profile is fixed to `agent_runtime:codex_sdk`,
`model_provider:openai_compatible`, and `gpt-5.6-luna`, using saved ChatGPT login
with API-key use disabled. Resolve and start must sanitize `OPENAI_API_KEY`,
`CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, `OPENAI_BASE_URL`, `PRAXIST_CODEX_BIN`,
`MODEL`, and `PRAXIST_MODEL` in that subprocess. No credential belongs in this
task. A0 approval remains separately required before start.
