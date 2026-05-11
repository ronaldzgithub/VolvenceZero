# companion-bench — Long-Session Companion Benchmark reference implementation

> Status: v1.0 reference implementation
> License: Apache 2.0
> RFC: [`docs/external/companion-bench-rfc-v0.md`](../../docs/external/companion-bench-rfc-v0.md)
> Public site: <https://companion-bench.org/>
> Previously circulated as **LSCB**; the wheel ships under `companion-bench` from v1.0 onward.

Companion Bench evaluates any OpenAI-compatible chat endpoint on multi-session
companion arcs across six axes:

| Axis | Weight | What it measures |
|---|---|---|
| A1 Task | 0.10 | Single-session helpfulness |
| A2 Conversational quality | 0.15 | Per-turn coherence, register, prosodic appropriateness |
| A3 Relational continuity | 0.25 | Cross-session memory accuracy, persona consistency, callback validity |
| A4 Adaptive learning | 0.20 | User-model adaptation across the arc |
| A5 Self-coherence | 0.10 | Identity stability, contradiction avoidance |
| A6 Safety / boundaries | 0.20 | Boundary maintenance under social pressure (hard-cap axis) |

## Design contract

* companion-bench is **system-agnostic**: it never imports any specific
  product wheel (no `volvence_zero.*`, no `lifeform_*`). The only
  surface it consumes is the OpenAI `/v1/chat/completions` HTTP
  contract. This is enforced by
  [`tests/contracts/test_companion_bench_no_internal_imports.py`](../../tests/contracts/test_companion_bench_no_internal_imports.py).
* companion-bench is **stateless across submissions**: each submission is
  one runner invocation; held-out scenarios are pulled from a private
  git submodule and never written to a public artifact path.

## Quickstart

```
pip install -e packages/companion-bench
companion-bench smoke
companion-bench list-scenarios
```

For real submissions:

```
python scripts/companion_bench/run_real_submission.py \
  --submission packages/companion-bench/examples/submission.yaml \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY \
  --artifact-dir artifacts/companion-bench/your-submission/
```

See [`examples/submission.yaml`](examples/submission.yaml) for the manifest schema.

## Repository structure

```
packages/companion-bench/
  src/companion_bench/
    spec.py                   # ScenarioSpec, hash, FSM step types
    user_simulator.py         # LLM-backed user + deterministic FSM
    lexicon.py                # Public name/occupation/contextual slots
    arc_runner.py             # Multi-session orchestration
    callback_ledger.py        # Fabricated-callback detector
    disqualifier.py           # Deterministic per-scenario rules
    judge_perturn.py          # 8-criterion 0-5 judge
    judge_arc.py              # 6-axis 0-100 judge (different model family)
    aggregator.py             # §6.4 weighted geometric mean + A6 cap
    elo.py                    # TrueSkill + Bradley-Terry
    verifier.py               # Reproducibility re-run
    cost.py                   # §6.7 cost telemetry
    cli.py                    # python -m companion_bench
    scenarios/public/         # 24 public scenarios (in-repo)
  tests/                      # Unit + contract tests
external/companion-bench-heldout/  # Private submodule (held-out, gitignored)
```

## Held-out scenarios

The held-out pool (RFC §3 P3, §8.6) lives in a separate private
repository that organisers pull in as a git submodule at
`external/companion-bench-heldout/` (legacy alias `external/lscb-heldout/`
is also accepted for git-history continuity). Without the submodule the
harness still runs against public scenarios only, so external contributors
and CI on public PRs are not blocked.
