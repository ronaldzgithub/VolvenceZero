# companion-ref-harness — Reference Companion Harness for CompanionBench

> Status: H-A SHADOW (v0.1) — packet [`docs/moving forward/companion-ref-harness-packet.md`](../../docs/moving%20forward/companion-ref-harness-packet.md)
> License: Apache 2.0
> Public site (planned): <https://companionbench.com/ref-harness>

## What this is

`companion-ref-harness` is a thin, vendor-neutral **agent wrapper** that sits
between a CompanionBench runner and any OpenAI-compatible chat completions
endpoint. It exposes its own `POST /v1/chat/completions` route on
`http://localhost:8500/v1/chat/completions` (default port), so a
CompanionBench submission YAML can point at the harness and the harness
forwards each turn to the upstream substrate (GPT-5, Claude Opus, Gemini,
DeepSeek, Llama, Qwen, ...) with **cross-session memory blended into the
prompt**.

## What this is NOT

* **NOT a production-grade companion AI framework**. This wheel is a
  benchmark reference baseline, not a deployable system.
* **NOT a LangChain / Dify / agent-builder SDK**. There is no plugin
  registry, no quickstart-app template, no developer onboarding. For
  production deployments see VolvenceZero Lifeform (separate product).
* **NOT a way to game CompanionBench scores**. The algorithm, prompts,
  and storage schema are all open. Submissions that override the
  open algorithm without declaring it violate the submission
  protocol's `harness_attestation` red line.

## Why it exists

CompanionBench evaluates multi-session companion behaviour. A raw
`/v1/chat/completions` request has **no cross-session memory** by
contract — session N+1 starts with an empty transcript. This means
the `closed-api` track of the leaderboard, by default, measures
"raw API has no memory" rather than "is this model a good companion
substrate".

This wrapper closes that operational gap. Every closed-api substrate
gets the same minimal agent infrastructure (4 optional components
listed below), so the comparison isolates substrate contribution
from agent-infra contribution.

Background reading:

* [`docs/external/companion-bench-rfc-v0.md`](../../docs/external/companion-bench-rfc-v0.md) §7.4 — leaderboard category
* [`docs/moving forward/companion-ref-harness-packet.md`](../../docs/moving%20forward/companion-ref-harness-packet.md) §1 — problem statement

## Components

| Component | H-A | H-B | H-C | What it does |
|---|---|---|---|---|
| `summary` | yes | yes | yes | Per-session structured summary, injected at the head of the next session as a system-message prefix |
| `embed` | no | yes | yes | OSS embedding-based retrieval (BGE family by default) over prior turns, injected as a markdown block before the current user turn |
| `user_model` | no | no | yes | LLM-extracted typed key-value dict of user facts (occupation, preferences, boundaries, ...), injected at session head |
| `episodic` | no | no | yes | LLM-extracted typed events with source-turn anchors, injected alongside retrieval block |

Each component is **independently togglable** via the `--components`
CLI flag:

```
--components                                 # passthrough (no blend)
--components summary                         # H-A default
--components summary,embed                   # H-B default
--components summary,embed,user_model,episodic  # H-C full
```

This is the SSOT for ablation: the same wheel, same upstream,
same prompts, different component sets — the difference in axis
scores is exactly the marginal contribution of each component.

## Quickstart

```
pip install -e packages/companion-ref-harness
```

Boot a harness server that forwards to OpenAI GPT-5 with the
`summary` component on (H-A default):

```
export OPENAI_API_KEY=sk-...
companion-ref-harness serve \
    --upstream-base-url https://api.openai.com/v1 \
    --upstream-model openai/gpt-5 \
    --upstream-key-env OPENAI_API_KEY \
    --port 8500 \
    --components summary
```

Boot in passthrough mode (no blend — useful for sanity-checking that
the harness does not corrupt requests):

```
companion-ref-harness serve \
    --upstream-base-url https://api.openai.com/v1 \
    --upstream-model openai/gpt-5 \
    --upstream-key-env OPENAI_API_KEY \
    --port 8500 \
    --components ""
```

A CompanionBench submission YAML then points at the harness instead
of directly at the substrate:

```yaml
base_url: http://localhost:8500/v1
model_identifier: openai/gpt-5
api_key_env: OPENAI_API_KEY    # the harness forwards this key upstream
```

## How session boundaries are detected

The harness recognises a session boundary in one of three ways
(checked in order):

1. **Explicit close**: client calls `POST /v1/sessions/{id}/close`.
2. **Lazy on new session**: a request arrives whose `metadata.session_id`
   differs from the most-recent `session_id` for the same
   `metadata.user_id`. Any older sessions of that user get their
   summary extracted on first need.
3. **Inactivity timeout**: a session inactive longer than
   `--session-inactivity-seconds` (default `300`) is closed and
   summarised on the next request from the same user.

Session boundaries trigger a single LLM call to the
`--summary-extractor-base-url` / `--summary-extractor-model` upstream
(defaults to the same as `--upstream-*` if not overridden).

**Cross-family default recommendation**: if your upstream is
GPT-5, set `--summary-extractor-base-url` to Claude / Gemini /
Qwen so the summary extractor and the SUT are different model
families. This mirrors the CompanionBench RFC v0.2 cross-family
judge protocol and avoids "the substrate writes its own
crib-notes" criticism.

## Scope keys

Cross-session memory is grouped per user, not per session. The
scope key is derived as:

```
scope_key = metadata.user_id            (if present)
         or X-Companion-User-Id header  (fallback)
         or "anon:" + hash(headers)     (last-resort surrogate)
```

The `(scope_key, session_id)` pair is the primary key of every
storage table.

## Reproducibility contract

* All prompts (summary extractor, retrieval block format, ...) are
  defined **inline** in the corresponding Python module's top-level
  docstring. No prompt lives in a binary file or a remote config.
* The default embedding model is OSS-only (BGE family); using a
  closed-API embedding service violates the contract.
* The SQLite store schema is documented in
  [`src/companion_ref_harness/store/sqlite_store.py`](src/companion_ref_harness/store/sqlite_store.py).
* The harness does **not** carry any `x-lifeform-*`, `x-volvence-*`,
  or `x-companionbench-*` headers in its response, so it is shape-
  indistinguishable from a raw OpenAI-compat endpoint to the
  CompanionBench runner.

## Repo boundary

`companion_ref_harness.*` does **not** import:

* `volvence_zero.*`
* `lifeform_*` (any sub-package)
* `companion_bench.*`

This is enforced by
[`tests/contracts/test_companion_ref_harness_no_internal_imports.py`](../../tests/contracts/test_companion_ref_harness_no_internal_imports.py).

## Testing

```
pip install -e 'packages/companion-ref-harness[test]'
pytest packages/companion-ref-harness/tests/
pytest tests/contracts/test_companion_ref_harness_no_internal_imports.py
pytest tests/contracts/test_apache_license_header_present.py
```

All tests run without network access. Real upstream calls happen
only when the CLI `serve` command is invoked with real env vars.
