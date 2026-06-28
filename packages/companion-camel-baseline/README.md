# companion-camel-baseline

A vendor-neutral **agent-framework baseline** for Companion Bench.

It boots an OpenAI-compatible `POST /v1/chat/completions` endpoint that wraps
an upstream substrate with the [CAMEL](https://github.com/camel-ai/camel)
open-source agent framework (a `ChatAgent` plus a cross-session memory layer).
The point is fairness: when you compare a cognitive system on a multi-session
companion benchmark, the "off-the-shelf agent" column should be a *real agent
framework on the same substrate*, not bare `/v1/chat/completions` (which has no
cross-session memory and therefore scores the absence of memory, not the model).

This wheel is the third row of the same-substrate ablation described in
`docs/specs/companion-ablation.md`:

| Track | Layer over the frozen substrate |
|---|---|
| `raw` | nothing (bare model) |
| `ref-harness` | minimal memory wrapper (`companion-ref-harness`) |
| **`camel`** | **standard open-source agent framework (this wheel)** |
| `volvence-cold` | full cognitive pipeline, no trained bootstraps |
| `volvence` | full cognitive pipeline + trained bootstraps |

All five rows MUST point at the **same frozen substrate weights**. In practice
the substrate is served once (e.g. `lifeform-serve --enable-openai-compat
--substrate-mode hf-shared`) and this wheel's `--upstream-base-url` points at
that endpoint's `?mode=raw` path. That guarantees byte-identical weights across
tracks, so any score delta is attributable to the layer, not the substrate.

## Design rules

- **Same substrate, fair fight.** The CAMEL backend's LLM is configured to call
  the same upstream model as the `raw` track. There is no implicit fallback to
  GPT-4 / any other model. If the upstream is misconfigured the server fails
  loud (no silent substitution).
- **Isolation.** This wheel never imports `volvence_zero.*`, `lifeform_*`,
  `companion_bench.*`, or `companion_ref_harness.*`. Enforced by
  `tests/test_no_internal_imports.py`.
- **Shape-indistinguishable.** Responses carry no `x-camel-*` / `x-volvence-*`
  / `x-companionbench-*` headers; the body is exactly the upstream
  `chat.completion` envelope so the Companion Bench runner cannot tell a wrapped
  endpoint from a raw one.
- **No swallowed errors.** Upstream HTTP failures map to `502
  camel_baseline_upstream_error`; backend-internal failures map to `502
  camel_baseline_internal`. Invalid request bodies map to `400`.
- **Prompts inline.** Any prompt the memory compactor sends is defined as a
  module-level constant in source, never hidden in a remote config.

## Cross-session memory model

CAMEL's `ChatAgent` keeps an in-context memory window *within* a session. For
*cross-session* continuity we persist a compact per-session memory record into
a SQLite store keyed by `scope_key = metadata.user_id`. At the start of a new
session for the same user, the prior records are re-seeded into the agent's
system context. Session boundaries are detected exactly as `companion-ref-harness`
does:

1. Explicit: `POST /v1/sessions/{session_id}/close`.
2. Lazy: when a new `session_id` arrives for a user, prior open sessions of that
   user are compacted first.

Cross-user isolation is structural: records are keyed by `scope_key`, and one
user's records are never seeded into another user's context.

## Backends

- `EchoCamelBackend` — deterministic, no network, no `camel-ai` dependency. Used
  by unit tests and `--backend echo` smoke runs. It demonstrably carries
  cross-session memory so the plumbing can be asserted end-to-end.
- `CamelChatAgentBackend` — production backend. Lazily imports `camel-ai`
  (optional `[camel]` extra) and configures a `ChatAgent` whose model is an
  OpenAI-compatible client pointed at the upstream substrate.

## Boot

```bash
# Deterministic smoke (no keys, no camel-ai):
companion-camel-baseline serve --backend echo --store-mode memory --port 8600

# Real CAMEL agent on the same Qwen the raw track uses:
companion-camel-baseline serve \
    --backend camel \
    --upstream-base-url http://127.0.0.1:8000/v1?mode=raw \
    --upstream-model lifeform-raw \
    --upstream-key-env LIFEFORM_LOCAL_API_KEY \
    --store-mode sqlite \
    --store-path ./companion-camel-baseline.sqlite3 \
    --port 8600
```

Then point the Companion Bench runner's `base_url` at `http://127.0.0.1:8600/v1`.
