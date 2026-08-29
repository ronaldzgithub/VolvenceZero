# lifeform-service

Product-layer skeleton: the **shell** that exposes the lifeform to users (HTTP / WS / CLI), enforces tenant isolation, and persists per-tenant state.

Today it ships:

- An `aiohttp` server with a small versioned API (`/v1/health`, `/v1/info`, `/v1/sessions`, `/v1/sessions/{id}/turns`, `/v1/sessions/{id}/end-scene`, `/v1/sessions/{id}/state`)
- A `SessionManager` with LRU + idle eviction, multi-tenant session isolation, and a single-shared-substrate hand-off
- A vertical registry that auto-discovers any installed `lifeform-domain-*` wheel

Run it via the `lifeform-serve` console script.

## Coding Brain API

Sessions created with `vertical="coding"` expose two additional routes:

```text
POST /v1/sessions/{session_id}/coding/context-packs
POST /v1/sessions/{session_id}/coding/outcomes
```

The first returns an ACTIVE memory-first Context Pack plus a separately marked
SHADOW advisor. The second accepts only the frozen test/review/merge enums from
`coding-outcome-report.v1`; unknown fields and invalid kind/source pairs fail
closed. Replaying the same id and payload is idempotent (`200`); the first
publication returns `201`; reusing an id with different content returns `409`.

Example request sequence:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/sessions/coding-1/coding/context-packs \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"req-1",
    "project_id":"project-1",
    "repository_id":"repo-1",
    "task_id":"task-1",
    "task_kind":"bugfix",
    "task_summary":"Fix checkpoint restoration",
    "target_paths":["src/state.py"]
  }'
```

Submit the resulting `context_pack_id` with a typed outcome. Deterministic
`task_verified/task_regressed` results accept only `test_suite`, `build_gate`,
or `ci`; review outcomes require `code_review`; merge/revert require `vcs`.
Closed-alpha coding sessions use the existing identity-scoped memory root, so a
new session for the same user can recall prior outcomes. The service retains
only bounded live idempotency lineage; Memory remains the cognitive owner.

## One Qwen, many tenants — substrate sharing

When you deploy on a single GPU server, every session must share **one** in-memory copy of the open-weight model. The service supports this directly:

```bash
# Default (no GPU, no model weights, fast tests):
lifeform-serve --vertical companion --substrate-mode synthetic

# One-GPU production deployment:
lifeform-serve \
  --vertical companion \
  --substrate-mode hf-shared \
  --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
  --substrate-device auto

# Companion Bench same-substrate ablation:
# one frozen runtime, six reviewed lifeform verticals selected with
# X-Compat-Vertical / ?vertical= on the OpenAI-compatible route.
lifeform-serve \
  --ablation-bundle \
  --substrate-mode hf-shared \
  --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
  --substrate-device cuda \
  --enable-openai-compat
```

The 1.5B model is the minimum product acceptance baseline for structured user-profile
fact extraction and recall. The 0.5B variant is suitable only for mechanism smoke
tests; it does not reliably satisfy the profile-memory output contract.

In `hf-shared` mode the model is **eagerly loaded once at service startup** and the same `TransformersOpenWeightResidualRuntime` Python object is passed into every Brain that the service constructs. Concurrent sessions take turns on it through the asyncio event loop's single-threaded execution model — `runtime.generate(...)` is a blocking torch call, so there is no parallelism inside one process and no need for an explicit lock. Throughput is "one decode in flight at a time"; if you need more, run multiple service processes (one model copy each) behind a load balancer, or graduate to a vLLM-backed runtime.

`/v1/info` reports `substrate_shared`, `substrate_model_id`, and `substrate_runtime_origin` so clients can see which deployment mode is live.

`--ablation-bundle` is intentionally narrower than general vertical discovery:
it registers only the reviewed Companion Bench verticals (`companion`,
`companion-cold`, and the four component arms) in one process. This keeps the
benchmark's owner boundary explicit while still sharing a single frozen model.

## Frozen-substrate invariant (R2)

A shared runtime **must** be frozen — `supports_live_substrate_mutation == False` (the default). Sharing a mutation-capable runtime would let one session's adapter-delta updates corrupt every other session's weights. `create_app(substrate_runtime=...)` enforces this at construction time and raises if the invariant is violated. If you genuinely need per-session adapter weights, the path forward is to refactor that mutable state out of the runtime and into the per-session `SubstrateAdapter`, not to flip the flag.

## What's still TODO

- WebSocket route for token-streamed responses
- Authentication, rate limiting, per-tenant session quotas
- Persistence (currently sessions live in-memory; restart loses them)
- General-purpose multi-vertical product hosting policy beyond the reviewed
  Companion Bench ablation bundle
- Optional vLLM-backed runtime for higher concurrent throughput

Future home of EmoGPT's `emoGPTservice/interface/*`, `emoGPTservice/tenant/*`, `emoGPTservice/persistence/*` and the `start_all_services.{ps1,sh}` / `supervisord.conf` deploy scripts.

Kept separate from `lifeform-core` so in-process embeddings (notebooks, benchmarks, tests) do not pull HTTP/DB dependencies.
