# lifeform-service

Product-layer skeleton: the **shell** that exposes the lifeform to users (HTTP / WS / CLI), enforces tenant isolation, and persists per-tenant state.

Today it ships:

- An `aiohttp` server with a small versioned API (`/v1/health`, `/v1/info`, `/v1/sessions`, `/v1/sessions/{id}/turns`, `/v1/sessions/{id}/end-scene`, `/v1/sessions/{id}/state`)
- A `SessionManager` with LRU + idle eviction, multi-tenant session isolation, and a single-shared-substrate hand-off
- A vertical registry that auto-discovers any installed `lifeform-domain-*` wheel

Run it via the `lifeform-serve` console script.

## One Qwen, many tenants — substrate sharing

When you deploy on a single GPU server, every session must share **one** in-memory copy of the open-weight model. The service supports this directly:

```bash
# Default (no GPU, no model weights, fast tests):
lifeform-serve --vertical companion --substrate-mode synthetic

# One-GPU production deployment:
lifeform-serve \
  --vertical companion \
  --substrate-mode hf-shared \
  --substrate-model-id Qwen/Qwen2.5-0.5B-Instruct \
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
