# lifeform-service

Product-layer skeleton: the **shell** that exposes the lifeform to users (HTTP / WS / CLI), enforces tenant isolation, and persists per-tenant state.

Future home of EmoGPT's:
- `emoGPTservice/interface/*`
- `emoGPTservice/tenant/*`
- `emoGPTservice/persistence/*`
- `start_all_services.{ps1,sh}`, `supervisord.conf`, deploy scripts

Kept separate from `lifeform-core` so in-process embeddings (notebooks, benchmarks, tests) do not pull HTTP/DB dependencies.
