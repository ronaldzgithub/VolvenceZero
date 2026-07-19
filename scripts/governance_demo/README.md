# Governance Demo Driver

Drive a running `lifeform-service` with a `companion-bench` LLM-backed
user simulator so the chat UI's Governance Trace panel has signal to
render across multi-session arcs.

This is the **CLI sibling** of the chat UI's "Run Simulator" button.
Both paths use the same `companion_bench.user_simulator.UserSimulator`
internally; the CLI is for headless / scripted / evidence runs, the UI
button is for live investor demos.

## Quick start

1. Start the chat service with a real local substrate (any `userId` works):

   ```powershell
   .\start_browser_chat_qwen.ps1
   ```

   For a realistic synthetic user, also set the same OpenRouter key the
   protocol-uptake routes use:

   ```powershell
   $env:PROTOCOL_LLM_API_KEY = "sk-or-..."
   $env:PROTOCOL_LLM_PROVIDER = "openrouter"        # default
   $env:PROTOCOL_LLM_BASE_URL = "https://openrouter.ai/api/v1"
   $env:PROTOCOL_LLM_MODEL = "openai/gpt-4o-mini"
   ```

2. From another terminal, drive an arc:

   ```powershell
   python scripts\governance_demo\drive_session_arc.py `
     --base-url http://127.0.0.1:8765 `
     --user-id alice `
     --scenario F2-repair-002 `
     --paraphrase-seed 0 `
     --backend openrouter
   ```

3. Open `http://127.0.0.1:8765/chat`, type `alice` in the userId field,
   create a session — the Governance Trace panel (right side) shows the
   compounded `rupture_repair_count` / `observed_repair_count` /
   regime / PE signals that the CLI run produced for the same scope.

## What the driver does, in order

- Loads the scenario YAML (from the `companion-bench` wheel's
  `scenarios/public/` resource, or from a custom file path).
- Builds a deterministic per-session-turn schedule with PRNG seeded
  on `(scenario_id, paraphrase_seed)`; matches the schedule the
  in-server simulator route builds, so CLI and UI runs produce
  identical arc lengths for the same `(scenario, seed)` pair.
- For each arc session:
  1. `POST /v1/sessions` (binds `X-Alpha-User: <user>`)
  2. Loop turns: `UserSimulator.next_turn(...)` → `POST /v1/sessions/{sid}/turns`
     → `simulator.append_assistant(response_text)` → coloured console line
  3. `POST /v1/sessions/{sid}/end-scene` (drains slow loop, emits evidence ref)
  4. `GET /v1/users/me/relationship-summary` (prints rupture/repair banner)
  5. `DELETE /v1/sessions/{sid}` (so the simulator cache for that
     session evicts and the next arc-session gets its own scope)
- Writes a JSONL trace to
  `artifacts/governance_demo/<user_id>/<scenario>-<seed>-<ts>.jsonl`
  with one record per turn (full turn payload + governance fields +
  FSM step + bot reply).

## Cross-session governance — what to watch

Every time the driver bridges to a new arc-session it prints:

```
ruptures=3 repaired=2 kinds=scope,commitment
```

That line is the runtime materialisation of Slide 7's "repair primitive
across sessions" — and it is produced by **the same alpha mode
`MemoryStore`** the manual chat UI uses. Switch `--user-id alice` →
`--user-id bob` and observe that bob starts at `ruptures=0`. That is
the runtime materialisation of Slide 7's "non-transferable scope".

## Choosing a backend

| `--backend` | Behaviour |
| --- | --- |
| `openrouter` (default) | Reads `PROTOCOL_LLM_*` env vars, talks to OpenRouter by default. Hard-fail when no key. |
| `fake` | `companion_bench.user_simulator.DeterministicFakeUtteranceClient` — hash-derived utterances. Stable, free, looks like a robot. Use for CI / smoke tests. |
| `auto` | OpenRouter when keys are present, fake otherwise. Convenient but silent. |
| `qwen` | Compatibility alias for older commands; still reads `PROTOCOL_LLM_*`, so it can also route to OpenRouter when the provider env says so. |

## Recommended scenarios

| Scenario | Why it's good for the Governance demo |
| --- | --- |
| `F2-repair-002` | Cleanest rupture-repair arc; 4 sessions; produces the largest swing in `rupture_repair_count` between sessions. Use for the "repair is the primitive" demo. |
| `F1-continuity-001` | Establish-pattern -> callback_probe across 3 sessions; demonstrates "what accrues" (commitments / open_loops compounding). |
| `F4-long-absence-001` | 4 sessions with longer gaps; demonstrates "what decays" — open_loop counters and scene-close behaviour. |

## Notes

- The CLI driver imports `companion_bench` directly (Apache 2.0 API)
  and `lifeform_service.openai_utterance_client` (proprietary). The
  reverse direction — `companion_bench` importing internal wheels — is
  forbidden by `tests/contracts/test_companion_bench_no_internal_imports.py`.
- The driver does **not** call the in-server `/v1/sessions/{sid}/simulator/*`
  endpoints. Those exist for the chat UI's "Run Simulator" button.
  Both paths converge at the same `companion_bench.UserSimulator`
  semantics, so traces from either path are comparable.
