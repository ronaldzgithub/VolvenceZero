# Interactive Cognition Entry

How an interactive turn enters the VolvenceZero cognition kernel, advances
state, publishes snapshots, and becomes observable. This is the end-to-end
"entry path" SSOT that ties together the two front doors, the session +
kernel layer, and the observability surfaces. It is deliberately narrow:
it does NOT re-specify the kernel's internal cognition (see
`cognitive-regime.md`, `prediction-error-loop.md`, `expression-layer.md`),
only how a turn gets *into* it and how operators *see* it afterwards.

Owner boundaries (R8 / R12): the platform layer (`dlaas-platform-api`,
`lifeform-openai-compat`) routes, dispatches, and reads-out. It never
re-derives cognition or reinterprets owner-internal fields; it only
forwards a typed input to the kernel and surfaces what the kernel
published.

## 1. Two front doors, one kernel

There are exactly two interactive entry points, and both terminate in the
same kernel call (`LifeformSession.run_turn` / the typed
`BrainSession.submit_*` dispatch). They differ only in wire shape and in
who calls them.

| | Native typed envelope | OpenAI-compat |
|---|---|---|
| Route | `POST /dlaas/v1/instances/{ai_id}/interactions` | `POST /v1/chat/completions` |
| Wire shape | `InteractionEnvelope` (typed) | OpenAI ChatCompletion |
| Handler | `app._handle_interaction` → `_dispatch_envelope_to_instance` | `lifeform_openai_compat.router._dispatch_lifeform` |
| Session id | `envelope.session_id` (required) | derived / explicit (`metadata.session_id`) |
| `ai_id` | path segment | `metadata["dlaas.ai_id"]` (opt-in) |
| Interaction kinds | chat / observe / report / feedback / teach / task / command | chat only (single user turn) |
| Cognition snapshot | always (`source="interaction"`) | when `ai_id` bound (`source="openai_compat"`) |

Both doors are equally first-class. The OpenAI door exists so external
chat clients (EQ-Bench, the OpenAI Python client, family-memorial /
einstein consumers) can talk to a lifeform without learning the typed
envelope. It is a *thin facade*: it never imports `dlaas-platform-api`,
and DLaaS never imports the OpenAI DTOs — they meet only through opaque
`aiohttp` app keys (see §4).

## 2. Native typed-envelope path

```
client → POST /dlaas/v1/instances/{ai_id}/interactions
       → _parse_envelope          (InteractionEnvelope.from_json; 400 on bad shape)
       → _dispatch_envelope_to_instance
            → operator-takeover / pause check
            → multi-pod forward  OR  local _resolve_session_manager
            → _get_or_create_session(session_id, end_user_ref)
            → dispatch_envelope    (switch on interaction_type → kernel)
       → _maybe_record_cognition_snapshot   (source="interaction")
       → _record_audit / _record_usage
       → 200 OutputAct  (or SSE frames when output_contract.stream)
```

Key contracts:

* `InteractionEnvelope` REQUIRES non-empty `contract_id`, `session_id`,
  `end_user_ref`, and a typed `interaction_type`. `_parse_envelope` does
  NOT auto-fill these; a body missing them is `400 invalid_envelope`.
  `contract_id` is used for audit + pause keying only — session resolution
  is by `session_id` + `end_user_ref`.
* `interaction_type` is a typed enum and the platform switches on it. It
  MUST NOT inspect `human_brief` to guess the type (R8 + no-keyword rule).
* `mode = apprentice` marks teach/task turns so vitals apply the
  apprentice override (drive deviation does not feed slow PE). See
  §5 (apprentice teaching).

## 3. OpenAI-compat path

```
client → POST /v1/chat/completions
       → _authorize_request           (optional Bearer)
       → ChatCompletionRequest.from_payload
       → _resolve_mode                (lifeform | raw)
       → [lifeform] _dispatch_lifeform
            → _resolve_lifeform_manager_for_request  (metadata["dlaas.ai_id"] → launcher)
            → lifeform_complete
                 → derive_session_id   (explicit | fresh | derived)
                 → _get_or_create_session
                 → session.run_turn(latest user message)
            → _invoke_on_turn_hook     (app["openai_compat_on_turn"], §4)
            → x-lifeform-* telemetry headers
       → 200 ChatCompletion  (or SSE re-emission when stream=true)
```

Session statefulness bridge: OpenAI is conceptually stateless (the client
re-sends history each call), the kernel is stateful per `session_id`. The
adapter derives a deterministic session id from
`(model, system_concat, first_user_message)` so a continuing arc reuses
one kernel session, while different openings map to different sessions.
Only the LATEST user message is sent to the kernel — the kernel's own
memory carries prior turns. `metadata.session_id` forces an explicit id.

Telemetry headers surface the turn's cognition without changing the
OpenAI body: `x-lifeform-regime`, `x-lifeform-abstract-action`,
`x-lifeform-pe-magnitude`, `x-lifeform-expression-intent`,
`x-lifeform-confidence` (kernel-PE calibrated; absent header = no PE
snapshot this turn — consumers must NOT fabricate a value).

## 4. Observability parity: the post-turn hook

Historically the OpenAI door advanced kernel state but wrote no
`CognitionSnapshot`, so `/cognition/health`, `/readouts`, and `/explain`
were blind to it. This is closed by an additive, decoupled seam:

* `lifeform_openai_compat.router` exposes an optional app key
  `app["openai_compat_on_turn"]` (`_ON_TURN_APP_KEY`). After a successful
  lifeform turn it calls, best-effort, `hook(request, *, ai_id, session,
  session_id)`. Absent key = no-op (a bare lifeform-service host).
* `dlaas-platform-api` registers `_record_openai_compat_snapshot` on that
  key inside `attach_dlaas_routes` / `attach_dlaas_full_stack`
  (`_register_openai_compat_cognition_hook`). It records a snapshot with
  `source="openai_compat"`, skipping silently when no `ai_id` is bound
  (nothing to attribute to — fabricating one would be dishonest).

Result: both doors write a per-turn `CognitionSnapshot` into the same
store, so every read-out below observes both uniformly. The wheel
boundary holds — the value on the app key is an opaque callable; neither
package imports the other's internals.

## 5. Apprentice teaching (online tuning)

Teaching is the same entry path with `interaction_type` in
`{teach, feedback, task}` and `mode=apprentice`:

* `teach` → apprentice `run_turn` (the kernel learns from the operator's
  demonstration).
* `feedback` → `submit_dialogue_outcome` (reward signal on the last
  response; carries a typed `FeedbackPayload`).
* `corpus` → reviewed ingestion via
  `POST /dlaas/v1/instances/{ai_id}/training/corpus` (NOT the interaction
  dispatch — corpus is reviewed knowledge, not a live turn).

Deploy-side surfaces: the `dlaas-portal` workshop console
(`/workshop/retrain/{ai_id}`) and the soul console detail page each build
a complete envelope through one shared helper
(`apps/dlaas-portal/.../interaction-envelope.ts`) and forward it — they
hold no cognition logic (R8).

## 6. Reading a turn back

| Surface | Route | What it shows |
|---|---|---|
| Readouts | `GET /dlaas/v1/instances/{ai_id}/readouts?view=summary\|full` | current published snapshots, summarized |
| Explain | `GET /dlaas/v1/instances/{ai_id}/explain?turn_index=latest\|<int>` | the decision chain for a turn |
| Raw snapshots | `GET .../snapshots` (admin auth) | redacted raw snapshot objects |
| Cognition health | `GET /dlaas/v1/cognition/health?ai_id=&window=` | health signals across recorded snapshots |
| Snapshot list | `GET /dlaas/v1/cognition/snapshots` | persisted per-turn snapshots (both sources) |

### `/explain` turn resolution (honest historical turns)

* `turn_index=latest` (default) → reads the kernel's *live* published
  snapshots and builds the full owner-internal chain
  (`_build_explain_chain`). Response `resolved.source = "live"`.
* `turn_index=<int>` → resolves a real recorded turn from the persisted
  cognition snapshots for this `(ai_id, session_id)`, oldest first.
  Non-negative is 0-based from the first recorded turn; negative counts
  from the end (`-1` = most recent recorded turn). The chain is rebuilt
  from the persisted readout bundle (`_explain_chain_from_readout`) and is
  honestly a *summary* projection, not a re-derivation of owner internals.
  Response `resolved.source = "historical"` with `captured_at_ms`,
  `snapshot_id`, `snapshot_source`, and `available_turns`.
* Out-of-range / no history → `404 turn_not_found`. The endpoint NEVER
  silently returns the latest live state mislabeled with a requested
  index.

## 7. Rollback

Every step here is additive and individually reversible:

* OpenAI snapshot parity: register a no-op hook before attaching DLaaS
  routes, or simply do not mount the OpenAI route — both restore the
  prior "OpenAI door is invisible to cognition reads" behaviour with no
  code change.
* `/explain` historical turns: passing only `turn_index=latest` reproduces
  the original live-only behaviour; the historical branch is reached only
  for an explicit integer index.
* Apprentice surfaces are flag-gated in the portal
  (`DLAAS_PORTAL_WORKSHOP_ENABLED`, `DLAAS_PORTAL_SOULS_ENABLED`).

## See also

* `dlaas-api-v1.md` — full DLaaS API surface (bake plane, third-party LLM,
  persona lifecycle).
* `dlaas-platform.md` — platform topology, auth, and store ownership.
* `cognitive-regime.md`, `prediction-error-loop.md`, `expression-layer.md`
  — the kernel internals this path feeds.
