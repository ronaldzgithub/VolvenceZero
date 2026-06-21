# DLaaS API v1

> Status: SHADOW contract
> Last updated: 2026-06-12
> Owner: `dlaas-platform-*`
> Related specs: `dlaas-platform.md`, `environment-interface.md`, `protocol-runtime.md`, `multi-timescale-learning.md`

## Purpose

DLaaS v1 exposes Volvence lives as externally addressable `ai_id` instances while preserving the core ownership split:

- `vz-*` owns cognition, memory, temporal control, substrate contracts, and runtime snapshots.
- `lifeform-*` adapts external events into the lifeform facade.
- `dlaas-platform-*` owns tenants, adoption contracts, instance lifecycle, protocol/training intake, ops, and HTTP routing.

The API has four planes:

1. **OpenAI-compatible chat** for broad SDK compatibility.
2. **Native runtime envelope** for typed chat / observe / feedback / teach / task / report / command.
3. **Adoption contract** for choosing vertical, substrate profile, protocol set, memory scope, tools, ops, and training policy.
4. **Protocol/training intake** for reviewed protocols, reviewed corpus ingestion, and gated rare-heavy artifact jobs.

## Invariants

- OpenAI-compatible requests must remain valid OpenAI chat completions requests; DLaaS metadata is optional.
- `InteractionEnvelope.interaction_type` is the only dispatch key. No natural-language keyword routing.
- Adoption choices are versioned contract state, not per-turn options.
- Protocols may hot-load after review; corpus enters through ingestion/reviewed event sinks; rare-heavy artifacts require offline gates and are not online base-model updates.
- Wake/sleep/status are launcher/platform lifecycle state. They do not become cognitive owner state.
- Feedback becomes typed outcome evidence; evaluation readouts do not write learning state.

## OpenAI-Compatible Surface

```http
POST /v1/chat/completions
```

Request remains OpenAI-shaped. Optional DLaaS metadata:

```json
{
  "model": "growth_advisor/cheng-laoshi",
  "messages": [{"role": "user", "content": "我今天有点焦虑"}],
  "metadata": {
    "session_id": "sess_001",
    "user_id": "wechat_user_abc",
    "dlaas.ai_id": "ai_cheng_laoshi_001",
    "dlaas.contract_id": "contract_123",
    "dlaas.end_user_ref": "wechat_user_abc"
  }
}
```

If `metadata["dlaas.ai_id"]` is present and the host app has an `InstanceManager`, the request routes to that `ai_id`'s `SessionManager`. Otherwise it follows the legacy single-service path.

### OpenAI Tool Calling

OpenAI-compatible tool calling is implemented as an adapter over the native
affordance loop:

- Request fields `tools`, `tool_choice`, and `parallel_tool_calls` are parsed
  when tool calling is enabled.
- Assistant messages may carry `tool_calls`; `tool` role messages carry
  `tool_call_id`, `name`, and `content`.
- Client-loop mode returns `finish_reason="tool_calls"` with assistant
  `tool_calls` and waits for the client to send `role="tool"` results.
- Server-loop mode invokes the selected affordance in-process through
  `ToolLoopOrchestrator` and returns the final assistant text unless the loop
  stops for confirmation, budget, or async task handoff.
- Tool results, whether supplied by OpenAI `role="tool"` or native DLaaS
  `observe/tool_result`, map to `LifeformSession.submit_tool_result(...)`.

`mode=raw` remains text-only unless the substrate runtime exposes a separate
tool-aware generation API.

## Native Runtime Envelope

Canonical route:

```http
POST /dlaas/v1/instances/{ai_id}/interactions
```

Compatibility alias:

```http
POST /dlaas/instances/{ai_id}/interactions
```

Envelope:

```json
{
  "protocol_version": "dlaas/v1",
  "contract_id": "contract_123",
  "session_id": "sess_001",
  "end_user_ref": "wechat_user_abc",
  "interaction_type": "chat",
  "mode": "live",
  "human_brief": "用户说他今天很焦虑",
  "target_person_ids": [],
  "lang": "cn",
  "structured_context": {},
  "output_contract": {
    "delivery_channel": "wechat",
    "format": "text",
    "stream": false
  }
}
```

Supported `interaction_type` values:

| Type | Runtime sink |
|---|---|
| `chat` | `LifeformSession.run_turn(USER_INPUT)` |
| `observe` | typed observation to reviewed event / ingestion / tool-result sinks |
| `feedback` | `LifeformSession.submit_dialogue_outcome(...)` |
| `teach` | apprentice-triggered `run_turn` |
| `task` | apprentice/task event path |
| `report` | scene closure + readout |
| `command` | typed allowlist only |

Native runtime responses can include tool-related output acts:

| Act type | Meaning |
|---|---|
| `tool_call` | Client should execute the described tool and return an `observe/tool_result` event. |
| `tool_task` | A long-running tool task was queued/running and should be polled or resumed later. |

These acts are platform delivery wrappers only. They do not create a second
tool owner; invocation outcomes still enter cognition through the canonical
environment/tool-result path.

### SSE response (`output_contract.stream=true`)

When the caller sets `output_contract.stream=true` and the interaction type is
streamable (`chat` / `teach` / `task`), the platform answers
`Content-Type: text/event-stream` instead of JSON
(`dlaas_platform_api/streaming.py`, partial close of known-debt #12). Frame
order:

```text
event: ack    -> {"ai_id", "session_id", "contract_id", "interaction_type"}
event: chunk  -> {"content": "<text segment>"}      (0..n, ordered)
event: act    -> one OutputAct JSON object          (1..n, structured authority)
event: done   -> full non-streaming JSON body       (terminal on success)
event: error  -> {"error", "detail", "status"}      (terminal on failure)
```

Contract invariants:

- `ack` is written **before** the kernel turn runs (immediate accept signal).
- Concatenated `chunk` contents equal the final text of every
  `act_type="text"` OutputAct, in order. Today the kernel produces turn text
  atomically (substrate streaming hooks are still debt #12), so chunks are
  platform-layer segments of the completed text; when substrate token hooks
  land, chunks become genuine increments without a wire change.
- `done` carries the exact body the JSON path would have returned (including
  the `extra` cognition readout below), so consumers persist from one shape.
- A typed `DispatchError` after `ack` becomes a terminal `error` frame —
  never a silent EOF. Errors raised before dispatch (envelope parse, unknown
  `ai_id`, paused session takeover) still return plain JSON.
- Non-streamable interaction types degrade silently to JSON per the
  `OutputContract` best-effort clause; clients must branch on the response
  `Content-Type` (see `DLAAS_README.md` §"Browser SSE Reader For
  Interactions").

### Interaction response `extra` (cognition readout)

Every `chat` / `teach` / `task` / `initiate_proactive_followup` response carries
an `extra` block. Beyond the response text in `output_acts`, the platform
surfaces the published cognitive state so a product client can drive the
visible avatar + relationship UI without a second round-trip:

| `extra` field | Source snapshot | Meaning |
|---|---|---|
| `active_regime` | turn result | Active regime id (persistent regime identity, R14). |
| `active_abstract_action` | turn result | Active ETA abstract action, when present. |
| `rationale_tags` | response | Response rationale tag list. |
| `expression_intent` | `response_assembly` | Published expression intent (e.g. `support-first`, `direct-answer`). Drives presence `PerformancePlan` when the client forwards it as `ExpressionIntentInput.source="lifeform"`. |
| `prediction_error` | `prediction_error` | Compact `{magnitude, relationship, task}` projection. Canonical source is the kernel snapshot's nested `error` axes (`error.magnitude` / `error.relationship_error` / `error.task_error`); flat-attribute values are accepted as a legacy fallback. Numeric axes only when present. |
| `confidence` | `prediction_error` | Kernel-calibrated forward-looking confidence `[0, 1]` (`next_prediction.confidence`, the PE owner's own readout). The single legal kernel-PE confidence source for a deploy-side collaboration gate (deploy debt `D-collab-pe`). Absent when no PE snapshot was published — the platform never fabricates it. |
| `confidence_origin` | `prediction_error` | Always `"kernel_pe"` when `confidence` is present; lets the consumer's gate distinguish a real kernel signal from its own `structural` / `fallback` literals. |
| `relationship_brief` | `relationship_state` | Publisher-authored `description` string only — never raw owner internals (same redaction posture as `/readouts`). |
| `plan_brief` | `plan_intent` | Compact task-progress projection: `{"active_goal": str, "active_step": str, "plan_revision_count": int, "continuity_score": float}`. `active_goal` / `active_step` are the owner-published free-text summaries (same exposure level as `relationship_brief`); the raw plan SemanticRecord lists are never surfaced here. |
| `open_loop_brief` | `open_loop` | Open-loop aggregates: `{"unresolved_count": int, "pending_confirmation_count": int, "closure_readiness": float, "stale_loop_count": int}`. Counts only — never the raw loop records. |
| `commitment_brief` | `commitment` | Commitment aggregates: `{"active_count": int, "at_risk_count": int, "due_followup_count": int, "stalled_count": int}`. Counts only — never the raw commitment records. |

The OpenAI-compat adapter mirrors the same kernel-PE confidence on the
`x-lifeform-confidence` response header (absent when unpublished), alongside
the existing `x-lifeform-pe-magnitude` / `x-lifeform-regime` /
`x-lifeform-expression-intent` telemetry headers.

The cognition fields are best-effort: they are present when the matching
snapshot was published for that turn, and absent otherwise. The chat turn is
load-bearing and never fails because a snapshot is missing. Clients MUST treat
every cognition field as optional. Full structured slots (e.g. the raw
`relationship_state` / `dual_track` owners) remain available only on the
admin snapshot-export route, per the readout-vs-raw split below.

## Adoption Contract

```http
POST /dlaas/v1/adoptions
```

`POST /dlaas/adopt` remains the compatibility route.

The adoption payload freezes the class of life being adopted:

```json
{
  "tenant_id": "tenant_growth_brand_001",
  "template_id": "template_growth_advisor_cheng_v1",
  "contract_id": "contract_123",
  "owner_user_id": "operator_001",
  "adoption_config": {
    "vertical": {
      "vertical_id": "growth_advisor",
      "runtime_template_id": "growth_advisor",
      "profile_id": "cheng_laoshi"
    },
    "substrate": {
      "substrate_profile_id": "qwen3-max-shared",
      "mode": "shared_frozen",
      "adapter_policy": "none",
      "allow_rare_heavy_refresh": false
    },
    "protocols": {
      "autoload": ["growth_advisor:cheng-laoshi"],
      "library_ids": ["customer:no-hard-sell-v1"],
      "activation_policy": "explicit_load",
      "allow_runtime_upload": true,
      "review_level_required": "L3"
    },
    "memory": {
      "scope_strategy": "tenant_ai_end_user",
      "retention_policy_id": "pipl-30d-pilot",
      "deletion_policy_id": "default-pilot-delete"
    },
    "tools": {
      "tool_policy_id": "growth-advisor-wechat-readonly",
      "allowed_capabilities": ["text", "handoff_ticket", "reviewed_knowledge"]
    },
    "ops": {
      "awake_strategy": "on_demand",
      "idle_sleep_seconds": 1800,
      "handoff_policy_id": "growth-advisor-standard",
      "pause_on_handoff": true
    },
    "training": {
      "allow_protocol_intake": true,
      "allow_corpus_intake": true,
      "allow_adapter_training": false,
      "promotion_gate": "reviewed_protocol_only"
    }
  }
}
```

During SHADOW rollout, this config is stored inside `ContractSpec.service_contract["adoption_config"]`. A later ACTIVE migration may split it into a dedicated `AdoptionSpec` table.

## Protocol And Training Intake

### Asset/File Intake

```http
POST /dlaas/v1/instances/{ai_id}/assets/intake
GET  /dlaas/v1/instances/{ai_id}/assets/intake/{asset_id}
```

This is the DLaaS entry for uploaded files, books, images, and inline
materials. The route accepts a typed JSON payload:

```json
{
  "contract_id": "contract_123",
  "session_id": "intake_sess_001",
  "end_user_ref": "operator_001",
  "intake_intent": "auto",
  "media_kind": "pdf",
  "title": "Customer Playbook",
  "mime_type": "application/pdf",
  "source_ref": "upload://customer-playbook.pdf",
  "content_base64": "...",
  "metadata": {"reviewed": true}
}
```

`intake_intent` values:

| Intent | Behavior |
|---|---|
| `storage_only` | Store platform asset metadata/reference only. No kernel turn. |
| `simple_ingest` | Parse text/PDF/DOCX and run `IngestionPipeline` now. |
| `deep_read` | Create a long-running corpus/deep-reading job. |
| `training_candidate` | Create an offline training/adapter candidate job. Promotion is gated. |
| `image_intake` | Store image and mark it pending a vision extractor. No image owner mutation. |
| `auto` | Use a schema-constrained intake classifier; no keyword routing. |

The route never writes cognitive owners directly. Text/PDF/DOCX ingestion goes
through `lifeform-ingestion`; image intake stays asset-only until a reviewed
vision extractor is wired; training candidates remain offline and gated.

### Protocol Submission

```http
POST /dlaas/v1/instances/{ai_id}/protocols/submissions
GET  /dlaas/v1/instances/{ai_id}/protocols/submissions
POST /dlaas/v1/instances/{ai_id}/protocols/submissions/{submission_id}/approve
POST /dlaas/v1/instances/{ai_id}/protocols/submissions/{submission_id}/reject
GET  /dlaas/v1/instances/{ai_id}/protocols/library
POST /dlaas/v1/instances/{ai_id}/protocols/library/{protocol_id}/load
POST /dlaas/v1/instances/{ai_id}/protocols/library/{protocol_id}/unload
```

`source_type` may be `pdf`, `markdown`, `text`, or `json_payload`. Approved submissions enter the reviewed protocol library and can be loaded into the instance's active set.

### Corpus Intake

```http
POST /dlaas/v1/instances/{ai_id}/training/corpus
```

Corpus intake is for reviewed knowledge/case/profile material. It must map to `observe` / `corpus_ingest` / reviewed event sinks and must not directly mutate memory stores.

### Training Jobs

```http
POST /dlaas/v1/instances/{ai_id}/training/jobs
GET  /dlaas/v1/instances/{ai_id}/training/jobs/{job_id}
POST /dlaas/v1/instances/{ai_id}/training/jobs/{job_id}/cancel
POST /dlaas/v1/instances/{ai_id}/training/jobs/{job_id}/promote
```

Job types:

- `protocol_extraction`
- `protocol_revision`
- `corpus_ingestion`
- `adapter_candidate`
- `eval_only`

Promotion:

- Protocol jobs promote to approved protocol library after review.
- Corpus jobs promote through ingestion/reviewed event sinks.
- Adapter candidates require offline gate evidence and produce versioned artifact references. They do not hot-update the frozen substrate.
- Eval-only jobs never promote into learning state.

## Lifecycle

```http
POST /dlaas/v1/instances/{ai_id}/wake
POST /dlaas/v1/instances/{ai_id}/sleep
GET  /dlaas/v1/instances/{ai_id}/status
GET  /dlaas/v1/instances
```

Lifecycle states: `provisioning`, `asleep`, `waking`, `awake`, `sleeping`, `paused`, `suspended`, `failed`.

Wake may acquire an instance if the request includes a `runtime_template_id`; otherwise it marks an already-adopted instance awake.

Sleep changes platform lifecycle state. It may optionally release the instance mapping when `release_instance=true`; the default keeps the manager available for fast re-wake.

## Convenience Environment And Feedback Routes

These are adapters to `InteractionEnvelope`, not separate dispatch owners:

```http
POST /dlaas/v1/instances/{ai_id}/environment/events
POST /dlaas/v1/instances/{ai_id}/environment/outcomes
POST /dlaas/v1/instances/{ai_id}/feedback
```

The dispatcher remains the SSOT for runtime routing.

## Tool Task Lifecycle

Long-running affordances use task handles instead of blocking the chat turn.
The task store is owned by the session-scoped `AffordanceInvoker`
(`lifeform_affordance.invoker`); these routes are thin HTTP adapters over
`get_task_handle` / `submit_deferred_result` and never re-implement task
state (`dlaas_platform_api.tool_tasks`).

```http
GET  /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}?session_id={session_id}
POST /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}/complete
```

Task status values: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
`session_id` is mandatory (query param on GET, body field on POST) because
the invoker holding the task handle is reached through the session; an
unknown session is a typed 404, never a silent create. Auth is the same
level as the interactions runtime route.

`GET` response (200):

```json
{
  "task_id": "task_1a2b3c",
  "descriptor_name": "vz-bundle.long_export",
  "status": "queued",
  "poll_after_ms": 1000,
  "plan_ref": "plan-42"
}
```

Unknown `task_id` → 404 `tool_task_not_found`; unknown `ai_id` / `session_id`
→ 404 `ai_id_not_found` / `session_not_found`; session without an affordance
invoker → 501 `tool_invoker_unavailable` (all in the standard
`{"status": "error", "error": code, "detail": ...}` envelope).

`POST .../complete` request body:

```json
{
  "session_id": "sess_001",
  "status": "succeeded",
  "payload": {"rows_exported": 1042},
  "latency_ms": 95000
}
```

- `status` is `"succeeded"` or `"failed"` (required).
- `payload` (optional object) only accompanies `"succeeded"`.
- `error` (required non-empty string for `"failed"`) maps to the invoker's
  `error_detail`; optional `error_class` defaults to
  `deferred_backend_failed`. A `"succeeded"` completion must not carry
  `error` / `error_class` (400 `conflicting_completion`).
- `latency_ms` (optional int) is forwarded for kernel cost accounting.

Completion is submitted through `AffordanceInvoker.submit_deferred_result`,
i.e. the same `session.submit_tool_result` path as fast tools — preserving PE
lineage and credit attribution via the `plan_ref` captured when the task was
queued. Re-completing a terminal task is a 409 `tool_task_already_terminal`
(deferred completion is submit-once; HTTP retries cannot double-feed the
kernel tool bus).

`POST .../complete` response (200) is the refreshed handle plus the
invocation result summary:

```json
{
  "task_id": "task_1a2b3c",
  "descriptor_name": "vz-bundle.long_export",
  "status": "succeeded",
  "poll_after_ms": 1000,
  "plan_ref": "plan-42",
  "result": {
    "status": "succeeded",
    "tool_event_ids": ["evt_77"],
    "kernel_summary_truncated": false
  }
}
```

(`result.status` is the invocation status: `succeeded` or `backend_failed`.)

`POST .../cancel` remains planned (the invoker already exposes
`cancel_task`; no HTTP adapter yet).

## Observability And Explainability

### Debug App Registration

```http
POST /dlaas/v1/debug/apps
GET  /dlaas/v1/debug/apps
GET  /dlaas/v1/debug/apps/{app_id}
POST /dlaas/v1/debug/apps/{app_id}/schemas
GET  /dlaas/v1/debug/apps/{app_id}/schemas
```

DLaaS debug registration is the platform contract for app-owned operational
facts. It is separate from Volvence Accounts OAuth app registration: OAuth
identifies hosted login/checkout clients, while `app_id` here identifies a
debuggable DLaaS integration.

App registration:

```json
{
  "app_id": "repair30",
  "display_name": "repair30",
  "tenant_id": "tenant_repair30",
  "allowed_ai_ids": ["ai_repair30_001"],
  "allowed_event_types": ["chat.completed", "feedback.submitted"],
  "default_retention_days": 30
}
```

Schema registration freezes the fields an app may submit. Every field must
carry its meaning so downstream analysis can explain evidence without
reconstructing app internals:

```json
{
  "schema_version": "repair30.debug.v1",
  "event_types": ["chat.completed"],
  "allow_extra_fields": false,
  "fields": [
    {
      "name": "status",
      "type": "number",
      "meaning": "HTTP status returned by the upstream DLaaS call.",
      "owner": "app",
      "privacy_level": "internal",
      "required": true
    }
  ]
}
```

Field types: `string`, `number`, `boolean`, `json`, `enum`.

Privacy levels: `public`, `internal`, `sensitive`, `secret`. Normal debug
event ingest rejects `secret` fields; secret material must stay in the owning
app or secret manager.

### Debug Event Ingest

```http
POST /dlaas/v1/debug/events
GET  /dlaas/v1/debug/events?app_id=repair30&ai_id=ai_repair30_001&session_id=repair30_123
```

Event envelope:

```json
{
  "app_id": "repair30",
  "schema_version": "repair30.debug.v1",
  "ai_id": "ai_repair30_001",
  "tenant_id": "tenant_repair30",
  "session_id": "repair30_enrollment_123",
  "end_user_ref": "user_123",
  "response_id": "chatcmpl_abc",
  "interaction_id": "turn_abc",
  "event_type": "chat.completed",
  "stage": "dlaas.chat",
  "fields": {
    "status": 200,
    "stream": true
  },
  "occurred_at": "2026-05-23T14:00:00Z"
}
```

Invariants:

- `app_id`, `schema_version`, `event_type`, `stage`, and `fields` are required.
- Unknown fields are rejected unless the schema sets `allow_extra_fields`.
- Registered `allowed_ai_ids` and `allowed_event_types` are enforced when present.
- Debug events are platform governance records and are mirrored into audit/event-stream entries.

### Debug Analysis

```http
POST /dlaas/v1/debug/analysis
GET  /dlaas/v1/debug/analysis/{analysis_id}
```

Analysis requests combine a human prompt with structured selectors. The prompt
is not an authorization layer; the selectors and registered field policies
control what evidence is available.

```json
{
  "prompt": "Why did this repair30 session fail to produce a helpful response?",
  "selectors": {
    "app_id": "repair30",
    "ai_id": "ai_repair30_001",
    "session_id": "repair30_enrollment_123",
    "event_types": ["chat.completed", "feedback.submitted"]
  },
  "include_readouts": true,
  "include_explain": true,
  "include_audit": true,
  "include_snapshots": false
}
```

The response stores a `debug_analysis` governance record and a
`debug_analysis` artifact. Evidence may include registered debug events,
platform audit events, curated readouts, explain traces, and admin-gated
redacted snapshots. Future LLM-backed analysis must use centralized prompt
templates and must not inline long prompts in route handlers.

### Curated Readouts

```http
GET /dlaas/v1/instances/{ai_id}/readouts?session_id=sess_001&view=summary
```

Normal tenant/runtime callers receive curated readouts grouped by
public capability area. They do not need to know internal slot schemas.

Response groups:

- `body`: lifecycle-like/vitals readout.
- `cognition`: active regime, abstract action, prediction error, temporal summary.
- `knowledge`: retrieval/domain/case summary counts and descriptions.
- `strategy`: playbook and selected strategy summaries.
- `protocol`: active protocol ids, activation weights, phase, strategy weights.
- `safety`: boundary decision/tags and safety protocol ids.
- `training`: pending protocol submissions and training job status.

### Admin Raw Snapshots

```http
GET /dlaas/v1/admin/instances/{ai_id}/snapshots?session_id=sess_001&slot=active_mixture&slot=boundary_policy
```

Rules:

- Requires control-plane or service auth.
- Returns selected immutable snapshot values from the session's latest active/shadow snapshots.
- Serializes dataclasses/enums/tuples/mappings into JSON-safe values.
- Does not expose mutable object references.
- Does not create any mutation path back into owners.

### Temporal Time-Node Snapshots And Historical Forks

> Status: SHADOW contract. Required for deploy debt
> `D-moonlight-temporal-fork`; not implemented by cognition aggregates.

Temporal fork is the platform contract for user-facing "return to a past
moment" experiences. It is intentionally separate from `/cognition/snapshots`:
cognition aggregates are observability rows, while `TimeNodeSnapshot` is an
owner-published, restorable checkpoint bundle.

```http
GET  /dlaas/v1/instances/{ai_id}/time-nodes?scope_key=family_123:subject_456&session_id=sess_current&since_ms=&until_ms=&limit=200
GET  /dlaas/v1/instances/{ai_id}/time-nodes/{time_node_id}
POST /dlaas/v1/instances/{ai_id}/sessions/fork
```

`TimeNodeSnapshot` response shape:

```json
{
  "time_node_id": "tn_20240401_0000",
  "ai_id": "lfd_subject_456",
  "scope_key": "family_123:subject_456",
  "source_session_id": "lfd_subject_456_current",
  "as_of_ms": 1711929600000,
  "captured_at_ms": 1711929600123,
  "snapshot_version": "tn.v1",
  "restore_status": "ready",
  "owner_slots": [
    "memory_checkpoint",
    "active_regime",
    "relationship_state",
    "semantic_state",
    "experience_receipts"
  ],
  "evidence": {
    "source_count": 18,
    "latest_source_captured_at_ms": 1711900000000,
    "watermark": "sha256:..."
  }
}
```

`restore_status` values:

| Status | Meaning |
|---|---|
| `pending` | The platform knows the point in time, but at least one required owner has not published a restorable bundle. |
| `ready` | All required owner bundles exist and the node can be forked. |
| `not_restorable` | The node is visible for audit/readiness but cannot hydrate a session. |
| `revoked` | Consent or scoped data removal invalidated the node. |
| `stale` | The node predates a contract migration and must be reverified before fork. |
| `error` | Readiness check failed loudly; clients must inspect the typed error. |

`POST .../sessions/fork` request:

```json
{
  "source_session_id": "lfd_subject_456_current",
  "fork_session_id": "lfd_moonlight_456_user_789_20240401",
  "time_node_id": "tn_20240401_0000",
  "scope_key": "family_123:subject_456",
  "mode": "historical_readonly",
  "metadata": {
    "moonlight.as_of_ms": 1711929600000,
    "requester_ref": "user_789"
  }
}
```

Successful fork response:

```json
{
  "status": "ok",
  "ai_id": "lfd_subject_456",
  "source_session_id": "lfd_subject_456_current",
  "fork_session_id": "lfd_moonlight_456_user_789_20240401",
  "time_node_id": "tn_20240401_0000",
  "snapshot_version": "tn.v1",
  "mode": "historical_readonly"
}
```

Contract invariants:

- `lifeform-service` / `SessionManager` owns fork hydration. DLaaS routes the
  request to the runtime owner; deploy apps never reconstruct state from
  transcript text.
- `historical_readonly` forks may run `chat`, but their learning, feedback,
  reflection, and memory writes must not mutate the forward/current session.
- A forked `session_id` is stable and distinct from the source session. Runtime
  routing must preserve `{ai_id, fork_session_id}` affinity.
- Owner bundles are published by their owning modules (`vz-memory`,
  `vz-cognition`, `vz-temporal`, semantic-state owners, experience receipts).
  Consumers receive readiness and evidence pointers, not mutable internals.
- Missing owner slots fail closed; clients must not fall back to a current
  session or prompt-only imitation.
- The rollout flag is `DLAAS_TEMPORAL_FORK=off|shadow|active`. `shadow` allows
  listing/readiness checks but rejects `sessions/fork`.

Typed errors:

| Code | HTTP | Meaning |
|---|---:|---|
| `time_node_not_found` | 404 | The requested node is absent for the selected `ai_id` / scope. |
| `snapshot_not_restorable` | 409 | The node exists but `restore_status` is not `ready`. |
| `scope_not_authorized` | 403 | The caller cannot access the requested scope or source session. |
| `owner_snapshot_missing` | 424 | At least one required owner bundle is missing or contract-incompatible. |
| `fork_route_unavailable` | 503 | Runtime placement or `SessionManager` fork support is unavailable. |
| `temporal_fork_disabled` | 503 | The upstream rollout flag is `off` or `shadow` for mutation. |

### Explain Trace

```http
GET /dlaas/v1/instances/{ai_id}/explain?session_id=sess_001&turn_index=latest
```

Returns a compact causal chain from already-published readouts:

```text
input event -> active regime -> active protocols -> boundary decision -> strategy -> retrieval/case hits -> response tags -> PE
```

Explainability never reconstructs producer internals. It reads only
snapshot descriptions and stable readout fields.

### Cognition Aggregates

Five tenant-scoped GET endpoints aggregate per-interaction cognition
snapshots so portals and downstream apps can render regime history,
learning-family distribution, ExperienceLoop throughput, and eval
trend without re-deriving owner state. All paths are gated through
the same tenant auth as the rest of `/dlaas/v1/...` and return
`{status, count, items}` (or `{status, totals, sample_count}` for
the radar aggregate).

Snapshots are written by the dispatch hook in `app.py` after every
successful `dispatch_envelope`. The recorded fields are the
`active_regime` id, prediction-error 4 axes (`task`, `relationship`,
`regime`, `action`, plus `magnitude`), a six-class learning-family
count derived from populated kernel slots, `eval_alert_count` /
`memory_entries`, and the full readout bundle for forward
compatibility. The store is in-memory aiohttp app state; the
platform-api restart drops history. Persistence is tracked as a
round-6 known debt in `apps/dlaas-portal/known-debts.md`.

The cognition surface honours the same R12 / R14 / R4 rules as the
other observability endpoints:

- R14: the regime owner is the kernel; this surface only records
  what the readout bundle already published.
- R4: visualisations are readouts (soft signals), not a model
  self-assessed capability score.
- R12: `eval-trend` is the eval PE readout; nothing here ever feeds
  back into a learning input.

```http
GET /dlaas/v1/cognition/snapshots?ai_id=ai_demo&since_ms=&until_ms=&window=7d&limit=200
```

Paginated raw rows. Accepts `ai_id`, `tenant_id`, `session_id`,
`source` (`interaction` / `session_end` / `sampler` / `manual`),
`since_ms` / `until_ms` epoch filters, and a free-form `window` like
`7d` / `24h` / `30m`. Default newest-first sort, max `limit=1000`.

Sample response:

```json
{
  "status": "ok",
  "count": 1,
  "items": [
    {
      "snapshot_id": "cog_4f3a91c0d712",
      "tenant_id": "tenant_demo",
      "ai_id": "ai_demo",
      "session_id": "sess_demo",
      "source": "interaction",
      "captured_at_ms": 1717049200000,
      "regime_id": "regime.calm",
      "prediction_error": {
        "magnitude": 0.18,
        "task": 0.05,
        "relationship": 0.12,
        "regime": 0.0,
        "action": 0.01
      },
      "learning_family": {
        "cognition": 3,
        "knowledge": 2,
        "strategy": 1,
        "protocol": 0,
        "safety": 1,
        "training": 0
      },
      "eval_alert_count": 0,
      "memory_entries": 12,
      "raw_readout": { "body": {}, "cognition": {} }
    }
  ]
}
```

```http
GET /dlaas/v1/cognition/timelines/regime?ai_id=ai_demo&window=7d
```

Coalesces consecutive same-regime snapshots into `RegimeTimelineSegment`
ranges so a strip chart can render one band per regime run instead
of one rect per turn.

```json
{
  "status": "ok",
  "count": 1,
  "items": [
    {
      "regime_id": "regime.calm",
      "started_at_ms": 1717048800000,
      "ended_at_ms": 1717049200000,
      "duration_ms": 400000,
      "sample_count": 7
    }
  ]
}
```

```http
GET /dlaas/v1/cognition/learning-family?ai_id=ai_demo&window=7d
```

Sums the six-class learning-family counts over the window so a
radar chart has a single point per family. Mirrors the slot family
buckets used internally by `_build_readout_bundle`.

```json
{
  "status": "ok",
  "sample_count": 24,
  "totals": {
    "cognition": 72,
    "knowledge": 48,
    "strategy": 24,
    "protocol": 0,
    "safety": 24,
    "training": 0
  }
}
```

```http
GET /dlaas/v1/cognition/experience-throughput?window=30d&group_by=binding,day
```

Reads from the existing `debug_events` store and groups
`experience.receipt.v1` + `experience.reflection.v1` envelopes by
ExperienceLoop binding and ISO day. No new persistence; the
endpoint is a thin aggregator over the debug events the platform
already records.

```json
{
  "status": "ok",
  "count": 1,
  "items": [
    {
      "binding": "marketing.bench",
      "day": "2026-05-26",
      "receipts": 12,
      "reflections": 4
    }
  ]
}
```

```http
GET /dlaas/v1/cognition/eval-trend?ai_id=ai_demo&window=30d
```

Reads from the `eval_runs` store and projects daily run count, mean
score, and pass rate (score >= 0.5). The endpoint is read-only and
never edits a run; it is a pure readout. The `runs` and
`average_score` figures match what `eval/runs/{run_id}` returns row
by row.

```json
{
  "status": "ok",
  "count": 1,
  "items": [
    {
      "day": "2026-05-26",
      "runs": 4,
      "average_score": 0.78,
      "pass_rate": 0.75
    }
  ]
}
```

### Eval Runs CRUD

The eval surface lives on the same auth tier as readouts but is
called out separately because `cognition/eval-trend` depends on it
and it is otherwise undocumented.

```http
POST /dlaas/v1/eval/runs
GET  /dlaas/v1/eval/runs/{run_id}
POST /dlaas/v1/eval/runs/{run_id}/approve
```

`POST /eval/runs` accepts `{ gate_id, ai_id, contract_id, score }`
and returns a `EvalRun` with status `pending`. `GET ../{run_id}`
returns the run; `POST ../{run_id}/approve` stamps
`status=approved` and `decision=PromotionDecision.ALLOW`. The
`score` field is a soft signal under R12 — it is a readout of PE,
not a learning input.

### Cognition Health

The cognition aggregates above are descriptive (what the snapshots
say). The health surface is the **normative** layer: it answers
"is this digital life OK" by deriving signals from the same snapshot
fields and rolling them up into an `ok` / `watch` / `alert` verdict.
It adds no new cognition concept and never re-derives owner state
(R12/R14); thresholds are an operational policy (env-overridable),
not a kernel fact.

```http
GET /dlaas/v1/cognition/health?ai_id=ai_demo&window=7d
GET /dlaas/v1/cognition/health/overview?window=7d
```

Signals (all derived from existing snapshot fields):

- `regime_instability` — regime switches per sample over the window
  (thrash). Needs >= 4 samples.
- `pe_elevation` — mean `prediction_error.magnitude` over the window.
- `eval_alerts` — summed `eval_alert_count` over the window.
- `staleness` — wall-clock gap since the most recent snapshot.

Each signal yields `watch` or `alert`; the verdict is the max
severity (no signals → `ok`). Thresholds:
`DLAAS_COG_PE_WATCH` / `DLAAS_COG_PE_ALERT` /
`DLAAS_COG_REGIME_THRASH` / `DLAAS_COG_EVAL_ALERT` /
`DLAAS_COG_STALE_HOURS`.

Single-verdict response:

```json
{
  "status": "ok",
  "health": {
    "ai_id": "ai_demo",
    "status": "alert",
    "signals": [
      {
        "name": "eval_alerts",
        "severity": "alert",
        "detail": "5 eval alerts in window >= 3",
        "value": 5
      }
    ],
    "sample_count": 12,
    "last_seen_ms": 1717049200000,
    "latest_session_id": "sess_demo",
    "computed_at_ms": 1717049260000
  }
}
```

`GET /cognition/health/overview` groups every ai_id in scope by
status and returns counts + a worst-first list. **Security: the
overview (and the single-verdict endpoint) NEVER include
`raw_readout`** — only the verdict + signal names/details — so a
cross-tenant platform operator cannot read tenant-private readout
content through the health surface.

```json
{
  "status": "ok",
  "computed_at_ms": 1717049260000,
  "counts": { "ok": 8, "watch": 2, "alert": 1 },
  "items": [
    {
      "ai_id": "ai_hot",
      "tenant_id": "tenant_demo",
      "status": "alert",
      "signals": [{ "name": "pe_elevation", "severity": "alert", "detail": "...", "value": 0.82 }],
      "sample_count": 30,
      "last_seen_ms": 1717049200000,
      "latest_session_id": "sess_hot",
      "computed_at_ms": 1717049260000
    }
  ]
}
```

### Application Status

The cognition surfaces above monitor a single ai_id (a digital life).
The application-status surface monitors a single **application** (a
consumer deployment: einstein, coread, repair30, ...). It reuses the
exact verdict shape as cognition health (`status` in
`ok`/`watch`/`alert` + `signals`) — only the entity differs.

It adds no new app-side reporting: the verdict is derived from the
existing debug-app registry (`/dlaas/v1/debug/apps`) unioned with
recent debug-event activity (`/dlaas/v1/debug/events`).

```http
GET /dlaas/v1/apps/status?window=7d&tenant_id=
GET /dlaas/v1/apps/status/{app_id}?window=7d
```

Signals (derived from existing fields):

- `silent` (alert) — a known app (registered or previously active)
  with zero events in the window, or whose most recent event is older
  than `DLAAS_APP_STALE_HOURS` (default 24h). Distinguishes a dead app
  from a healthy idle one.
- `error_rate` (watch/alert) — fraction of recent events that look
  like errors (`fields.ok === false`, `fields.status >= 400`, or an
  event_type naming a failure). Heuristic; calibrate per app.
  Suppressed below `DLAAS_APP_MIN_EVENTS_FOR_ERROR` events.

Overview response (grouped, worst-first; no event bodies):

```json
{
  "status": "ok",
  "computed_at_ms": 1717049260000,
  "counts": { "ok": 9, "watch": 1, "alert": 2 },
  "items": [
    {
      "app_id": "coread",
      "display_name": "Coread",
      "tenant_id": "tenant_demo",
      "registered": true,
      "status": "alert",
      "signals": [{ "name": "silent", "severity": "alert", "detail": "no events in window", "value": 0 }],
      "event_count": 0,
      "event_types": [],
      "last_event_ms": null,
      "computed_at_ms": 1717049260000
    }
  ]
}
```

`GET /apps/status/{app_id}` adds a `breakdown` of recent events by
`event_type` and `stage`. Thresholds:
`DLAAS_APP_STALE_HOURS` / `DLAAS_APP_ERROR_RATE_WATCH` /
`DLAAS_APP_ERROR_RATE_ALERT` / `DLAAS_APP_MIN_EVENTS_FOR_ERROR`.

## Multi-Angle Bake

One platform-owned **bake plane** turns a single set of raw materials
(a work, a corpus, reviewed sources) into one or more grounded personas,
each cut at a distinct **angle**. A single request fans out into N
per-angle jobs that share the raw materials but route to different
verticals and produce one template each.

Angles (`dlaas_platform_contracts.bake.BakeAngleKind`):

| Angle | Routes to | Meaning |
|---|---|---|
| `author` | figure vertical | the real creator/author as a primary-source figure |
| `interpreter` | figure vertical | 诠释者 / narrator-commentator that explains the work |
| `character` | character vertical | an in-world 角色; one persona per named character |

```http
POST /dlaas/v1/bake                  # submit a multi-angle bake run (202)
GET  /dlaas/v1/bake                  # list runs (tenant-scoped)
GET  /dlaas/v1/bake/{run_id}         # run aggregate + per-angle job states
GET  /dlaas/v1/bake/{run_id}/events  # SSE progress stream (monitor)
POST /dlaas/v1/bake/{run_id}/cancel  # request cancellation
GET  /dlaas/v1/bake/{run_id}/result  # per-angle produced templates
```

Auth: operator secrets act cross-tenant; tenant credentials bake only
for their own tenant (`tenant_mismatch` 403 otherwise).

`POST /dlaas/v1/bake` body:

```json
{
  "source_ref": "work:dream-of-red-chamber",
  "corpus_mode": "curated",
  "runtime_template_id": "",
  "shared_profile": { },
  "angles": [
    { "kind": "author", "slug": "caoxueqin", "display_name": "曹雪芹" },
    { "kind": "interpreter", "slug": "narrator" },
    { "kind": "character", "slug": "jiabaoyu", "display_name": "贾宝玉",
      "style_prior": { }, "boundary_priors": [], "target_contexts": [],
      "time_window": "", "profile_overrides": { } }
  ],
  "raw_materials": [
    { "kind": "text", "text": "第一回 ...", "angle_slugs": ["narrator"] },
    { "kind": "uri", "ref": "https://example/hlm.txt" }
  ]
}
```

- `angles` is required and non-empty; each `(kind, slug)` must be unique.
- `raw_materials[].kind` is one of `text` | `asset_ref` | `uri`. A
  material with empty `angle_slugs` feeds every angle; a non-empty list
  routes it to specific angle slugs only.
- `corpus_mode` is `synthetic` (default) | `curated`.

Per-angle progress mirrors the proven figure pipeline so a shared
monitor renders the same bar for every app:

```text
queued → staging → cleaning → verifying → baking → registering → done
                                                              ↘ failed | cancelled
```

The run aggregate rolls up its angles (SSOT
`dlaas_platform_contracts.bake.rollup_run_status`): `running` while any
angle is non-terminal, then `done` (all done), `failed` (all failed),
`partial` (mixed), or `cancelled` (all cancelled).

`GET /dlaas/v1/bake/{run_id}/events` is an SSE stream emitting
`run` / `angle` / `done` / `error` events (`event:` + JSON `data:`),
replays the backlog for late subscribers, and closes with a
`: stream-end` comment when the run is terminal.

`GET /dlaas/v1/bake/{run_id}/result` returns one entry per finished
angle: `{ angle_kind, angle_slug, template_id, figure_artifact_id,
bundle_id, lifecycle_stage, integrity_hash }`.

Ownership (R8 / R12 / R15): the bake plane is **orchestration**, not a
second owner of the baked bundle. Each angle runs three explicit seams:

1. **解析 / analysis (third-party LLM)** — raw materials (饲料) are turned
   into a structured per-angle profile via the third-party LLM plane
   (`/dlaas/v1/third-party-llm/json`). The LLM ONLY does this
   decomposition.
2. **吸收 / absorption (VZ, never LLM)** — `VzBakeAngleCompiler` compiles
   the profile into a real figure/character artifact through
   `lifeform_domain_figure` / `lifeform_domain_character`. What substrate
   VZ uses underneath is the platform's own config.
3. **register (control plane)** — `RegistryBakeArtifactRegistrar`
   registers the figure bundle into the runtime store, mints the
   template, and advances the persona lifecycle to `pretrained`
   (tenant-scoped; deferred with a note when the run has no tenant).

**Default runner is `third_party_llm`** (`VZ_BAKE_RUNNER`); the
deterministic GPU-free `synthetic` runner is the explicit CI/dev
fallback (`VZ_BAKE_RUNNER=synthetic`, no LLM, no registration).
`VZ_BAKE_REGISTER=0` disables the register seam (write-only artifacts).
When `third_party_llm` is selected but the provider is unconfigured,
each angle fails loud with `third_party_llm_unconfigured` — never a
silent success. Contracts: `dlaas_platform_contracts.bake`; routes:
`dlaas_platform_api.bake`.

## Third-Party LLM Plane

The third-party LLM plane is compile-time infrastructure for bake and
future document/protocol extraction. It is deliberately separate from
the VZ/lifeform runtime `/v1/chat/completions` path.

```http
GET  /dlaas/v1/third-party-llm/status            # provider readiness, no secrets
POST /dlaas/v1/third-party-llm/chat/completions  # OpenAI-compatible proxy
POST /dlaas/v1/third-party-llm/json              # schema-enforced JSON output
```

Auth: operator/service credentials only (`X-Control-Plane-Secret` or
`X-Service-Secret`). The endpoint is not public user chat.

Environment:

- `THIRD_PARTY_LLM_PROVIDER` (`openai`, `qwen`, `dashscope`, `vllm`, or custom)
- `THIRD_PARTY_LLM_API_KEY`
- `THIRD_PARTY_LLM_BASE_URL` (optional for known providers)
- `THIRD_PARTY_LLM_MODEL`
- `THIRD_PARTY_LLM_TIMEOUT_SECONDS`
- `THIRD_PARTY_LLM_ALLOW_PROTOCOL_FALLBACK=1` permits explicit fallback
  to `PROTOCOL_LLM_*` credentials.

`POST /dlaas/v1/third-party-llm/json` body:

```json
{
  "system_prompt": "centralized prompt text loaded by the platform",
  "user_prompt": "raw materials / task context",
  "schema_name": "bake_author_profile",
  "schema": { "type": "object", "required": ["slug"] },
  "temperature": 0,
  "metadata": { "dlaas.bake.run_id": "..." }
}
```

Response: `{ status: "ok", content, provider, model, response_id, usage }`.
Errors are typed: `third_party_llm_unconfigured`,
`third_party_llm_invalid_request`, `third_party_llm_upstream_error`,
`third_party_llm_invalid_json`, `third_party_llm_schema_validation_failed`.

## Persona Training Lifecycle

One platform-owned governance record per persona (keyed by `template_id`)
tracks the unified cognitive-training pipeline across products
(Myriad figures, digital-employee templates, cultivated experts):

```text
draft → pretrained → studying → training → exam → interview → inducted
  └──────────┴──────────┴──────────┴────────┴────────┴──────────┴──▶ retired
```

```http
POST /dlaas/v1/personas/{template_id}/lifecycle           # create (stage=draft)
GET  /dlaas/v1/personas/{template_id}/lifecycle           # record + events + gates
POST /dlaas/v1/personas/{template_id}/lifecycle/advance   # forward move
POST /dlaas/v1/personas/{template_id}/lifecycle/rollback  # audited backwards move
GET  /dlaas/v1/personas/lifecycles                        # list (tenant-scoped)
```

Auth: operator secrets (`X-Control-Plane-Secret` / `X-Service-Secret`)
act cross-tenant; tenant credentials act only on templates their tenant
owns (`tenant_mismatch` 403 otherwise).

`advance` body: `{ to_stage, evidence: { ... } }`. Each target stage has
mandatory evidence keys — entering a stage asserts the pointed artifact
exists (fail loudly with `409 invalid_transition` when missing):

| Target stage | Required evidence | Points at |
|---|---|---|
| `pretrained` | `figure_bundle_id` | offline bake (corpus → bundle → LoRA → persona verification) |
| `studying` | `cultivation_id` | cultivation self-study record |
| `training` | `training_ref` | corpus intake / training job / teach session |
| `exam` | `exam_run_id` (+ `passed`) | eval-gate exam run (`POST /dlaas/exam_runs`) |
| `interview` | `interview_run_id` (+ `passed`) | interactive interview run (`POST /dlaas/interview_runs`) |
| `inducted` | `reviewer_id` | operator decision |
| `retired` | `reason` | withdrawal |

Rules:

- Forward-only: `advance` must move strictly later on the pipeline;
  skipping stages is allowed (some products have no offline bake) and
  the skip is visible in the event history.
- **Gate evidence is verified, not trusted**: advancing to `exam` /
  `interview` cross-checks the referenced run against the registry's
  `exam_runs` / `interview_runs` persistence. The run must exist,
  belong to the same `template_id`, have `status=completed`, and its
  recorded `passed` outcome must equal the evidence `passed` flag —
  a caller cannot assert a pass the grader/operator never recorded
  (`409 invalid_transition` otherwise). Exams graded outside the
  platform cannot enter the gate stages; they skip them and must carry
  an explicit `waiver_reason` at induction.
- **Induction gate**: advancing to `inducted` requires the latest
  recorded `exam` and `interview` evidence to carry `passed: true`. An
  explicit `waiver_reason` overrides — the waiver itself becomes part
  of the audit trail (no silent pass).
- `rollback` body `{ to_stage, reason }` is the only backwards move; a
  non-empty reason is mandatory and the rollback is persisted as an
  immutable event (R15).
- `GET .../lifecycle` returns the record, the full ordered event log,
  and a derived `gates` readout
  (`{stage: {reached, passed, evidence}}`) consumers render as-is.

Ownership (R8 / R12): the lifecycle stores **pointers to evidence
artifacts**, never cognition; no learning signal flows back from this
surface. Contracts: `dlaas_platform_contracts.persona_lifecycle`;
store: `dlaas_platform_registry.persona_lifecycle_store` (schema v9);
routes: `dlaas_platform_api.persona_lifecycle`.

## Exam Gate (Eval)

The eval gate (`dlaas-platform-eval`) owns exam questions, exam runs,
and the launch license. All endpoints require tenant credentials and
template ownership:

```http
POST /dlaas/exam_questions
POST /dlaas/exam_questions/batch
POST /dlaas/exam_questions/generate
GET  /dlaas/exam_questions
POST /dlaas/exam_runs
GET  /dlaas/exam_runs/{run_id}
POST /dlaas/exam_runs/{run_id}/complete
POST /dlaas/exam_runs/{run_id}/execute
GET  /dlaas/templates/{template_id}/exam_runs
POST /dlaas/exam_runs/{run_id}/signoff
GET  /dlaas/templates/{template_id}/license
POST /dlaas/templates/{template_id}/license/evaluate
```

### Question generation

`POST /dlaas/exam_questions/generate` authors scenario questions WITH
rubrics (2-3 criteria each) and reference answers, grounded only in
the supplied source material (semantic LLM generation, no keyword
bank):

```json
{
  "template_id": "template_abc",
  "source": {
    "topics": ["relativity"],
    "corpus_excerpts": ["..."],
    "signature_cases": [{"title": "...", "summary": "..."}]
  },
  "count": 5,
  "difficulty": "medium",
  "language": "en"
}
```

At least one of `source.topics` / `source.corpus_excerpts` /
`source.signature_cases` is required (`400 empty_source` otherwise).
Generated questions are persisted through
`EvalStore.create_exam_question` and returned as
`{"status": "ok", "questions": [...]}`. When the eval LLM env is not
configured the route answers `503 llm_not_configured` — there is
deliberately no stub question bank. Malformed LLM output answers
`502 question_generation_error`.

### Grader configuration

Exam-run scoring goes through the `RubricGrader` seam
(`dlaas_platform_eval.build_grader_from_env()`):

- When the env trio is configured, `LLMRubricGrader` judges each
  response per rubric criterion via an OpenAI-compatible
  `/chat/completions` endpoint with centralized prompts
  (`dlaas_platform_eval.prompts`), strict-JSON output, full criteria
  coverage validation, and scores clamped to `[0, max_score]`. Each
  breakdown entry carries a `grader_label` (`"llm:<model>"`).
- Env vars (each `EVAL_LLM_*` overrides its `PROTOCOL_LLM_*` sibling
  individually; no provider presets — `base_url` and `model` must be
  explicit): `EVAL_LLM_BASE_URL` / `EVAL_LLM_API_KEY` /
  `EVAL_LLM_MODEL` / `EVAL_LLM_TIMEOUT_S` (default 60), falling back
  to `PROTOCOL_LLM_BASE_URL` / `PROTOCOL_LLM_API_KEY` /
  `PROTOCOL_LLM_MODEL` / `PROTOCOL_LLM_TIMEOUT_SECONDS`.
- Without that env the gate stays on the **fail-closed**
  `DefaultRubricGrader` (flat 0.5, loud warning): automation can never
  grant a license, by design.
- A malformed or failed judge call raises a typed
  `GraderResponseError`: the `complete` / `execute` handlers answer
  `502 grader_error` and leave the run `failed` with the error
  recorded in `comment` — never a silent fallback score.
- The operator `POST /dlaas/exam_runs/{run_id}/complete` path (caller-
  supplied `ai_responses`, operator identity, signoff) remains the
  authoritative way to finalize a run; `execute` is the automated
  variant over the same grading path.

R12 / OA-1: grading and question generation are readouts. Scores,
justifications, and generated questions are persisted as exam
artifacts only; nothing flows back into kernel owners or learning
state.

## Interview Runs

The interview (面试) is the interactive gate between `exam` and
`inducted`: an interviewer — an LLM interviewer following a question
plan, a human operator, or both (`interviewer_kind: llm | operator |
mixed`) — asks multi-turn questions against the persona, each turn is
scored in `[0, 1]`, and the run completes with an aggregate verdict.

```http
POST /dlaas/interview_runs                              # create
GET  /dlaas/interview_runs/{run_id}                     # read
GET  /dlaas/templates/{template_id}/interview_runs      # list
POST /dlaas/interview_runs/{run_id}/turns               # record turn
POST /dlaas/interview_runs/{run_id}/turns/execute       # live turn
POST /dlaas/interview_runs/{run_id}/turns/{i}/score     # operator score
POST /dlaas/interview_runs/{run_id}/complete            # verdict
```

All routes are dual-auth: operator secrets (`X-Control-Plane-Secret` /
`X-Service-Secret`) act cross-tenant so the operator console can run and
score interviews for any tenant's persona; tenant credentials act only
on templates their tenant owns (mirrors the exam surface).

- Create body: `{ template_id, ai_id?, session_id?, interviewer_kind?,
  question_plan?: [str], pass_threshold? (default 0.6) }`. Question
  plans typically come from `POST /dlaas/exam_questions/generate` or
  the app's role spec.
- `turns` records a caller-supplied exchange (human interviewer /
  offline transcript), optionally pre-scored.
- `turns/execute` asks the next unasked `question_plan` entry (or an
  explicit `question`) against the run's live `ai_id`
  (`409 instance_not_awake` when it is not awake; `409 plan_exhausted`
  when the plan ran out). When a rubric grader is configured on the
  bundle the turn is auto-scored; otherwise the score stays empty for
  the operator.
- `complete` refuses (`409 incomplete_interview`) while any turn is
  unscored — the verdict can never be silently granted. Aggregate =
  mean of turn scores; `passed = aggregate >= pass_threshold`.
- Completed/failed runs are frozen: no further turns or score edits.

The completed run's `run_id` + `passed` outcome is the evidence the
persona lifecycle's `interview` stage cross-checks (see above). R12:
interview scores are readouts — no learning signal flows back.
Contracts: `dlaas_platform_contracts.interview`; store:
`dlaas_platform_registry.interview_store` (schema v10); routes:
`dlaas_platform_api.interview`.

## Expert Cultivation

Autonomous industry-expert cultivation ("行业专家自动养成") grows a default
system expert with minimal human interaction: an operator seeds a rough
persona, the engine researches the domain, and the expert converges onto a
single coherent school of thought before an operator inducts it as a default
expert template. The "school" is NOT a regime label — it is the agent's
Behavior Protocol active-mixture, so conflict resolution rides the kernel's
existing layered mechanism (Identity Core + boundary-union hard-block,
PE-utility soft-blend / arbitration, slow-reflection retirement). Cognition
stays kernel-owned; this surface only orchestrates intake cadence and reads
published readouts (R12) — it never resolves contradictions itself.

All routes are dual-mode. Operator credentials (`X-Control-Plane-Secret`
or `X-Service-Secret`) act cross-tenant: they create **system-owned**
cultivations (`tenant_id=""`), list every record, and graduate candidate
templates under `SYSTEM_TENANT_ID` (existing behaviour, unchanged).
Tenant credentials (`X-Tenant-Api-Key` + `X-Tenant-Api-Secret` — the path
app BFFs like Myriad / digital-employee use) create **tenant-owned**
cultivations: the record carries the authenticated `tenant_id`, the
`ai_id` is tenant-namespaced (`cultivation:{tenant_id}:{slug}[:{track_id}]`
so two tenants reusing one slug never share kernel state), `GET` list
returns only that tenant's records, and get/tick/graduate/induct (and
package views) on another tenant's or a system-owned record are a typed
403 `tenant_mismatch`. Graduating a tenant-owned cultivation creates the
candidate template under the tenant's own `tenant_id` instead of
`SYSTEM_TENANT_ID`. Cultivation responses carry the lifecycle-evidence
pointers (`cultivation_id`, `tenant_id`, `dlaas_template_id`,
`last_exam_run_id`) the persona lifecycle API consumes; cultivation
handlers never auto-advance lifecycles — apps own that orchestration.

```http
POST /dlaas/v1/cultivation
GET  /dlaas/v1/cultivation
GET  /dlaas/v1/cultivation/packages
GET  /dlaas/v1/cultivation/packages/{package_id}
GET  /dlaas/v1/cultivation/{cultivation_id}
GET  /dlaas/v1/cultivation/{cultivation_id}/events
POST /dlaas/v1/cultivation/{cultivation_id}/tick
POST /dlaas/v1/cultivation/{cultivation_id}/teach
POST /dlaas/v1/cultivation/{cultivation_id}/pause
POST /dlaas/v1/cultivation/{cultivation_id}/resume
POST /dlaas/v1/cultivation/{cultivation_id}/reject
POST /dlaas/v1/cultivation/{cultivation_id}/graduate
POST /dlaas/v1/cultivation/{cultivation_id}/induct
```

- `POST /cultivation` seeds a new expert:
  `{ slug, display_name, domain, role_archetype, focus?, value_boundaries?[],
  single_school_objective?, curriculum: { topics[], source_hints?[],
  coherence_threshold? } }`. The seed compiles into an ACTIVE **Identity Core**
  `BehaviorProtocol` (value boundaries become a hard-block union — the anchor
  that resists drift from shallow inputs) loaded into the instance, and binds
  a per-`ai_id` `ProtocolUptakeService`.
- **Adopted seed (empty vs adopted entry).** `POST /cultivation` also accepts
  an optional `source_template_id` (plus optional `source_kind` / `source_angle`
  overrides). When present, the new cultivation is seeded from an existing baked
  persona/template instead of an empty seed: the seed fields default from the
  source template's `persona_spec` (operator-supplied fields still win), and —
  when the source template carries a `cultivation_protocol_bundle` in its
  `seed_config` — that converged school is hydrated into the per-`ai_id`
  `ProtocolUptakeService` *before* acquire so the cultivation **continues** from
  the adopted persona's school rather than a blank Identity Core. The runtime
  vertical stays `cultivation.expert.v0` (the study loop is unchanged); only the
  starting cognition differs.
  - **Trust boundary.** In tenant-auth mode a `source_template_id` may only be a
    template the tenant owns or one that is `PUBLISHED` (the adoptable contract);
    otherwise the create fails with `403 source_template_forbidden`. A missing
    template is `404 source_template_not_found`; a malformed bundle is
    `400 invalid_source_bundle`. Control-plane / service callers adopt
    cross-tenant.
  - **Durable continuation.** The uptake service is in-process, so `tick`,
    `teach`, and `graduate` re-hydrate the source bundle when the service has
    lost it (restart / eviction) — an adopted seed never silently collapses to
    persona-metadata-only between runs.
  - **Provenance.** The record persists `source_template_id` and a `provenance`
    object `{ source_kind, source_angle, continuation_mode }`, where
    `continuation_mode` is `protocol_bundle` (real learned-state continuation) or
    `metadata_only` (persona anchor only). `source_kind` preserves whether the
    source was a `character` / `author` / `interpreter` / `expert`. Provenance is
    propagated into the graduated template `persona_spec.source_provenance` and
    surfaced on package track views.
- `POST .../tick { cycles? }` runs autonomous study cycles: research
  (`search_web` / `fetch_webpage`) → DocumentUptake (researched theory →
  `BehaviorProtocol` candidate → review → load, NOT raw corpus) → apprentice
  study turn → R6 slow-reflection. Returns a `progress` readout: active-mixture
  convergence `coherence_score`, dominant school protocol, and the protocols
  uptaken this tick.
- `GET .../{id}` returns the cultivation record: status machine
  `seeding → studying → converging → exam → ready_for_review → inducted`,
  `coherence_score`, and `coherence_detail` (`dominant_protocol`,
  `distinct_schools`, `identity_core_present`, `readout`).
- `POST .../graduate` creates an activated + published candidate template,
  runs an eval exam as evidence (reuses the eval gate), and — when the school
  has converged — moves the record to `ready_for_review`.
- `POST .../induct` is the **operator approval** that promotes the published
  template to a default system expert. When `CULTIVATION_BAKE_ON_INDUCT` is set
  and the bake plane is attached, induct also **composes a bake run** from the
  cultivated profile (angle resolved from `provenance.source_angle` /
  `source_kind`, default `author`) by submitting to the existing bake plane;
  the run id is persisted as `bake_run_id` on the record and returned in the
  induct response. The bake runs async and produces a figure/character template
  (the release snapshot) + advances the persona lifecycle to `pretrained`. This
  is additive and best-effort: a bake failure is logged and never blocks
  induction. R8: the `cultivation_protocol_bundle` template remains the
  online-learned artifact; the baked bundle is a derivative release linked by
  `bake_run_id`, not a second owner. (Cultivation thus both *consumes* a baked
  template as an adopted seed and *emits* a bake at induct.)

**Monitoring + supervision (self-learning workshop).** The operator console
needs to watch and steer an in-flight cultivation, not just poll status:

- `GET .../events { limit? }` returns the append-only study trail split into
  two lists, oldest first: `events` (per-cycle `{ cycle_index, topic,
  docs_researched, protocols_uptaken, active_regime, reflected }`, plus `teach`
  corrections and `pause` / `resume` / `reject` supervision actions) and
  `timeline` (one `progress` snapshot per tick: `{ cycle_index,
  coherence_score, readout_kind, dominant, distinct_schools, uptaken_protocols,
  converged }`) so the console can chart how the school converges over time. The
  response also echoes the current `coherence_score`, `coherence_detail`,
  `regime_history`, `provenance`, and `source_template_id`. This is a pure
  **readout** (R12): nothing here is read back as a learning signal.
- **Apprentice retraining** of an inducted/adopted instance uses the existing
  runtime surfaces: `POST /dlaas/v1/instances/{ai_id}/interactions` with
  `interaction_type: teach` (apprentice turn) or `feedback` (outcome), and
  `POST /dlaas/v1/instances/{ai_id}/training/corpus` for reviewed corpus / case
  material. The workshop console orchestrates these over HTTP and renders the
  `readouts`; no learning state is written from the deploy layer.
- `POST .../teach { text, source_label? }` injects an **operator apprentice
  correction** into a running cultivation. The text is re-homed through the
  same `ProtocolUptakeService` as autonomous research (it competes in the
  active mixture on PE utility — never a hardcoded rule, R4) and run as one
  apprentice study turn. It does not advance the cycle counter; the next
  `tick` recomputes coherence with the correction in the mixture. Recorded as
  a `teach` event. Allowed while `seeding | studying | converging | exam |
  paused`.
- `POST .../pause` holds a runnable cultivation (`seeding | studying |
  converging`) in `paused` so `tick` returns `409 cultivation_not_runnable`;
  `POST .../resume` returns it to `studying` (the next tick recomputes
  coherence). Accumulated progress is preserved.
- `POST .../reject { reason? }` abandons a cultivation / school track
  (status → `failed`). This is the multi-school selection path: the operator
  inducts the chosen self-consistent school from a package and rejects the
  others (R15). An `inducted` expert cannot be rejected through this path.

**Multi-direction packages.** A seed may carry an optional `directions[]`
list — each entry `{ track_id, display_name, topics[], source_hints?[],
objective?, coherence_threshold?, min_cycles_for_convergence? }`. When
present, the seed fans out into several **self-consistent school tracks**:
each direction becomes its own cultivation row with a distinct
`ai_id = cultivation:{slug}:{track_id}` and its own active-mixture
convergence + exam, but all tracks share one `package_id`. Schools never
cross-contaminate because each track has its own kernel session. Tracks are
ticked / graduated / inducted individually through the existing
`{cultivation_id}` routes; `GET .../packages` and
`GET .../packages/{package_id}` return the grouped
`cultivation.package.v1` view (tracks + published refs + provenance) the
operator console renders. Graduated templates carry `package_id` / `track_id`
in their `cultivation.persona.v1` `persona_spec` so siblings group back to
one seed and roll back together (R15). When `directions[]` is absent the
legacy single-expert path is unchanged.

**Cognition reflow (portable learned school).** Graduation does not just
record persona metadata — it exports the cultivation's *converged school*
(the approved `BehaviorProtocol` set: Identity Core + researched theory
protocols) into the candidate template's `seed_config` under
`cultivation_protocol_bundle` (schema `cultivation.protocol_bundle.v1`,
lossless via `protocol_to_payload`). On `/wake` with that `template_id`,
the platform hydrates a `ProtocolUptakeService` from the bundle and binds
it to the adopting `ai_id` *before* acquire, so the adopted instance
*starts* from the cultivated school and continues online ETA learning
(its own per-session α/β PE mixing + a fresh `revision_log`). The adopted
runtime never writes back to the published bundle, so the bundle is a
reviewed-frozen artifact, not a second cognition owner (R8); improving a
published expert means **re-inducting**, which produces a new template
version with a new bundle (R15 rollback via versioned templates). If the
source process restarted between `tick` and `graduate` (in-memory uptake
empty), the template still carries `persona_spec` metadata and the legacy
seed path applies. Owner: `lifeform_service.cultivation_bundle`.

Runtime vertical: `cultivation.expert.v0`. The protocol slow-loop
(`protocol_reflection` / `protocol_revision_queue`) is ACTIVE for these
instances so the mixture prunes failing strategies and converges onto one
school while the Identity Core stays fixed (R15 rollback retained). Web
research reuses the vz-bundle browse tools; researched theories never enter
as raw corpus. This web research is **creation-time / studio-time only**
(the cultivation loop, before induction): the inducted template's adopted
runtime never reaches the internet, mirroring the Persona Studio
`runtimeEgressAllowed = false` invariant. Because the loop *does* acquire
external corpus at creation time, the Persona Studio contract marks
`self_learning` with `enrichesFromExternalCorpus = true`, so publication
must carry provenance. Engine + readouts live in
`packages/lifeform-cultivation`; operator console in
`apps/dlaas-portal` → `/[locale]/cultivation`.

### Session Mentor Intake

Human mentor intervention during an active human-AI collaboration uses a
session-local intake route:

```http
POST /v1/sessions/{session_id}/mentor-intakes
```

Request shape:

```json
{
  "guidance": "When this user asks for a plan, first confirm the boundary and then give two options.",
  "mentor_id": "mentor:alice",
  "apply_mode": "apply_to_session",
  "protocol_id": "mentor:plan-boundary-v1",
  "advisor_name": "Planning Mentor",
  "reviewer_level": "l4"
}
```

`apply_mode`:

- `classify_only`: classify and return the routed owner; no state changes.
- `apply_to_session`: when classification is `protocol` or `boundary`, extract a `BehaviorProtocol` and load it into the current session's `ProtocolRegistryModule`, affecting the next turn via `ActiveMixtureSnapshot`.
- `submit_for_review`: reserved for submitting the extracted candidate to the service-level uptake review queue; it does not mutate the current session unless separately applied.

The route deliberately does **not** change `/v1/protocols/.../approve` semantics:
service-level approved protocols still seed new sessions only. Live mentor intake is a separate session-local path so operator guidance can affect the current collaboration without turning the approved registry into a hidden global runtime owner.

## Safety Protocol Aliases

Safety injection is protocol-only:

```http
POST /dlaas/v1/instances/{ai_id}/safety/protocols
GET  /dlaas/v1/instances/{ai_id}/safety/protocols
POST /dlaas/v1/instances/{ai_id}/safety/protocols/{submission_id}/approve
POST /dlaas/v1/instances/{ai_id}/safety/protocols/{protocol_id}/load
```

These routes are aliases to protocol submission / approve / library
load. The platform validates that a safety submission describes
boundary contracts before accepting it. It never writes
`boundary_policy` or `BoundaryPriorHint` directly.

## Life Blueprint Catalog

Blueprints are adoption-time composition recipes.

```http
GET /dlaas/v1/catalog/blueprints
GET /dlaas/v1/catalog/verticals
GET /dlaas/v1/catalog/substrate-profiles
GET /dlaas/v1/catalog/protocols
GET /dlaas/v1/catalog/tool-policies
GET /dlaas/v1/catalog/training-policies
```

### Substrate profiles — registry-backed (implemented)

`GET /dlaas/v1/catalog/substrate-profiles` is now served from the
`SubstrateProfileRegistry`
([`substrate_profiles.py`](../../packages/dlaas-platform-api/src/dlaas_platform_api/substrate_profiles.py))
rather than a hardcoded list, and returns `default_substrate_profile_id`
plus each profile's `mode` / `adapter_policy` / `runtime_backend` /
`allow_rare_heavy_refresh` / `model_id_hint`. Shipped profiles:
`shared-frozen`, `shared-frozen-persona-lora`, `synthetic-dev`,
`synthetic-dev-persona-lora`, and `vllm-shared-frozen-persona-lora`
(`runtime_backend="vllm"`).

Adoption-time enforcement (implemented in
[`control_plane.py`](../../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py)):

- An explicit `substrate_profile_id` that is unknown → `400
  unknown_substrate_profile`. An empty id stays mode-agnostic
  (back-compat) and resolves a permissive adapter policy.
- A profile whose `mode` does not match the running substrate (synthetic
  vs shared_frozen) → `409 substrate_profile_mismatch`.
- `adapter_policy` is now **enforced**: `none` binds the figure bundle
  for L1/L3/L4 but disables the L2 persona-LoRA overlay at the
  activation site; `persona_lora` permits it. The resolved policy is
  persisted on the service contract and threaded through the
  SessionManager to the synthesizer.
- A figure bundle whose `compatible_substrates` (substrate fingerprints)
  do not include the running substrate's `model_id` → `409
  figure_bundle_substrate_incompatible` (substrate-upgrade-protocol).
- `runtime_backend` (`transformers` / `vllm`) on the profile is now selected
  at pod startup via `build_runtime_for_profile`; an explicit profile whose
  backend differs from the running substrate → `409
  substrate_backend_mismatch`.

### Training jobs — executor (implemented, opt-in)

`adapter_candidate` / training jobs are no longer record-only. With
`VZ_TRAINING_WORKER=1` a background `TrainingJobExecutor`
([`training_executor.py`](../../packages/dlaas-platform-api/src/dlaas_platform_api/training_executor.py))
drains `pending` jobs through a pluggable runner and advances
`pending → running → succeeded/failed`, persisting to the registry
`training_jobs` table. Job create now enforces policy: `adapter_candidate`
requires `training.allow_adapter_training` and
`substrate.allow_rare_heavy_refresh`, else `403`. Promotion still requires
`gate_evidence` (R10 — no hot substrate update). Default (worker off) keeps the
legacy record-only behaviour.

### Per-end-user identity (implemented)

`end_user_ref` + `session_id` remain caller-supplied. Reusing one `session_id`
for a different `end_user_ref` now returns `409 session_end_user_mismatch`
(opt-out `VZ_ALLOW_SESSION_END_USER_REMAP=1`). Two-layer `tenant:end_user`
memory scope and per-`ai_id` memory roots are opt-in
(`VZ_TWO_LAYER_SCOPE=1`, `VZ_PER_AI_MEMORY_ROOT=1`); default stays single-layer
to avoid re-keying existing on-disk memory.

Example blueprint:

```json
{
  "blueprint_id": "growth-advisor/cheng-laoshi/private-domain-v1",
  "display_name": "Cheng Laoshi Growth Advisor",
  "vertical": {"vertical_id": "growth_advisor", "profile_id": "cheng_laoshi"},
  "substrate": {"substrate_profile_id": "qwen3-max-shared", "mode": "shared_frozen"},
  "protocols": {"autoload": ["growth_advisor:cheng-laoshi"]},
  "memory": {"scope_strategy": "tenant_ai_end_user"},
  "tools": {"tool_policy_id": "growth-advisor-wechat-readonly"},
  "ops": {"awake_strategy": "on_demand", "handoff_policy_id": "growth-advisor-standard"},
  "training": {"allow_protocol_intake": true, "allow_adapter_training": false},
  "evaluation_gates": ["boundary-baseline", "protocol-effective", "handoff-slo"]
}
```

Adoption may pass an explicit `adoption_config` or a `blueprint_id`
plus overrides. The resolved config is still frozen into the adoption
contract.

## Acceptance Gates

- Legacy `/v1/*` and `/dlaas/*` paths continue to pass.
- `/dlaas/v1/*` aliases pass equivalent tests.
- OpenAI `metadata["dlaas.ai_id"]` routes to the correct adopted instance.
- OpenAI `tools` / `tool_choice` parse and can return assistant `tool_calls`.
- OpenAI `role="tool"` messages route to `submit_tool_result`, not text turns.
- Adoption stores vertical/substrate/protocol/memory/tool/ops/training choices.
- Adoption tool policy filters visible/invokable affordances.
- Asset intake can store-only, simple-ingest text/PDF/DOCX, create deep-read jobs, create training-candidate jobs, or store images pending a vision extractor.
- Auto asset-intent routing uses schema-constrained classification, not natural-language keyword branching.
- Protocol submission -> approve -> library load affects the next session's protocol seed path.
- Feedback and environment outcomes produce typed evidence.
- Native chat may emit `tool_call` and `tool_task` output acts.
- Rare-heavy training jobs cannot promote without gate evidence.
- Normal clients receive curated readouts but cannot fetch raw snapshots.
- Admin/service callers can fetch selected raw snapshots.
- Safety boundary injection is accepted only through protocol submission aliases.
- Catalog endpoints expose composable life blueprints and adoption components.
