# DLaaS API v1

> Status: SHADOW contract
> Last updated: 2026-05-22
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

Long-running affordances use task handles instead of blocking the chat turn:

```http
GET  /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}
POST /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}/cancel
```

Task status values: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
Completion is submitted through the same result path as fast tools, preserving
PE lineage and credit attribution via `plan_ref`.

## Observability And Explainability

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
