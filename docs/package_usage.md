# Volvence Zero Package Usage

> Scope: local package use for other projects on the same machine.
> Status: draft
> Last updated: 2026-04-25

This guide explains how to use the Volvence Zero brain kernel as a local Python package. It does not publish the package, upload code, download model weights, or start a public service.

## Install Locally

From this repository:

```bash
cd /Users/mengfu/Documents/GitHub/VolvenceZero
python -m pip install -e . --no-deps
```

From another project on the same machine:

```bash
python -m pip install -e /Users/mengfu/Documents/GitHub/VolvenceZero --no-deps
```

You can also add this to another project's `requirements.txt`:

```text
-e /Users/mengfu/Documents/GitHub/VolvenceZero
```

## Stable API

Use `volvence_zero.brain` as the package-facing API:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(BrainConfig())
session = brain.create_session(session_id="local-session")
result = session.run_turn("I feel stuck and need help deciding.")

print(result.response.text)
```

The default config uses `substrate_mode="synthetic"`, so it does not require Qwen, Hugging Face model weights, `torch`, or `transformers`.

## Core Capabilities

The package exposes the Volvence Zero brain kernel as a session-oriented runtime. External products can use it for:

- **Multi-turn continuity**: each `BrainSession` carries memory, prediction error, regime, temporal control, semantic state, and session-post slow-loop state across turns.
- **ETA temporal abstraction**: temporal control publishes the active abstract action and controller state, so the session can continue, revise, clarify, execute, or repair instead of treating every turn as isolated text.
- **NL multi-timescale learning**: online-fast, session-medium, background-slow, and rare-heavy paths are kept separate. Fast state updates happen during the turn; session-post slow-loop evidence can later consolidate experience.
- **Continuous memory**: `memory` keeps transient, episodic, durable, and derived memory snapshots, plus retrieved entries and lifecycle metrics.
- **Dual-track state**: `dual_track` separates world/task pressure from self/relationship pressure, so task progress does not collapse relationship continuity.
- **Prediction error chain**: `prediction_error` publishes evaluated prediction, actual outcome, next prediction, and error signal. This is the primary learning signal, while evaluation remains readout/gating evidence.
- **Regime identity**: `regime` publishes the current social/cognitive interaction frame rather than a prompt label.
- **Semantic state owners**: first-class owners publish plan/intent, commitment, open-loop, user-model, execution-result, belief/assumption, relationship-state, goal/value, and boundary/consent snapshots.
- **External semantic adapters**: tool results, product profile/settings, task/calendar events, and reviewed knowledge enter semantic owners through typed events and proposals, not through prompt text.
- **Domain knowledge**: `domain_knowledge` publishes compact, source-aware knowledge hits and unresolved conflicts. External facts should enter as reviewed knowledge candidates or domain packages, not by mutating memory.
- **Case memory**: `case_memory` publishes similar interaction/case patterns, risk markers, continuum band information, and relevance scores.
- **Strategy playbook**: `strategy_playbook` publishes matched strategy priors and recommended response ordering derived from experience.
- **Boundary policy**: `boundary_policy` publishes clarification, citation, referral, disclaimer, and professional-scope constraints.
- **Domain experience injection**: vertical knowledge, cases, playbooks, and boundary hints can be provided as `DomainExperiencePackage` data.
- **Response assembly**: `response_assembly` grounds generated replies in compact public runtime state, including memory, domain knowledge, case memory, playbook hints, boundaries, and semantic-state residue.
- **Evaluation evidence**: each turn publishes readout scores for task, relationship, learning, abstraction, and safety evidence.

## Semantic Owners

The semantic owners are the main structured state surfaces for product integrations:

| Slot | What It Owns | How To Feed It |
|------|--------------|----------------|
| `plan_intent` | active plans, deferred intents, standing plans, candidate plans, completed plan refs | dialogue proposals, `submit_task_event()`, tool result `plan_ref` |
| `commitment` | promises, trust obligations, completed/at-risk commitments | `submit_task_event(commitment_ref=...)`, structured proposal runtime |
| `open_loop` | unresolved questions, pending confirmations, blocked work, follow-up items | failed tool results, pending/blocked task events, reviewed knowledge follow-up |
| `user_model` | stable preferences, working style, sensitive boundaries, durable user goals | `submit_profile_event(preferences=...)` |
| `execution_result` | attempted/completed/failed actions and artifact references | `submit_tool_result(...)`, completed/failed task events |
| `belief_assumption` | beliefs, assumptions, verification needs, contradictions | tool evidence and reviewed knowledge events |
| `relationship_state` | trust, continuity, repair pressure, rapport signals, tensions | profile relationship notes or structured proposal runtime |
| `goal_value` | explicit goals, value priorities, tradeoff notes, active goal | profile goals, reviewed knowledge relevance, structured proposal runtime |
| `boundary_consent` | granted/missing/denied consent, memory consent, external-action consent | profile consent grants/denials, structured proposal runtime |

All semantic owner snapshots are immutable public state. External products should submit events or proposals and then read the next turn's snapshots.

## Knowledge And Experience Owners

Knowledge and experience are separate from semantic state:

| Slot | What It Publishes | How To Feed It |
|------|-------------------|----------------|
| `domain_knowledge` | source-aware knowledge hits, citations, domains, conflicts | `DomainExperiencePackage`, reviewed knowledge candidates, rare-heavy reviewed imports |
| `case_memory` | similar cases, problem patterns, risk markers, continuum locations | case records in `DomainExperiencePackage`, session-post promoted cases |
| `strategy_playbook` | matched strategy rules, recommended ordering, pacing hints | playbook rules in `DomainExperiencePackage`, session-post/rare-heavy distilled playbooks |
| `experience_fast_prior` | fast-path biases derived from delayed outcomes and experience consolidation | session-post slow-loop results and delayed credit summaries |
| `experience_consolidation` | experience deltas, application prior updates, writeback reports | session-post slow-loop completions |

Keep factual knowledge in `domain_knowledge`; keep semantic task/user/plan state in semantic owners. A reviewed document can influence `belief_assumption` or `goal_value`, but the factual source itself remains owned by the knowledge layer.

## What You Can Read From A Turn

`run_turn()` returns an `AgentTurnResult`. Most product integrations only need `result.response.text`, but advanced callers can inspect public snapshots:

```python
result = session.run_turn("Help me continue the rollout plan.")

print(result.response.text)
print(result.active_regime)
print(result.active_abstract_action)

plan = result.active_snapshots["plan_intent"].value
open_loops = result.active_snapshots["open_loop"].value
execution = result.active_snapshots["execution_result"].value
consent = result.active_snapshots["boundary_consent"].value

print(plan.active_goal)
print(open_loops.unresolved_loops)
print(execution.completed_actions)
print(consent.external_action_consent)
```

Snapshot values are immutable dataclasses. Treat them as read-only public state; do not mutate owner internals from the product side.

Useful snapshot groups:

- Task and plan: `plan_intent`, `open_loop`, `execution_result`, `goal_value`
- User and relationship: `user_model`, `relationship_state`, `commitment`, `boundary_consent`
- Knowledge and experience: `domain_knowledge`, `case_memory`, `strategy_playbook`, `experience_fast_prior`
- Control and learning: `temporal_abstraction`, `regime`, `prediction_error`, `evaluation`, `credit`, `reflection`

## How To Provide Feedback

Feedback should enter through the owner-appropriate public path:

| Feedback Type | Preferred API | Affects |
|---------------|---------------|---------|
| Tool/API/file result | `session.submit_tool_result(...)` | `execution_result`, `belief_assumption`, `open_loop`, optional `plan_intent` |
| User setting/profile update | `session.submit_profile_event(...)` | `user_model`, `goal_value`, `boundary_consent`, optional `relationship_state` |
| Task/calendar/reminder status | `session.submit_task_event(...)` | `plan_intent`, `open_loop`, `commitment`, `execution_result` |
| Reviewed external fact | `session.submit_reviewed_knowledge_event(...)` | `belief_assumption`, `goal_value`, `open_loop`; factual store remains `domain_knowledge` |
| Domain package seed | `BrainConfig(domain_experience_packages=(...))` | `domain_knowledge`, `case_memory`, `strategy_playbook`, `boundary_policy`, rare-heavy application state |
| Conversation feedback | next `run_turn(user_input)` | semantic proposal runtime, memory, prediction error, regime, response assembly |
| Session outcome | keep using the same `BrainSession`; session-post slow loop runs at context boundaries | `experience_consolidation`, `experience_fast_prior`, reflection/writeback evidence |

Do not write directly into owner stores from product code. Submit typed events, domain packages, or a structured `SemanticProposalRuntime`; then read the public snapshots from the next `AgentTurnResult`.

## Scenario Examples

### Continue After A Tool Succeeds

Use this when your product runs a tool outside the brain kernel and wants the next turn to know what actually happened.

```python
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="deploy-session")

session.submit_tool_result(
    event_id="tool:deploy:42",
    tool_name="deploy",
    action_id="deploy:42",
    status="succeeded",
    summary="Production deploy completed",
    detail="The deploy finished and produced log artifact deploy-log-42.",
    artifact_refs=("deploy-log-42",),
    plan_ref="production-rollout-plan",
)

result = session.run_turn("Continue the rollout from here.")

execution = result.active_snapshots["execution_result"].value
plan = result.active_snapshots["plan_intent"].value

print(result.response.text)
print(execution.completed_actions)
print(plan.plan_revision_count)
```

Expected effect: `execution_result` records the completed action, `belief_assumption` receives tool evidence, and `plan_intent` can revise the active plan when `plan_ref` is provided.

### Recover After A Tool Fails

Use failed tool results to create an open loop instead of hiding the failure in a reply.

```python
session.submit_tool_result(
    event_id="tool:calendar:17",
    tool_name="calendar",
    action_id="create-event:17",
    status="failed",
    summary="Calendar event was not created",
    detail="The calendar API returned an authorization error.",
    plan_ref="schedule-review-plan",
)

result = session.run_turn("What should we do about the scheduling step?")

open_loop = result.active_snapshots["open_loop"].value
execution = result.active_snapshots["execution_result"].value

print(open_loop.unresolved_loops)
print(execution.failed_actions)
```

Expected effect: `execution_result` records the failure, `open_loop` tracks the unresolved follow-up, and response assembly can ask for the missing authorization or suggest a safe next step.

### Apply Product Profile And Consent

Use profile events for product-owned settings, preferences, goals, and consent. Do not encode these as hidden prompt text.

```python
session.submit_profile_event(
    event_id="profile:user:1",
    source="product-settings",
    preferences=("prefers concise action plans", "likes explicit tradeoffs"),
    goals=("ship safely", "avoid unnecessary risk"),
    consent_grants=("remember planning preferences",),
    consent_denials=("take external action without confirmation",),
    relationship_note="prefers calm, direct collaboration",
)

result = session.run_turn("Help me plan the next release.")

user_model = result.active_snapshots["user_model"].value
goals = result.active_snapshots["goal_value"].value
consent = result.active_snapshots["boundary_consent"].value
relationship = result.active_snapshots["relationship_state"].value

print(user_model.stable_preferences)
print(goals.explicit_goals)
print(consent.denied_boundaries)
print(relationship.rapport_signals)
```

Expected effect: user preferences enter `user_model`, goals enter `goal_value`, denied permissions enter `boundary_consent`, and relationship notes enter `relationship_state`.

### Track A Deferred Or Completed Task

Use task events for calendar items, reminders, workflow states, and future plans.

```python
session.submit_task_event(
    event_id="task:review:1",
    task_id="weekly-review",
    status="deferred",
    summary="Review metrics tomorrow",
    detail="The weekly metrics review should happen tomorrow morning.",
    due_hint="tomorrow morning",
    commitment_ref="follow up on weekly metrics",
)

first = session.run_turn("Keep that for later.")

session.submit_task_event(
    event_id="task:review:2",
    task_id="weekly-review",
    status="completed",
    summary="Weekly metrics reviewed",
    detail="The product dashboard and error logs were reviewed.",
    commitment_ref="follow up on weekly metrics",
)

second = session.run_turn("Summarize what changed.")

print(first.active_snapshots["plan_intent"].value.deferred_intents)
print(second.active_snapshots["commitment"].value.honored_commitment_refs)
print(second.active_snapshots["execution_result"].value.completed_actions)
```

Expected effect: deferred work stays in `plan_intent` / `open_loop`; completion updates `commitment` and `execution_result`.

### Add Reviewed Knowledge Without Overwriting Memory

Use reviewed knowledge events when an external reviewer or trusted pipeline has already validated a fact. This does not directly mutate `domain_knowledge`; factual stores should still be fed by domain packages or reviewed knowledge candidates.

```python
session.submit_reviewed_knowledge_event(
    event_id="knowledge:rollout:1",
    knowledge_id="reviewed:rollout-safety",
    summary="Staged rollout reduces blast radius",
    detail="The reviewed source recommends staged rollout before full release.",
    source_label="internal-reviewed-runbook",
    confidence=0.88,
    relevance_hint="release safety",
    needs_followup=True,
)

result = session.run_turn("Use the reviewed rollout note in our plan.")

beliefs = result.active_snapshots["belief_assumption"].value
open_loop = result.active_snapshots["open_loop"].value
domain_knowledge = result.active_snapshots["domain_knowledge"].value

print(beliefs.beliefs)
print(open_loop.unresolved_loops)
print(domain_knowledge.hits)
```

Expected effect: `belief_assumption` can use the reviewed summary, `open_loop` can track follow-up if needed, and `domain_knowledge` remains the owner of factual knowledge stores.

### Seed A Vertical Domain Package

Use `DomainExperiencePackage` when you want the brain to start with domain knowledge, case patterns, playbooks, and boundary hints.

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        domain_experience_packages=(my_domain_package,),
    )
)
session = brain.create_session(session_id="domain-example")

result = session.run_turn("I need help with a domain-specific decision.")

knowledge = result.active_snapshots["domain_knowledge"].value
cases = result.active_snapshots["case_memory"].value
playbook = result.active_snapshots["strategy_playbook"].value
boundary = result.active_snapshots["boundary_policy"].value

print(knowledge.hits)
print(cases.hits)
print(playbook.matched_rules)
print(boundary.active_decision)
```

Expected effect: the package compiles into existing application owners. It does not create a new runtime owner and does not hardcode vertical behavior into the core kernel.

### Close The Loop With User Feedback

User feedback should usually be passed as the next turn. The runtime will update memory, prediction error, semantic owners, regime, and evaluation evidence through the normal chain.

```python
first = session.run_turn("Make a careful migration plan.")

# Product executes or displays the plan, then the user responds.
second = session.run_turn(
    "That plan mostly works, but the database step must wait until next week."
)

prediction_error = second.active_snapshots["prediction_error"].value
plan = second.active_snapshots["plan_intent"].value
evaluation = second.active_snapshots["evaluation"].value

print(prediction_error.error)
print(plan.active_constraints)
print(evaluation.turn_scores)
```

Expected effect: conversation feedback enters the normal online-fast loop. If your product has an external task status or tool result, submit it as an event before the next `run_turn()`.

### Combine Tool, Profile, And Task Inputs In One Turn

You can enqueue several event types before a turn. The session drains them once and merges their proposals into the relevant owners.

```python
session.submit_profile_event(
    event_id="profile:release:1",
    source="product-settings",
    preferences=("wants risk called out explicitly",),
    consent_denials=("deploy without approval",),
)
session.submit_tool_result(
    event_id="tool:test:1",
    tool_name="test-runner",
    action_id="pytest:release",
    status="succeeded",
    summary="Release tests passed",
    detail="All release tests passed locally.",
    artifact_refs=("pytest-report",),
    plan_ref="release-plan",
)
session.submit_task_event(
    event_id="task:approval:1",
    task_id="release-approval",
    status="pending",
    summary="Approval still needed",
    detail="Human approval is required before deployment.",
    commitment_ref="wait for approval",
)

result = session.run_turn("Are we ready to release?")

print(result.active_snapshots["execution_result"].value.completed_actions)
print(result.active_snapshots["boundary_consent"].value.denied_boundaries)
print(result.active_snapshots["open_loop"].value.unresolved_loops)
print(result.response.text)
```

Expected effect: the response can account for completed tests, missing approval, and denied deployment consent without relying on hidden prompt state.

## Async Use

If your app already runs inside an event loop, use `run_turn_async()`:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(BrainConfig())
session = brain.create_session(session_id="async-session")

result = await session.run_turn_async("Help me slow down and think this through.")
```

## Use Hugging Face / Qwen Explicitly

Install the optional runtime dependencies:

```bash
python -m pip install -e "/Users/mengfu/Documents/GitHub/VolvenceZero[hf]"
```

Then opt in explicitly:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        substrate_mode="hf",
        substrate_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        substrate_local_files_only=True,
    )
)
```

Model weights are not part of the package. They must exist in the runtime environment or Hugging Face cache. Keep `substrate_local_files_only=True` when you want to prevent network model fetches.

## Inject Your Own Runtime

Products and services can provide their own substrate runtime:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(substrate_mode="injected"),
    substrate_runtime=my_runtime,
)
```

This is the preferred path for a service that uses a remote model endpoint or a separately managed inference process.

## Load Domain Experience Packages

Vertical experience should be passed as `DomainExperiencePackage` data, not hardcoded into the kernel:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        domain_experience_packages=(my_domain_package,),
    )
)
session = brain.create_session(session_id="domain-session")
```

Packages are compiled into existing application owners: domain knowledge, case memory, strategy playbook, boundary hints, and rare-heavy application state.

## Inject Semantic State Proposals

The core package includes the semantic-state owner contracts and the prompt/schema resources needed by an external structured runtime:

```python
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.semantic_state import (
    SemanticProposalRuntime,
    load_semantic_json_schema,
    load_semantic_prompt_template,
)

prompt = load_semantic_prompt_template()
schema = load_semantic_json_schema()

brain = Brain(
    BrainConfig(),
    semantic_proposal_runtime=my_runtime,  # implements SemanticProposalRuntime
)
```

The default synthetic path uses `NoOpSemanticProposalRuntime`, so local installs still run without a model-backed semantic extractor. Product runtimes can inject a structured proposal runtime without becoming a second owner of the semantic state; each owner still publishes its own immutable snapshot.

## Submit External Semantic Events

External information does not need to arrive through dialogue. Submit structured events before the next turn; the session will drain them into semantic owner proposals exactly once:

```python
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="product-session")

session.submit_tool_result(
    event_id="tool:deploy:1",
    tool_name="deploy",
    action_id="deploy:1",
    status="succeeded",
    summary="Deploy finished",
    detail="The deploy tool produced artifact deploy-log-1.",
    artifact_refs=("deploy-log-1",),
    plan_ref="launch-plan",
)

result = session.run_turn("Continue from the deployment result.")
```

The package also exposes `submit_profile_event()`, `submit_task_event()`, and `submit_reviewed_knowledge_event()`. These helpers enqueue typed external events only; semantic owners still merge and publish their own immutable snapshots.

Typical mappings:

- `submit_tool_result(...)` updates `execution_result`, `belief_assumption`, `open_loop`, and optionally `plan_intent`.
- `submit_profile_event(...)` updates `user_model`, `goal_value`, `boundary_consent`, and optionally `relationship_state`.
- `submit_task_event(...)` updates `plan_intent`, `open_loop`, `commitment`, and `execution_result`.
- `submit_reviewed_knowledge_event(...)` updates `belief_assumption`, `goal_value`, and `open_loop`; factual knowledge stores remain owned by `domain_knowledge`.

Example profile and task events:

```python
session.submit_profile_event(
    event_id="profile:1",
    source="product-settings",
    preferences=("prefers concise plans",),
    goals=("ship safely",),
    consent_denials=("external action without confirmation",),
)

session.submit_task_event(
    event_id="task:launch:1",
    task_id="launch-checklist",
    status="completed",
    summary="Launch checklist completed",
    detail="All launch checklist items were marked done.",
    commitment_ref="finish launch checklist",
)

result = session.run_turn("What should we do next?")
```

## Persistence

Persistence paths are outside the package:

```python
from volvence_zero.brain import Brain, BrainConfig

brain = Brain(
    BrainConfig(
        application_persistence_dir="/path/to/product/storage",
    )
)
```

Do not store product user data, secrets, or deployment state inside the package directory.

## Public vs Internal Surfaces

Stable package-facing API:

- `volvence_zero.brain.Brain`
- `volvence_zero.brain.BrainConfig`
- `volvence_zero.brain.BrainSession`
- `volvence_zero.application.domain_experience` package schema and compiler
- `volvence_zero.semantic_state` proposal runtime contracts and bundled schema/prompt resources

Internal or research-heavy surfaces:

- `volvence_zero.agent.dialogue_benchmark`
- `volvence_zero.agent.eta_proof_benchmark`
- broad re-exports from `volvence_zero.agent`
- internal integration helpers with leading underscores

These may remain useful for development and evaluation, but product projects should prefer `volvence_zero.brain`.

## What This Does Not Do

Using the package locally does not:

- upload to PyPI
- push to GitHub
- start a public HTTP service
- package Qwen or other model weights
- include product user data
- download model weights unless you explicitly configure a runtime path that does so

## Quick Smoke Test

```bash
python - <<'PY'
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="smoke")
result = session.run_turn("I need help making a careful decision.")
print(result.response.text)
PY
```
