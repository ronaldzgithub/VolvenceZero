# Real Open Dialogue Learning Loop

> Purpose: turn real/open companion conversations into durable PE, memory, credit, reflection, regime-prior, and ETA/NL learning evidence without bypassing owner boundaries.

## Bottom Line

Current companion benchmarks prove a mechanism slice, not a mature open-world companion. Real interaction already enters part of the learning chain through `run_turn()`, but open-world improvement will not become reliable unless we explicitly close five loops:

1. capture real turn telemetry and user-visible outcomes,
2. convert failures into prediction-error and delayed-credit evidence,
3. end scenes and drain slow reflection so NL can consolidate,
4. persist the right stores for the same lifeform/user scope,
5. replay and compare full ETA/NL against matched controls.

The goal is not to add prompt rules. The goal is to make real failures flow through existing owners and become reusable temporal/memory structure.

## What Is Already Automatic

When a real user turn goes through `AgentSessionRunner.run_turn(...)`, these paths already execute:

- substrate capture publishes residual/semantic feature readouts, including `semantic_*_pull` when the runtime does not provide a full semantic surface;
- evaluation reads substrate, memory, and dual-track snapshots and publishes turn metrics;
- prediction error consumes upstream state and exposes task / relationship / regime / action error;
- regime owner scores candidate regimes, records pending delayed outcomes, and publishes delayed attribution / rolling payoff readouts;
- response assembly reads owner snapshots and emits `expression_intent`, speech plan, and support/decision diagnostics;
- memory owner can write turn-level memory through its formal write API;
- joint loop sees traces and PE-derived signals, giving ETA a path toward abstract-action family reuse.

This is enough for real conversations to create observable traces. It is not enough by itself to guarantee durable learning.

## What Is Not Automatic Enough Yet

### User Failure Must Be Observable

The system cannot learn a failed emotional decision support move if the failure never becomes an outcome. Real users often signal failure indirectly:

- "That felt cold."
- "You are optimizing me."
- "I wanted to be heard."
- "That helped, I can choose now."
- silence, abandonment, or repeated clarification requests.

Next step: add a first-class real-dialogue outcome intake path that turns explicit user feedback and session-end outcomes into structured evidence for prediction error and delayed credit. This must not be a keyword router. The allowed sources are typed user feedback events, explicit environment outcomes, human review entries, or structured LLM/embedding proposal sources that enter owner-owned proposal paths.

### Scene Boundaries Must Be Fired

NL consolidation depends on scene/session boundaries. If we only call `run_turn()` forever, slow reflection and session-post writeback may stay underused.

Required runtime discipline:

- call `end_scene(reason=...)` at natural boundaries;
- call or await `drain_session_post_slow_loop()` when evaluating learning;
- preserve completed session reports for cross-session evidence;
- log whether slow-loop jobs were queued, completed, stale, or blocked.

### Persistence Must Be Scoped

Default memory isolation is a feature, not a bug. A companion should not learn one user's preferences into another user's session.

Open-world trials need explicit scope:

- same lifeform + same user: shared `MemoryStore` / persisted stores enabled;
- different user or anonymous run: isolated store;
- test replay: immutable exported artifacts, not live owner mutation.

Without this, "it did not remember me" and "it leaked memory across users" are both easy to misread.

### Credit Must Stay PE-First

Evaluation and human ratings can widen evidence, but they must not replace PE / credit as learning primitives.

Correct flow:

```text
real outcome -> prediction_error -> credit / regime delayed attribution -> reflection / gate evidence
```

Wrong flow:

```text
human rating -> directly change regime weights
```

Credit should continue to aggregate from prediction error, delayed attributions, rolling payoff, and abstract-action evidence. Human review can become gate context or outcome evidence, not a second owner.

### ETA Family Reuse Needs Repetition

ETA does not become robust from one moving conversation. It needs repeated trajectories where abstract actions are selected, delayed outcomes arrive, and family payoff becomes distinguishable.

Minimum useful evidence:

- the same abstract-action family appears across multiple cases;
- family reuse improves or stabilizes later turns;
- `eta-off` / `pe-drive-off` / `timescale-off` controls lose measurable trajectory quality;
- delayed outcomes can be traced back to source turn, regime, abstract action, and family version.

## Implementation Phases

### Phase 1: Real Dialogue Capture Packet

Add a lightweight export path for every real/open session:

- per-turn `SnapshotReplay` summary: substrate semantic pulls, evaluation metrics, PE components, active regime, candidate regimes, abstract action, switch evidence, expression intent;
- session boundary summary: closed scene reason, pending/completed slow-loop jobs, writeback result, memory deltas;
- outcome summary: explicit user feedback, environment outcome, or human review item if available.

Artifact target:

- `artifacts/open_dialogue/<session_id>/turns.jsonl`
- `artifacts/open_dialogue/<session_id>/session_summary.json`

Rules:

- read snapshots; do not mutate owners;
- do not store raw private user data in review bundles unless explicitly opted in;
- include provenance and scope metadata.

### Phase 2: Outcome Intake

Create a typed feedback path for real usage:

```python
submit_dialogue_outcome(
    turn_id=...,
    outcome_kind="helped" | "misread" | "too_cold" | "over_directive" | "decision_clearer" | "unsafe" | "abandoned",
    source="user_explicit" | "human_review" | "environment",
    confidence=...,
    evidence_ref=...,
)
```

This function should not directly update memory, regime, or ETA. It should create typed outcome evidence consumed by the prediction-error / credit path.

Acceptance:

- an explicit repair outcome increases relationship/regime PE;
- a successful decision-clarity outcome creates positive delayed credit for the source regime / abstract action;
- missing or ambiguous feedback is represented as low-confidence evidence, not silently treated as success.

### Phase 3: Session-Post NL Closure

Define a required open-dialogue session lifecycle:

1. run turns through normal `run_turn()`;
2. submit explicit outcomes when available;
3. call `end_scene(reason=...)`;
4. drain session-post slow loop;
5. export memory / reflection / regime evidence;
6. start the next session with the same scoped store only when user scope is intentionally shared.

Acceptance:

- reflection writeback produces bounded memory or strategy proposals;
- applied writes use checkpoints and are rollback-safe;
- memory stratum flow shows transient/episodic records becoming durable only when evidence supports it;
- no consumer reconstructs memory internals from raw text.

### Phase 4: ETA/NL Matched Controls

For each open-dialogue batch, run or simulate matched controls:

- `full-nl`: PE + ETA + reflection + memory active;
- `eta-off`: temporal abstraction reduced;
- `pe-drive-off`: PE visible as readout but not dominant in schedule/reward;
- `timescale-off`: no session-post slow-loop benefit;
- `no-reflection`: fast path only;
- `no-shared-memory`: same prompts without cross-session continuity.

Required reports:

- `dialogue_option_discovery_report.json`;
- `pe_counterfactual_closure_report.json`;
- `longitudinal_dialogue_report.json`;
- `nl_ablation_matrix_report.json`;
- `memory_stratum_flow_report.json`;
- `regime_lockin_report.json`.

Do not claim retain-level open companion ability unless the full path beats matched controls on trajectory metrics and does not increase safety/regression risk.

### Phase 5: Human Review Anchor

Internal metrics are not enough for emotional decision support. Add blinded review packets that ask reviewers to rate:

- felt heard before advice;
- decision clarity after the exchange;
- autonomy preserved;
- repair quality after misread;
- over-directiveness;
- warmth without generic therapy language.

Human review should feed evidence bundles and gates, not directly rewrite owners.

## Minimal Operational Protocol

For a real companion trial:

1. Create or load a scoped lifeform/user store.
2. Run the dialogue through normal `run_turn()`.
3. When the user reacts, submit typed outcome evidence.
4. End the scene at a meaningful boundary.
5. Drain session-post slow loop.
6. Export open-dialogue artifacts.
7. Replay the session against controls.
8. Review failures and only then promote memory/strategy/artifact changes.

If any step is skipped, the trial can still be useful as telemetry, but it should not be treated as proof of learning.

## Engineering Tasks

### T1: Add Open Dialogue Artifact Export

Owner: runtime / evaluation evidence layer.

Deliverables:

- session-level artifact writer;
- per-turn compact snapshot summary;
- provenance, user scope, and privacy flags;
- tests verifying export is read-only.

### T2: Add Typed Outcome Intake

Owner: environment / prediction-error boundary.

Deliverables:

- typed outcome dataclass or API;
- mapping into PE/action context without direct memory/regime mutation;
- tests for helped, misread, over-directive, decision-clearer, abandoned.

### T3: Add Session Lifecycle Helper

Owner: lifeform runtime facade.

Deliverables:

- helper that runs `end_scene`, drains slow loop, collects reports, and exports artifacts;
- explicit status for completed / stale / blocked writeback;
- tests proving no slow-loop writeback happens without owner apply path.

### T4: Add Open Companion Batch Runner

Owner: lifeform-evolution benchmark/evidence layer.

Deliverables:

- batch runner for real/open transcripts;
- matched-control execution where available;
- summary verdict that separates telemetry, weak evidence, and retain candidate.

### T5: Add Human Review Packet

Owner: evidence program.

Deliverables:

- blind packet export;
- CSV/JSON rating ingest;
- aggregate dimensions for emotional decision support;
- gate context only, no direct owner write.

## Acceptance Gates

### Gate A: Failure Becomes PE

Given a real/user-marked misread, the next snapshot must show non-zero relationship/regime PE and a traceable outcome record.

### Gate B: Outcome Becomes Delayed Credit

Given a later positive or negative outcome, the regime snapshot must publish delayed attribution that includes source turn, regime, abstract action, and family version when available.

### Gate C: Slow Loop Changes Memory Safely

After scene end and slow-loop drain, memory/reflection writeback may apply only via checkpointed owner APIs. Rollback evidence must exist.

### Gate D: Full Path Beats Controls

Across an open batch, full ETA/NL must beat at least `eta-off`, `pe-drive-off`, and `timescale-off` on trajectory quality without worsening safety.

### Gate E: Human Legibility

Blind review must show users are more likely to feel heard, less over-directed, and clearer about the decision after full-path responses.

## Non-Goals

- Do not add raw-text keyword matching for emotions or decisions.
- Do not make response assembly own relationship or decision state.
- Do not write memory directly from benchmark or review code.
- Do not treat a single real conversation as evidence of open-world competence.
- Do not turn human ratings into direct reward updates.

## Success Definition

The next milestone is not "the companion feels real" in a demo. The milestone is:

> A real/open conversation failure can be traced from user outcome to prediction error, delayed credit, reflection, memory/regime evidence, and ETA/NL reuse, then replayed against matched controls with a measurable full-path advantage.

Only after that should we claim that emotional decision support is becoming an open-environment capability rather than a benchmark-shaped behavior.

## First-Principles Engineering Plan

This section consolidates the matured engineering position after iterating on
the questions "is auto-feedback enough?", "should we use an LLM judge?", and
"is this problem solvable at all?". It supersedes earlier phases when they
conflict.

### Reframing the Problem

Companion AI is not a closed-form optimization problem. There is no objective
ground truth for emotional decision support, and no learnable reward function
can recover one without external grounding. Treating it as an ML benchmark
invites self-deception.

The realistic frame is relational, not optimization-shaped:

- there is no "solved" state, only an ongoing relationship;
- mistakes are unavoidable; what matters is whether they are visible, repaired,
  and not repeated;
- trust grows through rupture-repair cycles, not through perfect responses;
- the system must support being raised by a specific user over time, not
  trained once for all users.

The optimal engineering goal is therefore:

> Build the smallest system that can enter a long-term relationship: detect
> rupture, repair specifically, remember the repair, and grow with one person
> over time, anchored to sparse but real external signal.

### Why LLM-as-Judge Is Not the Answer

An LLM evaluator can scale up review and produce structured outcome
proposals. It cannot serve as ground truth for companion AI because:

- it has no access to user wellbeing, only to text patterns;
- it shares training distribution with the substrate, so its errors are
  correlated with the system being judged;
- it has known biases (verbosity, therapy-style preference, sycophancy);
- using it as reward trains the system to please the judge, producing
  text that "looks like good support" rather than text that helps.

LLM is allowed only as a low-confidence proposal source feeding typed outcome
evidence and as a triage filter for human review. It must not become a reward
signal or an owner.

### Why Auto-Feedback Alone Is Not Enough

PE / internal readouts measure how surprised the system is by what follows,
not how the user actually felt. They will silently reward engagement-shaped
behaviors that are not always healthy. Pure auto-feedback risks an internal
self-consistency loop that drifts away from real wellbeing.

Auto-feedback is therefore high-frequency but low-confidence. It drives
stability and rapid local adaptation, not long-term policy reshaping.

### The Five-Layer Plan

The plan below is ordered by dependency. Each layer should land before the
next is treated as load-bearing.

#### Layer 1: Rupture and Repair as a First-Class Owner Primitive

This is the keystone. Without it, no other layer is meaningful.

Owner contract:

- new owner or extension of `relationship_state`: `rupture_state`;
- snapshot fields: `rupture_signal_strength`, `rupture_kind`,
  `evidence_sources`, `confidence`.

`rupture_kind` is an evidence-bucket label, not an emotion classification.
It must be derived from named structured signals (typed user outcome,
behavioral pattern flag, owner self-check pattern) and never from text
keyword detection or free LLM classification. The v0 vocabulary
(`misread`, `over_directive`, `pushed_too_fast`, `cold`, `unsafe`,
`abandoned`) is a finite set of evidence buckets matched to existing typed
signals; new kinds may only be added by introducing a new typed signal
source first.

Multi-source detection (none of these alone are sufficient):

- internal: PE spike on relationship / regime / action components;
- behavioral: repeated unmet request, explicit correction, retreat,
  cross-turn return to same unresolved theme, abandonment;
- self-check: response assembly notices structural mismatch in its own
  prior turn (for example structure-first under high emotional load);
- LLM as proposal source: flag only, low confidence, never authoritative.

Repair primitive lives inside response assembly when `rupture_state` is
active:

- name what just happened specifically; do not emit generic apology;
- offer one concrete reversible adjustment;
- do not retry the original move in the same turn;
- wait for an explicit accept / decline signal from the user.

Memory effect:

- every rupture and its repair attempt becomes a `DurableMemoryEntry`
  recording rupture kind, repair move, observed outcome;
- this is the primary substrate from which long-term relational competence
  is grown.

Guardrail: `rupture_state` must not become a hidden judge.

The single largest implementation risk in this layer is collapsing the same
self-consistency loop we already rejected for LLM judge and pure
auto-feedback. `rupture_state` is an owner aggregating evidence about the
relationship; it is not a verdict the system passes on itself. Concretely:

- `rupture_state` must be aggregated from multiple owner snapshots and named
  evidence sources, never derived from a single internal signal;
- every snapshot must carry `confidence` and `evidence_sources`; "internal
  only" must be a possible and visible value, not the default;
- detection must distinguish `internally suspected` from `externally
  confirmed`; only externally confirmed rupture earns high confidence;
- internal-only suspicion may trigger the humility primitive (Layer 4) and a
  low-cost clarification or softening, but must not by itself drive durable
  memory writes, regime prior shifts, or ETA family deprecation;
- durable updates require corroboration: behavioral correction, explicit user
  outcome, long-arc check-in answer, or human review; internal PE alone is
  not enough;
- every `rupture_state` write must be checkpointed and rollback-safe, with
  audit trail of which sources contributed and at what confidence;
- the system must be allowed to publish "I don't know whether I did wrong";
  forcing a binary verdict is itself the failure mode;
- consumers downstream must respect the confidence label: low-confidence
  rupture must not silently propagate as fact into reflection or credit.

In short, `rupture_state` is the owner that asks the question and tracks the
evidence; it is not the owner that decides the answer alone. The answer comes
from the user, the environment, or a human reviewer, integrated through this
owner with explicit uncertainty.

#### Layer 2: Per-User Scoped Persistent Memory

Without this layer, every session restarts from zero and no PE / ETA gain
survives.

Requirements:

- explicit user identity binds to a persistent `MemoryStore` path;
- default remains isolated when no identity is provided;
- session start loads scoped store; session end writes back through slow
  reflection only;
- persistent content scope: preferences, rupture-repair history, long-term
  open loops and commitments, recurring user semantic patterns;
- raw conversation text is not persisted by default;
- the user must be able to inspect, correct, and delete persisted content.

#### Layer 3: Sparse Honest External Signal

Three intake channels, ordered from highest frequency to highest trust:

1. scene-end optional feedback: at most one structured question per scene,
   skippable, semantic options
   (`this helped`, `I felt heard`, `this missed me`, `too directive`,
   `decision is clearer`, `want to come back to this`);
   feedback enters as typed outcome with high confidence;
2. long-arc check-in: when appropriate, the system can ask
   "last time you were considering X, how did that go?";
   the user reply itself is outcome ground truth;
3. human review on a small random sample (about 5% of sessions) focused on
   rupture / repair quality and over-direction risk; results enter evidence
   bundles and gates only, never directly mutate owners.

LLM responsibilities are restricted to mapping natural-language reactions
into typed outcomes (with low confidence and audit trail) and to triaging
sessions for human review. LLM never sets reward.

#### Layer 4: Calibrated Uncertainty and Humility Primitive

The system must be able to act on "I am not sure I am reading this right"
rather than always committing.

Implementation:

- owners already publish `confidence`; aggregate into a turn-level
  uncertainty readout consumed by response assembly;
- when uncertainty is high or signals conflict, response assembly emits a
  structured clarification primitive instead of acting:
  not "what do you need from me?" but
  "I might be reading this as X, but it could be Y; which feels closer?";
- the user response to this primitive is itself a low-cost, high-signal
  ground truth event that flows into typed outcome and PE.

#### Layer 5: NL Slow Loop Carries Long-Term Growth

Reuse existing `session_post_slow_loop`, `ReflectionEngine`, `MemoryStore`,
and `RegimeIdentity` delayed attribution. No new module is required.

Per-session post processing must aggregate:

- list of detected ruptures and whether repair succeeded;
- typed outcomes received via Layer 3;
- PE main trajectory and delayed attributions;
- regime / abstract action family payoff for the session.

Updates allowed:

- per-user memory and regime prior may be updated via reflection writeback,
  using existing checkpoint and rollback paths;
- cross-user learning is allowed only when human-reviewed evidence has been
  attached and the modification gate explicitly approves;
- rare-heavy artifacts and bootstrap updates remain gated by external
  evidence; auto-feedback alone is never sufficient.

### v0 Scope Lock and Risk Mitigations

The Five-Layer plan describes the long-term shape. v0 must be much narrower
or it will silently drift into the failure modes we already named. Three
specific risks must be locked down before any code lands.

Risk 1: `rupture_kind` degenerating into hand-written emotion classification.

- mitigation: kind is an evidence-bucket label, not a free-text taxonomy;
- mitigation: kind may only be derived from named typed signals; no keyword
  matching, no LLM free classification;
- mitigation: adding a new kind requires adding a new typed signal source
  first, not a heuristic;
- mitigation: contract test enforces that every published kind has a
  resolvable typed signal in `evidence_sources`.

Risk 2: LLM outcome mapping silently becoming judge or reward.

- mitigation: v0 ships with LLM mapping disabled by default;
- mitigation: outcome intake accepts only explicit user signals, environment
  events, and human review entries in v0;
- mitigation: when LLM mapping is later enabled, it must publish to a
  separate proposal channel with low confidence and an audit trail; the PE
  / credit path consumes it only as proposal evidence, never as reward;
- mitigation: ablation requirement: with LLM mapping turned off, the system
  must still produce non-trivial rupture / outcome evidence on a real
  session; if not, the loop is depending on the LLM and that is a fail.

Risk 3: F1-F12 scope too broad, milestones lose focus.

- mitigation: F1-F12 is the long-term task surface, not the v0 scope;
- mitigation: v0 ships only the minimum required to prove the loop exists
  end-to-end on one user across two sessions;
- mitigation: subsequent expansion is gated by v0 acceptance.

The v0 gate (every item is required, no substitutions):

1. Shadow rupture snapshot.
   - `rupture_state` owner ships at `WiringLevel.SHADOW`;
   - publishes structured snapshots with `evidence_sources` and `confidence`;
   - does not mutate any other owner;
   - does not influence response assembly behavior;
   - exists purely for telemetry, audit, and replay in v0.
2. Typed external signal intake.
   - `submit_dialogue_outcome(...)` accepts only explicit user signal,
     environment event, or human review entry in v0;
   - LLM-as-proposal-source is implemented but disabled by default flag;
   - outcomes route into `prediction_error` and `credit` evidence; they do
     not directly mutate memory, regime, or ETA.
3. Owner writeback only.
   - any persistence triggered by v0 (memory entry for rupture-repair,
     regime prior nudge) goes through existing owner apply paths with
     checkpoint and rollback;
   - no new direct write path is introduced;
   - the loop must be verifiably reversible by restoring the checkpoint.
4. Artifact replay.
   - every v0 session exports a replayable artifact that includes per-turn
     rupture snapshot, received external signals, owner writeback decisions,
     and post-state evidence;
   - replay must be able to rerun the same trajectory with the rupture loop
     disabled as a matched control.
5. Same-user cross-session evidence.
   - v0 is considered passing only when at least one real (or honest
     synthetic with explicit external signals) two-session trajectory
     demonstrates that a repair recorded in session A measurably changes
     behavior or memory retrieval in session B for the same user;
   - the matched control (no shared memory) must show that this change does
     not happen without the loop;
   - cross-user leakage must be verified absent.

What v0 deliberately does not include:

- LLM-driven outcome inference;
- automatic long-term regime prior reshaping from internal signals;
- rare-heavy artifact updates triggered by this loop;
- response-assembly-side repair primitive at `ACTIVE` (Layer 1 repair stays
  shadowed in v0; only the snapshot is published);
- humility / clarification primitive (Layer 4) at `ACTIVE`;
- cross-user generalization of any kind.

Mapping v0 to the F-task table:

| v0 gate item | Required from F-tasks | Notes |
|--------------|-----------------------|-------|
| Shadow rupture snapshot | F1 (snapshot only), F2 (detection wiring at SHADOW) | no F3 active; no memory write yet |
| Typed external signal intake | F6 (scene-end intake), partial F7 (deferred), explicit F11 disabled | LLM mapping flag off |
| Owner writeback only | F4 (rupture-repair memory entry kind), F10 (slow-loop aggregation) | reuse existing apply paths |
| Artifact replay | F8 (export side, no human ingest required for v0) | replay first, ingest later |
| Same-user cross-session evidence | F5 (per-user scoped store) | must include matched control |

Everything else in F1-F12 (calibrated uncertainty primitive, cross-user
review, rare-heavy gate update, full LLM proposal adapter, humility
primitive, etc.) is post-v0 and must not be required for v0 acceptance.

This lock is the entire point of v0: prove the smallest honest loop works,
then expand. If we cannot satisfy the v0 gate without reaching for LLM
judging, hand-written emotion taxonomy, or scope expansion, the design is
not yet ready and must be cut further, not patched.

### Concrete Engineering Tasks

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| F1 | `rupture_state` owner contract and snapshot | vz-cognition | extend or alongside `relationship_state` |
| F2 | Multi-source rupture detection wiring | vz-cognition | PE + behavioral + self-check + LLM proposal |
| F3 | Repair primitive in response assembly | vz-application | structured acknowledge + one concrete move |
| F4 | Rupture-repair durable memory entry kind | vz-memory | new `MemoryEntryKind`, persistence rules |
| F5 | Per-user scoped `MemoryStore` lifecycle | vz-memory + lifeform-core | identity binding, persistence, deletion |
| F6 | Scene-end typed outcome intake | lifeform-core + vz-runtime | minimal UI hook; routes to PE / credit |
| F7 | Long-arc check-in scheduler | lifeform-core | uses existing followup manager |
| F8 | Human review packet export and ingest | lifeform-evolution | evidence-only, no owner write |
| F9 | Calibrated uncertainty aggregator and clarification primitive | vz-application | reads owner confidence, gates assembly |
| F10 | Slow-loop aggregation of rupture / outcome / payoff | vz-cognition + vz-runtime | feeds reflection writeback |
| F11 | LLM-as-proposal-source adapter | lifeform-ingestion or new adapter | low-confidence typed outcome only |
| F12 | External-anchored rare-heavy gate update | vz-cognition + vz-runtime | block auto-only promotion |

These tasks intentionally avoid inventing new algorithms; they wire the
existing PE / ETA / NL / owner / reflection stack into a relationship-shaped
loop.

### Minimum Viable Companion Form

The system has reached its minimum viable form for first real users when:

- it has a named identity bound to a specific user;
- it remembers that user across sessions within scope;
- it can detect when it has misread or pushed too hard;
- it can repair specifically and remember what worked;
- it can act on "I am not sure" rather than always committing;
- it can occasionally let the user say in one short signal whether it
  understood;
- nothing about its long-term self-modification happens without external
  evidence.

This is achievable on the current technology stack without new algorithmic
inventions.

### What Not To Do

- Do not introduce LLM judge as reward or ground truth.
- Do not require per-turn user feedback; it destroys the relational frame.
- Do not allow cross-user automatic learning.
- Do not let response assembly become a hidden relationship owner.
- Do not claim relationship capability from benchmark pass rates alone.
- Do not perform long-term self-modification without external anchoring.

### Honest Promise

This plan does not promise an emotionally intelligent companion. It promises
a system that:

- can enter a long-term relationship with a specific user;
- will make mistakes;
- will notice many of those mistakes;
- will repair specifically rather than apologize generically;
- will not repeat the same repaired mistake;
- will grow slowly with that user;
- will not silently drift toward optimizing its own internal metrics.

Anything beyond this requires real time with real users and is not something
a benchmark or an LLM judge can substitute for.
