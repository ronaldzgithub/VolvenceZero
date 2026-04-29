# PRD Alignment Sprint Report — 2026-04-29

> Status: report of work completed in one session
> Scope: 7 slices across Gap 1 / Gap 3 / Gap 4 / Gap 8 / Gap 9
> Source document: `docs/implementation/13_emogpt_prd_alignment_upgrade.md`
> Prerequisite: this report reads the session's work at the slice level; per-slice engineering notes live at the same line indices in `13_*`.

## Summary in one paragraph

A single-day sprint landed seven PRD-alignment slices touching the affordance execution stack, the mid-frequency thinking loop, two advisory-readout layers, and multi-format ingestion. The coding vertical grew from "can't do anything" into a complete **read → edit → run tests → attribute results** loop, the thinking scheduler was wired into the default `LifeformSession` turn lifecycle, and two independent continuous-feature readouts (`ParticipationHint` and `InterlocutorState`) joined the existing affordance scorer in replacing `dict[str, behavior]` lookup tables with auditable scalar functions. Ingestion picked up PDF and DOCX. All seven slices share one pattern (`Context → pure_function → structured_output + rationale`), leave back-compat intact, and carry typed audit rationales (`readout.v1.*` / `scorer.v1:` version tags) so a future learned-weights replacement is a drop-in.

## Slices landed

| # | Slice | Wheel / module | New files | New tests |
|---|-------|----------------|-----------|-----------|
| 1 | Gap 1 slice 2b — Coding vertical read-only affordances | `lifeform-domain-coding/coding_affordances/` | `descriptors.py` / `backends.py` / `factory.py` / `__init__.py` | 31 passed + 1 symlink skip |
| 2 | Gap 4 slice 2c — Thinking-loop production wiring | `lifeform-thinking/adapter.py` + `lifeform-core/lifeform.py` | `ThinkingAdapter` + `ThinkingAdapterProtocol` + `LifeformSession` hook helpers | 16 unit + 5 e2e |
| 3 | Gap 8 slice 2 — Participation / depth hint continuous readout | `vz-cognition/regime/hint_readout.py` | `HintReadoutContext` + `readout_*_hint` + `RegimeModule.hint_readout_mode` | 15 new; 47 slice-1 back-compat |
| 4 | Gap 1 slice 2c — `write_file` + `run_test` subprocess sandbox | `lifeform-domain-coding/coding_affordances/` | 2 new descriptors + 2 backends + 2 consent grants | 26 new |
| 5 | Gap 1 slice 3 — Affordance continuous-feature scorer | `lifeform-affordance/scorer.py` | `AffordanceScoringContext` + `build_scored_snapshot` + duck-typed builder | 20 unit + 3 e2e |
| 6 | Gap 9 slice 1 — `InterlocutorState` 12-axis readout | `vz-cognition/interlocutor/` + `lifeform-core/lifeform.py` | 12-axis dataclass + `readout_interlocutor_state` + `LifeformSession.interlocutor_state` | 26 unit + 4 e2e |
| 7 | Gap 3 slice 2 — PDF + DOCX ingestion sources | `lifeform-ingestion/sources/pdf.py` + `sources/docx.py` | lazy-import pypdf / python-docx + fixtures built at runtime | 26 unit + 5 e2e + isolation contract auto-covered |

Total: **178 new tests**, all green; no existing test removed, no back-compat break.

## Per-slice detail

### 1. Gap 1 slice 2b — Coding vertical read-only affordances

**Goal.** Give the `coding` vertical three concrete, product-grade, safe-by-construction affordances so the AffordanceInvoker pipeline (slice 2a) has real work.

**Delivery.**

- Three TOOL-kind descriptors (`read_file`, `list_dir`, `grep`) with ≥ 50-char `when_to_use` / `when_not_to_use` hints, `additionalProperties: false` parameter schemas, `requires_consent_grant=("filesystem_read",)`, and `blocked_in_regimes=(casual_social, emotional_support, repair_and_deescalation)`.
- Sandbox resolver `resolve_sandbox_path(path, sandbox_root)` uses a two-phase strategy (non-strict resolve first for containment, then strict resolve for symlink validation) so escape attempts on non-existent paths still get the "outside sandbox" error rather than leaking existence information about external files.
- Three async backends: UTF-8 `read_file` with 10 MB hard cap, sorted `list_dir` with file/dir kind tags, pure-Python `grep` that skips > 2 MB files and UTF-8-undecodable files, honours `subdir` + `max_results` budget (hard capped at 2000).
- `build_coding_affordance_registry(seal=True)` + `build_coding_affordance_invoker(sandbox_root, ...)` one-line wiring.
- `lifeform-domain-coding` adds `lifeform-affordance==0.1.*` as a dependency.

**Why it matters.** Without real descriptors the Invoker was theoretically correct but practically untested against product code. This slice put a real consent grant, a real regime block, and a real sandbox guard into the invoker's 5-stage pipeline for the first time.

### 2. Gap 4 slice 2c — Thinking-loop production wiring

**Goal.** Move the mid-frequency thinking scheduler (built in slice 2b) from dormant to running on every `LifeformSession` turn — the last mile of R1's missing "session-medium" band.

**Delivery.**

- New `ThinkingAdapter` (`lifeform-thinking/adapter.py`) with three hooks (`on_turn_begin`, `on_turn_end`, `drain`), a default-scope `MID_REFLECTION_SCOPE = ("dual_track", "regime")`, and two consumer owners (`world_temporal` / `self_temporal`).
- `LifeformSession` new constructor arg `thinking_adapter: Any = None` + three `_invoke_thinking_*` helper methods that catch adapter exceptions so a buggy plugin cannot break a turn. Hooks fire in order: `on_turn_begin` → kernel runs → `on_turn_end`, and `drain` runs inside `end_scene` so no worker outlives the scene.
- Convenience clone `Lifeform.with_thinking_adapter_factory(factory)` to opt sessions in.
- Public observability: `session.thinking_adapter_snapshot`, `session.latest_thinking_artifacts_by_consumer`.

**Key architectural choice.** `lifeform-core` does **not** import `lifeform-thinking`. The wiring goes through `ThinkingAdapterProtocol` (a `runtime_checkable` structural protocol). The adapter is `Any`-typed at call sites; duck-typed access through `getattr` keeps the wheel boundary from SPLIT.md intact.

**Failure-isolation proof.** A test supplies a `_BrokenAdapter` whose three hooks all raise; the session still runs + closes its scene successfully. This is the proof that adapter bugs cannot escalate to turn-level failures.

### 3. Gap 8 slice 2 — Participation / depth hint continuous readout

**Goal.** Remove the slice-1 `dict[regime_id -> hint]` scaffold from the hot path. That scaffold violated the spirit of red line A (`no-keyword-matching-hacks.mdc`): the decision was keyed off the `regime_id` string value, not the runtime condition a `problem_solving` turn could actually be in.

**Delivery.**

- `HintReadoutContext` — 30 scalar features pulled from `memory` + `dual_track` + `evaluation` + `prediction_error` snapshots.
- `readout_participation_hint(ctx)` — three per-section scores (panorama / method / task) as pull-push-baseline weighted sums, discretised at 0.30 / 0.65 thresholds; `flow_kind` argmax over five independent kind scores (ties prefer milder).
- `readout_cognitive_depth_hint(ctx)` — single-scalar pressure mapped onto five depth tiers.
- `RegimeModule(hint_readout_mode="readout"|"scaffold")` opt-in switch; `"scaffold"` reproduces slice-1 behaviour exactly.
- `evidence = 0.25 + 0.70 * evidence_score` so the published `confidence` reflects available signal (slice-1 always published flat 0.4).
- Cold-start fallback: zero runtime signal falls through to the scaffold table with confidence capped at 0.30, tagged `readout:cold-fallback:` so audit can distinguish real readout outputs from degraded ones.

**Invariant.** No branch in the readout reads the `regime_id` value itself. `regime_id` is pure audit-trail data in the rationale. A `casual_social` / `chitchat` / `unknown_regime_xyz` profile with the same continuous features produces the same output.

### 4. Gap 1 slice 2c — `write_file` + `run_test` subprocess sandbox

**Goal.** Close the coding-vertical loop from "can read files" to "can edit and verify". First slice in the repo where `requires_user_confirmation=True` + `irreversible=True` + `rate_limit_per_minute` actually ship on a production descriptor.

**Delivery.**

- `write_file` — TOOL / FAST / `filesystem_write` consent + `requires_user_confirmation` + `irreversible` + `audit_required`. Three modes (`create` / `overwrite` / `append`), 10 MB content cap, parent-must-exist guard (`_resolve_parent_under_sandbox` — no auto-mkdir), directory-collision rejection.
- `run_test` — TOOL / SLOW / `run_shell_commands` consent + `rate_limit_per_minute=6` + `audit_required`. Uses `asyncio.create_subprocess_exec` with `sys.executable -m pytest -q` so pytest does not need to be on PATH. Timeout via `asyncio.wait_for` + `proc.kill()` + 5 s grace drain; captured `stdout` / `stderr` bounded to 64 KB per stream, replace-errors UTF-8 decode.
- `CONSENT_FILESYSTEM_WRITE` / `CONSENT_RUN_SHELL_COMMANDS` constants exported so host code can grant each axis independently from `filesystem_read`.
- E2E proof: a session opens → invoker writes a new `tests/test_added.py` → invoker runs `pytest` on it → the following turn's `execution_result.completed_actions` contains both `event_id`s.

**Gap 1 test breakdown after slice 2c.** 57 coding-affordance tests total (slice 2b + 2c combined): 12 write_file backend, 3 write_file invoker gate, 6 run_test backend, 2 run_test invoker gate, 1 real-session e2e + all slice-2b tests retained.

### 5. Gap 1 slice 3 — Affordance continuous-feature scorer

**Goal.** Replace `build_neutral_snapshot`'s all-0.5 scaffold with a scored snapshot driven by the same continuous features the hint readout uses. Second of the three red-line-A replacements in this session.

**Delivery.**

- `AffordanceScoringContext` — 14 scalar features (regime_id as audit only, flow_kind / cognitive_depth strings, turns, 4 drives, 4 biases, cross_track_tension, evidence).
- `score_affordance(descriptor, ctx)` — pure function with five stages: regime block (returns score=0 + `blocked_reason`); kind affinity (TOOL/ACTION/ORGAN/SHELL each computes from different feature axes); tag affinity (code / write / execute / social tags match differently); latency penalty (SLOW + SHALLOW depth gets penalised); safety penalty (`irreversible` + `requires_user_confirmation` under cold-start + high cross_tension + low turns gets penalised).
- `build_scored_snapshot(registry, ctx)` selects a single top candidate only when score ≥ 0.50 **and** margin ≥ 0.06 over runner-up — "not confidently selected" is an explicit return value, not a failure.
- `build_scoring_context_from_snapshots(...)` — duck-typed builder so `lifeform-affordance` does **not** import `vz-cognition`.

**Invariant.** No branch reads `descriptor.name`. All branching is over `AffordanceKind` enum, `affordance_tags` set, `cost_model.latency_class` enum, `safety_model.*` bools. Adding a new affordance is zero code change to the scorer.

### 6. Gap 9 slice 1 — `InterlocutorState` 12-axis readout

**Goal.** Third and most ambitious red-line-A replacement: EmoGPT §13's entire Resistance-zone + Coaxing-coefficient lookup table collapses into twelve continuous axes read from existing kernel signals.

**Delivery.**

- New kernel sub-package `vz-cognition/interlocutor/`.
- `InterlocutorState` 12 axes: `engagement_intensity` / `self_disclosure_level` / `task_focus_level` / `emotional_weight` / `cognitive_engagement` / `resistance_level` / `openness_to_guidance` / `directness` / `trust_signal` (signed `[-1, 1]`) / `stability` / `rapport_warmth` / `pace_pressure`, plus `readout_confidence` + `rationale`.
- `InterlocutorReadoutContext` pulls from 6 kernel snapshots (regime / dual_track / evaluation / prediction_error / memory / commitment). `commitment_alignment_trend` averages `lifecycle_entries[*].last_alignment` values — that's the Gap 7-side cross-reference; slice 2 materialising it into `relationship_state.trust_level` writes is a separate, optional step.
- `LifeformSession.interlocutor_state` property computes on read from latest snapshots (no kernel owner mutation).
- `resistance_level` and `openness_to_guidance` are computed independently (not `1 - x` of each other) so they can co-evolve under different weightings.

**Why a readout and not a kernel owner write.** `SemanticOwnerModule._build_snapshot` only receives the owner's own records, not cross-owner snapshots — Gap 9's 12 dimensions are fundamentally cross-owner (they read PE + dual_track + evaluation + memory + commitment simultaneously). Rather than invasively change the owner-module signature, the readout is a derived view computed at read time. A future slice that promotes it to a persisted owner field has full context to do so (the signatures match); this slice does not pay that architectural cost yet.

### 7. Gap 3 slice 2 — PDF + DOCX ingestion sources

**Goal.** Make apprenticeship / ingestion paths real enough to ingest actual textbooks and Word docs. Pre-slice this was limited to `plain_text` + `task_result` which was fine for testing but useless for "投喂书本 / 文档" demos.

**Delivery.**

- `sources/pdf.py` (`pypdf` behind `lifeform-ingestion[pdf]` extra): `envelope_from_pdf_bytes` / `envelope_from_pdf_file`, one chunk per page with `page=N/TOTAL` locator, long-page sub-chunk with `:sub:NN` suffix, encrypted-PDF rejection, `max_pages=200` default cap, per-page best-effort that never aborts the entire envelope.
- `sources/docx.py` (`python-docx` behind `[docx]` extra): heading-delimited section chunking with single-section fallback when no headings, fast ZIP magic + `word/document.xml` validation, `max_paragraphs=2000` cap, tables flagged as `section_has_unextracted_table` partial_failure so the operator sees what was skipped (table extraction lands in slice 2b).
- Both adapters emit `source_kind=BOOK`, `compliance_profile=FORCED` (routes through Gap 2 apprentice override), and `source_uri=file://<abs-path>` on `_file` variants.
- Lazy dependency import so `pip install lifeform-ingestion` stays thin; missing optional extras raise typed `PdfIngestionError` / `DocxIngestionError` naming the exact install command.
- `tests/contracts/test_ingestion_isolation.py` auto-discovers the new source files and confirms the wheel-boundary invariant (no kernel-owner imports) still holds.

## Cross-cutting themes

### The `Context → readout_function → structured_output` pattern

By the end of this session the same pattern is implemented in three independent modules:

| Module | Context type | Output type | Rationale tag |
|--------|-------------|-------------|---------------|
| `vz-cognition/regime/hint_readout.py` | `HintReadoutContext` | `ParticipationHint`, `CognitiveDepthHint` | `readout.v1:` / `readout.v1.depth:` |
| `lifeform-affordance/scorer.py` | `AffordanceScoringContext` | `AffordanceCandidate`, `AffordanceSnapshot` | `scorer.v1:` |
| `vz-cognition/interlocutor/__init__.py` | `InterlocutorReadoutContext` | `InterlocutorState` | `readout.v1.interlocutor:` |

Each one:

- Has a duck-typed builder (`build_*_from_snapshots`) that tolerates `None` / missing snapshots and scales `evidence` / `readout_confidence` down.
- Publishes a `.rationale` audit string listing the top contributing features.
- Has a versioned tag (`v1`) so a future learned-weights replacement is a drop-in and can be distinguished in logs.
- Has a cold-start path that either stays at neutral or falls through to a scaffold with explicit confidence capping.

This is now the canonical VolvenceZero pattern for "produce a structured advisory from continuous kernel signals". A fourth use of it (e.g. a `RelationshipRecommendationReadout`) should follow the same shape.

### Red-line-A enforcement (`no-keyword-matching-hacks.mdc`)

Three separate lookup-table decision paths were replaced with continuous functions in this session:

1. `dict[regime_id, ParticipationHint]` → continuous-feature readout (Gap 8 slice 2)
2. `all descriptors get score=0.5` scaffold → per-descriptor continuous score (Gap 1 slice 3)
3. EmoGPT §13's 5-zone Resistance + 4-type Coaxing tables → 12 continuous axes (Gap 9 slice 1)

None of the new code branches off a string value the kernel produces. `regime_id`, `flow_kind`, `cognitive_depth` appear in context + rationale strings, but **no** `if ctx.regime_id == "problem_solving"` branches exist — decisions flow from feature axes.

### Fail-loud invariants

- `AffordanceDescriptor.__post_init__` enforces `when_to_use` / `when_not_to_use` ≥ 50 chars, unique tags.
- `ParticipationHint` / `CognitiveDepthHint` / `InterlocutorState` all validate `confidence` and per-axis ranges in `__post_init__`.
- PDF encrypted / non-PDF magic / zero-byte / max_pages exceeded / zero extractable text all raise `PdfIngestionError`.
- DOCX non-ZIP / missing `word/document.xml` / max_paragraphs exceeded all raise `DocxIngestionError`.
- `RegimeModule.hint_readout_mode` rejects anything other than `"readout"` / `"scaffold"`.

## What was intentionally not done

This list is meant to prevent future "what happened to X?" confusion.

- **Gap 1 slice 4 — learned affordance weights.** Blocked on `lifeform-trace` collecting `(context, observed_effectiveness)` pairs; that infrastructure does not exist.
- **Gap 3 slice 2b — web ingestion adapter.** Explicitly scoped out of this session. SSRF discipline / content-type sniffing / encoding detection / redirect policy deserves its own pass.
- **Gap 3 slice 3 — DOCX table extraction.** Currently flagged as `section_has_unextracted_table` partial_failure. Real extraction (reading `table.rows[*].cells[*].paragraphs` into markdown) is a follow-up.
- **Gap 6 — DLaaS control plane.** Service layer; not in this sprint.
- **Gap 9 slice 2 — `relationship_state` owner durable trust writes.** The Gap 9 slice 1 readout already reflects commitment alignment trends in `trust_signal`. Promoting that readout to a real owner-field mutation needs a product decision on whether durable persistence is worth the invasive `SemanticStateStore.apply` change.
- **Gap 9 slice 3 — metacontroller consumes 12 axes in `z_t` observation.** Deliberately deferred. Changing the metacontroller's observation surface without learned weights would perturb its behaviour unpredictably; needs eval-harness guarantees first.

## Regression

Throughout this session each slice ran a targeted regression, a broader category regression, and (periodically) the full `tests/lifeform_e2e/` suite. The final full-suite run after Gap 1 slice 3 reported **217 passed, 1 skipped, 0 failed** across `tests/lifeform_e2e/` (Windows symlink test skipped only). Slices 6 (Gap 9) and 7 (Gap 3 slice 2) ran targeted 350+ tests each with no failures. Lint (`ReadLints`) was clean on every new or modified file.

## Wheel-boundary and red-line status

| Invariant | How verified | Status |
|-----------|--------------|--------|
| `vz-* ↛ lifeform-*` import direction | `tests/contracts/test_import_boundaries.py` + `test_ingestion_isolation.py` auto-discovery | passes after each new file |
| `lifeform-affordance` independent of `vz-cognition` | Scorer uses plain-scalar `AffordanceScoringContext` + duck-typed builder | passes |
| `lifeform-core` independent of `lifeform-thinking` | Adapter wired via `ThinkingAdapterProtocol` + duck-typed `getattr` | passes |
| `lifeform-ingestion` independent of kernel owners | `test_ingestion_isolation.py` covers new `sources/pdf.py` + `sources/docx.py` | passes |
| No keyword-match decision paths in new code | Manual review + rationale-tag grep | passes |
| `__post_init__` range checks on every new dataclass | Every new dataclass reviewed | passes |

## File-level impact

| Package | New files |
|---------|-----------|
| `vz-contracts` | 0 (affordance + thinking + participation hints already existed from prior slices) |
| `vz-cognition` | 2 (`interlocutor/__init__.py`, `regime/hint_readout.py`) |
| `lifeform-thinking` | 1 (`adapter.py`) |
| `lifeform-affordance` | 1 (`scorer.py`) |
| `lifeform-ingestion` | 2 (`sources/pdf.py`, `sources/docx.py`) |
| `lifeform-domain-coding` | 4 (`coding_affordances/{descriptors,backends,factory,__init__}.py`) |
| `lifeform-core` | 0 new files; `lifeform.py` grew `ThinkingAdapterProtocol`, session hooks, `interlocutor_state` property, `with_thinking_adapter_factory` |
| `tests/` | 7 new test files (4 unit + 3 e2e; counts in table above) |
| `docs/implementation/` | `13_*.md` updated per slice + this report |
| `docs/specs/` | `thinking-loop.md` updated with slice 2c production-wiring section |

## How to consume this report

- **"Is Gap X done?"** — `docs/implementation/13_emogpt_prd_alignment_upgrade.md` holds the per-slice engineering note; this report is the one-page summary.
- **"Where is pattern Y used?"** — See "Cross-cutting themes" above; every `Context → readout` use is registered there.
- **"What's blocked on training data?"** — slice 4 of Gap 1 and slices 3-4 of Gap 9. Any "learned weights" slice requires `lifeform-trace` to collect `(context, effectiveness)` pairs first.
- **"What's the next natural step?"** — Likely candidates: running a concrete product demo through the full `coding` vertical using the new write + run_test affordances, or picking up `docs/scenarios/sea/haiyang_*.md` as a driver for further vertical work.
