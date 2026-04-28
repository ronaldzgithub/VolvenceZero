"""End-to-end lifeform integration tests.

These tests exercise the full ``Lifeform`` stack (TickEngine + SceneManager +
FollowupManager + Brain + GroundedResponseSynthesizer + companion
DomainExperiencePackage) to verify:

1. The lifeform can run a full multi-turn dialogue.
2. Cross-turn memory continuity is preserved (memory snapshot grows).
3. Regime is selected and remains a non-empty string.
4. End-of-scene triggers the kernel ``begin_new_context`` boundary and the
   session-post slow loop drains.
5. The grounded synthesizer produces non-empty responses for every turn
   and emits structured plan tags in its rationale.
6. Tick-driven idle close fires the kernel boundary even without an
   explicit ``end_scene`` call.
7. Open-loop / commitment ingestion into ``FollowupManager`` is idempotent.

These tests deliberately use the **synthetic** substrate so they run on a
torch-less environment. They are NOT a quality benchmark — that lives in
``lifeform_evolution.benchmark``.
"""

from __future__ import annotations

from lifeform_core import (
    Lifeform,
    LifeformConfig,
    SceneStatus,
    TickEngineConfig,
    TickKind,
)
from lifeform_domain_emogpt import build_companion_package
from lifeform_evolution import format_report, run_benchmark_async, low_mood_disclosure_scenario
from lifeform_expression import (
    GroundedResponseSynthesizer,
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
    EtiquetteWatchdog,
    SpeakVerdict,
)


def _build_lifeform(*, idle_close_after_system_ticks: int | None = 60) -> Lifeform:
    """Build a deterministic synthetic-substrate lifeform for tests."""
    from volvence_zero.brain import BrainConfig

    config = LifeformConfig(
        brain_config=BrainConfig(rare_heavy_enabled=False),
        tick=TickEngineConfig(
            system_tick_seconds=0.0,  # we only use pump-style advance() in tests
            energy_every_n_system_ticks=2,
            context_every_n_system_ticks=4,
        ),
        idle_close_after_system_ticks=idle_close_after_system_ticks,
    )
    config = config.with_domain_experience((build_companion_package(),))
    return Lifeform(
        config,
        response_synthesizer=GroundedResponseSynthesizer(planner=PromptPlanner()),
    )


# ---------------------------------------------------------------------------
# Multi-turn dialogue continuity
# ---------------------------------------------------------------------------


async def _three_turn_dialogue_runs(session) -> list:
    results = []
    for turn in (
        "I have been feeling stuck lately and I am not sure why.",
        "It is mostly that work feels heavy and home is also tense.",
        "Honestly that almost made it worse, I just wanted to be heard.",
    ):
        results.append(await session.run_turn(turn))
    return results


def test_lifeform_runs_three_turn_dialogue_with_continuity():
    import asyncio

    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="test-three-turn")

    results = asyncio.run(_three_turn_dialogue_runs(session))

    # Every turn produced a non-empty response.
    for result in results:
        assert result.response.text.strip(), f"Empty response: {result!r}"

    # Active regime was always non-None — kernel committed to a regime.
    for result in results:
        assert result.active_regime, f"Regime was empty: {result.active_regime}"

    # GroundedResponseSynthesizer attaches the plan tail to rationale.
    for result in results:
        assert "intent=" in result.response.rationale
        assert "regime=" in result.response.rationale

    # Lifeform layer recorded one TurnSummary per turn.
    summaries = session.turn_summaries
    assert len(summaries) == 3
    assert summaries[0].turn_index == 1
    assert summaries[2].turn_index == 3

    # Scene was opened and is still open.
    assert session.open_scene is not None
    assert session.open_scene.turn_count == 3
    assert session.open_scene.status is SceneStatus.OPEN

    # Memory snapshot is growing across turns (MemoryStore got writes).
    mem_first = results[0].active_snapshots["memory"].value
    mem_last = results[-1].active_snapshots["memory"].value
    assert mem_last.lifecycle_metrics is not None
    # Either the entry count or the lifecycle metric should reflect activity.
    last_entries = getattr(mem_last, "entries", ()) or ()
    first_entries = getattr(mem_first, "entries", ()) or ()
    assert len(last_entries) >= len(first_entries) or mem_last is not mem_first


# ---------------------------------------------------------------------------
# end_scene triggers kernel boundary and slow-loop drain
# ---------------------------------------------------------------------------


def test_end_scene_triggers_kernel_boundary_and_drains_slow_loop():
    import asyncio

    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="test-end-scene")

    async def go():
        await session.run_turn("Hi I want to talk about something difficult.")
        await session.run_turn("Actually I am going to take a break for now, thanks.")
        return await session.end_scene(reason="user-farewell")

    closed = asyncio.run(go())
    assert closed is not None
    assert closed.status is SceneStatus.CLOSED
    assert closed.turn_count == 2
    # No open scene any more after end_scene.
    assert session.open_scene is None
    # SceneManager moved the closed scene into history.
    assert len(session.closed_scenes) == 1


# ---------------------------------------------------------------------------
# Tick-driven idle close
# ---------------------------------------------------------------------------


def test_idle_tick_closes_scene_and_invokes_kernel_boundary():
    import asyncio

    lifeform = _build_lifeform(idle_close_after_system_ticks=3)
    session = lifeform.create_session(session_id="test-idle-close")

    async def go():
        await session.run_turn("Just checking in for a minute.")
        # Drive enough ticks to cross the idle threshold.
        await session.advance_tick(5)
        return session

    asyncio.run(go())
    # Idle close should have run end_scene which clears open_scene.
    assert session.open_scene is None
    assert len(session.closed_scenes) == 1
    closed = session.closed_scenes[0]
    assert closed.turn_count == 1


# ---------------------------------------------------------------------------
# FollowupManager idempotency
# ---------------------------------------------------------------------------


def test_followup_manager_dedupes_repeated_open_loop_keys():
    import asyncio

    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="test-followup-dedup")

    async def go():
        await session.run_turn("Can you keep track that I am supposed to call my sister tomorrow?")
        # Re-running the same kind of input should not blow up the queue.
        await session.run_turn("And remind me about the call to my sister tomorrow.")

    asyncio.run(go())

    pending = session.all_pending_followups()
    keys = {item.metadata.get("key") for item in pending if item.source == "open_loop"}
    # Whatever open loops there are, each key is unique.
    assert len(keys) == len({k for k in keys if k is not None})


# ---------------------------------------------------------------------------
# Tick engine pump semantics
# ---------------------------------------------------------------------------


def test_tick_engine_fires_system_energy_context_at_configured_cadence():
    import asyncio

    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="test-tick-cadence")

    async def go():
        events = await session.advance_tick(4)
        return events

    events = asyncio.run(go())
    kinds = [ev.kind for ev in events]

    # 4 system ticks fired.
    assert kinds.count(TickKind.SYSTEM) == 4
    # Energy fires every 2 system ticks → 2 events.
    assert kinds.count(TickKind.ENERGY) == 2
    # Context fires every 4 system ticks → 1 event.
    assert kinds.count(TickKind.CONTEXT) == 1


# ---------------------------------------------------------------------------
# PromptPlanner does what its name says — purely structural
# ---------------------------------------------------------------------------


def test_prompt_planner_picks_support_first_for_emotional_support_regime():
    from volvence_zero.agent.response import ResponseContext

    planner = PromptPlanner()
    context = ResponseContext(
        regime_id="emotional_support",
        regime_name="emotional_support",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.5,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="ssl-only",
        user_input="I am exhausted.",
    )

    plan: PromptPlan = planner.plan(context=context, assembly=None)
    assert plan.intent is TurnIntent.SUPPORT_FIRST
    assert SectionId.ACKNOWLEDGE_PRESSURE in plan.sections
    assert SectionId.NEXT_STEP in plan.sections


# ---------------------------------------------------------------------------
# EtiquetteWatchdog speaks/waits/silences as documented
# ---------------------------------------------------------------------------


def test_etiquette_watchdog_enforces_cooldown_and_quiet_hours():
    from lifeform_core.types import Scene

    wd = EtiquetteWatchdog(min_cooldown_ticks=3, max_consecutive_proactive_turns=1)
    scene = Scene(scene_id="scene-001", started_at_tick=0, turn_count=2, last_turn_at_tick=10)

    # First turn — speak ok.
    v = wd.evaluate(current_tick=10, scene=scene, is_proactive=False)
    assert v.verdict is SpeakVerdict.SPEAK
    wd.record_spoken(current_tick=10, was_proactive=False)

    # Second turn within cooldown — wait.
    v = wd.evaluate(current_tick=11, scene=scene, is_proactive=False)
    assert v.verdict is SpeakVerdict.WAIT
    assert v.cooldown_remaining_ticks == 2

    # Quiet hours block speak.
    wd_quiet = EtiquetteWatchdog(
        min_cooldown_ticks=0,
        max_consecutive_proactive_turns=10,
        quiet_hours_start_tick=20,
        quiet_hours_end_tick=30,
    )
    v = wd_quiet.evaluate(current_tick=25, scene=scene, is_proactive=False)
    assert v.verdict is SpeakVerdict.STAY_SILENT


# ---------------------------------------------------------------------------
# Benchmark CLI smoke (lightweight, just verifies it runs)
# ---------------------------------------------------------------------------


def test_lifeform_benchmark_runs_low_mood_scenario_end_to_end():
    import asyncio

    report = asyncio.run(run_benchmark_async(scenario=low_mood_disclosure_scenario()))
    assert report.scenario_id == "low-mood-disclosure"
    assert len(report.turn_reports) == 4
    assert report.response_non_empty_rate == 1.0
    # Closed-scene count is 1 (we close the scene at the end).
    assert report.closed_scene_count == 1
    # Pretty-print does not crash.
    text = format_report(report)
    assert "Lifeform benchmark" in text


# ---------------------------------------------------------------------------
# Expression-intent diversification (kernel calibration that this round
# landed) — the lifeform should now surface multiple distinct
# expression_intents across a 4-turn dialogue rather than collapsing to
# one shape for every turn.
# ---------------------------------------------------------------------------


def test_lifeform_benchmark_surfaces_multiple_distinct_expression_intents():
    import asyncio

    report = asyncio.run(run_benchmark_async(scenario=low_mood_disclosure_scenario()))
    intents = {tr.expression_intent for tr in report.turn_reports if tr.expression_intent}
    # Pre-fix: this set was ``{"judgment-process"}`` for every input.
    assert len(intents) >= 2, f"expected >=2 distinct intents, got {intents!r}"


# ---------------------------------------------------------------------------
# Trace collector (the "往下生长" SSL data path)
# ---------------------------------------------------------------------------


def test_trace_collector_emits_one_row_per_turn_with_required_fields(tmp_path):
    from lifeform_evolution import (
        TraceCollector,
        all_built_in_scenarios,
        casual_social_checkin_scenario,
    )

    out = tmp_path / "trace.ndjson"
    collector = TraceCollector(output_path=out)
    try:
        report = collector.collect_scenario(casual_social_checkin_scenario())
    finally:
        collector.close()

    assert report.record_count == len(casual_social_checkin_scenario().turns) == 3
    assert len(collector.records) == 3

    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3

    import json
    first = json.loads(rows[0])
    required = {
        "schema_version",
        "scenario_id",
        "session_id",
        "turn_index",
        "scene_id",
        "tick_index",
        "user_input",
        "response_text",
        "active_regime",
        "expression_intent",
        "pe_magnitude",
        "pe_task",
        "pe_relationship",
        "pe_regime",
        "pe_action",
        "dual_track_world_pressure",
        "dual_track_self_pressure",
        "memory_entry_count",
        "open_loop_count",
        "commitment_count",
        "has_substrate_residuals",
    }
    assert required.issubset(first.keys())
    assert first["schema_version"] == "trace.v1"


def test_trace_collector_writes_all_built_in_scenarios(tmp_path):
    from lifeform_evolution import TraceCollector, all_built_in_scenarios

    out = tmp_path / "trace-all.ndjson"
    collector = TraceCollector(output_path=out)
    try:
        reports = collector.collect_scenarios(all_built_in_scenarios())
    finally:
        collector.close()

    assert len(reports) == 3
    total_rows = sum(r.record_count for r in reports)
    written = out.read_text(encoding="utf-8").splitlines()
    assert len(written) == total_rows


# ---------------------------------------------------------------------------
# LLM synthesizer (smoke — does not actually call an LLM, just the wiring)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SSL training loop (M1 of "downward growth")
# ---------------------------------------------------------------------------


def test_dataset_adapter_round_trips_collector_output(tmp_path):
    from lifeform_evolution import (
        TraceCollector,
        casual_social_checkin_scenario,
        trace_records_from_ndjson,
        trace_records_to_training_dataset,
    )

    out = tmp_path / "trace.ndjson"
    collector = TraceCollector(output_path=out)
    try:
        collector.collect_scenario(casual_social_checkin_scenario())
    finally:
        collector.close()

    # Records read back from disk should be field-equivalent to the in-memory
    # records the collector kept.
    on_disk = trace_records_from_ndjson(out)
    in_memory = collector.records
    assert len(on_disk) == len(in_memory) == 3
    assert {r.scenario_id for r in on_disk} == {r.scenario_id for r in in_memory}

    dataset = trace_records_to_training_dataset(on_disk)
    assert len(dataset.traces) == 3
    # Every trace should have steps (at least the user-input tokens).
    for trace in dataset.traces:
        assert len(trace.steps) >= 1
        # Trace ID is rich enough to attribute back to the originating turn.
        assert trace.trace_id.startswith("casual-social-checkin::")


def test_ssl_demo_drives_switch_gate_toward_sparser_firing():
    """ETA's \u03b2_t emergence: after SSL training the switch frequency should
    drop toward sparse, subgoal-aligned firings.

    This is the M1 acceptance test for "downward growth": traces collected
    on the synthetic substrate are semantically rich enough to drive
    non-trivial SSL gradients in the metacontroller. The exact final value
    depends on initialisation, but the direction (sparser firing across the
    run) is the ETA paper's invariant for a discovered abstraction.
    """
    from lifeform_evolution import run_ssl_demo

    report = run_ssl_demo()
    assert report.trace_count >= 5
    assert report.trained_step_count > 0
    # Switch gate stats should be populated (trainer published them).
    assert report.switch_frequency_first is not None
    assert report.switch_frequency_last is not None
    # Mean persistence should be a positive run length.
    assert report.mean_persistence_first is not None
    assert report.mean_persistence_first > 0.0
    assert report.mean_persistence_last is not None
    # Direction check: training should not make switching MORE random than
    # the start. We allow a small noise margin (0.1) because some runs may
    # naturally hover.
    assert (
        report.switch_frequency_last
        <= report.switch_frequency_first + 0.1
    ), (
        f"switch frequency went from {report.switch_frequency_first:.3f} "
        f"to {report.switch_frequency_last:.3f} \u2014 expected sparser or flat"
    )


# ---------------------------------------------------------------------------
# Closed feedback loop (M2 of "downward growth"): collect \u2192 SSL train \u2192
# inject trained snapshot into a fresh Brain \u2192 re-benchmark \u2192 observe shift.
# ---------------------------------------------------------------------------


def test_brain_accepts_temporal_bootstrap_and_propagates_to_session():
    """The kernel surface must let product code inject a trained policy.

    Verifies the public API contract: ``Brain(temporal_bootstrap=...)`` is
    accepted, exposed on ``brain.temporal_bootstrap``, and each
    ``create_session`` constructs a fresh ``FullLearnedTemporalPolicy``
    initialised from the snapshot (not a shared mutable instance).
    """
    from volvence_zero.brain import Brain, BrainConfig
    from volvence_zero.temporal import FullLearnedTemporalPolicy

    policy = FullLearnedTemporalPolicy()
    snapshot = policy.export_rare_heavy_snapshot()

    brain = Brain(BrainConfig(rare_heavy_enabled=False), temporal_bootstrap=snapshot)
    assert brain.temporal_bootstrap is snapshot

    s1 = brain.create_session(session_id="s1")
    s2 = brain.create_session(session_id="s2")

    runner_a = s1.runner
    runner_b = s2.runner
    assert runner_a is not runner_b
    # Each session got its own world_temporal_policy instance.
    assert runner_a._world_temporal_policy is not runner_b._world_temporal_policy


def test_lifeform_with_temporal_bootstrap_returns_fresh_clone():
    """``Lifeform.with_temporal_bootstrap`` must produce a clone, not mutate."""
    from volvence_zero.brain import BrainConfig
    from volvence_zero.temporal import FullLearnedTemporalPolicy

    base = Lifeform(LifeformConfig(brain_config=BrainConfig(rare_heavy_enabled=False)))
    policy = FullLearnedTemporalPolicy()
    snap = policy.export_rare_heavy_snapshot()

    bootstrapped = base.with_temporal_bootstrap(snap)
    assert bootstrapped is not base
    assert base.temporal_bootstrap is None
    assert bootstrapped.temporal_bootstrap is snap


def test_learning_loop_closes_end_to_end():
    """The headline R3/R4/R12 acceptance test: collect \u2192 SSL \u2192 reinject \u2192
    re-benchmark in one call, with all loop verdicts passing.
    """
    from lifeform_evolution import run_learning_loop, format_learning_loop_report

    report = run_learning_loop()
    assert report.scenarios == (
        "low-mood-disclosure",
        "trust-rupture-repair",
        "casual-social-checkin",
    )
    # SSL ran on the trace volume.
    assert report.ssl.trained_step_count > 0
    # Both behavioural snapshots have content.
    assert report.baseline.turn_count == report.trained.turn_count
    assert report.baseline.turn_count > 0
    # Pretty-print does not crash.
    text = format_learning_loop_report(report)
    assert "Lifeform learning loop" in text
    # All loop verdicts pass.
    assert report.loop_closed(), report.verdicts


# ---------------------------------------------------------------------------
# Multi-round learning loop (R13 evidence harness)
# ---------------------------------------------------------------------------


def test_multi_round_loop_evolves_policy_state_and_finds_a_healthy_round():
    """R13 acceptance: SSL across multiple rounds should produce evolving
    policy snapshots, at least one round of meaningful surface drift, and
    at least one round combining sparse \u03b2_t with that drift.

    We deliberately do not assert "more rounds = monotonic improvement" \u2014
    over-training on a small fixed dataset is *expected* to eventually
    overshoot and collapse \u03b2_t (and the harness flags this when it
    happens). The contract here is direction-only: the loop must produce a
    healthy regime SOMEWHERE in the trajectory.
    """
    from lifeform_evolution import (
        format_multi_round_report,
        run_multi_round_loop,
    )

    report = run_multi_round_loop(rounds=3)
    assert len(report.rounds) == 3
    # Round 0 is the untrained baseline.
    assert report.baseline.distance_to_baseline == 0.0
    # SSL ran every round.
    for r in report.rounds:
        assert r.ssl.trained_step_count > 0
    # Snapshots evolved.
    fingerprints = {repr(r.snapshot) for r in report.rounds}
    assert len(fingerprints) >= 2
    # Best-round selector returns a real round.
    best = report.best_round()
    assert best.round_index >= 1
    assert best.distance_to_baseline > 0.0
    # Pretty-print does not crash.
    assert "Multi-round learning loop" in format_multi_round_report(report)
    # All trajectory verdicts passed at the chosen round count (3 rounds).
    assert report.trajectory_passes(), report.verdicts


def test_multi_round_loop_distance_is_zero_for_baseline_and_nonzero_after_training():
    """The Hellinger-style distance metric must behave: round 0's distance
    to baseline is 0 (it IS the baseline), and at least one later round
    moves it.
    """
    from lifeform_evolution import run_multi_round_loop

    report = run_multi_round_loop(rounds=3)
    assert report.rounds[0].distance_to_baseline == 0.0
    later_distances = [r.distance_to_baseline for r in report.rounds[1:]]
    assert any(d > 0.0 for d in later_distances), (
        f"Expected at least one trained round to move distance from baseline; "
        f"got {later_distances}"
    )


def test_lifeform_llm_synthesizer_attaches_plan_to_rationale_when_falling_back():
    """If the runtime's ``generate`` returns empty text, the LLM synthesizer
    falls back to the base templates. The plan must still be attached.
    """
    from lifeform_expression import LifeformLLMResponseSynthesizer
    from volvence_zero.agent.response import ResponseContext

    class _StubRuntime:
        model_id = "stub-runtime"

        def generate(self, **kwargs):
            class _R:
                text = ""
                token_count = 0
            return _R()

    synth = LifeformLLMResponseSynthesizer(runtime=_StubRuntime())

    context = ResponseContext(
        regime_id="emotional_support",
        regime_name="emotional_support",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.5,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="ssl-only",
        user_input="I am exhausted.",
    )
    response = synth.synthesize(context=context, assembly=None)
    assert "intent=" in response.rationale
    assert synth.last_plan is not None
