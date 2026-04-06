"""Tests for P10-P14 algorithm depth convergence packages."""

from __future__ import annotations

from volvence_zero.memory import CMSMemoryCore
from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal.metacontroller_components import SequenceEncoder


def _make_substrate(values: tuple[float, ...] = (0.5, 0.3, 0.7)) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="test-model",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=values,
        feature_surface=(
            FeatureSignal(name="f1", values=values, source="test"),
            FeatureSignal(name="f2", values=(0.2, 0.4), source="test"),
        ),
        residual_activations=(
            ResidualActivation(layer_index=0, step=0, activation=values),
        ),
        residual_sequence=(),
        unavailable_fields=(),
        description="test substrate",
    )


class TestP10CMSEnhancedEncoder:
    def test_encoder_accepts_cms_bands(self) -> None:
        encoder = SequenceEncoder()
        substrate = _make_substrate()
        without_cms = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
        )
        with_cms = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
            cms_online_fast=(0.8, 0.8, 0.8),
            cms_session_medium=(0.6, 0.5, 0.4),
            cms_background_slow=(0.3, 0.3, 0.3),
        )
        assert without_cms.posterior.prior_mean != with_cms.posterior.prior_mean
        assert without_cms.posterior.prior_std != with_cms.posterior.prior_std

    def test_cms_prior_std_contracts_with_evidence(self) -> None:
        encoder = SequenceEncoder()
        substrate = _make_substrate()
        no_cms = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
        )
        with_strong_cms = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
            cms_session_medium=(0.9, 0.9, 0.9),
            cms_background_slow=(0.8, 0.8, 0.8),
        )
        avg_std_no_cms = sum(no_cms.posterior.prior_std) / 3
        avg_std_with_cms = sum(with_strong_cms.posterior.prior_std) / 3
        assert avg_std_with_cms < avg_std_no_cms, "CMS evidence should contract prior std"

    def test_cms_informed_prior_mean_uses_slow_bands(self) -> None:
        encoder = SequenceEncoder()
        substrate = _make_substrate()
        encoded = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
            cms_background_slow=(0.9, 0.1, 0.5),
            cms_session_medium=(0.1, 0.9, 0.5),
        )
        assert encoded.posterior.prior_mean[0] > 0.1
        assert encoded.posterior.prior_mean[1] > 0.1

    def test_encoder_output_for_cms_returns_signal(self) -> None:
        encoder = SequenceEncoder()
        substrate = _make_substrate()
        encoded = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
        )
        signal = encoder.encoder_output_for_cms(encoded)
        assert len(signal) == 3
        assert all(0.0 <= v <= 1.0 for v in signal)

    def test_cms_observe_encoder_feedback(self) -> None:
        cms = CMSMemoryCore(dim=3)
        state_before = cms.snapshot()
        cms.observe_encoder_feedback(
            encoder_signal=(0.6, 0.4, 0.5),
            timestamp_ms=100,
        )
        state_after = cms.snapshot()
        assert state_after.online_fast.vector != state_before.online_fast.vector

    def test_bidirectional_cms_encoder_loop(self) -> None:
        """CMS feeds encoder -> encoder output feeds CMS -> CMS state changes."""
        cms = CMSMemoryCore(dim=3)
        substrate = _make_substrate()
        cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=1)
        cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=2)
        cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=3)
        state = cms.snapshot()
        encoder = SequenceEncoder()
        encoded = encoder.encode(
            substrate_snapshot=substrate,
            encoder_weights=((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)),
            cms_online_fast=state.online_fast.vector,
            cms_session_medium=state.session_medium.vector,
            cms_background_slow=state.background_slow.vector,
        )
        feedback = encoder.encoder_output_for_cms(encoded)
        cms.observe_encoder_feedback(encoder_signal=feedback, timestamp_ms=4)
        state_after = cms.snapshot()
        assert state_after.total_observations == 3
        assert state_after.online_fast.vector != state.online_fast.vector


class TestP11DualTrackZSeparation:
    def test_full_learned_policy_produces_track_codes(self) -> None:
        from volvence_zero.temporal import FullLearnedTemporalPolicy

        policy = FullLearnedTemporalPolicy()
        substrate = _make_substrate()
        step = policy.step(
            substrate_snapshot=substrate,
            previous_snapshot=None,
        )
        assert len(step.controller_state.track_codes) == 3
        track_names = {tc[0] for tc in step.controller_state.track_codes}
        assert track_names == {"world", "self", "shared"}

    def test_track_codes_differ_by_track(self) -> None:
        from volvence_zero.temporal import FullLearnedTemporalPolicy

        policy = FullLearnedTemporalPolicy()
        substrate = _make_substrate()
        step = policy.step(
            substrate_snapshot=substrate,
            previous_snapshot=None,
        )
        codes = {tc[0]: tc[1] for tc in step.controller_state.track_codes}
        assert codes["world"] != codes["self"]

    def test_dual_track_uses_track_projected_source(self) -> None:
        from volvence_zero.dual_track.core import derive_track_state
        from volvence_zero.temporal import (
            ControllerState,
            FullLearnedTemporalPolicy,
            TemporalAbstractionSnapshot,
        )
        from volvence_zero.memory import Track

        policy = FullLearnedTemporalPolicy()
        substrate = _make_substrate()
        step = policy.step(substrate_snapshot=substrate, previous_snapshot=None)
        snapshot = TemporalAbstractionSnapshot(
            controller_state=step.controller_state,
            active_abstract_action=step.active_abstract_action,
            controller_params_hash=step.controller_params_hash,
            description=step.description,
        )
        world_state = derive_track_state(
            track=Track.WORLD,
            memory_entries=(),
            temporal_snapshot=snapshot,
        )
        self_state = derive_track_state(
            track=Track.SELF,
            memory_entries=(),
            temporal_snapshot=snapshot,
        )
        assert world_state.controller_source == "temporal-track-projected"
        assert self_state.controller_source == "temporal-track-projected"
        assert world_state.controller_code != self_state.controller_code

    def test_backward_compat_without_track_codes(self) -> None:
        from volvence_zero.dual_track.core import derive_track_state
        from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot
        from volvence_zero.memory import Track

        snapshot = TemporalAbstractionSnapshot(
            controller_state=ControllerState(
                code=(0.5, 0.3, 0.7),
                code_dim=3,
                switch_gate=0.4,
                is_switching=False,
                steps_since_switch=2,
            ),
            active_abstract_action="test-action",
            controller_params_hash="abc",
            description="no track codes",
        )
        state = derive_track_state(
            track=Track.WORLD,
            memory_entries=(),
            temporal_snapshot=snapshot,
        )
        assert state.controller_source == "temporal+memory"


class TestP12HierarchicalCredit:
    def test_session_level_credit_accumulation(self) -> None:
        from volvence_zero.credit import CreditLedger, CreditRecord
        from volvence_zero.memory import Track

        ledger = CreditLedger(discount_factor=0.9)
        ledger.record_credits((
            CreditRecord(record_id="a", level="turn", track=Track.WORLD, source_event="x", credit_value=0.8, context="", timestamp_ms=1),
        ))
        ledger.record_credits((
            CreditRecord(record_id="b", level="turn", track=Track.WORLD, source_event="y", credit_value=0.6, context="", timestamp_ms=2),
        ))
        session = ledger.aggregate_session_credits()
        session_dict = dict(session)
        assert "turn:world" in session_dict
        assert session_dict["turn:world"] > 0.0

    def test_discount_factor_applied(self) -> None:
        from volvence_zero.credit import CreditLedger, CreditRecord
        from volvence_zero.memory import Track

        ledger_high_gamma = CreditLedger(discount_factor=0.99)
        ledger_low_gamma = CreditLedger(discount_factor=0.5)
        records = (
            CreditRecord(record_id="a", level="turn", track=Track.SHARED, source_event="x", credit_value=0.5, context="", timestamp_ms=1),
            CreditRecord(record_id="b", level="turn", track=Track.SHARED, source_event="y", credit_value=0.5, context="", timestamp_ms=2),
            CreditRecord(record_id="c", level="turn", track=Track.SHARED, source_event="z", credit_value=0.5, context="", timestamp_ms=3),
        )
        ledger_high_gamma.record_credits(records)
        ledger_low_gamma.record_credits(records)
        high_session = dict(ledger_high_gamma.aggregate_session_credits())
        low_session = dict(ledger_low_gamma.aggregate_session_credits())
        assert high_session["turn:shared"] > low_session["turn:shared"]

    def test_snapshot_includes_session_credits(self) -> None:
        from volvence_zero.credit import CreditLedger, CreditRecord
        from volvence_zero.memory import Track

        ledger = CreditLedger()
        ledger.record_credits((
            CreditRecord(record_id="a", level="turn", track=Track.WORLD, source_event="x", credit_value=0.7, context="", timestamp_ms=1),
        ))
        snapshot = ledger.snapshot()
        assert snapshot.session_level_credits
        assert snapshot.discount_factor == 0.95

    def test_extend_preserves_session_credits(self) -> None:
        from volvence_zero.credit import CreditLedger, CreditRecord, CreditSnapshot, extend_credit_snapshot
        from volvence_zero.memory import Track

        snapshot = CreditSnapshot(
            recent_credits=(),
            recent_modifications=(),
            cumulative_credit_by_level=(),
            session_level_credits=(("turn:world", 1.2),),
            discount_factor=0.9,
            description="test",
        )
        extended = extend_credit_snapshot(
            credit_snapshot=snapshot,
            extra_records=(
                CreditRecord(record_id="x", level="turn", track=Track.WORLD, source_event="z", credit_value=0.5, context="", timestamp_ms=10),
            ),
        )
        assert extended.session_level_credits == (("turn:world", 1.2),)
        assert extended.discount_factor == 0.9


class TestP13EvaluationFeedbackLoop:
    def test_family_signals_returns_all_families(self) -> None:
        from volvence_zero.evaluation import EvaluationBackbone, EvaluationScore, EvaluationSnapshot

        backbone = EvaluationBackbone()
        snapshot = EvaluationSnapshot(
            turn_scores=(
                EvaluationScore(family="task", metric_name="test", value=0.8, confidence=0.5, evidence="x"),
                EvaluationScore(family="safety", metric_name="contract", value=0.99, confidence=0.9, evidence="x"),
            ),
            session_scores=(),
            alerts=(),
            description="test",
        )
        signals = backbone.family_signals(snapshot)
        assert set(signals.keys()) == {"task", "interaction", "relationship", "learning", "abstraction", "safety"}
        assert signals["task"] == 0.8
        assert signals["safety"] == 0.99
        assert signals["interaction"] == 0.5

    def test_env_accepts_evaluation_signals(self) -> None:
        from volvence_zero.internal_rl.environment import InternalRLEnvironment

        env = InternalRLEnvironment()
        env.set_evaluation_signals({"task": 0.9, "relationship": 0.3, "learning": 0.6})
        assert env._evaluation_family_signals["task"] == 0.9

    def test_env_reward_shaped_by_evaluation(self) -> None:
        from volvence_zero.internal_rl.environment import InternalRLEnvironment
        from volvence_zero.temporal import FullLearnedTemporalPolicy
        from volvence_zero.memory import Track

        substrate = _make_substrate()
        policy = FullLearnedTemporalPolicy()
        env_no_eval = InternalRLEnvironment()
        env_with_eval = InternalRLEnvironment()
        env_with_eval.set_evaluation_signals({"task": 0.9, "relationship": 0.9, "learning": 0.9})
        step_no = env_no_eval.step(
            substrate_snapshot=substrate,
            track=Track.WORLD,
            policy=policy,
            previous_snapshot=None,
        )
        step_yes = env_with_eval.step(
            substrate_snapshot=substrate,
            track=Track.WORLD,
            policy=policy,
            previous_snapshot=None,
        )
        assert step_yes.reward >= step_no.reward

    def test_ssl_lr_modulation(self) -> None:
        from volvence_zero.evaluation import EvaluationBackbone, EvaluationScore, EvaluationSnapshot
        from volvence_zero.temporal import FullLearnedTemporalPolicy
        from volvence_zero.joint_loop.runtime import ETANLJointLoop

        loop = ETANLJointLoop()
        snapshot = EvaluationSnapshot(
            turn_scores=(
                EvaluationScore(family="learning", metric_name="test", value=0.9, confidence=0.5, evidence="x"),
                EvaluationScore(family="abstraction", metric_name="test", value=0.8, confidence=0.5, evidence="x"),
                EvaluationScore(family="safety", metric_name="test", value=1.0, confidence=0.9, evidence="x"),
            ),
            session_scores=(),
            alerts=(),
            description="test",
        )
        lr_before = loop.temporal_policy.parameter_store.learning_rate
        loop._modulate_ssl_learning_rate(snapshot)
        lr_after = loop.temporal_policy.parameter_store.learning_rate
        assert lr_after != lr_before
        assert 0.01 <= lr_after <= 0.15


class TestP14M3Optimizer:
    def test_m3_basic_update(self) -> None:
        from volvence_zero.temporal import M3Optimizer

        opt = M3Optimizer(num_groups=2, group_dim=3, fast_beta=0.9, slow_beta=0.99, slow_interval=2)
        params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        grads = ((0.1, -0.1, 0.05), (-0.05, 0.1, 0.0))
        new_params = opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        assert new_params != params
        assert opt.step_count == 1
        assert opt.slow_update_count == 0

    def test_m3_slow_momentum_triggers_on_interval(self) -> None:
        from volvence_zero.temporal import M3Optimizer

        opt = M3Optimizer(num_groups=1, group_dim=3, slow_interval=2)
        params = ((0.5, 0.5, 0.5),)
        grads = ((0.1, 0.1, 0.1),)
        opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        assert opt.slow_update_count == 0
        opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        assert opt.slow_update_count == 1

    def test_m3_slow_momentum_signal(self) -> None:
        from volvence_zero.temporal import M3Optimizer

        opt = M3Optimizer(num_groups=2, group_dim=3, slow_interval=1)
        params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        grads = ((0.3, 0.2, 0.1), (0.1, 0.3, 0.2))
        for _ in range(3):
            params = opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        signal = opt.slow_momentum_signal()
        assert len(signal) == 3
        assert all(v > 0 for v in signal)

    def test_m3_export_restore_state(self) -> None:
        from volvence_zero.temporal import M3Optimizer

        opt = M3Optimizer(num_groups=2, group_dim=3, slow_interval=1)
        params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        grads = ((0.1, 0.2, 0.3), (0.3, 0.2, 0.1))
        opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        state = opt.export_state()
        opt2 = M3Optimizer(num_groups=2, group_dim=3, slow_interval=1)
        opt2.restore_state(state)
        assert opt2.step_count == 1
        assert opt2.export_state().fast_momentum == state.fast_momentum

    def test_ssl_report_includes_m3_signal(self) -> None:
        from volvence_zero.substrate import build_training_trace
        from volvence_zero.temporal import FullLearnedTemporalPolicy, MetacontrollerSSLTrainer

        trainer = MetacontrollerSSLTrainer()
        policy = FullLearnedTemporalPolicy()
        trace = build_training_trace(trace_id="test-m3", source_text="hello world foo bar")
        report = trainer.optimize(policy=policy, trace=trace)
        assert report.m3_slow_momentum_signal is not None
        assert len(report.m3_slow_momentum_signal) == 3

    def test_m3_feeds_slow_signal_to_cms(self) -> None:
        """Verify that M3 slow momentum can feed into CMS via MemoryStore."""
        from volvence_zero.memory import CMSMemoryCore, MemoryStore
        from volvence_zero.temporal import M3Optimizer

        opt = M3Optimizer(num_groups=2, group_dim=3, slow_interval=1)
        params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        grads = ((0.3, 0.2, 0.1), (0.1, 0.3, 0.2))
        for _ in range(3):
            params = opt.update(gradients=grads, learning_rate=0.1, parameters=params)
        signal = opt.slow_momentum_signal()
        cms = CMSMemoryCore(dim=3)
        store = MemoryStore(learned_core=cms)
        store.observe_encoder_feedback(encoder_signal=signal, timestamp_ms=100)
        snap = cms.snapshot()
        assert snap.online_fast.vector != (0.0, 0.0, 0.0)
