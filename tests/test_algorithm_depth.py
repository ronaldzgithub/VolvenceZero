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


# =========================================================================
#  P15: N-dim Tensor Core
# =========================================================================


class TestP15TensorOps:
    """Verify pure-Python tensor operations."""

    def test_gru_cell_output_shape(self) -> None:
        from volvence_zero.temporal.tensor_ops import gru_cell, init_gru_params, zeros

        params = init_gru_params(8, 16, seed=0)
        x = tuple(0.5 for _ in range(8))
        h = zeros(16)
        h_next = gru_cell(
            x=x, h_prev=h,
            W_z=params["W_z"], U_z=params["U_z"], b_z=params["b_z"],
            W_r=params["W_r"], U_r=params["U_r"], b_r=params["b_r"],
            W_h=params["W_h"], U_h=params["U_h"], b_h=params["b_h"],
        )
        assert len(h_next) == 16
        assert all(-1.0 <= v <= 1.0 for v in h_next)

    def test_ffn_2layer_output_shape(self) -> None:
        from volvence_zero.temporal.tensor_ops import ffn_2layer, init_ffn_params

        params = init_ffn_params(8, 12, 4, seed=0)
        x = tuple(0.3 for _ in range(8))
        out = ffn_2layer(x=x, W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"])
        assert len(out) == 4

    def test_vec_ops_basic(self) -> None:
        from volvence_zero.temporal.tensor_ops import vec_add, vec_mul, vec_scale, vec_sigmoid

        a = (0.5, -0.5, 1.0)
        b = (0.1, 0.2, 0.3)
        assert len(vec_add(a, b)) == 3
        assert len(vec_mul(a, b)) == 3
        assert len(vec_scale(a, 2.0)) == 3
        sig = vec_sigmoid(a)
        assert all(0.0 <= v <= 1.0 for v in sig)


class TestP15NdimSequenceEncoder:
    """Verify GRU-based encoder at n_z=16."""

    def test_encode_produces_correct_dim(self) -> None:
        from volvence_zero.temporal.metacontroller_components import NdimSequenceEncoder

        enc = NdimSequenceEncoder(n_z=16)
        substrate = _make_substrate()
        result = enc.encode(substrate_snapshot=substrate)
        assert len(result.posterior.z_tilde) == 16
        assert len(result.posterior.hidden_state) == 16
        assert len(result.posterior.posterior_mean) == 16
        assert len(result.posterior.posterior_std) == 16
        assert result.posterior.posterior_drift >= 0.0

    def test_recurrence_updates_hidden(self) -> None:
        from volvence_zero.temporal.metacontroller_components import NdimSequenceEncoder

        enc = NdimSequenceEncoder(n_z=8)
        substrate = _make_substrate()
        r1 = enc.encode(substrate_snapshot=substrate)
        r2 = enc.encode(substrate_snapshot=substrate, previous_hidden_state=r1.posterior.hidden_state)
        assert r1.posterior.hidden_state != r2.posterior.hidden_state


class TestP15NdimSwitchUnit:
    """Verify element-wise switch gate."""

    def test_switch_produces_ndim_beta(self) -> None:
        from volvence_zero.temporal.metacontroller_components import NdimSwitchUnit

        sw = NdimSwitchUnit(n_z=16)
        z = tuple(0.5 for _ in range(16))
        prev = tuple(0.0 for _ in range(16))
        beta_cont, beta_bin, scalar = sw.compute(z_tilde=z, previous_code=prev)
        assert len(beta_cont) == 16
        assert len(beta_bin) == 16
        assert all(0.0 <= v <= 1.0 for v in beta_cont)
        assert all(v in (0.0, 1.0) for v in beta_bin)
        assert 0.0 <= scalar <= 1.0


class TestP15NdimResidualDecoder:
    """Verify 2-layer FFN decoder."""

    def test_decode_produces_correct_dim(self) -> None:
        from volvence_zero.temporal.metacontroller_components import NdimResidualDecoder

        dec = NdimResidualDecoder(n_z=16)
        latent = tuple(0.4 for _ in range(16))
        result = dec.decode(latent_code=latent)
        assert len(result.applied_control) == 16
        assert all(0.0 <= v <= 1.0 for v in result.applied_control)


class TestP15FullLearnedNdimPolicy:
    """Verify FullLearnedTemporalPolicy at n_z=16 uses ndim path."""

    def test_full_policy_ndim_step(self) -> None:
        from volvence_zero.temporal import FullLearnedTemporalPolicy, MetacontrollerParameterStore

        store = MetacontrollerParameterStore(n_z=16)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        substrate = _make_substrate()
        step = policy.step(substrate_snapshot=substrate, previous_snapshot=None)
        assert len(step.controller_state.code) == 16
        assert step.controller_state.code_dim == 16
        assert "ndim" in step.description.lower()

    def test_full_policy_legacy_dim3(self) -> None:
        from volvence_zero.temporal import FullLearnedTemporalPolicy, MetacontrollerParameterStore

        store = MetacontrollerParameterStore(n_z=3)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        substrate = _make_substrate()
        step = policy.step(substrate_snapshot=substrate, previous_snapshot=None)
        assert len(step.controller_state.code) == 3
        assert step.controller_state.code_dim == 3

    def test_ndim_ssl_trainer(self) -> None:
        from volvence_zero.temporal import MetacontrollerParameterStore, FullLearnedTemporalPolicy
        from volvence_zero.temporal.ssl import MetacontrollerSSLTrainer
        from volvence_zero.substrate import build_training_trace

        store = MetacontrollerParameterStore(n_z=8)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        trace = build_training_trace(trace_id="ndim-ssl", source_text="hello world ndim test")
        trainer = MetacontrollerSSLTrainer(n_z=8)
        report = trainer.optimize(policy=policy, trace=trace)
        assert report.trained_steps >= 1
        assert len(report.m3_slow_momentum_signal) == 8

    def test_ndim_causal_policy_rollout(self) -> None:
        from volvence_zero.temporal import MetacontrollerParameterStore, FullLearnedTemporalPolicy
        from volvence_zero.internal_rl.sandbox import InternalRLSandbox
        from volvence_zero.memory import Track

        store = MetacontrollerParameterStore(n_z=16)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        sandbox = InternalRLSandbox(policy=policy)
        substrates = tuple(_make_substrate() for _ in range(3))
        rollout = sandbox.rollout(
            rollout_id="ndim-test",
            substrate_steps=substrates,
            track=Track.SHARED,
        )
        assert len(rollout.transitions) == 3
        for t in rollout.transitions:
            assert len(t.latent_code) == 16
            assert len(t.hidden_state) == 16

    def test_parameter_store_ndim_property(self) -> None:
        from volvence_zero.temporal import MetacontrollerParameterStore

        store3 = MetacontrollerParameterStore(n_z=3)
        assert store3.n_z == 3
        store16 = MetacontrollerParameterStore(n_z=16)
        assert store16.n_z == 16
        assert len(store16.latest_latent_mean) == 16
        assert len(store16.encoder_weights) == 16


# =========================================================================
#  P16: Non-Causal Sequence Embedder
# =========================================================================


class TestP16NonCausalEmbedder:
    """Verify bidirectional s(e_{1:T}) embedder."""

    def test_embed_produces_correct_dim(self) -> None:
        from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder

        embedder = NonCausalSequenceEmbedder(n_z=16)
        substrate = _make_substrate()
        result = embedder.embed(substrate_snapshot=substrate)
        assert len(result.summary_vector) == 16
        assert len(result.forward_final) == 16
        assert len(result.backward_final) == 16
        assert 0.0 <= result.information_content <= 1.0

    def test_embed_from_steps(self) -> None:
        from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder

        embedder = NonCausalSequenceEmbedder(n_z=8)
        steps = tuple(tuple(float(i + j) * 0.1 for j in range(8)) for i in range(5))
        result = embedder.embed_from_steps(step_vectors=steps)
        assert len(result.summary_vector) == 8
        assert result.sequence_length == 5

    def test_bidirectional_differs_from_forward_only(self) -> None:
        from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder

        embedder = NonCausalSequenceEmbedder(n_z=8)
        substrate = _make_substrate()
        result = embedder.embed(substrate_snapshot=substrate)
        assert result.forward_final != result.backward_final

    def test_posterior_enrichment(self) -> None:
        from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder

        embedder = NonCausalSequenceEmbedder(n_z=8)
        substrate = _make_substrate()
        embedding = embedder.embed(substrate_snapshot=substrate)
        causal_mean = tuple(0.5 for _ in range(8))
        causal_std = tuple(0.3 for _ in range(8))
        enrichment = embedder.enrich_posterior(
            causal_mean=causal_mean,
            causal_std=causal_std,
            embedding=embedding,
        )
        assert len(enrichment.enriched_mean) == 8
        assert len(enrichment.enriched_std) == 8
        assert enrichment.kl_tightening >= 0.0
        for s in enrichment.enriched_std:
            assert s >= 0.05

    def test_ssl_trainer_reports_noncausal_metrics(self) -> None:
        from volvence_zero.temporal import MetacontrollerParameterStore, FullLearnedTemporalPolicy
        from volvence_zero.temporal.ssl import MetacontrollerSSLTrainer
        from volvence_zero.substrate import build_training_trace

        store = MetacontrollerParameterStore(n_z=8)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        trace = build_training_trace(trace_id="noncausal-test", source_text="alpha beta gamma delta")
        trainer = MetacontrollerSSLTrainer(n_z=8)
        report = trainer.optimize(policy=policy, trace=trace)
        assert report.noncausal_information_content >= 0.0
        assert report.noncausal_kl_tightening >= 0.0
        assert "kl_tightening" in report.description

    def test_information_asymmetry(self) -> None:
        """Non-causal embedder sees full sequence; causal encoder sees prefix only."""
        from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder
        from volvence_zero.temporal.metacontroller_components import NdimSequenceEncoder

        n_z = 8
        embedder = NonCausalSequenceEmbedder(n_z=n_z)
        encoder = NdimSequenceEncoder(n_z=n_z)
        substrate = _make_substrate()
        full_embed = embedder.embed(substrate_snapshot=substrate)
        causal_result = encoder.encode(substrate_snapshot=substrate)
        assert full_embed.summary_vector != causal_result.posterior.posterior_mean


# =========================================================================
#  P17: Unified SSL→RL Training Pipeline
# =========================================================================


class TestP17SSLRLPipeline:
    """Verify the two-phase training orchestrator."""

    def _make_traces(self, count: int = 8) -> tuple:
        from volvence_zero.substrate import build_training_trace
        return tuple(
            build_training_trace(
                trace_id=f"pipe-{i}",
                source_text=f"word{i} alpha beta gamma delta epsilon zeta",
            )
            for i in range(count)
        )

    def test_pipeline_runs_both_phases(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=4, rl_max_steps=3)
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(8)
        result = pipeline.run_pipeline(traces=traces)
        assert result.ssl_steps_completed >= 2
        assert result.rl_steps_completed >= 1
        assert result.transition_step >= 0
        ssl_reports = [r for r in result.phase_reports if r.phase == "ssl"]
        rl_reports = [r for r in result.phase_reports if r.phase == "rl"]
        assert len(ssl_reports) >= 2
        assert len(rl_reports) >= 1

    def test_pipeline_respects_convergence(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(
            n_z=8,
            ssl_min_steps=2,
            ssl_max_steps=10,
            ssl_convergence_threshold=999.0,
            rl_max_steps=5,
        )
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(6)
        result = pipeline.run_pipeline(traces=traces)
        assert result.ssl_steps_completed >= 2
        assert result.transition_step >= 1

    def test_pipeline_phase_reports_have_metrics(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=2)
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(6)
        result = pipeline.run_pipeline(traces=traces)
        for report in result.phase_reports:
            if report.phase == "ssl":
                assert report.ssl_loss >= 0.0
            elif report.phase == "rl":
                assert report.total_reward != 0.0 or report.policy_objective != 0.0

    def test_pipeline_rollback_to_ssl(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=2)
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(6)
        pipeline.run_pipeline(traces=traces)
        if pipeline._ssl_checkpoint is not None:
            assert pipeline.rollback_to_ssl_checkpoint()

    def test_pipeline_binary_gate_rl(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=2, binary_gate_rl=True)
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(6)
        result = pipeline.run_pipeline(traces=traces)
        rl_reports = [r for r in result.phase_reports if r.phase == "rl"]
        assert len(rl_reports) >= 1

    def test_pipeline_noncausal_metrics_flow(self) -> None:
        from volvence_zero.joint_loop.pipeline import SSLRLTrainingPipeline, PipelineConfig

        cfg = PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=1)
        pipeline = SSLRLTrainingPipeline(config=cfg)
        traces = self._make_traces(5)
        result = pipeline.run_pipeline(traces=traces)
        ssl_reports = [r for r in result.phase_reports if r.phase == "ssl"]
        assert any(r.noncausal_kl_tightening >= 0.0 for r in ssl_reports)


# =========================================================================
#  P18: Propagation Topo-Sort + Guard Closure
# =========================================================================


class TestP18TopoSort:
    """Verify topological sorting and cycle detection."""

    def test_topo_sort_linear_chain(self) -> None:
        import asyncio
        from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel, topo_sort_modules
        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class _Val:
            x: int

        class ModA(RuntimeModule[_Val]):
            slot_name = "a"
            owner = "A"
            value_type = _Val
            dependencies = ()

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=1))

        class ModB(RuntimeModule[_Val]):
            slot_name = "b"
            owner = "B"
            value_type = _Val
            dependencies = ("a",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=2))

        class ModC(RuntimeModule[_Val]):
            slot_name = "c"
            owner = "C"
            value_type = _Val
            dependencies = ("b",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=3))

        modules = [ModC(), ModA(), ModB()]
        sorted_mods = topo_sort_modules(modules)
        slots = [m.slot_name for m in sorted_mods]
        assert slots.index("a") < slots.index("b")
        assert slots.index("b") < slots.index("c")

    def test_cycle_detection(self) -> None:
        from volvence_zero.runtime import RuntimeModule, Snapshot, detect_dependency_cycle
        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class _Val:
            x: int

        class ModX(RuntimeModule[_Val]):
            slot_name = "x"
            owner = "X"
            value_type = _Val
            dependencies = ("y",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=1))

        class ModY(RuntimeModule[_Val]):
            slot_name = "y"
            owner = "Y"
            value_type = _Val
            dependencies = ("x",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=2))

        cycle = detect_dependency_cycle([ModX(), ModY()])
        assert cycle is not None
        assert len(cycle) > 0

    def test_cycle_graceful_fallback(self) -> None:
        from volvence_zero.runtime import RuntimeModule, Snapshot, topo_sort_modules
        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class _Val:
            x: int

        class ModX(RuntimeModule[_Val]):
            slot_name = "x"
            owner = "X"
            value_type = _Val
            dependencies = ("y",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=1))

        class ModY(RuntimeModule[_Val]):
            slot_name = "y"
            owner = "Y"
            value_type = _Val
            dependencies = ("x",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=2))

        modules = [ModX(), ModY()]
        sorted_mods = topo_sort_modules(modules)
        assert len(sorted_mods) == 2

    def test_propagation_with_auto_sort(self) -> None:
        import asyncio
        from volvence_zero.runtime import RuntimeModule, Snapshot, propagate
        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class _Val:
            x: int

        class ModA(RuntimeModule[_Val]):
            slot_name = "a_sort"
            owner = "A_sort"
            value_type = _Val
            dependencies = ()

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=1))

        class ModB(RuntimeModule[_Val]):
            slot_name = "b_sort"
            owner = "B_sort"
            value_type = _Val
            dependencies = ("a_sort",)

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                a_val = upstream["a_sort"].value
                return self.publish(_Val(x=a_val.x + 1))

        modules = [ModB(), ModA()]
        result = asyncio.run(propagate(modules, auto_sort=True))
        assert "a_sort" in result
        assert "b_sort" in result
        assert result["b_sort"].value.x == 2

    def test_guard_closure_runs(self) -> None:
        import asyncio
        from volvence_zero.runtime import RuntimeModule, Snapshot, EventRecorder, propagate
        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class _Val:
            x: int

        class ModA(RuntimeModule[_Val]):
            slot_name = "guard_a"
            owner = "Guard_A"
            value_type = _Val
            dependencies = ()

            async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[_Val]:
                return self.publish(_Val(x=1))

        recorder = EventRecorder()
        asyncio.run(propagate([ModA()], recorder=recorder))
        closure_events = [e for e in recorder.events if e.event_type == "guard.closure.passed"]
        assert len(closure_events) == 1


# =========================================================================
#  P19: N-dim CMS + Gradient-Style Updates
# =========================================================================


class TestP19NdimCMS:
    """Verify gradient-style CMS with momentum and anti-forgetting."""

    def test_gradient_update_changes_vector(self) -> None:
        from volvence_zero.memory import CMSMemoryCore
        from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind

        cms = CMSMemoryCore(dim=8, online_lr=0.5, session_lr=0.3, background_lr=0.1)
        substrate = _make_substrate()
        cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=1)
        snap = cms.snapshot()
        assert any(v != 0.0 for v in snap.online_fast.vector)
        assert len(snap.online_fast.vector) == 8

    def test_momentum_accumulates(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=4, momentum_beta=0.9)
        substrate = _make_substrate()
        for i in range(5):
            cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=i)
        snap = cms.snapshot()
        assert any(m != 0.0 for m in snap.online_fast.momentum)

    def test_anti_forgetting_backflow(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=4, anti_forgetting=0.5, background_cadence=1)
        substrate = _make_substrate()
        for i in range(10):
            cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=i)
        snap_before = cms.snapshot()
        bg = snap_before.background_slow.vector
        online = snap_before.online_fast.vector
        for i in range(len(bg)):
            if bg[i] > 0.01:
                assert abs(online[i] - bg[i]) < abs(0.5 - bg[i]) + 0.3

    def test_per_band_learning_rates_in_snapshot(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=4, online_lr=0.7, session_lr=0.4, background_lr=0.15)
        snap = cms.snapshot()
        assert snap.online_fast.learning_rate == 0.7
        assert snap.session_medium.learning_rate == 0.4
        assert snap.background_slow.learning_rate == 0.15

    def test_encoder_feedback_with_ndim(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=16)
        signal = tuple(0.5 for _ in range(16))
        cms.observe_encoder_feedback(encoder_signal=signal, timestamp_ms=1)
        snap = cms.snapshot()
        assert any(v != 0.0 for v in snap.online_fast.vector)
        assert len(snap.online_fast.vector) == 16

    def test_encoder_feedback_dim_mismatch_projects(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=8)
        signal = tuple(0.5 for _ in range(3))
        cms.observe_encoder_feedback(encoder_signal=signal, timestamp_ms=1)
        snap = cms.snapshot()
        assert len(snap.online_fast.vector) == 8
        assert any(v != 0.0 for v in snap.online_fast.vector)

    def test_anti_forgetting_strength_in_snapshot(self) -> None:
        from volvence_zero.memory import CMSMemoryCore

        cms = CMSMemoryCore(dim=4, anti_forgetting=0.25)
        snap = cms.snapshot()
        assert snap.online_fast.anti_forgetting_strength == 0.25
        assert snap.background_slow.anti_forgetting_strength == 0.25
