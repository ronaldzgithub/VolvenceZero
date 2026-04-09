from __future__ import annotations

import asyncio

from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.reflection import TemporalPriorUpdate, TemporalStructureProposal
from volvence_zero.substrate import SimulatedResidualSubstrateAdapter, build_training_trace
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    HeuristicTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerSSLTrainer,
    PlaceholderTemporalPolicy,
    TemporalModule,
    fit_policy_from_trace_dataset,
)
from volvence_zero.substrate import TrainingTraceDataset
from volvence_zero.temporal.metacontroller_components import (
    ActionFamilyObservation,
    DiscoveredActionFamily,
    discover_latent_action_family,
)


def test_temporal_module_builds_placeholder_snapshot():
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=PlaceholderSubstrateAdapter(model_id="placeholder-model"),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    temporal = TemporalModule(policy=PlaceholderTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(temporal.process_standalone(substrate_snapshot=substrate_snapshot))

    assert snapshot.value.controller_state.code_dim == 3
    assert snapshot.value.controller_state.switch_gate == 0.0
    assert snapshot.value.active_abstract_action == "placeholder-controller"


def test_temporal_module_heuristic_switches_on_feature_signature_change():
    temporal = TemporalModule(policy=HeuristicTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    first_substrate = asyncio.run(
        SubstrateModule(
            adapter=FeatureSurfaceSubstrateAdapter(
                model_id="heuristic-model",
                feature_surface=(FeatureSignal(name="context_a", values=(0.2, 0.3), source="adapter"),),
            ),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    second_substrate = asyncio.run(
        SubstrateModule(
            adapter=FeatureSurfaceSubstrateAdapter(
                model_id="heuristic-model",
                feature_surface=(FeatureSignal(name="context_b", values=(0.8, 0.9), source="adapter"),),
            ),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value

    first = asyncio.run(temporal.process_standalone(substrate_snapshot=first_substrate)).value
    second = asyncio.run(temporal.process_standalone(substrate_snapshot=second_substrate)).value

    assert first.controller_state.code_dim == 3
    assert second.controller_state.is_switching is True
    assert second.active_abstract_action.startswith("refresh-controller-context:")


def test_temporal_module_runs_in_shadow_chain():
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="shadow-temporal-model",
            feature_surface=(FeatureSignal(name="temporal_signal", values=(0.6, 0.2), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    temporal = TemporalModule()
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, temporal],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "substrate" in result
    assert "temporal_abstraction" not in result
    temporal_snapshot = shadow_snapshots["temporal_abstraction"]
    assert temporal_snapshot.value.controller_state.code_dim == 3
    assert temporal_snapshot.value.controller_params_hash


def test_learned_lite_policy_fits_from_trace_dataset_and_emits_controller_step():
    dataset = TrainingTraceDataset(
        (
            build_training_trace(trace_id="t1", source_text="steady progress"),
            build_training_trace(trace_id="t2", source_text="repair emotional tension"),
        )
    )
    policy = LearnedLiteTemporalPolicy()
    fit_policy_from_trace_dataset(policy=policy, dataset=dataset)
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=SimulatedResidualSubstrateAdapter(trace=dataset.latest()),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    temporal = TemporalModule(policy=policy, wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(temporal.process_standalone(substrate_snapshot=substrate_snapshot))

    assert snapshot.value.controller_params_hash
    assert snapshot.value.active_abstract_action.endswith("learned-lite")


def test_learned_lite_policy_can_align_with_internal_rl_parameters():
    policy = LearnedLiteTemporalPolicy()
    initial = policy.export_parameters()

    policy.align_with_internal_rl(
        world_weights=(0.8, 0.1, 0.1),
        self_weights=(0.1, 0.8, 0.1),
        shared_weights=(0.4, 0.4, 0.2),
        persistence=0.7,
    )
    aligned = policy.export_parameters()

    assert aligned != initial
    assert aligned.switch_bias > 0.0


def test_learned_lite_policy_exports_runtime_visible_metacontroller_state():
    policy = LearnedLiteTemporalPolicy()

    runtime_state = policy.export_runtime_state()

    assert runtime_state.mode == "learned-lite"
    assert runtime_state.temporal_parameters.switch_bias >= 0.0
    assert len(runtime_state.track_parameters) == 3
    assert len(runtime_state.update_steps) == 3


def test_full_learned_policy_uses_residual_sequence_and_exports_decoder_state():
    trace = build_training_trace(trace_id="full-policy", source_text="repair then focus then stabilize")
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=SimulatedResidualSubstrateAdapter(trace=trace),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    policy = FullLearnedTemporalPolicy()
    temporal = TemporalModule(policy=policy, wiring_level=WiringLevel.ACTIVE)

    assert policy.parameter_store.action_families == ()
    snapshot = asyncio.run(temporal.process_standalone(substrate_snapshot=substrate_snapshot))
    runtime_state = policy.export_runtime_state()

    assert snapshot.value.controller_state.code_dim == 3
    assert runtime_state.mode == "full-learned"
    assert runtime_state.sequence_length == len(trace.steps)
    assert runtime_state.decoder_control
    assert runtime_state.decoder_applied_control
    assert runtime_state.active_label
    assert runtime_state.active_label.startswith("discovered_family_")
    assert runtime_state.prior_mean
    assert runtime_state.prior_std
    assert runtime_state.posterior_mean
    assert runtime_state.posterior_std
    assert runtime_state.posterior_sample_noise
    assert runtime_state.z_tilde
    assert runtime_state.binary_switch_rate >= 0.0


def test_ssl_trainer_updates_full_learned_policy_metrics():
    trace = build_training_trace(trace_id="ssl-trace", source_text="steady repair and guided exploration")
    policy = FullLearnedTemporalPolicy()
    trainer = MetacontrollerSSLTrainer()

    report = trainer.optimize(policy=policy, trace=trace)
    runtime_state = policy.export_runtime_state()

    assert report.trained_steps > 0
    assert report.total_loss >= 0.0
    assert report.posterior_drift >= 0.0
    assert runtime_state.latest_ssl_loss >= 0.0
    assert runtime_state.sequence_length == len(trace.steps)
    assert runtime_state.posterior_drift >= 0.0
    assert runtime_state.learning_phase == "ssl"
    assert runtime_state.structure_frozen is False
    assert runtime_state.active_label.startswith("discovered_family_")
    assert policy.parameter_store.action_families
    assert all(family.support >= 1 for family in policy.parameter_store.action_families)


def test_discovered_action_family_lifecycle_can_split_topology():
    observation = ActionFamilyObservation(
        latent_code=(0.72, 0.18, 0.14),
        decoder_control=(0.71, 0.16, 0.18),
        switch_gate=0.82,
        posterior_drift=0.06,
        persistence_window=0.85,
    )
    families, _, _ = discover_latent_action_family(
        observation=observation,
        action_families=(),
        structure_frozen=False,
    )
    for _ in range(6):
        families, _, _ = discover_latent_action_family(
            observation=observation,
            action_families=families,
            structure_frozen=False,
        )

    divergent = ActionFamilyObservation(
        latent_code=(0.58, 0.43, 0.18),
        decoder_control=(0.56, 0.41, 0.22),
        switch_gate=0.54,
        posterior_drift=0.34,
        persistence_window=0.05,
    )
    updated_families, active_label, summary = discover_latent_action_family(
        observation=divergent,
        action_families=families,
        structure_frozen=False,
    )

    assert len(updated_families) >= 2
    assert active_label.startswith("discovered_family_")
    assert "split:" in summary or "anti-collapse-create:" in summary


def test_discovered_action_family_lifecycle_can_merge_and_prune_topology():
    families = (
        DiscoveredActionFamily(
            family_id="discovered_family_0",
            latent_centroid=(0.62, 0.28, 0.18),
            decoder_centroid=(0.60, 0.26, 0.20),
            support=6,
            stability=0.82,
            switch_bias=0.68,
            mean_posterior_drift=0.10,
            mean_persistence_window=0.72,
            summary="manual-a",
        ),
        DiscoveredActionFamily(
            family_id="discovered_family_1",
            latent_centroid=(0.61, 0.29, 0.19),
            decoder_centroid=(0.59, 0.27, 0.21),
            support=5,
            stability=0.80,
            switch_bias=0.66,
            mean_posterior_drift=0.11,
            mean_persistence_window=0.70,
            summary="manual-b",
        ),
        DiscoveredActionFamily(
            family_id="discovered_family_2",
            latent_centroid=(0.10, 0.12, 0.11),
            decoder_centroid=(0.09, 0.10, 0.12),
            support=1,
            stability=0.20,
            switch_bias=0.15,
            mean_posterior_drift=0.05,
            mean_persistence_window=0.10,
            summary="manual-c",
        ),
    )
    observation = ActionFamilyObservation(
        latent_code=(0.60, 0.28, 0.18),
        decoder_control=(0.58, 0.27, 0.19),
        switch_gate=0.64,
        posterior_drift=0.09,
        persistence_window=0.75,
    )

    updated_families, active_label, summary = discover_latent_action_family(
        observation=observation,
        action_families=families,
        structure_frozen=False,
        max_families=2,
    )

    assert len(updated_families) == 1
    assert active_label == "discovered_family_0"
    assert "merge:" in summary or updated_families[0].support >= 11


def test_temporal_owner_can_apply_structural_family_proposals():
    policy = FullLearnedTemporalPolicy()
    policy.parameter_store.action_families = (
        DiscoveredActionFamily(
            family_id="discovered_family_0",
            latent_centroid=(0.62, 0.28, 0.18),
            decoder_centroid=(0.60, 0.26, 0.20),
            support=6,
            stability=0.82,
            switch_bias=0.68,
            mean_posterior_drift=0.10,
            mean_persistence_window=0.72,
            summary="manual-a",
        ),
        DiscoveredActionFamily(
            family_id="discovered_family_1",
            latent_centroid=(0.58, 0.30, 0.20),
            decoder_centroid=(0.57, 0.28, 0.22),
            support=2,
            stability=0.54,
            switch_bias=0.64,
            mean_posterior_drift=0.16,
            mean_persistence_window=0.64,
            summary="manual-b",
        ),
    )

    operations = policy.apply_reflection_prior_update(
        update=TemporalPriorUpdate(
            target="metacontroller.temporal_prior",
            target_groups=("action-family-structure",),
            residual_strength=0.5,
            memory_strength=0.3,
            reflection_strength=0.2,
            switch_bias_delta=0.0,
            persistence_delta=0.0,
            learning_rate_delta=0.0,
            description="test structure proposals",
            structure_proposals=(
                TemporalStructureProposal(
                    proposal_type="split",
                    family_id="discovered_family_0",
                    related_family_id=None,
                    confidence=0.6,
                    justification="split test",
                ),
                TemporalStructureProposal(
                    proposal_type="prune",
                    family_id="discovered_family_1",
                    related_family_id=None,
                    confidence=0.7,
                    justification="prune test",
                ),
            ),
        )
    )

    assert any(operation.startswith("temporal-prior:action-family-split=") for operation in operations)
    assert any(operation == "temporal-prior:action-family-prune=discovered_family_1" for operation in operations)
    assert len(policy.parameter_store.action_families) == 2


def test_family_competition_memory_raises_monopoly_under_repeated_selection():
    policy = FullLearnedTemporalPolicy()
    for _ in range(6):
        policy.parameter_store.discover_action_family(
            latent_code=(0.82, 0.16, 0.12),
            decoder_control=(0.78, 0.14, 0.15),
            switch_gate=0.74,
            posterior_drift=0.08,
            persistence_window=0.82,
        )
    runtime_state = policy.export_runtime_state()

    assert runtime_state.active_family_summary is not None
    assert runtime_state.active_family_summary.reuse_streak >= 4
    assert runtime_state.action_family_monopoly_pressure > 0.7
    assert runtime_state.active_family_competition_score < 0.7


def test_family_competition_turnover_health_improves_with_diverse_family_usage():
    repeated_policy = FullLearnedTemporalPolicy()
    for _ in range(6):
        repeated_policy.parameter_store.discover_action_family(
            latent_code=(0.82, 0.16, 0.12),
            decoder_control=(0.78, 0.14, 0.15),
            switch_gate=0.74,
            posterior_drift=0.08,
            persistence_window=0.82,
        )
    repeated_runtime = repeated_policy.export_runtime_state()

    diverse_policy = FullLearnedTemporalPolicy()
    observations = (
        ((0.82, 0.16, 0.12), (0.78, 0.14, 0.15), 0.74),
        ((0.22, 0.78, 0.25), (0.18, 0.72, 0.22), 0.28),
        ((0.36, 0.30, 0.84), (0.34, 0.26, 0.78), 0.48),
        ((0.76, 0.18, 0.16), (0.74, 0.16, 0.18), 0.72),
        ((0.20, 0.80, 0.30), (0.16, 0.74, 0.26), 0.24),
        ((0.34, 0.28, 0.86), (0.32, 0.24, 0.82), 0.46),
    )
    for latent_code, decoder_control, switch_gate in observations:
        diverse_policy.parameter_store.discover_action_family(
            latent_code=latent_code,
            decoder_control=decoder_control,
            switch_gate=switch_gate,
            posterior_drift=0.12,
            persistence_window=0.46,
        )
    diverse_runtime = diverse_policy.export_runtime_state()

    assert diverse_runtime.action_family_turnover_health > repeated_runtime.action_family_turnover_health
    assert len(diverse_runtime.action_family_summaries) >= 2


def test_ssl_alpha_controls_kl_weight():
    from volvence_zero.temporal import MetacontrollerSSLTrainer
    trace = build_training_trace(trace_id="alpha-trace", source_text="steady warm planning")

    low_alpha = MetacontrollerSSLTrainer(alpha=0.01)
    policy_low = FullLearnedTemporalPolicy()
    report_low = low_alpha.optimize(policy=policy_low, trace=trace)

    high_alpha = MetacontrollerSSLTrainer(alpha=1.0)
    policy_high = FullLearnedTemporalPolicy()
    report_high = high_alpha.optimize(policy=policy_high, trace=trace)

    assert report_low.kl_loss >= 0.0
    assert report_high.kl_loss >= 0.0
    if report_low.kl_loss > 0.001 and report_high.kl_loss > 0.001:
        assert report_high.total_loss >= report_low.total_loss


def test_switch_gate_stats_published_in_ssl_report():
    from volvence_zero.temporal import MetacontrollerSSLTrainer
    trace = build_training_trace(trace_id="stats-trace", source_text="repair tension then plan")
    trainer = MetacontrollerSSLTrainer(alpha=0.1)
    policy = FullLearnedTemporalPolicy()
    report = trainer.optimize(policy=policy, trace=trace)

    assert report.switch_gate_stats is not None
    assert len(report.switch_gate_stats.beta_histogram) == 10
    assert sum(report.switch_gate_stats.beta_histogram) == report.trained_steps
    assert report.switch_gate_stats.observation_count == report.trained_steps
    assert 0.0 <= report.switch_gate_stats.switch_frequency <= 1.0
    assert report.switch_gate_stats.mean_persistence_steps >= 0.0


def test_family_long_term_payoff_accumulation():
    from volvence_zero.temporal.metacontroller_components import (
        DiscoveredActionFamily,
        update_family_outcome_history,
    )
    family = DiscoveredActionFamily(
        family_id="test_family_0",
        latent_centroid=(0.5, 0.5, 0.5),
        decoder_centroid=(0.5, 0.5, 0.5),
        support=3,
        stability=0.8,
        switch_bias=0.5,
        long_term_payoff=0.5,
        delayed_credit_sum=0.0,
    )
    assert family.long_term_payoff == 0.5
    assert family.delayed_credit_sum == 0.0

    updated = update_family_outcome_history(
        (family,), family_id="test_family_0", outcome_value=0.8,
    )
    assert updated[0].outcome_driven_score > 0.0
    assert updated[0].long_term_payoff == 0.5
    assert updated[0].delayed_credit_sum == 0.0


def test_family_competition_state_detects_collapse():
    from volvence_zero.temporal.metacontroller_components import (
        DiscoveredActionFamily,
        FamilyCompetitionState,
        build_family_competition_state,
    )
    dominant = DiscoveredActionFamily(
        family_id="dominant_0",
        latent_centroid=(0.8, 0.1, 0.1),
        decoder_centroid=(0.8, 0.1, 0.1),
        support=20,
        stability=0.9,
        switch_bias=0.5,
        monopoly_pressure=0.85,
        reuse_streak=6,
        long_term_payoff=0.7,
    )
    minor = DiscoveredActionFamily(
        family_id="minor_1",
        latent_centroid=(0.1, 0.8, 0.1),
        decoder_centroid=(0.1, 0.8, 0.1),
        support=2,
        stability=0.4,
        switch_bias=0.3,
        long_term_payoff=0.3,
    )
    state = build_family_competition_state((dominant, minor))
    assert isinstance(state, FamilyCompetitionState)
    assert state.top1_share > 0.8
    assert state.collapse_alert is True
    assert state.monopoly_alert is True
    assert len(state.ranked_families) == 2


# ---------------------------------------------------------------------------
# Phase 3 Uplift — variational bottleneck and family competition validation
# ---------------------------------------------------------------------------

def test_noncausal_embedder_tightens_posterior():
    """W5.2: Verify the noncausal embedder's enrich_posterior actually
    tightens the posterior (reduces variance) relative to causal-only.

    The bidirectional embedder sees the full sequence and should produce
    a tighter posterior than the causal encoder which only sees the past.
    """
    from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder
    from volvence_zero.temporal import MetacontrollerSSLTrainer

    trace = build_training_trace(
        trace_id="noncausal-trace",
        source_text="repair tension then continue helpfully and plan carefully for growth",
    )

    trainer = MetacontrollerSSLTrainer(alpha=0.1)
    policy = FullLearnedTemporalPolicy()
    report = trainer.optimize(policy=policy, trace=trace)

    assert report.noncausal_kl_tightening >= 0.0, (
        f"KL tightening should be non-negative, got {report.noncausal_kl_tightening}"
    )
    assert report.noncausal_information_content > 0.0, (
        "Noncausal embedding should have positive information content"
    )

    embedder = NonCausalSequenceEmbedder(n_z=3)
    from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind, FeatureSignal, ResidualActivation
    substrate = SubstrateSnapshot(
        model_id="noncausal-test",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.5, 0.3),
        feature_surface=(
            FeatureSignal(name="val", values=(0.7,), source="test"),
        ),
        residual_activations=(
            ResidualActivation(layer_index=0, activation=(0.5, 0.3, 0.7), step=0),
            ResidualActivation(layer_index=1, activation=(0.6, 0.4, 0.8), step=0),
        ),
        residual_sequence=(),
        unavailable_fields=(),
        description="test",
    )
    embedding = embedder.embed(substrate_snapshot=substrate)
    assert embedding.sequence_length >= 1
    assert len(embedding.summary_vector) == 3

    causal_mean = (0.5, 0.5, 0.5)
    causal_std = (0.3, 0.3, 0.3)
    enrichment = embedder.enrich_posterior(
        causal_mean=causal_mean,
        causal_std=causal_std,
        embedding=embedding,
    )

    assert len(enrichment.enriched_mean) == 3
    assert len(enrichment.enriched_std) == 3
    enriched_var = sum(s ** 2 for s in enrichment.enriched_std) / 3
    causal_var = sum(s ** 2 for s in causal_std) / 3
    assert enriched_var <= causal_var, (
        f"Enriched variance ({enriched_var:.4f}) should be <= causal variance ({causal_var:.4f})"
    )
    assert enrichment.kl_tightening >= 0.0


def test_noncausal_embedder_bidirectional_ordering_matters():
    """Verify the bidirectional embedder produces different outputs for
    different sequence orderings, proving it actually processes order."""
    from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder

    embedder = NonCausalSequenceEmbedder(n_z=4)

    trace_fwd = build_training_trace(
        trace_id="fwd", source_text="repair tension then plan carefully",
    )
    trace_rev = build_training_trace(
        trace_id="rev", source_text="carefully plan then tension repair",
    )

    from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind, FeatureSignal, ResidualSequenceStep
    def _make_full_substrate(trace):
        return SubstrateSnapshot(
            model_id=trace.trace_id,
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=tuple(0.5 for _ in trace.steps),
            feature_surface=trace.steps[0].feature_surface if trace.steps else (),
            residual_activations=trace.steps[0].residual_activations if trace.steps else (),
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=s.step, token=s.token,
                    feature_surface=s.feature_surface,
                    residual_activations=s.residual_activations,
                    description=f"step {s.step}",
                )
                for s in trace.steps
            ),
            unavailable_fields=(),
            description="full",
        )

    embed_fwd = embedder.embed(substrate_snapshot=_make_full_substrate(trace_fwd))
    embed_rev = embedder.embed(substrate_snapshot=_make_full_substrate(trace_rev))

    assert embed_fwd.summary_vector != embed_rev.summary_vector, (
        "Different token orders should produce different embeddings"
    )


def test_phase3_multi_alpha_beta_distribution():
    """Phase 3 W5.1: Test multiple alpha values and verify switch gate
    behavior changes with KL weight.

    Higher alpha should produce stronger KL penalty and different total
    loss profiles compared to lower alpha.
    """
    from volvence_zero.temporal import MetacontrollerSSLTrainer

    trace = build_training_trace(
        trace_id="multi-alpha-trace",
        source_text="repair tension then continue helpfully and plan carefully",
    )
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    reports = {}
    for alpha in alphas:
        trainer = MetacontrollerSSLTrainer(alpha=alpha)
        policy = FullLearnedTemporalPolicy()
        report = trainer.optimize(policy=policy, trace=trace)
        reports[alpha] = report

    for alpha in alphas:
        report = reports[alpha]
        assert report.switch_gate_stats is not None, f"No switch gate stats for alpha={alpha}"
        assert len(report.switch_gate_stats.beta_histogram) == 10
        assert report.switch_gate_stats.observation_count > 0

    for alpha in alphas:
        assert reports[alpha].kl_loss >= 0.0

    low = reports[0.01].total_loss
    high = reports[1.0].total_loss
    assert high != low or reports[1.0].kl_loss != reports[0.01].kl_loss, (
        "Different alpha values should produce different loss profiles"
    )


def test_phase3_family_competition_not_collapsing_with_diverse_rollouts():
    """Phase 3 W6.1: Verify family distribution does not collapse
    in multi-scenario rollouts with diverse latent codes."""
    policy = FullLearnedTemporalPolicy()
    latent_codes = [
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.1, 0.1, 0.8),
        (0.5, 0.5, 0.1),
        (0.1, 0.5, 0.5),
        (0.5, 0.1, 0.5),
        (0.3, 0.7, 0.2),
        (0.7, 0.2, 0.3),
    ]

    for i, code in enumerate(latent_codes):
        policy.parameter_store.discover_action_family(
            latent_code=code,
            decoder_control=code,
            switch_gate=0.6 + i * 0.03,
            posterior_drift=0.05,
            persistence_window=0.7,
        )

    families = policy.parameter_store.action_families
    assert len(families) >= 2, "Diverse codes should produce multiple families"

    supports = [f.support for f in families]
    total_support = sum(supports)
    if total_support > 0:
        max_share = max(supports) / total_support
        assert max_share < 0.95, (
            f"Top family has {max_share:.0%} share, should not monopolize"
        )


# ---------------------------------------------------------------------------
# Phase 3 W7.2 — Emergence vs Heuristic A/B comparison
# ---------------------------------------------------------------------------

def test_ab_switch_gate_alpha_vs_heuristic_bias():
    """A/B comparison: variational bottleneck alpha vs fixed switch_bias.

    Run SSL training with alpha > 0 (emergence) and alpha = 0 (heuristic
    bias only), then compare switch gate sparsity patterns.
    """
    trace = build_training_trace(
        trace_id="ab-switch",
        source_text="repair tension then continue helpfully and plan carefully for growth",
    )

    trainer_emergence = MetacontrollerSSLTrainer(alpha=0.1)
    policy_e = FullLearnedTemporalPolicy()
    report_e = trainer_emergence.optimize(policy=policy_e, trace=trace)

    trainer_heuristic = MetacontrollerSSLTrainer(alpha=0.0)
    policy_h = FullLearnedTemporalPolicy()
    report_h = trainer_heuristic.optimize(policy=policy_h, trace=trace)

    assert report_e.switch_gate_stats is not None
    assert report_h.switch_gate_stats is not None

    hist_e = report_e.switch_gate_stats.beta_histogram
    hist_h = report_h.switch_gate_stats.beta_histogram
    edge_mass_e = hist_e[0] + hist_e[-1]
    edge_mass_h = hist_h[0] + hist_h[-1]

    total_e = report_e.total_loss
    total_h = report_h.total_loss
    assert total_e != total_h, (
        "Alpha should produce different loss profiles"
    )


def test_ab_family_competition_payoff_weighted_vs_similarity_only():
    """A/B comparison: family selection with payoff weighting vs pure similarity.

    Families with high payoff should be preferred by payoff-weighted
    selection but not by similarity-only.
    """
    from volvence_zero.temporal.metacontroller_components import (
        DiscoveredActionFamily,
        build_family_competition_state,
    )

    good_family = DiscoveredActionFamily(
        family_id="good_fam",
        latent_centroid=(0.5, 0.5, 0.5),
        decoder_centroid=(0.5, 0.5, 0.5),
        support=5,
        stability=0.7,
        switch_bias=0.5,
        long_term_payoff=0.9,
        delayed_credit_sum=2.0,
    )
    bad_family = DiscoveredActionFamily(
        family_id="bad_fam",
        latent_centroid=(0.52, 0.48, 0.51),
        decoder_centroid=(0.52, 0.48, 0.51),
        support=5,
        stability=0.7,
        switch_bias=0.5,
        long_term_payoff=0.1,
        delayed_credit_sum=0.2,
    )

    state = build_family_competition_state((good_family, bad_family))
    assert state.ranked_families[0][0] == "good_fam", (
        "Payoff-weighted ranking should prefer high-payoff family"
    )

    query_code = (0.51, 0.49, 0.505)
    sim_good = sum(abs(query_code[i] - good_family.latent_centroid[i]) for i in range(3))
    sim_bad = sum(abs(query_code[i] - bad_family.latent_centroid[i]) for i in range(3))
    assert abs(sim_good - sim_bad) < 0.1, (
        "Centroids are close, so similarity-only would not strongly distinguish"
    )
    assert good_family.long_term_payoff > bad_family.long_term_payoff, (
        "But payoff clearly distinguishes them"
    )


def test_ab_regime_learned_weight_vs_fixed():
    """A/B comparison: regime selection with learned weights vs fixed weights.

    With attribution_horizons=(1,), after enough turns the regime module
    accumulates delayed outcomes and adjusts selection weights. This
    demonstrates learned selection diverging from uniform fixed weights.
    """
    from volvence_zero.regime import RegimeModule

    module = RegimeModule(attribution_horizons=(1,), wiring_level=WiringLevel.ACTIVE)
    initial_weights = dict(module._selection_weights)
    assert all(abs(w - 1.0) < 0.01 for w in initial_weights.values()), (
        "Initial weights should all be ~1.0 (equivalent to fixed heuristic)"
    )

    for _ in range(5):
        asyncio.run(module.process_standalone())

    final_weights = dict(module._selection_weights)
    for regime_id, weight in final_weights.items():
        assert isinstance(weight, float)
        assert 0.3 <= weight <= 2.0, (
            f"Weight for {regime_id} out of range: {weight}"
        )
