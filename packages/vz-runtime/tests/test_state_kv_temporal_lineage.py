from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

from volvence_zero.conditioning_bank_contracts import (
    ConditioningLineageRef,
    ConditioningRevocationState,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.agent.session_observation import (
    SessionObservationMixin,
    _substrate_conditioning_lineage_from_previous,
)
from volvence_zero.substrate import (
    FeatureSignal,
    OpenWeightResidualStreamSubstrateAdapter,
    OpenWeightRuntimeCapture,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal import (
    ControllerState,
    LearnedLiteTemporalPolicy,
    PlaceholderTemporalPolicy,
    TemporalAbstractionSnapshot,
    TemporalModule,
    build_temporal_aggregate_snapshot,
)


def _personal_conditioning(*, confidence: float = 0.7) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(0.4 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2), ("boundary_consent", 3)),
        source_fingerprint="state-kv-lineage-personal-v1",
        confidence=confidence,
        is_cold_start=False,
        description="Personal conditioning state for State KV lineage test.",
        rendered_statement="Personal state readout for lineage test.",
    )


def _lineage(*, carrier: str = "prefix_kv") -> ConditioningLineageRef:
    ref = _substrate_conditioning_lineage_from_previous(
        previous_conditioning=_personal_conditioning(),
        user_scope="user-a",
        session_scope="session-a",
        personal_conditioning_wiring=WiringLevel.ACTIVE,
        personal_conditioning_mode=carrier,
    )
    assert ref is not None
    return ref


def _substrate_snapshot(
    *,
    lineage: ConditioningLineageRef | None = None,
) -> SubstrateSnapshot:
    step = ResidualSequenceStep(
        step=0,
        token="hello",
        feature_surface=(
            FeatureSignal(name="residual_sequence_present", values=(1.0,), source="test"),
        ),
        residual_activations=(
            ResidualActivation(layer_index=0, activation=(0.1, 0.2), step=0),
        ),
        description="test residual step",
    )
    return SubstrateSnapshot(
        model_id="test-open-weight",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(step,),
        unavailable_fields=(),
        description="test substrate snapshot",
        conditioning_lineage=lineage,
    )


class _Runtime:
    model_id = "test-open-weight"
    is_frozen = True
    runtime_origin = "unit-test"
    fallback_active = False
    capture_source = "unit-test"

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        del source_text
        snapshot = _substrate_snapshot()
        return OpenWeightRuntimeCapture(
            token_logits=snapshot.token_logits,
            feature_surface=snapshot.feature_surface,
            residual_activations=snapshot.residual_activations,
            residual_sequence=snapshot.residual_sequence,
            description="unit test capture",
        )

    def capture_conditioned(
        self,
        *,
        source_text: str,
        personal_conditioning: PersonalConditioningSnapshot,
        personal_conditioning_carrier: str,
    ) -> OpenWeightRuntimeCapture:
        del source_text, personal_conditioning
        assert personal_conditioning_carrier == "prefix_kv"
        snapshot = _substrate_snapshot()
        conditioned = tuple(
            ResidualActivation(
                layer_index=activation.layer_index,
                activation=tuple(value + 0.4 for value in activation.activation),
                step=activation.step,
            )
            for activation in snapshot.residual_activations
        )
        step = replace(
            snapshot.residual_sequence[0],
            residual_activations=conditioned,
        )
        return OpenWeightRuntimeCapture(
            token_logits=snapshot.token_logits,
            feature_surface=snapshot.feature_surface,
            residual_activations=conditioned,
            residual_sequence=(step,),
            description="conditioned unit test capture",
            personal_conditioning_applied=True,
        )


def test_state_kv_pre_capture_lineage_is_active_only_and_scoped() -> None:
    active_ref = _substrate_conditioning_lineage_from_previous(
        previous_conditioning=_personal_conditioning(),
        user_scope="user-a",
        session_scope="session-a",
        personal_conditioning_wiring=WiringLevel.ACTIVE,
        personal_conditioning_mode="prefix_kv",
    )

    assert active_ref is not None
    assert active_ref.session_scope == "session-a"
    assert active_ref.selected_bank_set == ("personal",)
    assert active_ref.bank_fingerprints[0][0] == "personal"
    assert active_ref.carrier == "prefix_kv"
    assert active_ref.delivery_phase == "substrate-capture"

    assert (
        _substrate_conditioning_lineage_from_previous(
            previous_conditioning=_personal_conditioning(),
            user_scope="user-a",
            session_scope="session-a",
            personal_conditioning_wiring=WiringLevel.SHADOW,
            personal_conditioning_mode="prefix_kv",
        )
        is None
    )
    assert (
        _substrate_conditioning_lineage_from_previous(
            previous_conditioning=_personal_conditioning(),
            user_scope="user-a",
            session_scope="session-a",
            personal_conditioning_wiring=WiringLevel.ACTIVE,
            personal_conditioning_mode="text",
        )
        is None
    )


def test_open_weight_adapter_copies_lineage_to_substrate_and_steps() -> None:
    ref = _lineage()
    adapter = OpenWeightResidualStreamSubstrateAdapter(
        runtime=_Runtime(),
        default_source_text="hello",
        conditioning_lineage=ref,
    )

    snapshot = asyncio.run(adapter.capture())

    assert snapshot.conditioning_lineage == ref
    assert snapshot.residual_sequence
    assert snapshot.residual_sequence[0].conditioning_lineage == ref


def test_state_kv_conditioned_capture_changes_temporal_code_and_switch_gate() -> None:
    ref = _lineage()
    runtime = _Runtime()
    baseline = asyncio.run(
        OpenWeightResidualStreamSubstrateAdapter(
            runtime=runtime,
            default_source_text="hello",
        ).capture()
    )
    conditioned = asyncio.run(
        OpenWeightResidualStreamSubstrateAdapter(
            runtime=runtime,
            default_source_text="hello",
            conditioning_lineage=ref,
            personal_conditioning=_personal_conditioning(),
            personal_conditioning_carrier="prefix_kv",
        ).capture()
    )

    baseline_temporal = asyncio.run(
        TemporalModule(policy=LearnedLiteTemporalPolicy()).process_standalone(
            substrate_snapshot=baseline
        )
    )
    conditioned_temporal = asyncio.run(
        TemporalModule(policy=LearnedLiteTemporalPolicy()).process_standalone(
            substrate_snapshot=conditioned
        )
    )

    assert baseline.personal_conditioning_applied is False
    assert conditioned.personal_conditioning_applied is True
    assert conditioned.residual_activations != baseline.residual_activations
    assert (
        conditioned_temporal.value.controller_state.code
        != baseline_temporal.value.controller_state.code
    )
    assert (
        conditioned_temporal.value.controller_state.switch_gate
        != baseline_temporal.value.controller_state.switch_gate
    )
    assert conditioned_temporal.value.conditioning_lineage_refs == (ref,)


def test_session_observation_passes_previous_state_into_physical_capture() -> None:
    runner = SimpleNamespace(
        _substrate_adapter_factory=None,
        _previous_personal_conditioning_snapshot=_personal_conditioning(),
        _personal_conditioning_revocation_state=(
            ConditioningRevocationState.ACTIVE
        ),
        user_scope="user-a",
        _session_id="session-a",
        _config=SimpleNamespace(
            personal_conditioning=WiringLevel.ACTIVE,
            personal_conditioning_mode="prefix_kv",
        ),
        _default_residual_runtime=_Runtime(),
    )

    adapter = SessionObservationMixin._build_substrate_adapter(
        runner,
        user_input="hello",
    )
    snapshot = asyncio.run(adapter.capture())

    assert snapshot.personal_conditioning_applied is True
    assert snapshot.conditioning_lineage is not None
    assert snapshot.conditioning_lineage.carrier == "prefix_kv"


def test_session_observation_revocation_blocks_capture_and_lineage() -> None:
    runner = SimpleNamespace(
        _substrate_adapter_factory=None,
        _previous_personal_conditioning_snapshot=_personal_conditioning(),
        _personal_conditioning_revocation_state=(
            ConditioningRevocationState.REVOKED
        ),
        user_scope="user-a",
        _session_id="session-a",
        _config=SimpleNamespace(
            personal_conditioning=WiringLevel.ACTIVE,
            personal_conditioning_mode="prefix_kv",
        ),
        _default_residual_runtime=_Runtime(),
    )

    adapter = SessionObservationMixin._build_substrate_adapter(
        runner,
        user_input="hello",
    )
    snapshot = asyncio.run(adapter.capture())

    assert snapshot.personal_conditioning_applied is False
    assert snapshot.conditioning_lineage is None


def test_temporal_snapshot_preserves_substrate_lineage() -> None:
    ref = _lineage()
    module = TemporalModule(policy=PlaceholderTemporalPolicy())

    temporal_snapshot = asyncio.run(
        module.process_standalone(substrate_snapshot=_substrate_snapshot(lineage=ref))
    )

    assert temporal_snapshot.value.conditioning_lineage_refs == (ref,)


def test_temporal_aggregate_deduplicates_conditioning_lineage_refs() -> None:
    ref = _lineage(carrier="residual")
    controller_state = ControllerState(
        code=(0.1, 0.2, 0.3),
        code_dim=3,
        switch_gate=0.0,
        is_switching=False,
        steps_since_switch=1,
    )
    world = TemporalAbstractionSnapshot(
        controller_state=controller_state,
        active_abstract_action="world-action",
        controller_params_hash="world",
        description="world",
        conditioning_lineage_refs=(ref,),
    )
    self_snapshot = TemporalAbstractionSnapshot(
        controller_state=controller_state,
        active_abstract_action="self-action",
        controller_params_hash="self",
        description="self",
        conditioning_lineage_refs=(ref,),
    )

    aggregate = build_temporal_aggregate_snapshot(
        world_snapshot=world,
        self_snapshot=self_snapshot,
    )

    assert aggregate.conditioning_lineage_refs == (ref,)
