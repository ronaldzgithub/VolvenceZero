from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceSafety,
)


_HINT = (
    "Use this test affordance when validating that the lifeform publishes "
    "a z_t-scored affordance snapshot after every completed turn."
)


def _descriptor() -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name="publication_probe",
        kind=AffordanceKind.TOOL,
        version="1.0.0",
        display_name="Publication probe",
        description="Verifies lifeform-side affordance snapshot publication.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Do not use outside this integration test.",
        parameters_schema={"type": "object", "properties": {}},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=AffordanceLatencyClass.INSTANT,
        ),
        safety_model=AffordanceSafety(),
    )


async def test_lifeform_publishes_affordance_snapshot_after_turn() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    registry, _ = lifeform.ensure_affordance_registry()
    registry.register(_descriptor())
    session = lifeform.create_session(session_id="affordance-publication")

    await session.run_turn("Start a normal turn.")

    snapshot = session.latest_active_snapshots["affordance"]
    assert snapshot.owner == "AffordanceModule"
    assert tuple(item.name for item in snapshot.value.available) == (
        "publication_probe",
    )
    assert snapshot.value.candidates_for_turn[0].descriptor_name == "publication_probe"
    assert "src=z_t_projection" in snapshot.value.candidates_for_turn[0].rationale
