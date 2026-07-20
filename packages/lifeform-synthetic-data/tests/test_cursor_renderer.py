from __future__ import annotations

import json
from pathlib import Path

from lifeform_synthetic_data.contracts import GenerationTier, ScenarioBlueprint
from lifeform_synthetic_data.cursor_renderer import (
    ASSET_SCHEMA_VERSION,
    CursorAuthoredJsonClient,
    validate_cursor_assets,
)
from lifeform_synthetic_data.llm import BudgetLedger, RateCard
from lifeform_synthetic_data.renderer import render_trajectory
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import compile_structural_trajectory


def test_cursor_authored_client_renders_without_api_or_cost(
    tmp_path: Path,
) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    _write_asset(tmp_path, blueprint)
    report = validate_cursor_assets((blueprint,), root=tmp_path)
    client = CursorAuthoredJsonClient(asset_root=tmp_path)
    structural = client.enrich_truth(
        compile_structural_trajectory(
            blueprint,
            replicate_index=0,
            seed=23,
            run_id="cursor-render-test",
            created_at="2026-07-20T00:00:00Z",
            git_sha="test",
        )
    )

    rendered, completion, _ = render_trajectory(
        structural,
        client=client,
        budget=BudgetLedger(
            max_cost_usd=0.0,
            rate_card=RateCard(0.0, 0.0),
        ),
        max_output_tokens=256,
    )

    assert report.family_count == 1
    assert report.scenario_count == 1
    assert report.variant_count == sum(blueprint.turns_per_session) * 4
    assert rendered.generation_tier is GenerationTier.RENDERED
    assert rendered.provenance.model_id == client.model_id
    assert rendered.provenance.source_kind == "synthetic_world_fsm_plus_text_render"
    first_facts = {item.key: item.value for item in rendered.truth_frames[0].observable_facts}
    assert first_facts["fact"] == "session 0 turn 0 variant 0"
    assert "scenario_anchor" in first_facts
    assert dict((item.key, item.value) for item in rendered.metadata)["cursor_render_asset_hash"] == report.asset_hash
    assert completion.cost_usd == 0.0
    assert completion.usage.total_tokens == 0
    assert all(not turn.text.startswith("[structural") for session in rendered.sessions for turn in session.turns)


def test_cursor_variant_selection_is_unique_for_128_replicates(
    tmp_path: Path,
) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    _write_asset(tmp_path, blueprint)
    client = CursorAuthoredJsonClient(asset_root=tmp_path)
    transcripts: set[tuple[str, ...]] = set()
    for replicate_index in range(128):
        structural = client.enrich_truth(
            compile_structural_trajectory(
                blueprint,
                replicate_index=replicate_index,
                seed=17072026 + replicate_index,
                run_id="cursor-variant-uniqueness",
                created_at="2026-07-20T00:00:00Z",
                git_sha="test",
            )
        )
        rendered, _, _ = render_trajectory(
            structural,
            client=client,
            budget=BudgetLedger(
                max_cost_usd=0.0,
                rate_card=RateCard(0.0, 0.0),
            ),
            max_output_tokens=256,
        )
        transcripts.add(tuple(turn.text for session in rendered.sessions for turn in session.turns))

    assert len(transcripts) == 128


def _write_asset(root: Path, blueprint: ScenarioBlueprint) -> None:
    payload = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "family": blueprint.family,
        "scenarios": [
            {
                "scenario_id": blueprint.scenario_id,
                "language": blueprint.language,
                "sessions": [
                    {
                        "session_index": session_index,
                        "turns": [
                            {
                                "turn_index": turn_index,
                                "role": ("user" if turn_index % 2 == 0 else "assistant"),
                                "variants": [
                                    (f"session {session_index} turn {turn_index} variant {variant_index}")
                                    for variant_index in range(4)
                                ],
                            }
                            for turn_index in range(turn_count)
                        ],
                    }
                    for session_index, turn_count in enumerate(blueprint.turns_per_session)
                ],
            }
        ],
    }
    (root / f"{blueprint.family}.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
