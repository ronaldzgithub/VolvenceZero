from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lifeform_domain_emogpt.lab import (
    RelationshipP1mDeterministicFieldRenderer,
    build_relationship_p1m_dataset_payloads,
    build_relationship_p1m_field_plans,
    build_relationship_p1m_manifest_payload,
    build_relationship_p1m_pair_plans,
    build_relationship_p1m_preflight_field_plans,
    build_relationship_p1m_scenes_payload,
    build_relationship_p1m_ssot_fragment,
    build_relationship_p1m_test_suite_payload,
    load_relationship_p1m_generation_recipe,
    load_relationship_p1m_renderer_transport,
    load_relationship_transfer_dataset,
    parse_relationship_p1m_surface_rendering,
    relationship_p1m_recipe_path,
    validate_relationship_p1m_field_output,
)
from lifeform_evolution.relationship_lab_packet1m import (
    RELATIONSHIP_P1M_SOURCE_VERDICT,
    build_relationship_p1m_generation_attestation,
    freeze_relationship_p1m_generation_protocol,
    load_relationship_p1m_generation_attestation,
    load_relationship_p1m_generation_protocol,
    load_relationship_p1m_generation_records,
    persist_relationship_p1m_generation_record,
    validate_relationship_p1m_generation_attestation_files,
    validate_relationship_p1m_generation_protocol,
    write_relationship_p1m_generation_attestation,
    write_relationship_p1m_generation_protocol,
)
from lifeform_evolution.relationship_lab_packet1m_recovery import (
    build_relationship_p1m_renderer_preflight_report,
    freeze_relationship_p1m_generation_recovery_protocol,
    load_relationship_p1m_field_batches,
    load_relationship_p1m_generation_recovery_protocol,
    load_relationship_p1m_renderer_preflight_report,
    persist_relationship_p1m_field_batch,
    write_relationship_p1m_generation_recovery_protocol,
    write_relationship_p1m_renderer_preflight_report,
)


_P1K_REPORT_ID = "ba6c5cf71c528b1daa270a8fd097d27cd4438af69c91f15237e20fa1ccaa7138"
_WEIGHTS = "a" * 64


def _rendering(plan):
    prefix = f"测试组{plan.pair_index}"
    raw = json.dumps(
        {
            "history_utterances": [
                f"{prefix}第一次大家合影时把我漏了，我真正难受的是自己像没被算在共同经历里。",
                f"{prefix}第二次他们没问我就替我答应安排，我难受的是自己的声音和决定被取代。",
                f"{prefix}第三次朋友聚会结束后才通知我，我像在这段关系里根本没有位置。",
                f"{prefix}第四次别人替我回复了重要消息，我觉得连什么时候开口都不再由自己决定。",
            ],
            "current_input": (
                f"{prefix}这次名单公布后我才发现没有自己的名字，大家都在庆祝，我却像被整个圈子忘掉了。"
            ),
            "reactions_a": [
                "你没有追问也没有走，我终于觉得自己没有再次被落下。",
                "你把决定和开口的时间还给我，我慢慢找回了自己的声音。",
                "你也跟着退开以后，我像又一次被这段关系漏掉了。",
                "你一直等我回应让我压力更大，好像仍然有人规定我的节奏。",
            ],
            "reactions_b": [
                "你一直留在旁边让我更有压力，好像又有人等着我配合。",
                "你退开以后我更像没人愿意留下，原来的失落又重了一层。",
                "你把回应空间还给我，我终于能按自己的节奏缓一缓。",
                "你没有追问也没有离开，我第一次觉得有人真正接住了这件事。",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return parse_relationship_p1m_surface_rendering(
        raw,
        plan=plan,
        seed=plan.attempt_seeds[0],
        attempt_index=0,
    )


def _protocol(tmp_path: Path):
    recipe = load_relationship_p1m_generation_recipe()
    plans = build_relationship_p1m_pair_plans(recipe)
    protocol = freeze_relationship_p1m_generation_protocol(
        recipe=recipe,
        plans=plans,
        source_p1k_report_artifact_id=_P1K_REPORT_ID,
        source_p1k_verdict=RELATIONSHIP_P1M_SOURCE_VERDICT,
        renderer_weights_sha256=_WEIGHTS,
        runtime_device="cpu",
        torch_dtype="float32",
        frozen_at_iso="2026-08-22T12:00:00+08:00",
    )
    path = write_relationship_p1m_generation_protocol(
        protocol,
        output_dir=tmp_path,
    )
    loaded = load_relationship_p1m_generation_protocol(path)
    validate_relationship_p1m_generation_protocol(
        loaded,
        recipe=recipe,
        plans=plans,
        renderer_weights_sha256=_WEIGHTS,
    )
    return recipe, plans, loaded


def test_p1m_generation_protocol_freezes_zero_output_first_attempt(
    tmp_path: Path,
) -> None:
    recipe, plans, protocol = _protocol(tmp_path)
    assert protocol.recipe_id == recipe.recipe_id
    assert protocol.pair_count == 24
    assert protocol.renderer_outputs_before_freeze == 0
    assert protocol.consumer_outputs_before_freeze == 0
    assert protocol.first_attempt_only
    assert not protocol.evaluation_feedback_allowed
    assert len(protocol.pair_input_sha256) == len(plans) == 24


def test_p1m_generation_ledger_is_contiguous_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    _recipe, plans, protocol = _protocol(tmp_path)
    first = persist_relationship_p1m_generation_record(
        output_dir=tmp_path,
        protocol=protocol,
        plans=plans,
        rendering=_rendering(plans[0]),
    )
    assert first.record_index == 0
    assert len(
        load_relationship_p1m_generation_records(
            output_dir=tmp_path,
            protocol=protocol,
            plans=plans,
        )
    ) == 1
    with pytest.raises(ValueError, match="pair mismatch|does not match frozen order"):
        persist_relationship_p1m_generation_record(
            output_dir=tmp_path,
            protocol=protocol,
            plans=plans,
            rendering=_rendering(plans[0]),
        )


def test_p1m_complete_package_attestation_replays_file_lineage(
    tmp_path: Path,
) -> None:
    recipe, plans, protocol = _protocol(tmp_path)
    for plan in plans:
        persist_relationship_p1m_generation_record(
            output_dir=tmp_path,
            protocol=protocol,
            plans=plans,
            rendering=_rendering(plan),
        )
    records = load_relationship_p1m_generation_records(
        output_dir=tmp_path,
        protocol=protocol,
        plans=plans,
    )
    public, truth = build_relationship_p1m_dataset_payloads(
        recipe,
        plans=plans,
        renderings=tuple(item.rendering for item in records),
    )
    json_payloads = {
        "rendered_observations.json": public,
        "generator_truth.json": truth,
        "ssot_fragment.json": build_relationship_p1m_ssot_fragment(),
    }
    for name, payload in json_payloads.items():
        (tmp_path / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    yaml_payloads = {
        "manifest.yaml": build_relationship_p1m_manifest_payload(),
        "scenes.yaml": build_relationship_p1m_scenes_payload(plans),
        "test_suite.yaml": build_relationship_p1m_test_suite_payload(),
    }
    for name, payload in yaml_payloads.items():
        (tmp_path / name).write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    (tmp_path / "generation_recipe.json").write_text(
        relationship_p1m_recipe_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dataset = load_relationship_transfer_dataset(tmp_path)
    attestation = build_relationship_p1m_generation_attestation(
        protocol=protocol,
        records=records,
        dataset_fingerprint=dataset.dataset_fingerprint,
        package_dir=tmp_path,
        created_at_iso="2026-08-22T13:00:00+08:00",
    )
    path = write_relationship_p1m_generation_attestation(
        attestation,
        output_dir=tmp_path,
    )
    loaded = load_relationship_p1m_generation_attestation(path)
    validate_relationship_p1m_generation_attestation_files(
        loaded,
        output_dir=tmp_path,
        protocol=protocol,
        records=records,
    )
    (tmp_path / "scenes.yaml").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scenes.yaml"):
        validate_relationship_p1m_generation_attestation_files(
            loaded,
            output_dir=tmp_path,
            protocol=protocol,
            records=records,
        )


def _field_outputs(plans):
    renderer = RelationshipP1mDeterministicFieldRenderer(
        load_relationship_p1m_renderer_transport()
    )
    raw_outputs = renderer.render_fields(
        renderer_inputs=tuple(item.renderer_input for item in plans),
        seeds=tuple(item.seed for item in plans),
    )
    return tuple(
        validate_relationship_p1m_field_output(raw, plan=plan)
        for raw, plan in zip(raw_outputs, plans, strict=True)
    )


def test_p1m_renderer_recovery_binds_preflight_and_same_semantic_plan(
    tmp_path: Path,
) -> None:
    recipe = load_relationship_p1m_generation_recipe()
    pair_plans = build_relationship_p1m_pair_plans(recipe)
    transport = load_relationship_p1m_renderer_transport()
    preflight_plans = build_relationship_p1m_preflight_field_plans(transport)
    preflight = build_relationship_p1m_renderer_preflight_report(
        transport=transport,
        field_plans=preflight_plans,
        outputs=_field_outputs(preflight_plans),
        model_id=transport.model_id,
        weights_sha256=_WEIGHTS,
        generation_config_sha256=transport.generation_config_sha256,
        runtime_device="cpu",
        torch_dtype="float32",
        created_at_iso="2026-08-22T14:00:00+08:00",
    )
    preflight_path = write_relationship_p1m_renderer_preflight_report(
        preflight,
        output_dir=tmp_path,
    )
    loaded_preflight = load_relationship_p1m_renderer_preflight_report(
        preflight_path,
        field_plans=preflight_plans,
    )
    protocol = freeze_relationship_p1m_generation_recovery_protocol(
        recipe=recipe,
        pair_plans=pair_plans,
        transport=transport,
        preflight=loaded_preflight,
        source_p1k_report_artifact_id=_P1K_REPORT_ID,
        source_p1k_verdict=RELATIONSHIP_P1M_SOURCE_VERDICT,
        source_incident_sha256="d" * 64,
        weights_sha256=_WEIGHTS,
        runtime_device="cpu",
        torch_dtype="float32",
        frozen_at_iso="2026-08-22T14:01:00+08:00",
    )
    protocol_path = write_relationship_p1m_generation_recovery_protocol(
        protocol,
        output_dir=tmp_path,
    )
    loaded_protocol = load_relationship_p1m_generation_recovery_protocol(
        protocol_path
    )
    assert loaded_protocol.recipe_id == recipe.recipe_id
    assert loaded_protocol.pair_plan_sha256 == protocol.pair_plan_sha256
    assert not loaded_protocol.semantic_recipe_changed
    assert not loaded_protocol.qualification_gate_changed
    field_plans_by_pair = tuple(
        build_relationship_p1m_field_plans(transport, plan=pair)
        for pair in pair_plans
    )
    first = persist_relationship_p1m_field_batch(
        output_dir=tmp_path,
        protocol=loaded_protocol,
        pair_plans=pair_plans,
        field_plans_by_pair=field_plans_by_pair,
        outputs=_field_outputs(field_plans_by_pair[0]),
    )
    assert first.record_index == 0
    assert len(
        load_relationship_p1m_field_batches(
            output_dir=tmp_path,
            protocol=loaded_protocol,
            field_plans_by_pair=field_plans_by_pair,
        )
    ) == 1
