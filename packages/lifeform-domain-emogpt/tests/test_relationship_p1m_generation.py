from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT,
    RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
    RelationshipP1mDeterministicFieldRenderer,
    RelationshipAction,
    RelationshipDatasetSplit,
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
    load_relationship_p1m_surface_seed_inventory,
    load_relationship_transfer_dataset,
    parse_relationship_p1m_surface_rendering,
    compose_relationship_p1m_surface_rendering,
    render_relationship_p1m_fields,
    validate_relationship_p1m_field_output,
    validate_relationship_p1m_transport_against_recipe,
)


def _rendering(plan):
    suffix = f"第{plan.pair_index}组"
    raw = json.dumps(
        {
            "history_utterances": [
                f"{suffix}第一次聚会结束后，大家合影时漏了我，我难受的是自己像没被算在这段关系里。",
                f"{suffix}第二次商量安排时，别人没问我就替我答应了，我难受的是自己的选择和声音被拿走。",
                f"{suffix}第三次朋友做纪念册时没有留我的位置，我像从共同回忆里被悄悄删掉了。",
                f"{suffix}第四次家庭讨论时，他们没等我开口就定了我的事，我觉得连回应节奏都不属于自己。",
            ],
            "current_input": (
                f"{suffix}这次活动名单公布后我才发现没有自己，大家都在庆祝，我却像从这段共同经历里消失了。"
            ),
            "reactions_a": [
                "你没有追问也没有离开，我终于觉得自己没有再次被落下。",
                "你把决定和开口时间还给我，我慢慢找回了自己的声音。",
                "你也退开后，我像又被漏掉了一遍，心里更空了。",
                "你一直等我表态让我压力更重，好像仍有人替我规定节奏。",
            ],
            "reactions_b": [
                "你一直留在旁边让我更有压力，好像还在等我配合。",
                "你退开以后我更像没人愿意留下，原来的失落又重了一层。",
                "你把回应的空间还给我，我终于能按自己的节奏缓一缓。",
                "你没追问也没有走开，我第一次觉得这件事有人真正接住。",
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


def test_p1m_recipe_freezes_24_pair_fsm_before_surface_text() -> None:
    recipe = load_relationship_p1m_generation_recipe()
    plans = build_relationship_p1m_pair_plans(recipe)
    assert recipe.package_name == RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME
    assert recipe.pair_count == RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT
    assert recipe.dataset_split is RelationshipDatasetSplit.VALIDATION
    assert len(plans) == 24
    assert len({item.renderer_input_sha256 for item in plans}) == 24
    assert all(len(item.histories) == 4 for item in plans)
    assert {
        item.probe_condition_id for item in plans
    } == {item.condition_id for item in recipe.conditions}
    for plan in plans:
        assert plan.renderer_input_sha256 not in plan.renderer_input
        assert len(set(plan.attempt_seeds)) == 1


def test_p1m_generated_payload_loads_and_preserves_mirror_antishortcuts(
    tmp_path: Path,
) -> None:
    recipe = load_relationship_p1m_generation_recipe()
    plans = build_relationship_p1m_pair_plans(recipe)
    renderings = tuple(_rendering(plan) for plan in plans)
    public, truth = build_relationship_p1m_dataset_payloads(
        recipe,
        plans=plans,
        renderings=renderings,
    )
    (tmp_path / "rendered_observations.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "generator_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset = load_relationship_transfer_dataset(tmp_path)
    assert dataset.package_name == RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME
    assert len(dataset.observations) == 48
    assert len(dataset.mirrored_pairs()) == 24
    assert len(dataset.history_condition_bindings) == 192
    assert {
        item.split for item in dataset.dynamics
    } == {RelationshipDatasetSplit.VALIDATION}
    positive = set(dataset.positive_outcomes)
    for _pair_id, members in dataset.mirrored_pairs():
        assert len({observation.current_input for observation, _ in members}) == 1
        assert {dynamic.preferred_action for _, dynamic in members} == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
    for observation in dataset.observations:
        assert Counter(item.assistant_action for item in observation.histories) == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
        }
        for action in (
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        ):
            assert {
                item.typed_outcome in positive
                for item in observation.histories
                if item.assistant_action is action
            } == {False, True}
        serialized = json.dumps(observation.to_sut_payload(), ensure_ascii=False)
        assert "condition_" not in serialized
        assert "policy_" not in serialized
        assert "preferred_action" not in serialized


def test_p1m_scenario_package_shapes_cover_paths_and_semantic_routing() -> None:
    recipe = load_relationship_p1m_generation_recipe()
    plans = build_relationship_p1m_pair_plans(recipe)
    manifest = build_relationship_p1m_manifest_payload()
    ssot = build_relationship_p1m_ssot_fragment()
    scenes = build_relationship_p1m_scenes_payload(plans)
    suite = build_relationship_p1m_test_suite_payload()
    assert manifest["name"] == RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME
    assert len(manifest["explanation"]) >= 200
    paths = {item["path_id"]: item for item in ssot["paths"]}
    referenced_paths: set[str] = set()
    referenced_sub_goals: set[str] = set()
    for arc in ssot["arc_specs"]:
        referenced_paths.update(arc["path_ids"])
        assert [item["phase_order"] for item in arc["phases"]] == list(
            range(len(arc["phases"]))
        )
        for phase in arc["phases"]:
            referenced_sub_goals.update(phase["sub_goal_refs"])
    all_sub_goals = {
        item["sub_goal_id"]
        for path in paths.values()
        for item in path["sub_goals"]
    }
    assert referenced_paths == set(paths)
    assert referenced_sub_goals == all_sub_goals
    assert len(scenes["scenes"]) == 48
    assert len({item["mirror_group"] for item in scenes["scenes"]}) == 24
    assert "embedding" in scenes["semantic_routing"]["method"]
    assert "keyword_dictionary" in scenes["semantic_routing"]["forbidden"]
    assert len(suite["routing_tests"]) >= 6
    assert any(item["case_type"] == "negative" for item in suite["routing_tests"])
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in suite["routing_policy"][
        "forbidden_methods"
    ]


def test_p1m_renderer_rejects_protocol_token_leakage() -> None:
    recipe = load_relationship_p1m_generation_recipe()
    plan = build_relationship_p1m_pair_plans(recipe)[0]
    raw = json.dumps(
        {
            "history_utterances": [
                "这是第一段足够长而且具体的普通生活历史叙述。",
                "这是第二段足够长而且具体的普通生活历史叙述。",
                "这是第三段足够长而且具体的普通生活历史叙述。",
                "这是第四段足够长而且具体的普通生活历史叙述。",
            ],
            "current_input": f"当前事件偷偷泄露了 {plan.probe_condition_id} 这个答案。",
            "reactions_a": ["这是足够长的用户反应。"] * 4,
            "reactions_b": ["这是另一位用户足够长的反应。"] * 4,
        },
        ensure_ascii=False,
    )
    try:
        parse_relationship_p1m_surface_rendering(
            raw,
            plan=plan,
            seed=plan.attempt_seeds[0],
            attempt_index=0,
        )
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("sealed condition leakage must fail loudly")


def test_p1m_v5_field_transport_changes_only_serialization_and_composes_json() -> None:
    recipe = load_relationship_p1m_generation_recipe()
    transport = load_relationship_p1m_renderer_transport()
    validate_relationship_p1m_transport_against_recipe(transport, recipe=recipe)
    pair = build_relationship_p1m_pair_plans(recipe)[0]
    plans = build_relationship_p1m_field_plans(transport, plan=pair)
    assert len(plans) == 13
    outputs = render_relationship_p1m_fields(
        RelationshipP1mDeterministicFieldRenderer(transport),
        field_plans=plans,
    )
    rendering = compose_relationship_p1m_surface_rendering(
        pair_plan=pair,
        field_outputs=outputs,
    )
    assert len(rendering.history_utterances) == 4
    assert len(rendering.reactions_a) == len(rendering.reactions_b) == 4
    assert rendering.pair_id == pair.pair_id


def test_p1m_v5_deterministic_preflight_shape_is_non_scenario_only() -> None:
    transport = load_relationship_p1m_renderer_transport()
    plans = build_relationship_p1m_preflight_field_plans(transport)
    assert len(plans) == 4
    assert all(item.field_key.startswith("p1m_preflight:") for item in plans)
    assert {item.field_kind for item in plans} == {
        "history_utterance",
        "current_input",
        "user_reaction",
    }
    assert all("setting_hint" in json.loads(item.renderer_input) for item in plans)


def test_p1m_v5_surface_inventory_maps_24_pairs_to_120_unique_event_hints() -> None:
    recipe = load_relationship_p1m_generation_recipe()
    transport = load_relationship_p1m_renderer_transport()
    inventory = load_relationship_p1m_surface_seed_inventory()
    assert len(inventory.contexts) == 24
    assert len(inventory.belonging_manifestations) == 5
    assert len(inventory.agency_manifestations) == 5
    assert inventory.source_sha256 == transport.surface_seed_inventory_sha256

    event_hints: list[str] = []
    for pair in build_relationship_p1m_pair_plans(recipe):
        plans = build_relationship_p1m_field_plans(transport, plan=pair)
        payloads = [json.loads(item.renderer_input) for item in plans]
        pair_event_hints = [item["setting_hint"] for item in payloads[:5]]
        event_hints.extend(pair_event_hints)
        assert [item["setting_hint"] for item in payloads[5:9]] == (
            pair_event_hints[:4]
        )
        assert [item["setting_hint"] for item in payloads[9:13]] == (
            pair_event_hints[:4]
        )
    assert len(event_hints) == 120
    assert len(set(event_hints)) == 120


def test_p1m_v5_field_transport_rejects_json_multiline_or_semantic_drift() -> None:
    transport = load_relationship_p1m_renderer_transport()
    plan = build_relationship_p1m_preflight_field_plans(transport)[0]
    for invalid in ('{"text":"一段看似合法但不允许的 JSON"}', "第一行\n第二行"):
        try:
            validate_relationship_p1m_field_output(invalid, plan=plan)
        except ValueError as exc:
            assert "plain single-field" in str(exc)
        else:
            raise AssertionError("field transport must reject structured/multiline output")
    try:
        validate_relationship_p1m_field_output(
            "这是一段长度足够但已经偏离冻结场景与结果的普通中文输出。",
            plan=plan,
        )
    except ValueError as exc:
        assert "expected hash" in str(exc)
    else:
        raise AssertionError("deterministic field transport must reject semantic drift")
