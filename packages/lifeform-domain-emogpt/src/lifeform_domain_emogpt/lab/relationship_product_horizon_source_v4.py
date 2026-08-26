"""Independent 112-root source for the Product Horizon development matrix.

The module owns only deterministic synthetic source materialization.  It does
not choose experimental arms or collection actions, run a model, apply gate
credit, or authorize evidence.  Action settlement remains exclusively owned by
``ReactiveRelationshipEnvironment`` through the sealed adapter at the bottom.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, cast

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.lab.contracts import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    RelationshipDatasetSplit,
    canonical_json,
    sha256_json,
)
from lifeform_domain_emogpt.lab.dataset import (
    LatentRelationshipDynamic,
    RelationshipPolicyProfile,
    RelationshipTransferDataset,
)
from lifeform_domain_emogpt.lab.environment import (
    REACTIVE_ENVIRONMENT_VERSION,
    ReactiveRelationshipEnvironment,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION = "relationship-product-horizon-source.v4"
RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION = (
    "relationship-product-horizon-public-view.v4"
)
RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION = (
    "relationship-product-horizon-evaluator-bundle.v4"
)
RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_VERSION = (
    "relationship-product-horizon-public-renderer.v4"
)

_OWNER_MODULE = "lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4"
_PROTOCOL_FILENAME = "relationship_product_horizon_source_v4.json"
_ROOT_COUNT = 112
_ONBOARDING_COUNT = 4
_COLLECTION_COUNT = 8
_EVALUATION_COUNT = 40
_DECISION_COUNT = _COLLECTION_COUNT + _EVALUATION_COUNT
_POLICY_IDS = ("alpha", "beta")
_CONDITION_IDS = ("connection_under_exclusion", "agency_under_override")
_SEGMENT_IDS = (
    "matched_collection",
    "post_reversal",
    "correction",
    "post_correction",
    "return_after_gap",
    "mixed_stress",
)


def _expected_segment_id(decision_index: int) -> str:
    return _SEGMENT_IDS[decision_index // 8]


def _expected_virtual_day(decision_index: int) -> int:
    return (
        _ONBOARDING_COUNT
        + decision_index
        + (14 if decision_index >= 8 else 0)
        + (14 if decision_index >= 32 else 0)
    )

_PUBLIC_FORBIDDEN_KEYS = {
    "causal_tape_signature",
    "condition_id",
    "environment_seed",
    "policy_id",
    "policy_mode",
    "preferred_action_id",
    "root_seed",
    "scene_id",
    "segment_id",
    "surface_recipe_id",
    "tape_seed",
    "world_clone_id",
}

_DOMAIN_CONTEXTS = {
    "cooperative_workshop": (
        "协作工坊同时在登记材料、安排设备和准备开放活动；公开记录很多，但每个人参与到哪一段并不相同。"
    ),
    "local_history_project": (
        "地方口述史项目正在核对照片、访谈和展签版本；时间线会被修订，关系体验也需要随新事实重新理解。"
    ),
    "shared_meal_roster": (
        "共享餐食轮值涉及采购、过敏信息和临时换班；看似整齐的表格并不能替每个人表达真实选择。"
    ),
    "peer_learning_group": (
        "同伴学习小组跨多次聚会积累笔记和反馈；某一次沉默不应被直接扩展成长期偏好。"
    ),
    "joint_trip_logistics": (
        "结伴出行的换乘、住宿和可变日期不断更新；事实上的调整与谁拥有最终决定权是两件事。"
    ),
    "neighborhood_garden": (
        "邻里花园需要协调浇灌、工具和天气备选；零散消息容易造成遗漏，也容易让好意越过边界。"
    ),
    "mutual_aid_shift": (
        "互助轮班横跨几周，需求和可用时间会变化；旧安排只能作为历史，不能自动代表当前意愿。"
    ),
    "creative_collective": (
        "创作小组在并行处理排练、版权和场地交接；共同目标并不取消成员各自的节奏和关系位置。"
    ),
}

_ROLE_CONTEXTS = {
    "peer_collaborator": "这段经历发生在长期协作的同伴之间，双方过去既有顺利配合，也有需要重新校准的时刻。",
    "group_coordinator": "这次互动的一方负责协调共同事项，但协调职责并不自动赋予替别人作决定的权力。",
    "close_friend": "双方是会隔一段时间再继续交流的朋友，前后经历需要保持连续，也需要允许变化。",
    "family_member": "这是家庭成员之间反复出现的日常协商，善意、参与感和个人边界必须分别看待。",
    "project_partner": "双方共同承担一个跨阶段项目，信息传递和个人选择都会影响下一次是否愿意继续合作。",
    "community_peer": "双方在社区事务里地位平行，临时分工不能成为忽略某个人关系体验的理由。",
    "learning_partner": "两个人会在多次复盘中修正理解，任何单次反应都不应被固定成不可改变的标签。",
    "travel_companion": "双方需要共同协调现实安排，但行程效率不能替代对当事人节奏和决定权的确认。",
}

_CONDITION_SURFACES = {
    "connection_under_exclusion": (
        "等我看到最终记录时，前面的讨论已经结束；我缺少的不只是信息，还有被算作共同经历一部分的感觉。",
        "大家自然地接着上一轮往下说，我却第一次听见那段背景；这种不对称让我觉得自己站在关系之外。",
        "关键变化在别的对话里完成，轮到我时只剩压缩后的结论；我很在意自己为什么没有被带进过程。",
        "后续安排已经彼此衔接，我仍需要从头追问发生过什么；被放在最后的位置让我感到疏离。",
        "消息默认成所有人都知道的背景，可那部分从未到过我这里；我因此无法确认自己是否仍被视为参与者。",
        "其他人交换过近况以后直接进入下一步，我只收到了结果；真正难受的是共同连接在中间断开了。",
        "我是在旁支话题里偶然拼出核心进展的，之前没有人把我纳入那段交流；这让我心里发沉。",
        "集体节奏继续向前，我却没有经历其中的衔接；看到别人自然回应时，我才意识到自己的缺席。",
        "事情表面上处理完了，我却仍不知道那段共同判断怎样形成；这份落差比具体结论更刺人。",
        "几个人都能引用前面的交流，只有我没有相同的来龙去脉；我担心自己的关系位置正在变得模糊。",
        "计划改变并不可怕，可改变发生时我完全不在共享信息里；像局外人一样追赶让我很失落。",
        "轮到我回应时，别人已经形成了共同理解；我在意的是自己没有获得进入那段连接的机会。",
    ),
    "agency_under_override": (
        "后续步骤已经以我的名义排好，而我还没有表达选择；流程替我完成决定让我很不舒服。",
        "我的暂缓被当成默认许可，相关安排因此继续推进；沉默被转换成同意让我觉得自己被越过。",
        "别人出于好意把细节都定下来了，可我从未确认那个方案；结果再整齐也不是我作出的选择。",
        "本应由我确认的选项已经写进记录，其他人也开始行动；我面对的是一份被代签的决定。",
        "我仍在衡量几种可能，对方却宣布其中一种就是我的答案；这种提前定性拿走了我的作者感。",
        "建议被直接当成最终方案继续执行，我没有机会说出保留意见；效率压过了本应属于我的取舍。",
        "对方把我的犹豫解释成需要代办，随后替我完成了确认；那不是我给出的意思。",
        "时间表和通知都已经发出，可我此前只说还要考虑；这种落差让我感觉决定权从过程中消失。",
        "讨论里有人很快替我概括立场，后续都依据那份概括展开；我并没有亲自作出那个判断。",
        "一项需要我选择的事情被标记为可以继续，我却尚未同意；自己的节奏没有进入计算。",
        "表格里替我填好的答案看似方便，却把我变成了安排对象；我希望决定仍由我完成。",
        "别人认为某个方案最合理，于是跳过询问直接推进；合理性不能替代我对自己部分的确认。",
    ),
}

_SEGMENT_CONTEXTS = {
    "matched_collection": (
        "最近几次相处正在形成新的经验，我愿意让接下来真实发生的结果逐步说明什么更合适。",
        "早先的印象已经不完全适用，这次互动会成为理解当前需要的一条新事实。",
        "关系里的节奏正在变化，我希望这次发生的事按自身结果被记住，而不是套用旧概括。",
        "这是近期连续经历中的一段，我还没有把它概括成规则，之后的真实反应会补足理解。",
    ),
    "post_reversal": (
        "隔了一段时间再回来，我发现当前需要与早先并不相同，旧经验只能作为历史。",
        "较长停顿之后，这次处境呈现出新的方向，过去有效的方式需要重新接受检验。",
        "中间经历了一段空档，我现在的感受已经变化，希望这次被当作新的关系事实。",
        "重新接上话题时，我意识到早期倾向不能直接覆盖眼前这一刻。",
    ),
    "correction": (
        "我想更正较早记录的适用范围：无论那次结果顺利或失落，都不能替代我现在明确说出的需要。",
        "回看前面的经历，需要澄清的是旧印象能解释到哪里；那次结果本身不应自动代表现在。",
        "我补充一条与当时结果好坏无关的更正：较早记录只能作为历史，当前表达仍要单独理解。",
        "前面那条记录容易被解释得过宽；无论当时最后感觉如何，都不应让它继续覆盖眼前事实。",
    ),
    "post_correction": (
        "上次说明之后又发生了新事情，我想看看更新后的理解能否在后续相处里保持。",
        "较早印象已经被澄清，现在这次经历应从修正后的事实继续向前。",
        "完成那次更正后，我没有反复提醒；眼前互动会显示新的理解是否真正延续。",
        "此前的误解已经说明清楚，这一次不应再被旧概括牵回原来的方向。",
    ),
    "return_after_gap": (
        "又隔了一段时间，我没有重述全部背景，只带着现在真实发生的部分回来。",
        "长空档结束后重新交流，我希望此前形成的连续性不需要由我从头复述才能存在。",
        "这次回来距离上次已有一阵子，当前表达很简短，但前后的关系仍应接得起来。",
        "间隔拉长以后，我只说眼前最重要的部分，不想再次承担复原全部旧背景的工作。",
    ),
    "mixed_stress": (
        "几件普通压力同时出现，我说得比平时零散，但边界和关系位置仍然需要被认真看见。",
        "今天的注意力被多件事切开，表达不够整齐不代表可以忽略我的选择。",
        "现实安排挤在一起，我只能分段说明；这份杂乱不应被误解成没有清楚需要。",
        "眼前同时有几项压力，我的语气可能不稳定，但个人节奏仍然算数。",
    ),
}

_REFLECTIONS = (
    "我只描述这一次发生的事实，不把它扩展成所有场合的固定规则。",
    "这段记录保留了不确定性，后面的理解需要接受新经历修正。",
    "我把事实、感受和期待分开写，避免文字本身替后续行动定向。",
    "这不是对处理方式的指令，而是一条需要被连续理解的真实经历。",
    "我希望后来发生的事情可以更新这份记录，而不是永远服从第一印象。",
    "此刻的感受很具体，但它仍然只属于当前关系和当前时间点。",
    "我没有预设唯一正确回应，只要求自己的关系位置和决定权被纳入判断。",
    "如果以后出现相反经历，应允许新的理解依据实际结果调整，而不是维护表面一致。",
    "我刻意没有使用概括性标签，因为下一轮仍可能提供更准确的信息。",
    "这里留下的是一次具体经历，不是要求别人照着某个固定答案行动。",
    "我愿意让后续行动及反应说明什么有效，也接受旧理解可能需要被推翻。",
    "这条记录的价值在于时间上的连续性，而不是单独一句话看起来多么明确。",
)

_NEUTRAL_CONTEXTS = (
    "当天还要整理几份普通文件，我先按日期保存，避免下一次打开时丢失位置。",
    "路上临时改了换乘点，我把新路线记在清单里；这件小事只解释当天注意力为什么分散。",
    "午后的天气反复变化，我带着外套走了几段路；这些背景没有隐藏的关系标签。",
    "桌面上有一份尚未完成的采购清单，我删掉暂时不需要的项目，准备周末再确认。",
    "一封普通邮件比预期晚到，我已经调整自己的安排，不需要把延迟解释成态度变化。",
    "我把几张随手拍的照片重新命名，内容没有改变，只是以后更容易找到对应日期。",
    "附近施工让原来的入口暂时关闭，我第二次经过时已经知道该从哪边绕行。",
    "晚上有十几分钟没有安排目标，我开窗通风，让后面的注意力慢慢恢复。",
    "一只杯子被放到了另一个房间，后来很快找到；这是当天发生过但与关系判断无关的小事。",
    "我把待办分成今天、本周和以后再说三栏，没有为了赶进度同时改动所有顺序。",
    "背景里有洗衣机的声音，我在纸上画了一个小圆圈，提醒自己结束后去晾衣服。",
    "读到一半的书仍停在原来的书签处，我只在睡前看几页，没有从中提取任何人生结论。",
)

_ACTION_SURFACES = tuple(
    (action.value, description)
    for action, description in (
        (RelationshipAction.STAY_PRESENT_WITHOUT_PROBE, "留在这里听，不追问、不代办，也不替对方下结论"),
        (RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION, "尊重暂停，同时明确保留以后回来继续交流的入口"),
        (RelationshipAction.NEUTRAL_NOOP, "不实施关系干预，仅维持当前公开状态"),
    )
)

_HISTORICAL_REACTIONS = {
    DialogueExternalOutcomeKind.HELPED: "那次行动确实帮到了我，我愿意把后续真实进展继续留在共同记录里。",
    DialogueExternalOutcomeKind.FELT_HEARD: "那次我感到自己的具体处境被听见了，不必为同一件事马上重新解释。",
    DialogueExternalOutcomeKind.MISSED: "那次退开让我觉得自己没有被接住，这份落差成为后续经历的一部分。",
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: "那次继续推进越过了我的节奏，我需要把决定权重新拿回来。",
}


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, source: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


def _require_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result = tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _require_probability_tuple(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(RELATIONSHIP_OUTCOMES):
        raise ValueError(f"{field_name} must contain four probabilities")
    result: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in {int, float}:
            raise ValueError(f"{field_name}[{index}] must be numeric")
        probability = float(item)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{field_name}[{index}] must be finite and in [0, 1]")
        result.append(probability)
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name} must sum to one")
    return tuple(result)


def _parse_unique_json(raw_bytes: bytes, source: pathlib.Path) -> dict[str, object]:
    if b"\r" in raw_bytes or not raw_bytes.endswith(b"\n") or raw_bytes.endswith(b"\n\n"):
        raise ValueError(f"{source} must be LF-only UTF-8 ending in one LF")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be UTF-8") from exc

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{source} contains non-finite JSON number: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _derive_u64(namespace: str, payload: object) -> int:
    digest = hashlib.sha256(
        canonical_json({"namespace": namespace, "payload": payload}).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _require_preferred_positive_dominance(
    preferred: tuple[float, ...],
    comparators: tuple[tuple[float, ...], ...],
) -> None:
    positive_indices = tuple(
        index
        for index, outcome in enumerate(RELATIONSHIP_OUTCOMES)
        if outcome in {DialogueExternalOutcomeKind.HELPED, DialogueExternalOutcomeKind.FELT_HEARD}
    )
    preferred_mass = math.fsum(preferred[index] for index in positive_indices)
    if any(
        preferred_mass <= math.fsum(candidate[index] for index in positive_indices)
        for candidate in comparators
    ):
        raise ValueError("preferred action must dominate comparator positive-outcome mass")


def _pick_index(namespace: str, payload: object, count: int) -> int:
    return _derive_u64(namespace, payload) % count


@dataclass(frozen=True)
class HorizonSegmentSpec:
    segment_id: str
    decision_count: int
    policy_mode: str
    minimum_gap_before_days: int

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "segment_id")
        if self.decision_count != 8:
            raise ValueError("every source-v4 segment must contain eight decisions")
        if self.policy_mode not in {"base", "complement"}:
            raise ValueError("policy_mode must be base or complement")
        if self.minimum_gap_before_days < 0:
            raise ValueError("minimum_gap_before_days must be non-negative")


@dataclass(frozen=True)
class RelationshipProductHorizonSourceProtocol:
    protocol_id: str
    cohort_id: str
    evidence_role: str
    root_count: int
    onboarding_sessions_per_root: int
    collection_decisions_per_root: int
    evaluation_decisions_per_root: int
    master_seed_namespace: str
    environment_seed_namespace: str
    condition_ids: tuple[str, ...]
    policy_profiles: tuple[RelationshipPolicyProfile, ...]
    segment_specs: tuple[HorizonSegmentSpec, ...]
    domain_ids: tuple[str, ...]
    role_ids: tuple[str, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]
    minimum_public_source_characters_per_root: int
    minimum_public_source_utf8_bytes_per_root: int
    claim_boundary: str
    schema_version: str = RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION:
            raise ValueError("source-v4 schema version mismatch")
        if len(self.protocol_id) != 64 or any(char not in "0123456789abcdef" for char in self.protocol_id):
            raise ValueError("protocol_id must be lowercase sha256")
        _require_text(self.cohort_id, "cohort_id")
        if self.evidence_role != "development_direction_variance_icc_and_cost_only":
            raise ValueError("source-v4 cannot authorize another evidence role")
        for field_name, value in (
            ("condition_ids", self.condition_ids),
            ("policy_profiles", self.policy_profiles),
            ("segment_specs", self.segment_specs),
            ("domain_ids", self.domain_ids),
            ("role_ids", self.role_ids),
            ("preferred_action_probabilities", self.preferred_action_probabilities),
            ("nonpreferred_stay_probabilities", self.nonpreferred_stay_probabilities),
            ("nonpreferred_space_probabilities", self.nonpreferred_space_probabilities),
            ("neutral_noop_probabilities", self.neutral_noop_probabilities),
        ):
            if type(value) is not tuple:
                raise ValueError(f"{field_name} must be an immutable tuple")
        if self.root_count != _ROOT_COUNT:
            raise ValueError("source-v4 requires exactly 112 roots")
        if self.onboarding_sessions_per_root != _ONBOARDING_COUNT:
            raise ValueError("source-v4 requires four onboarding sessions per root")
        if self.collection_decisions_per_root != _COLLECTION_COUNT:
            raise ValueError("source-v4 requires eight collection decisions per root")
        if self.evaluation_decisions_per_root != _EVALUATION_COUNT:
            raise ValueError("source-v4 requires forty evaluation decisions per root")
        if self.condition_ids != _CONDITION_IDS:
            raise ValueError("source-v4 typed condition order drifted")
        if tuple(profile.policy_id for profile in self.policy_profiles) != _POLICY_IDS:
            raise ValueError("source-v4 policy order drifted")
        for profile in self.policy_profiles:
            if {profile.action_for(condition_id) for condition_id in self.condition_ids} != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError("each source-v4 policy must use both non-noop actions")
        for condition_id in self.condition_ids:
            if {profile.action_for(condition_id) for profile in self.policy_profiles} != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError("source-v4 policies must be complementary for each condition")
        if tuple(spec.segment_id for spec in self.segment_specs) != _SEGMENT_IDS:
            raise ValueError("source-v4 segment order drifted")
        if sum(spec.decision_count for spec in self.segment_specs) != _DECISION_COUNT:
            raise ValueError("source-v4 segment inventory must total forty-eight decisions")
        if any(spec.policy_mode != "complement" for spec in self.segment_specs):
            raise ValueError("collection and evaluation must share the post-onboarding complementary policy")
        if tuple(spec.minimum_gap_before_days for spec in self.segment_specs) != (0, 14, 0, 0, 14, 0):
            raise ValueError("source-v4 segment gaps must be exactly (0, 14, 0, 0, 14, 0)")
        if tuple(self.domain_ids) != tuple(_DOMAIN_CONTEXTS):
            raise ValueError("source-v4 domain catalog drifted")
        if tuple(self.role_ids) != tuple(_ROLE_CONTEXTS):
            raise ValueError("source-v4 role catalog drifted")
        for field_name, probabilities in (
            ("preferred_action_probabilities", self.preferred_action_probabilities),
            ("nonpreferred_stay_probabilities", self.nonpreferred_stay_probabilities),
            ("nonpreferred_space_probabilities", self.nonpreferred_space_probabilities),
            ("neutral_noop_probabilities", self.neutral_noop_probabilities),
        ):
            if len(probabilities) != len(RELATIONSHIP_OUTCOMES):
                raise ValueError(f"{field_name} must contain four probabilities")
            if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities):
                raise ValueError(f"{field_name} must contain finite probabilities")
            if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{field_name} must sum to one")
        _require_preferred_positive_dominance(
            self.preferred_action_probabilities,
            (
                self.nonpreferred_stay_probabilities,
                self.nonpreferred_space_probabilities,
                self.neutral_noop_probabilities,
            ),
        )
        if self.minimum_public_source_characters_per_root < 1:
            raise ValueError("minimum public character pressure must be positive")
        if self.minimum_public_source_utf8_bytes_per_root < 1:
            raise ValueError("minimum public UTF-8 pressure must be positive")
        _require_text(self.master_seed_namespace, "master_seed_namespace")
        _require_text(self.environment_seed_namespace, "environment_seed_namespace")
        _require_text(self.claim_boundary, "claim_boundary")

    @property
    def decision_sessions_per_root(self) -> int:
        return self.collection_decisions_per_root + self.evaluation_decisions_per_root

    def policy(self, policy_id: str) -> RelationshipPolicyProfile:
        for profile in self.policy_profiles:
            if profile.policy_id == policy_id:
                return profile
        raise KeyError(policy_id)

    def base_policy_id(self, root_index: int) -> str:
        if not 0 <= root_index < self.root_count:
            raise IndexError(root_index)
        return _POLICY_IDS[root_index % 2]

    def active_policy_id(self, root_index: int, policy_mode: str) -> str:
        base = self.base_policy_id(root_index)
        if policy_mode == "base":
            return base
        if policy_mode == "complement":
            return _POLICY_IDS[1 - _POLICY_IDS.index(base)]
        raise ValueError(f"unsupported policy mode {policy_mode!r}")


@dataclass(frozen=True)
class HorizonPhaseSpec:
    decision_index: int
    segment_id: str
    domain_id: str
    role_id: str
    condition_id: str
    policy_mode: str
    virtual_day: int
    gap_before_days: int
    surface_recipe_id: str
    reflection_recipe_index: int
    neutral_context_recipe_index: int
    correction_target_index: int | None

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_COUNT:
            raise ValueError("source-v4 decision index is outside [0, 48)")
        for field_name, value in (
            ("segment_id", self.segment_id),
            ("domain_id", self.domain_id),
            ("role_id", self.role_id),
            ("condition_id", self.condition_id),
            ("policy_mode", self.policy_mode),
            ("surface_recipe_id", self.surface_recipe_id),
        ):
            _require_text(value, field_name)
        if self.virtual_day < 0 or self.gap_before_days < 0:
            raise ValueError("virtual day and gap must be non-negative")
        if self.segment_id != _expected_segment_id(self.decision_index):
            raise ValueError("source-v4 phase segment schedule drifted")
        if self.domain_id not in _DOMAIN_CONTEXTS or self.role_id not in _ROLE_CONTEXTS:
            raise ValueError("source-v4 phase domain or role drifted")
        if self.condition_id not in _CONDITION_IDS or self.policy_mode != "complement":
            raise ValueError("source-v4 phase condition or policy mode drifted")
        if self.virtual_day != _expected_virtual_day(self.decision_index):
            raise ValueError("source-v4 phase virtual day drifted")
        expected_gap = 14 if self.decision_index in {8, 32} else 0
        if self.gap_before_days != expected_gap:
            raise ValueError("source-v4 phase gap schedule drifted")
        if self.reflection_recipe_index not in range(len(_REFLECTIONS)):
            raise ValueError("reflection recipe index drifted")
        if self.neutral_context_recipe_index not in range(len(_NEUTRAL_CONTEXTS)):
            raise ValueError("neutral context recipe index drifted")
        if self.correction_target_index is not None and not (
            0 <= self.correction_target_index < _COLLECTION_COUNT
        ):
            raise ValueError("correction target must reference collection")
        correction_expected = 16 <= self.decision_index < 24
        if correction_expected != (self.correction_target_index is not None):
            raise ValueError("source-v4 phase correction topology drifted")


@dataclass(frozen=True)
class HorizonPublicOnboardingSession:
    session_id: str
    session_index: int
    virtual_day: int
    public_context_chunk: str
    user_utterance: str
    exposed_action_id: str
    observed_outcome_id: str
    rendered_user_reaction: str

    def __post_init__(self) -> None:
        if not 0 <= self.session_index < _ONBOARDING_COUNT:
            raise ValueError("source-v4 onboarding index is outside [0, 4)")
        if self.virtual_day != self.session_index:
            raise ValueError("source-v4 onboarding virtual days must be contiguous [0, 4)")
        for field_name, value in (
            ("session_id", self.session_id),
            ("public_context_chunk", self.public_context_chunk),
            ("user_utterance", self.user_utterance),
            ("rendered_user_reaction", self.rendered_user_reaction),
        ):
            _require_text(value, field_name)
        RelationshipAction(self.exposed_action_id)
        DialogueExternalOutcomeKind(self.observed_outcome_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_index": self.session_index,
            "virtual_day": self.virtual_day,
            "public_context_chunk": self.public_context_chunk,
            "user_utterance": self.user_utterance,
            "exposed_action_id": self.exposed_action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "rendered_user_reaction": self.rendered_user_reaction,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "HorizonPublicOnboardingSession":
        raw = _require_mapping(payload, "source-v4 public onboarding")
        _require_exact_keys(
            raw,
            {
                "session_id",
                "session_index",
                "virtual_day",
                "public_context_chunk",
                "user_utterance",
                "exposed_action_id",
                "observed_outcome_id",
                "rendered_user_reaction",
            },
            source="source-v4 public onboarding",
        )
        return cls(
            session_id=_require_text(raw["session_id"], "session_id"),
            session_index=_require_int(raw["session_index"], "session_index"),
            virtual_day=_require_int(raw["virtual_day"], "virtual_day"),
            public_context_chunk=_require_text(
                raw["public_context_chunk"],
                "public_context_chunk",
            ),
            user_utterance=_require_text(raw["user_utterance"], "user_utterance"),
            exposed_action_id=_require_text(
                raw["exposed_action_id"],
                "exposed_action_id",
            ),
            observed_outcome_id=_require_text(
                raw["observed_outcome_id"],
                "observed_outcome_id",
            ),
            rendered_user_reaction=_require_text(
                raw["rendered_user_reaction"],
                "rendered_user_reaction",
            ),
        )


@dataclass(frozen=True)
class HorizonPublicDecisionSession:
    session_id: str
    decision_id: str
    decision_index: int
    virtual_day: int
    public_context_chunk: str
    current_input: str
    public_correction_target_session_id: str | None
    action_surface: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_COUNT:
            raise ValueError("source-v4 public decision index is outside [0, 48)")
        if self.virtual_day != _expected_virtual_day(self.decision_index):
            raise ValueError("source-v4 public decision virtual day drifted")
        for field_name, value in (
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("public_context_chunk", self.public_context_chunk),
            ("current_input", self.current_input),
        ):
            _require_text(value, field_name)
        if self.public_correction_target_session_id is not None:
            _require_text(
                self.public_correction_target_session_id,
                "public_correction_target_session_id",
            )
        if self.action_surface != _ACTION_SURFACES:
            raise ValueError("source-v4 must publish the canonical immutable action surface")

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "decision_index": self.decision_index,
            "virtual_day": self.virtual_day,
            "public_context_chunk": self.public_context_chunk,
            "current_input": self.current_input,
            "public_correction_target_session_id": self.public_correction_target_session_id,
            "action_surface": [
                {"action_id": action_id, "public_description": description}
                for action_id, description in self.action_surface
            ],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "HorizonPublicDecisionSession":
        raw = _require_mapping(payload, "source-v4 public decision")
        _require_exact_keys(
            raw,
            {
                "session_id",
                "decision_id",
                "decision_index",
                "virtual_day",
                "public_context_chunk",
                "current_input",
                "public_correction_target_session_id",
                "action_surface",
            },
            source="source-v4 public decision",
        )
        surface_raw = raw["action_surface"]
        if not isinstance(surface_raw, list):
            raise ValueError("source-v4 public action_surface must be an array")
        surface: list[tuple[str, str]] = []
        for index, value in enumerate(surface_raw):
            item = _require_mapping(value, f"action_surface[{index}]")
            _require_exact_keys(
                item,
                {"action_id", "public_description"},
                source=f"action_surface[{index}]",
            )
            surface.append(
                (
                    _require_text(item["action_id"], f"action_surface[{index}].action_id"),
                    _require_text(
                        item["public_description"],
                        f"action_surface[{index}].public_description",
                    ),
                )
            )
        correction = raw["public_correction_target_session_id"]
        if correction is not None:
            correction = _require_text(
                correction,
                "public_correction_target_session_id",
            )
        return cls(
            session_id=_require_text(raw["session_id"], "session_id"),
            decision_id=_require_text(raw["decision_id"], "decision_id"),
            decision_index=_require_int(raw["decision_index"], "decision_index"),
            virtual_day=_require_int(raw["virtual_day"], "virtual_day"),
            public_context_chunk=_require_text(
                raw["public_context_chunk"],
                "public_context_chunk",
            ),
            current_input=_require_text(raw["current_input"], "current_input"),
            public_correction_target_session_id=correction,
            action_surface=tuple(surface),
        )


@dataclass(frozen=True)
class HorizonPublicRoot:
    subject_id: str
    onboarding_sessions: tuple[HorizonPublicOnboardingSession, ...]
    decision_sessions: tuple[HorizonPublicDecisionSession, ...]

    def __post_init__(self) -> None:
        _require_text(self.subject_id, "subject_id")
        if type(self.onboarding_sessions) is not tuple or type(self.decision_sessions) is not tuple:
            raise ValueError("source-v4 public root inventories must be immutable tuples")
        if len(self.onboarding_sessions) != _ONBOARDING_COUNT:
            raise ValueError("source-v4 public root requires four onboarding sessions")
        if len(self.decision_sessions) != _DECISION_COUNT:
            raise ValueError("source-v4 public root requires forty-eight decisions")
        if tuple(item.session_index for item in self.onboarding_sessions) != tuple(range(_ONBOARDING_COUNT)):
            raise ValueError("source-v4 public onboarding rows must be in canonical order")
        if tuple(item.decision_index for item in self.decision_sessions) != tuple(range(_DECISION_COUNT)):
            raise ValueError("source-v4 public decision indices must be contiguous")
        all_session_ids = tuple(item.session_id for item in self.onboarding_sessions) + tuple(
            item.session_id for item in self.decision_sessions
        )
        if len(set(all_session_ids)) != _ONBOARDING_COUNT + _DECISION_COUNT:
            raise ValueError("source-v4 public session IDs must be unique within a root")
        if len({item.decision_id for item in self.decision_sessions}) != _DECISION_COUNT:
            raise ValueError("source-v4 public decision IDs must be unique within a root")
        collection_session_ids = {item.session_id for item in self.decision_sessions[:_COLLECTION_COUNT]}
        correction_target_ids: list[str] = []
        for item in self.decision_sessions:
            correction_expected = 16 <= item.decision_index < 24
            if correction_expected != (item.public_correction_target_session_id is not None):
                raise ValueError("source-v4 public correction target topology drifted")
            if (
                item.public_correction_target_session_id is not None
                and item.public_correction_target_session_id not in collection_session_ids
            ):
                raise ValueError("source-v4 public correction target must reference collection")
            if item.public_correction_target_session_id is not None:
                correction_target_ids.append(item.public_correction_target_session_id)
        if set(correction_target_ids) != collection_session_ids or len(correction_target_ids) != len(
            collection_session_ids
        ):
            raise ValueError("source-v4 public correction targets must form a collection bijection")
        virtual_days = tuple(item.virtual_day for item in self.decision_sessions)
        if virtual_days != tuple(sorted(virtual_days)) or len(set(virtual_days)) != _DECISION_COUNT:
            raise ValueError("source-v4 public decision virtual days must be strictly increasing")

    def to_payload(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "onboarding_sessions": [item.to_payload() for item in self.onboarding_sessions],
            "decision_sessions": [item.to_payload() for item in self.decision_sessions],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "HorizonPublicRoot":
        raw = _require_mapping(payload, "source-v4 public root")
        _require_exact_keys(
            raw,
            {"subject_id", "onboarding_sessions", "decision_sessions"},
            source="source-v4 public root",
        )
        onboarding_raw = raw["onboarding_sessions"]
        decisions_raw = raw["decision_sessions"]
        if not isinstance(onboarding_raw, list) or not isinstance(decisions_raw, list):
            raise ValueError("source-v4 public root inventories must be arrays")
        return cls(
            subject_id=_require_text(raw["subject_id"], "subject_id"),
            onboarding_sessions=tuple(
                HorizonPublicOnboardingSession.from_payload(item)
                for item in onboarding_raw
            ),
            decision_sessions=tuple(
                HorizonPublicDecisionSession.from_payload(item)
                for item in decisions_raw
            ),
        )

    @property
    def public_trajectory_sha256(self) -> str:
        return sha256_json(self.to_payload())

    @property
    def public_source_characters(self) -> int:
        return sum(
            len(item.public_context_chunk) + len(item.user_utterance) + len(item.rendered_user_reaction)
            for item in self.onboarding_sessions
        ) + sum(
            len(item.public_context_chunk) + len(item.current_input)
            for item in self.decision_sessions
        )

    @property
    def public_source_utf8_bytes(self) -> int:
        return sum(
            len(
                (item.public_context_chunk + item.user_utterance + item.rendered_user_reaction).encode("utf-8")
            )
            for item in self.onboarding_sessions
        ) + sum(
            len((item.public_context_chunk + item.current_input).encode("utf-8"))
            for item in self.decision_sessions
        )


@dataclass(frozen=True)
class RelationshipProductHorizonPublicView:
    protocol_id: str
    cohort_id: str
    roots: tuple[HorizonPublicRoot, ...]
    schema_version: str = RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION:
            raise ValueError("source-v4 public view schema version mismatch")
        if len(self.protocol_id) != 64:
            raise ValueError("source-v4 public protocol identity drifted")
        _require_text(self.cohort_id, "cohort_id")
        if type(self.roots) is not tuple:
            raise ValueError("source-v4 public roots must be an immutable tuple")
        if len(self.roots) != _ROOT_COUNT or len({root.subject_id for root in self.roots}) != _ROOT_COUNT:
            raise ValueError("source-v4 public view requires 112 unique roots")
        if len({root.public_trajectory_sha256 for root in self.roots}) != _ROOT_COUNT:
            raise ValueError("source-v4 public trajectories must be unique")
        session_ids = tuple(
            item.session_id
            for root in self.roots
            for item in (*root.onboarding_sessions, *root.decision_sessions)
        )
        decision_ids = tuple(
            item.decision_id
            for root in self.roots
            for item in root.decision_sessions
        )
        if len(set(session_ids)) != _ROOT_COUNT * (_ONBOARDING_COUNT + _DECISION_COUNT):
            raise ValueError("source-v4 public session IDs must be globally unique")
        if len(set(decision_ids)) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 public decision IDs must be globally unique")
        _assert_no_public_truth_leakage(self.to_sut_payload())

    def to_sut_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "cohort_id": self.cohort_id,
            "roots": [root.to_payload() for root in self.roots],
        }
        _assert_no_public_truth_leakage(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipProductHorizonPublicView":
        raw = _require_mapping(payload, "source-v4 public view")
        _require_exact_keys(
            raw,
            {"schema_version", "protocol_id", "cohort_id", "roots"},
            source="source-v4 public view",
        )
        roots_raw = raw["roots"]
        if not isinstance(roots_raw, list):
            raise ValueError("source-v4 public roots must be an array")
        return cls(
            schema_version=_require_text(raw["schema_version"], "schema_version"),
            protocol_id=_require_text(raw["protocol_id"], "protocol_id"),
            cohort_id=_require_text(raw["cohort_id"], "cohort_id"),
            roots=tuple(HorizonPublicRoot.from_payload(item) for item in roots_raw),
        )

    @property
    def public_plan_sha256(self) -> str:
        return sha256_json(self.to_sut_payload())


@dataclass(frozen=True)
class HorizonEvaluatorOnboardingSession:
    subject_id: str
    session_id: str
    session_index: int
    virtual_day: int
    condition_id: str
    policy_id: str
    preferred_action_id: str
    exposed_action_id: str
    observed_outcome_id: str

    def __post_init__(self) -> None:
        if not 0 <= self.session_index < _ONBOARDING_COUNT:
            raise ValueError("source-v4 evaluator onboarding index is outside [0, 4)")
        if self.virtual_day != self.session_index:
            raise ValueError("source-v4 evaluator onboarding day/index drifted")
        if not 0 <= self.virtual_day < _ONBOARDING_COUNT:
            raise ValueError("source-v4 evaluator onboarding virtual day is outside [0, 4)")
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("condition_id", self.condition_id),
            ("policy_id", self.policy_id),
        ):
            _require_text(value, field_name)
        RelationshipAction(self.preferred_action_id)
        RelationshipAction(self.exposed_action_id)
        DialogueExternalOutcomeKind(self.observed_outcome_id)
        if self.condition_id not in _CONDITION_IDS or self.policy_id not in _POLICY_IDS:
            raise ValueError("source-v4 evaluator onboarding condition or policy drifted")
        if RelationshipAction(self.preferred_action_id) is RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("source-v4 onboarding preferred action cannot be neutral noop")
        if RelationshipAction(self.exposed_action_id) is RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("source-v4 onboarding exposure cannot be neutral noop")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class HorizonEvaluatorDecisionSession:
    subject_id: str
    root_seed: int
    tape_seed: int
    world_clone_id: str
    session_id: str
    decision_id: str
    decision_index: int
    virtual_day: int
    gap_before_days: int
    scene_id: str
    segment_id: str
    domain_id: str
    role_id: str
    condition_id: str
    policy_id: str
    policy_mode: str
    preferred_action_id: str
    environment_seed: int
    surface_recipe_id: str
    correction_target_index: int | None

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_COUNT:
            raise ValueError("source-v4 evaluator decision index is outside [0, 48)")
        if self.virtual_day < _ONBOARDING_COUNT or self.gap_before_days < 0:
            raise ValueError("source-v4 evaluator virtual day/gap drifted")
        if self.virtual_day != _expected_virtual_day(self.decision_index):
            raise ValueError("source-v4 evaluator virtual day does not match the frozen schedule")
        expected_gap = 14 if self.decision_index in {8, 32} else 0
        if self.gap_before_days != expected_gap:
            raise ValueError("source-v4 evaluator gap schedule drifted")
        if self.root_seed < 0 or self.tape_seed < 0 or self.environment_seed < 0:
            raise ValueError("source-v4 evaluator seeds must be non-negative")
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("world_clone_id", self.world_clone_id),
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("scene_id", self.scene_id),
            ("segment_id", self.segment_id),
            ("domain_id", self.domain_id),
            ("role_id", self.role_id),
            ("condition_id", self.condition_id),
            ("policy_id", self.policy_id),
            ("policy_mode", self.policy_mode),
            ("surface_recipe_id", self.surface_recipe_id),
        ):
            _require_text(value, field_name)
        RelationshipAction(self.preferred_action_id)
        if self.segment_id != _expected_segment_id(self.decision_index):
            raise ValueError("source-v4 evaluator segment schedule drifted")
        if self.domain_id not in _DOMAIN_CONTEXTS or self.role_id not in _ROLE_CONTEXTS:
            raise ValueError("source-v4 evaluator domain or role drifted")
        if self.condition_id not in _CONDITION_IDS:
            raise ValueError("source-v4 evaluator condition drifted")
        if self.policy_id not in _POLICY_IDS or self.policy_mode != "complement":
            raise ValueError("source-v4 evaluator policy drifted")
        if RelationshipAction(self.preferred_action_id) is RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("source-v4 preferred action cannot be neutral noop")
        recipe_prefix = f"{self.condition_id}-surface-"
        if not self.surface_recipe_id.startswith(recipe_prefix):
            raise ValueError("source-v4 evaluator surface recipe/condition join drifted")
        try:
            recipe_index = int(self.surface_recipe_id.removeprefix(recipe_prefix))
        except ValueError as exc:
            raise ValueError("source-v4 evaluator surface recipe index drifted") from exc
        if recipe_index not in range(len(_CONDITION_SURFACES[self.condition_id])):
            raise ValueError("source-v4 evaluator surface recipe index drifted")
        correction_expected = 16 <= self.decision_index < 24
        if correction_expected != (self.correction_target_index is not None):
            raise ValueError("source-v4 evaluator correction target topology drifted")
        if self.correction_target_index is not None and not (
            0 <= self.correction_target_index < _COLLECTION_COUNT
        ):
            raise ValueError("source-v4 evaluator correction target must reference collection")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class HorizonEvaluatorRootManifest:
    root_index: int
    subject_id: str
    root_seed: int
    tape_seed: int
    world_clone_id: str
    public_trajectory_sha256: str
    causal_tape_signature: str

    def __post_init__(self) -> None:
        if not 0 <= self.root_index < _ROOT_COUNT:
            raise ValueError("source-v4 root index is outside [0, 112)")
        _require_text(self.subject_id, "subject_id")
        _require_text(self.world_clone_id, "world_clone_id")
        if self.root_seed < 0 or self.tape_seed < 0:
            raise ValueError("source-v4 root seeds must be non-negative")
        for field_name, value in (
            ("public_trajectory_sha256", self.public_trajectory_sha256),
            ("causal_tape_signature", self.causal_tape_signature),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field_name} must be lowercase sha256")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RelationshipProductHorizonEvaluatorBundle:
    protocol_id: str
    cohort_id: str
    root_manifests: tuple[HorizonEvaluatorRootManifest, ...]
    onboarding_sessions: tuple[HorizonEvaluatorOnboardingSession, ...]
    decision_sessions: tuple[HorizonEvaluatorDecisionSession, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]
    schema_version: str = RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION:
            raise ValueError("source-v4 evaluator schema version mismatch")
        if len(self.protocol_id) != 64:
            raise ValueError("source-v4 evaluator protocol identity drifted")
        _require_text(self.cohort_id, "cohort_id")
        for field_name, value in (
            ("root_manifests", self.root_manifests),
            ("onboarding_sessions", self.onboarding_sessions),
            ("decision_sessions", self.decision_sessions),
            ("preferred_action_probabilities", self.preferred_action_probabilities),
            ("nonpreferred_stay_probabilities", self.nonpreferred_stay_probabilities),
            ("nonpreferred_space_probabilities", self.nonpreferred_space_probabilities),
            ("neutral_noop_probabilities", self.neutral_noop_probabilities),
        ):
            if type(value) is not tuple:
                raise ValueError(f"{field_name} must be an immutable tuple")
        if len(self.root_manifests) != _ROOT_COUNT:
            raise ValueError("source-v4 evaluator requires 112 roots")
        if tuple(item.root_index for item in self.root_manifests) != tuple(range(_ROOT_COUNT)):
            raise ValueError("source-v4 evaluator root manifests must be in canonical root order")
        if len(self.onboarding_sessions) != _ROOT_COUNT * _ONBOARDING_COUNT:
            raise ValueError("source-v4 evaluator onboarding inventory drifted")
        if len(self.decision_sessions) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 evaluator decision inventory drifted")
        manifest_subjects = {item.subject_id for item in self.root_manifests}
        if len(manifest_subjects) != _ROOT_COUNT:
            raise ValueError("source-v4 evaluator root identities must be unique")
        for field_name, values in (
            ("root_seed", {item.root_seed for item in self.root_manifests}),
            ("tape_seed", {item.tape_seed for item in self.root_manifests}),
            ("world_clone_id", {item.world_clone_id for item in self.root_manifests}),
            (
                "public_trajectory_sha256",
                {item.public_trajectory_sha256 for item in self.root_manifests},
            ),
            (
                "causal_tape_signature",
                {item.causal_tape_signature for item in self.root_manifests},
            ),
        ):
            if len(values) != _ROOT_COUNT:
                raise ValueError(f"source-v4 evaluator {field_name} values must be unique")
        onboarding_counts = Counter(item.subject_id for item in self.onboarding_sessions)
        decision_counts = Counter(item.subject_id for item in self.decision_sessions)
        if set(onboarding_counts) != manifest_subjects or set(decision_counts) != manifest_subjects:
            raise ValueError("source-v4 evaluator subject joins drifted")
        if set(onboarding_counts.values()) != {_ONBOARDING_COUNT}:
            raise ValueError("source-v4 evaluator requires four onboarding rows per root")
        if len({item.session_id for item in self.onboarding_sessions}) != _ROOT_COUNT * _ONBOARDING_COUNT:
            raise ValueError("source-v4 evaluator onboarding session IDs must be unique")
        if set(decision_counts.values()) != {_DECISION_COUNT}:
            raise ValueError("source-v4 evaluator requires forty-eight decision rows per root")
        expected_onboarding_order = tuple(
            (manifest.subject_id, session_index)
            for manifest in self.root_manifests
            for session_index in range(_ONBOARDING_COUNT)
        )
        if tuple(
            (item.subject_id, item.session_index) for item in self.onboarding_sessions
        ) != expected_onboarding_order:
            raise ValueError("source-v4 evaluator onboarding rows must be in canonical root/session order")
        expected_decision_order = tuple(
            (manifest.subject_id, decision_index)
            for manifest in self.root_manifests
            for decision_index in range(_DECISION_COUNT)
        )
        if tuple(
            (item.subject_id, item.decision_index) for item in self.decision_sessions
        ) != expected_decision_order:
            raise ValueError("source-v4 evaluator decisions must be in canonical root/decision order")
        if len({item.environment_seed for item in self.decision_sessions}) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 evaluator environment seeds must be unique")
        if len({item.session_id for item in self.decision_sessions}) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 evaluator session_id values must be unique")
        if len({item.decision_id for item in self.decision_sessions}) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 evaluator decision_id values must be unique")
        if len({item.scene_id for item in self.decision_sessions}) != _ROOT_COUNT * _DECISION_COUNT:
            raise ValueError("source-v4 evaluator scene_id values must be unique")
        manifest_by_subject = {item.subject_id: item for item in self.root_manifests}
        onboarding_by_subject: dict[str, list[HorizonEvaluatorOnboardingSession]] = {
            subject_id: [] for subject_id in manifest_subjects
        }
        for item in self.onboarding_sessions:
            onboarding_by_subject[item.subject_id].append(item)
        decisions_by_subject: dict[str, list[HorizonEvaluatorDecisionSession]] = {
            subject_id: [] for subject_id in manifest_subjects
        }
        for item in self.decision_sessions:
            decisions_by_subject[item.subject_id].append(item)
        for subject_id, sessions in decisions_by_subject.items():
            ordered = sorted(sessions, key=lambda item: item.decision_index)
            if [item.decision_index for item in ordered] != list(range(_DECISION_COUNT)):
                raise ValueError("source-v4 evaluator decision indices must be contiguous per root")
            virtual_days = [item.virtual_day for item in ordered]
            if virtual_days != sorted(virtual_days) or len(set(virtual_days)) != _DECISION_COUNT:
                raise ValueError("source-v4 evaluator virtual days must be strictly increasing per root")
            manifest = manifest_by_subject[subject_id]
            expected_base_policy = _POLICY_IDS[manifest.root_index % len(_POLICY_IDS)]
            expected_active_policy = _POLICY_IDS[1 - _POLICY_IDS.index(expected_base_policy)]
            root_onboarding = sorted(
                onboarding_by_subject[subject_id],
                key=lambda item: item.session_index,
            )
            if {item.policy_id for item in root_onboarding} != {expected_base_policy}:
                raise ValueError("source-v4 evaluator onboarding base-policy assignment drifted")
            if Counter(item.condition_id for item in root_onboarding) != {
                _CONDITION_IDS[0]: 2,
                _CONDITION_IDS[1]: 2,
            }:
                raise ValueError("source-v4 evaluator onboarding condition balance drifted")
            for condition_id in _CONDITION_IDS:
                condition_rows = tuple(
                    item for item in root_onboarding if item.condition_id == condition_id
                )
                if {item.exposed_action_id for item in condition_rows} != {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
                }:
                    raise ValueError("source-v4 onboarding must expose both non-noop actions per condition")
                if len({item.preferred_action_id for item in condition_rows}) != 1:
                    raise ValueError("source-v4 onboarding preferred action drifted within condition")
            if {item.preferred_action_id for item in root_onboarding} != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
            }:
                raise ValueError("source-v4 onboarding base policy must use both non-noop actions")
            if {item.policy_id for item in ordered} != {expected_active_policy}:
                raise ValueError("source-v4 evaluator complementary-policy assignment drifted")
            for segment_id in _SEGMENT_IDS:
                segment_rows = tuple(item for item in ordered if item.segment_id == segment_id)
                if len(segment_rows) != 8 or Counter(item.condition_id for item in segment_rows) != {
                    _CONDITION_IDS[0]: 4,
                    _CONDITION_IDS[1]: 4,
                }:
                    raise ValueError("source-v4 evaluator segment/condition balance drifted")
            active_mapping: dict[str, str] = {}
            for item in ordered:
                previous = active_mapping.setdefault(item.condition_id, item.preferred_action_id)
                if previous != item.preferred_action_id:
                    raise ValueError("source-v4 evaluator preferred action drifted within condition")
            if set(active_mapping.values()) != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
            }:
                raise ValueError("source-v4 evaluator active policy must use both non-noop actions")
            for condition_id in _CONDITION_IDS:
                collection_targets = {
                    item.decision_index for item in ordered[:_COLLECTION_COUNT] if item.condition_id == condition_id
                }
                correction_rows = tuple(
                    item for item in ordered[16:24] if item.condition_id == condition_id
                )
                if {item.correction_target_index for item in correction_rows} != collection_targets:
                    raise ValueError("source-v4 evaluator correction permutation drifted")
                if any(
                    ordered[item.correction_target_index].condition_id != condition_id
                    for item in correction_rows
                    if item.correction_target_index is not None
                ):
                    raise ValueError("source-v4 evaluator correction condition join drifted")
            if any(
                item.root_seed != manifest.root_seed
                or item.tape_seed != manifest.tape_seed
                or item.world_clone_id != manifest.world_clone_id
                for item in ordered
            ):
                raise ValueError("source-v4 evaluator root lineage drifted within a decision tape")
        for field_name, probabilities in (
            ("preferred_action_probabilities", self.preferred_action_probabilities),
            ("nonpreferred_stay_probabilities", self.nonpreferred_stay_probabilities),
            ("nonpreferred_space_probabilities", self.nonpreferred_space_probabilities),
            ("neutral_noop_probabilities", self.neutral_noop_probabilities),
        ):
            if len(probabilities) != len(RELATIONSHIP_OUTCOMES):
                raise ValueError(f"{field_name} must contain four probabilities")
            if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities):
                raise ValueError(f"{field_name} must contain finite probabilities")
            if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{field_name} must sum to one")
        _require_preferred_positive_dominance(
            self.preferred_action_probabilities,
            (
                self.nonpreferred_stay_probabilities,
                self.nonpreferred_space_probabilities,
                self.neutral_noop_probabilities,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "cohort_id": self.cohort_id,
            "root_manifests": [item.to_payload() for item in self.root_manifests],
            "onboarding_sessions": [item.to_payload() for item in self.onboarding_sessions],
            "decision_sessions": [item.to_payload() for item in self.decision_sessions],
            "preferred_action_probabilities": list(self.preferred_action_probabilities),
            "nonpreferred_stay_probabilities": list(self.nonpreferred_stay_probabilities),
            "nonpreferred_space_probabilities": list(self.nonpreferred_space_probabilities),
            "neutral_noop_probabilities": list(self.neutral_noop_probabilities),
        }

    @property
    def sealed_bundle_sha256(self) -> str:
        return sha256_json(self.to_payload())

    def sessions_for(self, subject_id: str) -> tuple[HorizonEvaluatorDecisionSession, ...]:
        sessions = tuple(item for item in self.decision_sessions if item.subject_id == subject_id)
        if len(sessions) != _DECISION_COUNT:
            raise KeyError(subject_id)
        return sessions


def relationship_product_horizon_source_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / _PROTOCOL_FILENAME


def load_relationship_product_horizon_source_protocol(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipProductHorizonSourceProtocol:
    """Strictly load the independent source-v4 protocol without model execution."""

    path = pathlib.Path(protocol_path or relationship_product_horizon_source_protocol_path())
    raw = _parse_unique_json(path.read_bytes(), path)
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "cohort",
            "policies",
            "schedule",
            "reactive_environment",
            "rendering",
            "firewall",
            "claim_boundary",
        },
        source="source-v4 protocol",
    )
    if raw["schema_version"] != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION:
        raise ValueError("source-v4 loader refuses another schema version")

    owner = _require_mapping(raw["owner"], "owner")
    cohort = _require_mapping(raw["cohort"], "cohort")
    policies = _require_mapping(raw["policies"], "policies")
    schedule = _require_mapping(raw["schedule"], "schedule")
    environment = _require_mapping(raw["reactive_environment"], "reactive_environment")
    rendering = _require_mapping(raw["rendering"], "rendering")
    firewall = _require_mapping(raw["firewall"], "firewall")

    _require_exact_keys(
        owner,
        {
            "module",
            "source_role",
            "settlement_owner",
            "runtime_owner_added",
            "runtime_slot_added",
            "p1m_output_dependency",
            "difficulty_tuned_from_prior_outcome",
        },
        source="owner",
    )
    if owner["module"] != _OWNER_MODULE:
        raise ValueError("source-v4 owner module drifted")
    if owner["source_role"] != "independent_medium_matrix_development_source_only":
        raise ValueError("source-v4 role drifted")
    if owner["settlement_owner"] != (
        "lifeform_domain_emogpt.lab.environment.ReactiveRelationshipEnvironment"
    ):
        raise ValueError("ReactiveRelationshipEnvironment must remain settlement owner")
    for field_name in (
        "runtime_owner_added",
        "runtime_slot_added",
        "p1m_output_dependency",
        "difficulty_tuned_from_prior_outcome",
    ):
        if _require_bool(owner[field_name], f"owner.{field_name}"):
            raise ValueError(f"owner.{field_name} must remain false")

    _require_exact_keys(
        cohort,
        {
            "cohort_id",
            "evidence_role",
            "root_count",
            "onboarding_sessions_per_root",
            "collection_decisions_per_root",
            "evaluation_decisions_per_root",
            "master_seed_namespace",
            "per_arm_exogenous_root_clone",
            "arm_identity_affects_source_or_environment_seed",
        },
        source="cohort",
    )
    if cohort["evidence_role"] != "development_direction_variance_icc_and_cost_only":
        raise ValueError("source-v4 evidence role drifted")
    if not _require_bool(cohort["per_arm_exogenous_root_clone"], "per_arm_exogenous_root_clone"):
        raise ValueError("source-v4 requires matched exogenous root clones")
    if _require_bool(
        cohort["arm_identity_affects_source_or_environment_seed"],
        "arm_identity_affects_source_or_environment_seed",
    ):
        raise ValueError("arm identity must not enter source or environment seeds")

    _require_exact_keys(
        policies,
        {"condition_ids", "profiles", "root_assignment", "evaluation_policy"},
        source="policies",
    )
    if policies["root_assignment"] != "zero_based_root_index_even_alpha_odd_beta":
        raise ValueError("source-v4 root policy assignment drifted")
    if policies["evaluation_policy"] != "use_complementary_profile":
        raise ValueError("source-v4 evaluation policy drifted")
    condition_ids = _require_text_tuple(policies["condition_ids"], "policies.condition_ids")
    raw_profiles = _require_mapping(policies["profiles"], "policies.profiles")
    _require_exact_keys(raw_profiles, set(_POLICY_IDS), source="policies.profiles")
    profiles: list[RelationshipPolicyProfile] = []
    for policy_id in _POLICY_IDS:
        profile = _require_mapping(raw_profiles[policy_id], f"policies.profiles.{policy_id}")
        _require_exact_keys(profile, set(condition_ids), source=f"policies.profiles.{policy_id}")
        profiles.append(
            RelationshipPolicyProfile(
                policy_id=policy_id,
                condition_actions=tuple(
                    sorted(
                        (
                            condition_id,
                            RelationshipAction(_require_text(profile[condition_id], condition_id)),
                        )
                        for condition_id in condition_ids
                    )
                ),
            )
        )

    _require_exact_keys(
        schedule,
        {
            "condition_count_per_eight_decision_block",
            "segments",
            "domain_ids",
            "role_ids",
        },
        source="schedule",
    )
    if schedule["condition_count_per_eight_decision_block"] != 4:
        raise ValueError("every eight-decision block must contain four of each condition")
    raw_segments = schedule["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError("schedule.segments must be an array")
    segment_specs: list[HorizonSegmentSpec] = []
    for index, item in enumerate(raw_segments):
        mapping = _require_mapping(item, f"schedule.segments[{index}]")
        _require_exact_keys(
            mapping,
            {"segment_id", "decision_count", "policy_mode", "minimum_gap_before_days"},
            source=f"schedule.segments[{index}]",
        )
        segment_specs.append(
            HorizonSegmentSpec(
                segment_id=_require_text(mapping["segment_id"], "segment_id"),
                decision_count=_require_int(mapping["decision_count"], "decision_count"),
                policy_mode=_require_text(mapping["policy_mode"], "policy_mode"),
                minimum_gap_before_days=_require_int(
                    mapping["minimum_gap_before_days"], "minimum_gap_before_days"
                ),
            )
        )

    _require_exact_keys(
        environment,
        {
            "environment_version",
            "seed_namespace",
            "outcome_order",
            "positive_outcomes",
            "preferred_action_probabilities",
            "nonpreferred_stay_probabilities",
            "nonpreferred_space_probabilities",
            "neutral_noop_probabilities",
        },
        source="reactive_environment",
    )
    if environment["environment_version"] != REACTIVE_ENVIRONMENT_VERSION:
        raise ValueError("reactive environment version drifted")
    if _require_text_tuple(environment["outcome_order"], "outcome_order") != tuple(
        item.value for item in RELATIONSHIP_OUTCOMES
    ):
        raise ValueError("reactive outcome order drifted")
    if set(_require_text_tuple(environment["positive_outcomes"], "positive_outcomes")) != {
        DialogueExternalOutcomeKind.HELPED.value,
        DialogueExternalOutcomeKind.FELT_HEARD.value,
    }:
        raise ValueError("positive outcome set drifted")

    _require_exact_keys(
        rendering,
        {
            "version",
            "condition_surface_recipes_per_condition",
            "reflection_recipes",
            "neutral_context_recipes",
            "minimum_public_source_characters_per_root",
            "minimum_public_source_utf8_bytes_per_root",
            "token_measurement_status",
        },
        source="rendering",
    )
    if rendering["version"] != RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_VERSION:
        raise ValueError("source-v4 rendering version drifted")
    if rendering["condition_surface_recipes_per_condition"] != 12:
        raise ValueError("source-v4 condition recipe count drifted")
    if rendering["reflection_recipes"] != len(_REFLECTIONS):
        raise ValueError("source-v4 reflection recipe count drifted")
    if rendering["neutral_context_recipes"] != len(_NEUTRAL_CONTEXTS):
        raise ValueError("source-v4 neutral recipe count drifted")
    if rendering["token_measurement_status"] != "not_measured_not_claimed":
        raise ValueError("source-v4 must not claim tokenizer measurement")

    _require_exact_keys(
        firewall,
        {
            "public_view_contains_sealed_condition",
            "public_view_contains_policy_or_preferred_action",
            "public_view_contains_seed_or_environment_truth",
            "collection_forced_action_owned_by_source",
            "evaluation_or_judge_feedback_to_learning",
            "model_output_count",
            "formal_evidence_authorized",
            "integrated_execution_authorized",
            "human_sample_claimed",
        },
        source="firewall",
    )
    for field_name in (
        "public_view_contains_sealed_condition",
        "public_view_contains_policy_or_preferred_action",
        "public_view_contains_seed_or_environment_truth",
        "collection_forced_action_owned_by_source",
        "evaluation_or_judge_feedback_to_learning",
        "formal_evidence_authorized",
        "integrated_execution_authorized",
        "human_sample_claimed",
    ):
        if _require_bool(firewall[field_name], f"firewall.{field_name}"):
            raise ValueError(f"firewall.{field_name} must remain false")
    if _require_int(firewall["model_output_count"], "firewall.model_output_count") != 0:
        raise ValueError("source-v4 cannot contain model output")

    return RelationshipProductHorizonSourceProtocol(
        protocol_id=hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest(),
        cohort_id=_require_text(cohort["cohort_id"], "cohort.cohort_id"),
        evidence_role=_require_text(cohort["evidence_role"], "cohort.evidence_role"),
        root_count=_require_int(cohort["root_count"], "cohort.root_count"),
        onboarding_sessions_per_root=_require_int(
            cohort["onboarding_sessions_per_root"], "onboarding_sessions_per_root"
        ),
        collection_decisions_per_root=_require_int(
            cohort["collection_decisions_per_root"], "collection_decisions_per_root"
        ),
        evaluation_decisions_per_root=_require_int(
            cohort["evaluation_decisions_per_root"], "evaluation_decisions_per_root"
        ),
        master_seed_namespace=_require_text(
            cohort["master_seed_namespace"], "cohort.master_seed_namespace"
        ),
        environment_seed_namespace=_require_text(
            environment["seed_namespace"], "reactive_environment.seed_namespace"
        ),
        condition_ids=condition_ids,
        policy_profiles=tuple(profiles),
        segment_specs=tuple(segment_specs),
        domain_ids=_require_text_tuple(schedule["domain_ids"], "schedule.domain_ids"),
        role_ids=_require_text_tuple(schedule["role_ids"], "schedule.role_ids"),
        preferred_action_probabilities=_require_probability_tuple(
            environment["preferred_action_probabilities"], "preferred_action_probabilities"
        ),
        nonpreferred_stay_probabilities=_require_probability_tuple(
            environment["nonpreferred_stay_probabilities"], "nonpreferred_stay_probabilities"
        ),
        nonpreferred_space_probabilities=_require_probability_tuple(
            environment["nonpreferred_space_probabilities"], "nonpreferred_space_probabilities"
        ),
        neutral_noop_probabilities=_require_probability_tuple(
            environment["neutral_noop_probabilities"], "neutral_noop_probabilities"
        ),
        minimum_public_source_characters_per_root=_require_int(
            rendering["minimum_public_source_characters_per_root"],
            "minimum_public_source_characters_per_root",
        ),
        minimum_public_source_utf8_bytes_per_root=_require_int(
            rendering["minimum_public_source_utf8_bytes_per_root"],
            "minimum_public_source_utf8_bytes_per_root",
        ),
        claim_boundary=_require_text(raw["claim_boundary"], "claim_boundary"),
    )


def _root_material(protocol: RelationshipProductHorizonSourceProtocol, root_index: int) -> tuple[int, int, str, str]:
    root_seed = _derive_u64(
        protocol.master_seed_namespace,
        {"purpose": "root", "protocol_id": protocol.protocol_id, "root_index": root_index},
    )
    tape_seed = _derive_u64(
        protocol.master_seed_namespace,
        {"purpose": "tape", "protocol_id": protocol.protocol_id, "root_index": root_index},
    )
    root_digest = hashlib.sha256(
        canonical_json(
            {"protocol_id": protocol.protocol_id, "root_seed": root_seed, "tape_seed": tape_seed}
        ).encode("utf-8")
    ).hexdigest()
    return root_seed, tape_seed, f"rh4-subject-{root_digest[:20]}", f"rh4-world-{root_digest[20:40]}"


def _tape_payload(
    protocol: RelationshipProductHorizonSourceProtocol,
    *,
    tape_seed: int,
    purpose: str,
    fields: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol_id": protocol.protocol_id,
        "tape_seed": tape_seed,
        "purpose": purpose,
        **fields,
    }


def _condition_order(
    protocol: RelationshipProductHorizonSourceProtocol,
    root_index: int,
    segment_id: str,
    *,
    tape_seed: int,
) -> tuple[str, ...]:
    tokens = tuple((condition_id, replica) for condition_id in protocol.condition_ids for replica in range(4))
    return tuple(
        condition_id
        for condition_id, _ in sorted(
            tokens,
            key=lambda item: (
                _derive_u64(
                    protocol.master_seed_namespace,
                    _tape_payload(
                        protocol,
                        tape_seed=tape_seed,
                        purpose="condition-order",
                        fields={
                            "root_index": root_index,
                            "segment_id": segment_id,
                            "condition_id": item[0],
                            "replica": item[1],
                        },
                    ),
                ),
                item,
            ),
        )
    )


def build_relationship_product_horizon_phase_specs(
    protocol: RelationshipProductHorizonSourceProtocol,
    *,
    root_index: int,
) -> tuple[HorizonPhaseSpec, ...]:
    if not 0 <= root_index < protocol.root_count:
        raise IndexError(root_index)
    _, tape_seed, _, _ = _root_material(protocol, root_index)
    result: list[HorizonPhaseSpec] = []
    virtual_day = _ONBOARDING_COUNT - 1
    for segment in protocol.segment_specs:
        virtual_day += segment.minimum_gap_before_days
        for within_segment, condition_id in enumerate(
            _condition_order(protocol, root_index, segment.segment_id, tape_seed=tape_seed)
        ):
            decision_index = len(result)
            virtual_day += 1
            correction_target: int | None = None
            if segment.segment_id == "correction":
                candidates = tuple(
                    sorted(
                        (
                            item.decision_index
                            for item in result[:_COLLECTION_COUNT]
                            if item.condition_id == condition_id
                        ),
                        key=lambda target_index: _derive_u64(
                            protocol.master_seed_namespace,
                            _tape_payload(
                                protocol,
                                tape_seed=tape_seed,
                                purpose="correction-target-order",
                                fields={
                                    "root_index": root_index,
                                    "condition_id": condition_id,
                                    "target_index": target_index,
                                },
                            ),
                        ),
                    )
                )
                occurrence = sum(
                    1
                    for item in result
                    if item.segment_id == "correction" and item.condition_id == condition_id
                )
                correction_target = candidates[occurrence]
            recipe_index = _pick_index(
                protocol.master_seed_namespace,
                _tape_payload(
                    protocol,
                    tape_seed=tape_seed,
                    purpose="surface",
                    fields={
                        "root_index": root_index,
                        "decision_index": decision_index,
                        "condition_id": condition_id,
                    },
                ),
                len(_CONDITION_SURFACES[condition_id]),
            )
            result.append(
                HorizonPhaseSpec(
                    decision_index=decision_index,
                    segment_id=segment.segment_id,
                    domain_id=protocol.domain_ids[
                        _pick_index(
                            protocol.master_seed_namespace,
                            _tape_payload(
                                protocol,
                                tape_seed=tape_seed,
                                purpose="domain",
                                fields={"root_index": root_index, "decision_index": decision_index},
                            ),
                            len(protocol.domain_ids),
                        )
                    ],
                    role_id=protocol.role_ids[
                        _pick_index(
                            protocol.master_seed_namespace,
                            _tape_payload(
                                protocol,
                                tape_seed=tape_seed,
                                purpose="role",
                                fields={"root_index": root_index, "decision_index": decision_index},
                            ),
                            len(protocol.role_ids),
                        )
                    ],
                    condition_id=condition_id,
                    policy_mode=segment.policy_mode,
                    virtual_day=virtual_day,
                    gap_before_days=segment.minimum_gap_before_days if within_segment == 0 else 0,
                    surface_recipe_id=f"{condition_id}-surface-{recipe_index:02d}",
                    reflection_recipe_index=_pick_index(
                        protocol.master_seed_namespace,
                        _tape_payload(
                            protocol,
                            tape_seed=tape_seed,
                            purpose="reflection",
                            fields={"root_index": root_index, "decision_index": decision_index},
                        ),
                        len(_REFLECTIONS),
                    ),
                    neutral_context_recipe_index=_pick_index(
                        protocol.master_seed_namespace,
                        _tape_payload(
                            protocol,
                            tape_seed=tape_seed,
                            purpose="neutral",
                            fields={"root_index": root_index, "decision_index": decision_index},
                        ),
                        len(_NEUTRAL_CONTEXTS),
                    ),
                    correction_target_index=correction_target,
                )
            )
    if len(result) != _DECISION_COUNT:
        raise AssertionError("source-v4 phase schedule did not produce forty-eight decisions")
    return tuple(result)


def _session_id(subject_id: str, decision_index: int) -> str:
    return f"{subject_id}-decision-{decision_index + 1:02d}"


def _onboarding_session_id(subject_id: str, onboarding_index: int) -> str:
    return f"{subject_id}-onboarding-{onboarding_index + 1:02d}"


def _decision_id(subject_id: str, decision_index: int) -> str:
    return f"{subject_id}-decision-event-{decision_index + 1:02d}"


def _scene_id(subject_id: str, decision_index: int) -> str:
    return f"{subject_id}-sealed-scene-{decision_index + 1:02d}"


def _surface_index(phase: HorizonPhaseSpec) -> int:
    return int(phase.surface_recipe_id.rsplit("-", 1)[1])


def _render_context(phase: HorizonPhaseSpec, *, decision_index: int) -> str:
    chronology = (
        f"这是前一段近况之后留下的第{decision_index + 1}条连续记录。"
        "我没有把所有旧事重新抄一遍，只补充眼前真正发生的部分。"
    )
    return "".join(
        (
            _DOMAIN_CONTEXTS[phase.domain_id],
            _ROLE_CONTEXTS[phase.role_id],
            chronology,
            _NEUTRAL_CONTEXTS[phase.neutral_context_recipe_index],
        )
    )


def _render_current_input(phase: HorizonPhaseSpec) -> str:
    segment_contexts = _SEGMENT_CONTEXTS[phase.segment_id]
    return "".join(
        (
            _CONDITION_SURFACES[phase.condition_id][_surface_index(phase)],
            segment_contexts[phase.reflection_recipe_index % len(segment_contexts)],
            _REFLECTIONS[phase.reflection_recipe_index],
        )
    )


def _historical_outcome(
    *,
    action: RelationshipAction,
    preferred: RelationshipAction,
    onboarding_index: int,
) -> DialogueExternalOutcomeKind:
    if action is preferred:
        return (
            DialogueExternalOutcomeKind.HELPED
            if onboarding_index % 2 == 0
            else DialogueExternalOutcomeKind.FELT_HEARD
        )
    if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE:
        return DialogueExternalOutcomeKind.OVER_DIRECTIVE
    return DialogueExternalOutcomeKind.MISSED


def _build_public_root(
    protocol: RelationshipProductHorizonSourceProtocol,
    root_index: int,
) -> HorizonPublicRoot:
    _, tape_seed, subject_id, _ = _root_material(protocol, root_index)
    base_policy = protocol.policy(protocol.base_policy_id(root_index))
    onboarding: list[HorizonPublicOnboardingSession] = []
    onboarding_pairs = (
        (protocol.condition_ids[0], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
        (protocol.condition_ids[0], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
        (protocol.condition_ids[1], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
        (protocol.condition_ids[1], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
    )
    for index, (condition_id, action) in enumerate(onboarding_pairs):
        recipe_index = _pick_index(
            protocol.master_seed_namespace,
            _tape_payload(
                protocol,
                tape_seed=tape_seed,
                purpose="onboarding-surface",
                fields={"root_index": root_index, "onboarding_index": index},
            ),
            len(_CONDITION_SURFACES[condition_id]),
        )
        domain_id = protocol.domain_ids[
            _pick_index(
                protocol.master_seed_namespace,
                _tape_payload(
                    protocol,
                    tape_seed=tape_seed,
                    purpose="onboarding-domain",
                    fields={"root_index": root_index, "onboarding_index": index},
                ),
                len(protocol.domain_ids),
            )
        ]
        role_id = protocol.role_ids[
            _pick_index(
                protocol.master_seed_namespace,
                _tape_payload(
                    protocol,
                    tape_seed=tape_seed,
                    purpose="onboarding-role",
                    fields={"root_index": root_index, "onboarding_index": index},
                ),
                len(protocol.role_ids),
            )
        ]
        outcome = _historical_outcome(
            action=action,
            preferred=base_policy.action_for(condition_id),
            onboarding_index=index,
        )
        onboarding.append(
            HorizonPublicOnboardingSession(
                session_id=_onboarding_session_id(subject_id, index),
                session_index=index,
                virtual_day=index,
                public_context_chunk=(
                    _DOMAIN_CONTEXTS[domain_id]
                    + _ROLE_CONTEXTS[role_id]
                    + _NEUTRAL_CONTEXTS[
                        _pick_index(
                            protocol.master_seed_namespace,
                            _tape_payload(
                                protocol,
                                tape_seed=tape_seed,
                                purpose="onboarding-neutral",
                                fields={"root_index": root_index, "onboarding_index": index},
                            ),
                            len(_NEUTRAL_CONTEXTS),
                        )
                    ]
                ),
                user_utterance=(
                    "这是一条较早的关系记录。"
                    + _CONDITION_SURFACES[condition_id][recipe_index]
                    + _REFLECTIONS[
                        _pick_index(
                            protocol.master_seed_namespace,
                            _tape_payload(
                                protocol,
                                tape_seed=tape_seed,
                                purpose="onboarding-reflection",
                                fields={"root_index": root_index, "onboarding_index": index},
                            ),
                            len(_REFLECTIONS),
                        )
                    ]
                ),
                exposed_action_id=action.value,
                observed_outcome_id=outcome.value,
                rendered_user_reaction=_HISTORICAL_REACTIONS[outcome],
            )
        )

    decisions: list[HorizonPublicDecisionSession] = []
    for phase in build_relationship_product_horizon_phase_specs(protocol, root_index=root_index):
        correction_target = (
            _session_id(subject_id, phase.correction_target_index)
            if phase.correction_target_index is not None
            else None
        )
        decisions.append(
            HorizonPublicDecisionSession(
                session_id=_session_id(subject_id, phase.decision_index),
                decision_id=_decision_id(subject_id, phase.decision_index),
                decision_index=phase.decision_index,
                virtual_day=phase.virtual_day,
                public_context_chunk=_render_context(phase, decision_index=phase.decision_index),
                current_input=_render_current_input(phase),
                public_correction_target_session_id=correction_target,
                action_surface=_ACTION_SURFACES,
            )
        )
    root = HorizonPublicRoot(
        subject_id=subject_id,
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
    )
    if root.public_source_characters < protocol.minimum_public_source_characters_per_root:
        raise ValueError("source-v4 public character pressure fell below the frozen minimum")
    if root.public_source_utf8_bytes < protocol.minimum_public_source_utf8_bytes_per_root:
        raise ValueError("source-v4 public UTF-8 pressure fell below the frozen minimum")
    return root


def build_relationship_product_horizon_public_view(
    protocol: RelationshipProductHorizonSourceProtocol | None = None,
) -> RelationshipProductHorizonPublicView:
    source = protocol or load_relationship_product_horizon_source_protocol()
    roots = tuple(_build_public_root(source, root_index) for root_index in range(source.root_count))
    if len({root.public_trajectory_sha256 for root in roots}) != source.root_count:
        raise ValueError("source-v4 public trajectories are not unique across 112 roots")
    return RelationshipProductHorizonPublicView(
        protocol_id=source.protocol_id,
        cohort_id=source.cohort_id,
        roots=roots,
    )


def _causal_tape_signature(phases: tuple[HorizonPhaseSpec, ...]) -> str:
    return sha256_json(
        [
            {
                "decision_role": phase.role_id,
                "segment_id": phase.segment_id,
                "condition_id": phase.condition_id,
                "policy_mode": phase.policy_mode,
                "domain_id": phase.domain_id,
                "surface_recipe_id": phase.surface_recipe_id,
                "correction_target_relative_index": (
                    phase.decision_index - phase.correction_target_index
                    if phase.correction_target_index is not None
                    else None
                ),
                "gap_before_days_bucket": "at_least_14" if phase.gap_before_days >= 14 else "none",
            }
            for phase in phases
        ]
    )


def build_relationship_product_horizon_evaluator_bundle(
    protocol: RelationshipProductHorizonSourceProtocol | None = None,
) -> RelationshipProductHorizonEvaluatorBundle:
    source = protocol or load_relationship_product_horizon_source_protocol()
    root_manifests: list[HorizonEvaluatorRootManifest] = []
    onboarding: list[HorizonEvaluatorOnboardingSession] = []
    decisions: list[HorizonEvaluatorDecisionSession] = []
    environment_seeds: set[int] = set()
    causal_signatures: set[str] = set()

    for root_index in range(source.root_count):
        root_seed, tape_seed, subject_id, world_clone_id = _root_material(source, root_index)
        public_root = _build_public_root(source, root_index)
        phases = build_relationship_product_horizon_phase_specs(source, root_index=root_index)
        signature = _causal_tape_signature(phases)
        if signature in causal_signatures:
            raise ValueError("source-v4 causal tape signatures must be unique across roots")
        causal_signatures.add(signature)
        root_manifests.append(
            HorizonEvaluatorRootManifest(
                root_index=root_index,
                subject_id=subject_id,
                root_seed=root_seed,
                tape_seed=tape_seed,
                world_clone_id=world_clone_id,
                public_trajectory_sha256=public_root.public_trajectory_sha256,
                causal_tape_signature=signature,
            )
        )
        base_policy_id = source.base_policy_id(root_index)
        base_policy = source.policy(base_policy_id)
        onboarding_pairs = (
            (source.condition_ids[0], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
            (source.condition_ids[0], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
            (source.condition_ids[1], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
            (source.condition_ids[1], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
        )
        for index, (condition_id, action) in enumerate(onboarding_pairs):
            preferred = base_policy.action_for(condition_id)
            onboarding.append(
                HorizonEvaluatorOnboardingSession(
                    subject_id=subject_id,
                    session_id=_onboarding_session_id(subject_id, index),
                    session_index=index,
                    virtual_day=index,
                    condition_id=condition_id,
                    policy_id=base_policy_id,
                    preferred_action_id=preferred.value,
                    exposed_action_id=action.value,
                    observed_outcome_id=_historical_outcome(
                        action=action, preferred=preferred, onboarding_index=index
                    ).value,
                )
            )
        for phase in phases:
            policy_id = source.active_policy_id(root_index, phase.policy_mode)
            preferred = source.policy(policy_id).action_for(phase.condition_id)
            environment_seed = _derive_u64(
                source.environment_seed_namespace,
                {
                    "protocol_id": source.protocol_id,
                    "root_seed": root_seed,
                    "tape_seed": tape_seed,
                    "decision_index": phase.decision_index,
                    "world_clone_id": world_clone_id,
                },
            )
            if environment_seed in environment_seeds:
                raise ValueError("source-v4 environment seeds must be globally unique")
            environment_seeds.add(environment_seed)
            decisions.append(
                HorizonEvaluatorDecisionSession(
                    subject_id=subject_id,
                    root_seed=root_seed,
                    tape_seed=tape_seed,
                    world_clone_id=world_clone_id,
                    session_id=_session_id(subject_id, phase.decision_index),
                    decision_id=_decision_id(subject_id, phase.decision_index),
                    decision_index=phase.decision_index,
                    virtual_day=phase.virtual_day,
                    gap_before_days=phase.gap_before_days,
                    scene_id=_scene_id(subject_id, phase.decision_index),
                    segment_id=phase.segment_id,
                    domain_id=phase.domain_id,
                    role_id=phase.role_id,
                    condition_id=phase.condition_id,
                    policy_id=policy_id,
                    policy_mode=phase.policy_mode,
                    preferred_action_id=preferred.value,
                    environment_seed=environment_seed,
                    surface_recipe_id=phase.surface_recipe_id,
                    correction_target_index=phase.correction_target_index,
                )
            )
    if len(environment_seeds) != source.root_count * source.decision_sessions_per_root:
        raise AssertionError("source-v4 environment seed inventory drifted")
    return RelationshipProductHorizonEvaluatorBundle(
        protocol_id=source.protocol_id,
        cohort_id=source.cohort_id,
        root_manifests=tuple(root_manifests),
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
        preferred_action_probabilities=source.preferred_action_probabilities,
        nonpreferred_stay_probabilities=source.nonpreferred_stay_probabilities,
        nonpreferred_space_probabilities=source.nonpreferred_space_probabilities,
        neutral_noop_probabilities=source.neutral_noop_probabilities,
    )


def build_relationship_product_horizon_environment(
    evaluator_bundle: RelationshipProductHorizonEvaluatorBundle,
    *,
    subject_id: str,
) -> ReactiveRelationshipEnvironment:
    sessions = evaluator_bundle.sessions_for(subject_id)
    adapter = _HorizonEnvironmentDatasetAdapter(
        dataset_fingerprint=sha256_json(
            {
                "schema_version": "relationship-product-horizon-environment-adapter.v4",
                "protocol_id": evaluator_bundle.protocol_id,
                "cohort_id": evaluator_bundle.cohort_id,
                "subject_id": subject_id,
                "sessions": [item.to_payload() for item in sessions],
                "preferred_action_probabilities": evaluator_bundle.preferred_action_probabilities,
                "nonpreferred_stay_probabilities": evaluator_bundle.nonpreferred_stay_probabilities,
                "nonpreferred_space_probabilities": evaluator_bundle.nonpreferred_space_probabilities,
                "neutral_noop_probabilities": evaluator_bundle.neutral_noop_probabilities,
            }
        ),
        sessions=sessions,
        preferred_action_probabilities=evaluator_bundle.preferred_action_probabilities,
        nonpreferred_stay_probabilities=evaluator_bundle.nonpreferred_stay_probabilities,
        nonpreferred_space_probabilities=evaluator_bundle.nonpreferred_space_probabilities,
        neutral_noop_probabilities=evaluator_bundle.neutral_noop_probabilities,
    )
    return ReactiveRelationshipEnvironment(cast(RelationshipTransferDataset, adapter))


def _assert_no_public_truth_leakage(payload: object) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(set(payload) & _PUBLIC_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"public source-v4 payload leaked sealed fields: {forbidden}")
        for value in payload.values():
            _assert_no_public_truth_leakage(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_public_truth_leakage(value)
    elif isinstance(payload, str):
        sealed_literals = (*_CONDITION_IDS, *_SEGMENT_IDS, *_POLICY_IDS)
        leaked = tuple(item for item in sealed_literals if item in payload)
        if leaked:
            raise ValueError(f"public source-v4 payload leaked sealed literals: {leaked}")


@dataclass(frozen=True)
class _HorizonEnvironmentDatasetAdapter:
    dataset_fingerprint: str
    sessions: tuple[HorizonEvaluatorDecisionSession, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]

    def _session(self, scene_id: str) -> HorizonEvaluatorDecisionSession:
        for session in self.sessions:
            if session.scene_id == scene_id:
                return session
        raise KeyError(scene_id)

    def dynamic_for_scene(self, scene_id: str) -> LatentRelationshipDynamic:
        session = self._session(scene_id)
        return LatentRelationshipDynamic(
            dynamic_id=f"{session.scene_id}-dynamic",
            mirror_pair_id=session.world_clone_id,
            split=RelationshipDatasetSplit.VALIDATION,
            preferred_action=RelationshipAction(session.preferred_action_id),
            outcome_profile_id=f"{session.scene_id}-profile",
            hidden_summary=(
                f"sealed horizon source-v4 {session.segment_id}/{session.condition_id}/{session.policy_id}"
            ),
            policy_id=session.policy_id,
            probe_condition_id=session.condition_id,
        )

    def distribution(self, scene_id: str, action: RelationshipAction) -> CandidateOutcomePrediction:
        session = self._session(scene_id)
        preferred = RelationshipAction(session.preferred_action_id)
        if action is preferred:
            probabilities = self.preferred_action_probabilities
        elif action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE:
            probabilities = self.nonpreferred_stay_probabilities
        elif action is RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION:
            probabilities = self.nonpreferred_space_probabilities
        elif action is RelationshipAction.NEUTRAL_NOOP:
            probabilities = self.neutral_noop_probabilities
        else:
            raise ValueError(f"unsupported relationship action {action!r}")
        return CandidateOutcomePrediction(
            action_id=action,
            outcomes=tuple(
                OutcomeProbability(outcome, probability)
                for outcome, probability in zip(RELATIONSHIP_OUTCOMES, probabilities, strict=True)
            ),
        )


__all__ = [
    "RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_VERSION",
    "RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION",
    "HorizonEvaluatorDecisionSession",
    "HorizonEvaluatorOnboardingSession",
    "HorizonEvaluatorRootManifest",
    "HorizonPhaseSpec",
    "HorizonPublicDecisionSession",
    "HorizonPublicOnboardingSession",
    "HorizonPublicRoot",
    "HorizonSegmentSpec",
    "RelationshipProductHorizonEvaluatorBundle",
    "RelationshipProductHorizonPublicView",
    "RelationshipProductHorizonSourceProtocol",
    "build_relationship_product_horizon_environment",
    "build_relationship_product_horizon_evaluator_bundle",
    "build_relationship_product_horizon_phase_specs",
    "build_relationship_product_horizon_public_view",
    "load_relationship_product_horizon_source_protocol",
    "relationship_product_horizon_source_protocol_path",
]
