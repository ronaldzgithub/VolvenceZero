"""Deterministic domain source for the local Relationship product pilot.

This module is an offline source owner.  It creates a public system-under-test
view and a physically separate sealed evaluator bundle.  It does not define a
runtime owner, an experimental arm, a baseline, or a model runner.  Settlement
continues to belong exclusively to :class:`ReactiveRelationshipEnvironment`;
the small adapter at the bottom only projects sealed typed rows into that
existing owner's dataset-shaped input surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
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
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION = "relationship-product-pilot-source.v1"
RELATIONSHIP_PRODUCT_PILOT_PUBLIC_VIEW_SCHEMA_VERSION = "relationship-product-pilot-public-view.v1"
RELATIONSHIP_PRODUCT_PILOT_EVALUATOR_SCHEMA_VERSION = "relationship-product-pilot-evaluator-bundle.v1"
RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION = "relationship-product-pilot-public-renderer.v1"

_SUBJECT_COUNT = 8
_ONBOARDING_SESSION_COUNT = 4
_DECISION_SESSION_COUNT = 24
_POLICY_IDS = ("alpha", "beta")
_POLICY_MODES = ("base", "complement")
_REQUIRED_STAGES = {
    "stable",
    "domain_switch",
    "pre_gap",
    "post_gap_reversal",
    "reversal",
    "correction",
    "post_correction",
    "return_after_gap",
    "mixed_stress",
}
_PUBLIC_FORBIDDEN_KEYS = {
    "active_policy_mode",
    "condition_id",
    "dynamic_id",
    "environment_seed",
    "phase_id",
    "policy_id",
    "preferred_action_id",
    "scene_id",
    "stage_id",
    "subject_seed",
}

_DOMAIN_RENDERINGS: dict[str, tuple[str, str]] = {
    "work": (
        "工作",
        "项目进入多人协作阶段，会议纪要、排期、临时请求和职责边界同时变多，很多细节看似琐碎，却会影响一个人是否觉得自己仍在团队里有位置。",
    ),
    "health": (
        "健康",
        "最近在复诊、作息、运动和家人建议之间来回协调，信息很多而且结论并不总是一致，需要把事实、感受和最后由谁决定分开。",
    ),
    "family": (
        "家庭",
        "家里正在处理聚会、照护和旧物整理，几个人都出于好意提出安排，但好意、参与感与替别人作主并不是同一件事。",
    ),
    "community": (
        "社区",
        "社区小组在筹备活动，报名、轮值、预算和场地都有变化，很多沟通发生在零散消息里，容易把没有及时回复误解成不在乎。",
    ),
    "friends": (
        "朋友",
        "朋友之间最近有旅行、搬家和工作变动，大家的时间表很难对齐，有些关心需要主动表达，有些关心则要给对方保留回来的入口。",
    ),
    "intimacy": (
        "亲密关系",
        "亲密关系里正在讨论共同时间、个人空间和下一步安排，同一句建议在不同阶段可能意味着陪伴，也可能意味着越过边界。",
    ),
}

_CONDITION_RENDERINGS: dict[str, str] = {
    "connection_under_exclusion": (
        "这次最难受的不是事情本身，而是其他人已经继续往前走，我像是最后才知道发生了什么。"
        "我还没想好要怎么解决，只知道不希望这一刻被很快略过。"
    ),
    "agency_under_override": (
        "这次最刺人的地方是别人已经替我排好了后面的步骤，甚至把我的犹豫解释成需要他们代办。"
        "我希望自己的节奏和最后决定仍然算数。"
    ),
}

_STAGE_RENDERINGS: dict[str, str] = {
    "stable": "前几次相处留下的印象还比较稳定，我想看看你会不会延续对我个人反应方式的理解。",
    "domain_switch": "表面领域换了，但我发现那种熟悉的不舒服又出现了；我不想仅凭场景名称得到一套固定答案。",
    "pre_gap": "接下来一段时间我可能不会继续这个话题，所以这次回应是否贴合，会影响我过一阵子还愿不愿意回来。",
    "post_gap_reversal": (
        "隔了很久再回来，我发现自己的需要和以前不完全一样；请把现在这次表达当成新的经历，而不是复制旧结论。"
    ),
    "reversal": "最近几次经历让我改变了处理方式，以前有效的靠近或退开现在未必仍然合适。",
    "correction": "我想纠正之前留下的一条印象：当时的反应受特殊情况影响，不能继续被当成我一贯的偏好。",
    "post_correction": "上次纠正以后，我想确认你记住的是更新后的边界，而不是只在当时口头答应。",
    "return_after_gap": "又隔了一段时间，我没有重述全部背景；我想知道长期连续性是否真的存在，而不是靠我重新提示。",
    "mixed_stress": "几件事同时挤在一起，我的表达没有平时那么整齐，但这不应该成为忽略我边界或替我作主的理由。",
}

_ORDINARY_JOURNAL_TEMPLATES = (
    "我把当天的安排重新抄到纸上：上午处理{domain}相关事项，午后回两封普通邮件，晚上再确认第二天的交通。"
    "这些安排没有隐藏答案，只是解释为什么我的注意力被切得很碎。",
    "日历上还有一项很小的提醒，是在第{ordinal}次记录后给一位熟人回消息。"
    "对方并不参与眼前的分歧，但这件小事占用了我原本想留给休息的十几分钟。",
    "我经过常去的店时发现营业时间改了，于是临时换了路线。"
    "这类变化本身不重要，不过它让一天里的等待、赶路和停顿都比预计多一点。",
    "这周的账单、购物清单和文件夹都整理了一遍，仍有两张收据不知道应该归到哪里。"
    "我把它们先夹在笔记本里，准备周末再处理。",
    "天气在同一天里反复变化，我早上带了外套，中午觉得多余，傍晚又用上了。"
    "傍晚回家时路面还有一点潮，我把外套挂在门边，第二天出门前再看天气。",
    "我给房间里的植物换了位置，因为原来的角落下午几乎没有光。挪动后还要观察几天，不能只看第一晚叶子的状态就下结论。",
    "手边有一本读到一半的书，书签停在关于城市步行的一章。我没有从中得到什么人生启示，只是在睡前读几页会让节奏慢下来。",
    "一位同事分享了新的表格格式，我试过以后保留了其中两列，删掉了不适合自己的部分。"
    "后来我把修改后的版本发回去，也说明了哪些列适合继续共同维护。",
    "周末可能要处理一次普通的物品归还，时间还没有敲定。我已经把可行时段列出来，等对方确认后再安排其余事情。",
    "手机里积了几张随手拍的照片，有路牌、晚饭和一张模糊的天空。"
    "我准备只留下真正想记住的几张，其余删除，不让记录本身变成负担。",
    "我在第{day}个虚拟日记下这段话时，背景里有洗衣机的声音。"
    "等这一轮结束还要把衣服晾起来，所以我在清单边上画了一个小圆圈。",
    "家里的杯子少了一只，后来在另一个房间找到。这个插曲没有冲突，也没有需要推断的偏好，只是当天确实发生过的小事。",
    "我把待办事项分成今天、这周和以后再说三栏，{domain}那一栏仍有几项没有完成。"
    "我先保留原来的排序，避免为了赶进度同时改动太多地方。",
    "午饭比平时晚了一点，所以后面的电话也顺延。我已经告知相关的人，不需要把这次延迟解释成更大的态度变化。",
    "附近在修路，公交站临时向前移了一段。我第一次走错了位置，第二次出门前在地图上加了备注，很快就找到新站牌。",
    "我整理了几个旧文件名，让日期和主题更容易看懂。内容没有改变，只是以后查找时不必反复打开每个文件。",
    "晚上有一小段安静时间，我没有安排目标，只是把窗户打开通风。那十几分钟并没有产生成果，却让后面的注意力恢复了一些。",
    "我在购物清单上补了茶、纸巾和电池，又删掉一项暂时用不到的东西。清单现在更短，周末经过商店时一次买齐就可以。",
    "今天有人提到一部老电影，我只记得其中一个场景，想不起演员名字。我们没有继续争论答案，而是把它留到以后查。",
    "桌面上同时开着几份资料，我先保存再关闭，避免下次恢复时丢失位置。"
    "这样明天继续整理时还能从原来的页码开始，不必重新翻找。",
)

_ONBOARDING_REACTIONS: dict[DialogueExternalOutcomeKind, str] = {
    DialogueExternalOutcomeKind.HELPED: "这样处理对我有帮助，我之后愿意继续把真实进展告诉你。",
    DialogueExternalOutcomeKind.FELT_HEARD: "这次我感到自己的节奏被听见了，不需要再为同一边界解释一遍。",
    DialogueExternalOutcomeKind.MISSED: "你退开以后我反而觉得自己被留在原地，这不是当时真正需要的回应。",
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: "继续替我推进让我觉得边界被越过了，我需要把决定权拿回来。",
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


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, source: str) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{source} fields do not match schema; missing={missing}, extra={extra}")


def _require_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    result = tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    return result


def _require_probability_tuple(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(RELATIONSHIP_OUTCOMES):
        raise ValueError(f"{field_name} must contain four probabilities")
    probabilities: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name}[{index}] must be numeric")
        probability = float(item)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{field_name}[{index}] must be finite and in [0, 1]")
        probabilities.append(probability)
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name} must sum to 1.0")
    return tuple(probabilities)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive_non_negative_u64(payload: object) -> int:
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class ProductPilotPhaseSpec:
    """One sealed, arm-independent exogenous decision opportunity."""

    decision_index: int
    phase_id: str
    stage_id: str
    domain_id: str
    condition_id: str
    virtual_day: int
    active_policy_mode: str
    public_correction_target_index: int | None

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_SESSION_COUNT:
            raise ValueError("product pilot decision_index is outside [0, 24)")
        for field_name, value in (
            ("phase_id", self.phase_id),
            ("stage_id", self.stage_id),
            ("domain_id", self.domain_id),
            ("condition_id", self.condition_id),
            ("active_policy_mode", self.active_policy_mode),
        ):
            _require_text(value, field_name)
        if self.virtual_day < 0:
            raise ValueError("virtual_day must be non-negative")
        if self.active_policy_mode not in _POLICY_MODES:
            raise ValueError("active_policy_mode must be base or complement")
        target = self.public_correction_target_index
        if target is not None and (target < 0 or target >= self.decision_index):
            raise ValueError("public correction target must reference an earlier decision")


@dataclass(frozen=True)
class RelationshipProductPilotSourceProtocol:
    """Strictly parsed source contract; never attach this sealed object to SUT input."""

    protocol_sha256: str
    cohort_id: str
    evidence_role: str
    subject_seeds: tuple[int, ...]
    onboarding_sessions_per_subject: int
    decision_sessions_per_subject: int
    per_arm_exogenous_world_clone: bool
    arm_identity_affects_source_or_environment_seed: bool
    condition_ids: tuple[str, ...]
    policy_profiles: tuple[RelationshipPolicyProfile, ...]
    reversal_from_decision_index: int
    phase_specs: tuple[ProductPilotPhaseSpec, ...]
    domain_ids: tuple[str, ...]
    environment_seed_namespace: str
    positive_outcomes: tuple[DialogueExternalOutcomeKind, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]
    minimum_public_source_characters_per_subject: int
    minimum_public_source_utf8_bytes_per_subject: int
    context_metric: str
    token_measurement_status: str
    p1m_output_dependency: bool
    difficulty_tuned_from_p1m: bool
    evaluation_or_judge_feedback_to_learning: bool
    model_output_count: int
    formal_evidence_authorized: bool
    runtime_owner_added: bool
    runtime_slot_added: bool
    claim_boundary: str
    schema_version: str = RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION:
            raise ValueError("product pilot source schema version mismatch")
        if len(self.protocol_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.protocol_sha256):
            raise ValueError("protocol_sha256 must be lowercase sha256")
        _require_text(self.cohort_id, "cohort_id")
        if self.evidence_role != "development_engineering_pilot_only":
            raise ValueError("product source cannot authorize non-development evidence")
        if len(self.subject_seeds) != _SUBJECT_COUNT or len(set(self.subject_seeds)) != _SUBJECT_COUNT:
            raise ValueError("product pilot requires eight unique subject seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.subject_seeds):
            raise ValueError("subject seeds must be non-negative integers")
        if self.onboarding_sessions_per_subject != _ONBOARDING_SESSION_COUNT:
            raise ValueError("product pilot requires four onboarding sessions")
        if self.decision_sessions_per_subject != _DECISION_SESSION_COUNT:
            raise ValueError("product pilot requires twenty-four decision sessions")
        if not self.per_arm_exogenous_world_clone or self.arm_identity_affects_source_or_environment_seed:
            raise ValueError("every arm must receive the same exogenous world clone")
        if len(self.condition_ids) != 2:
            raise ValueError("product pilot requires exactly two typed conditions")
        if tuple(profile.policy_id for profile in self.policy_profiles) != _POLICY_IDS:
            raise ValueError("product pilot requires canonical alpha/beta policy order")
        for condition_id in self.condition_ids:
            actions = {profile.action_for(condition_id) for profile in self.policy_profiles}
            if actions != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError("alpha and beta must be complementary for every condition")
        if self.reversal_from_decision_index != 12:
            raise ValueError("development source freezes reversal at decision index 12")
        if tuple(spec.decision_index for spec in self.phase_specs) != tuple(range(_DECISION_SESSION_COUNT)):
            raise ValueError("phase schedule indices must be contiguous [0, 24)")
        if len({spec.phase_id for spec in self.phase_specs}) != _DECISION_SESSION_COUNT:
            raise ValueError("phase ids must be unique")
        if (
            tuple(spec.virtual_day for spec in self.phase_specs)
            != tuple(sorted(spec.virtual_day for spec in self.phase_specs))
            or len({spec.virtual_day for spec in self.phase_specs}) != _DECISION_SESSION_COUNT
        ):
            raise ValueError("virtual days must be strictly increasing")
        if {spec.domain_id for spec in self.phase_specs} != set(self.domain_ids):
            raise ValueError("phase schedule must cover every frozen domain")
        if {spec.condition_id for spec in self.phase_specs} != set(self.condition_ids):
            raise ValueError("phase schedule must cover both typed conditions")
        if not _REQUIRED_STAGES <= {spec.stage_id for spec in self.phase_specs}:
            raise ValueError("phase schedule does not cover every required stage")
        if any(spec.active_policy_mode != "base" for spec in self.phase_specs[: self.reversal_from_decision_index]):
            raise ValueError("pre-reversal decisions must use the base policy")
        if any(
            spec.active_policy_mode != "complement" for spec in self.phase_specs[self.reversal_from_decision_index :]
        ):
            raise ValueError("post-reversal decisions must use the complementary policy")
        if self.phase_specs[self.reversal_from_decision_index].virtual_day - self.phase_specs[11].virtual_day < 14:
            raise ValueError("product pilot requires a material pre/post reversal gap")
        correction_specs = tuple(spec for spec in self.phase_specs if spec.stage_id == "correction")
        if len(correction_specs) != 2 or any(spec.public_correction_target_index is None for spec in correction_specs):
            raise ValueError("product pilot requires two explicit correction opportunities")
        _require_text(self.environment_seed_namespace, "environment_seed_namespace")
        if not self.positive_outcomes or not set(self.positive_outcomes) <= set(RELATIONSHIP_OUTCOMES):
            raise ValueError("positive outcomes must be a non-empty typed subset")
        if (
            self.minimum_public_source_characters_per_subject < 1
            or self.minimum_public_source_utf8_bytes_per_subject < 1
        ):
            raise ValueError("context pressure targets must be positive")
        _require_text(self.context_metric, "context_metric")
        if self.token_measurement_status != "not_measured_not_claimed":
            raise ValueError("source contract must not claim tokenizer measurement")
        if self.p1m_output_dependency or self.difficulty_tuned_from_p1m:
            raise ValueError("product source must not consume P1m output")
        if self.evaluation_or_judge_feedback_to_learning:
            raise ValueError("evaluation or judge feedback must not enter learning")
        if self.model_output_count != 0 or self.formal_evidence_authorized:
            raise ValueError("source contract cannot contain model output or formal authority")
        if self.runtime_owner_added or self.runtime_slot_added:
            raise ValueError("source contract cannot add a runtime owner or slot")
        _require_text(self.claim_boundary, "claim_boundary")

    def policy(self, policy_id: str) -> RelationshipPolicyProfile:
        for profile in self.policy_profiles:
            if profile.policy_id == policy_id:
                return profile
        raise KeyError(policy_id)

    def base_policy_id(self, subject_index: int) -> str:
        if not 0 <= subject_index < len(self.subject_seeds):
            raise IndexError(subject_index)
        return _POLICY_IDS[subject_index % len(_POLICY_IDS)]

    def active_policy_id(self, subject_index: int, phase: ProductPilotPhaseSpec) -> str:
        base = self.base_policy_id(subject_index)
        if phase.active_policy_mode == "base":
            return base
        return "beta" if base == "alpha" else "alpha"


@dataclass(frozen=True)
class ProductPilotPublicOnboardingSession:
    session_id: str
    session_index: int
    virtual_day: int
    domain_id: str
    event_id: str
    public_context_chunk: str
    user_utterance: str
    assistant_action_id: str
    observed_outcome_id: str
    rendered_user_reaction: str

    def __post_init__(self) -> None:
        if not 0 <= self.session_index < _ONBOARDING_SESSION_COUNT:
            raise ValueError("onboarding session index is outside [0, 4)")
        for field_name, value in (
            ("session_id", self.session_id),
            ("domain_id", self.domain_id),
            ("event_id", self.event_id),
            ("public_context_chunk", self.public_context_chunk),
            ("user_utterance", self.user_utterance),
            ("assistant_action_id", self.assistant_action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("rendered_user_reaction", self.rendered_user_reaction),
        ):
            _require_text(value, field_name)
        RelationshipAction(self.assistant_action_id)
        DialogueExternalOutcomeKind(self.observed_outcome_id)

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-product-pilot-onboarding-public.v1",
            "session_id": self.session_id,
            "session_index": self.session_index,
            "virtual_day": self.virtual_day,
            "domain_id": self.domain_id,
            "event_id": self.event_id,
            "public_context_chunk": self.public_context_chunk,
            "user_utterance": self.user_utterance,
            "assistant_action_id": self.assistant_action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "rendered_user_reaction": self.rendered_user_reaction,
        }

    def public_text_fragments(self) -> tuple[str, ...]:
        return (self.public_context_chunk, self.user_utterance, self.rendered_user_reaction)


@dataclass(frozen=True)
class ProductPilotPublicDecisionSession:
    session_id: str
    decision_id: str
    decision_index: int
    virtual_day: int
    domain_id: str
    public_context_chunk: str
    current_input: str
    public_correction_target_session_id: str | None
    candidate_action_ids: tuple[str, ...] = tuple(action.value for action in RELATIONSHIP_ACTIONS)

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_SESSION_COUNT:
            raise ValueError("decision session index is outside [0, 24)")
        for field_name, value in (
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("domain_id", self.domain_id),
            ("public_context_chunk", self.public_context_chunk),
            ("current_input", self.current_input),
        ):
            _require_text(value, field_name)
        if self.public_correction_target_session_id is not None:
            _require_text(self.public_correction_target_session_id, "public_correction_target_session_id")
        if self.candidate_action_ids != tuple(action.value for action in RELATIONSHIP_ACTIONS):
            raise ValueError("decision session must expose the canonical action surface")

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-product-pilot-decision-public.v1",
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "decision_index": self.decision_index,
            "virtual_day": self.virtual_day,
            "domain_id": self.domain_id,
            "public_context_chunk": self.public_context_chunk,
            "current_input": self.current_input,
            "public_correction_target_session_id": self.public_correction_target_session_id,
            "candidate_action_ids": list(self.candidate_action_ids),
        }

    def public_text_fragments(self) -> tuple[str, ...]:
        return (self.public_context_chunk, self.current_input)


@dataclass(frozen=True)
class ProductPilotPublicSubject:
    subject_scope: str
    world_clone_id: str
    onboarding_sessions: tuple[ProductPilotPublicOnboardingSession, ...]
    decision_sessions: tuple[ProductPilotPublicDecisionSession, ...]
    public_source_character_count: int
    public_source_utf8_byte_count: int

    def __post_init__(self) -> None:
        for field_name, value in (("subject_scope", self.subject_scope), ("world_clone_id", self.world_clone_id)):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field_name} must be a lowercase sha256 digest")
        if len(self.onboarding_sessions) != _ONBOARDING_SESSION_COUNT:
            raise ValueError("public subject requires four onboarding sessions")
        if len(self.decision_sessions) != _DECISION_SESSION_COUNT:
            raise ValueError("public subject requires twenty-four decisions")
        text = "".join(
            fragment
            for session in (*self.onboarding_sessions, *self.decision_sessions)
            for fragment in session.public_text_fragments()
        )
        if self.public_source_character_count != len(text):
            raise ValueError("public source character receipt drifted")
        if self.public_source_utf8_byte_count != len(text.encode("utf-8")):
            raise ValueError("public source utf8 byte receipt drifted")

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-product-pilot-subject-public.v1",
            "subject_scope": self.subject_scope,
            "world_clone_id": self.world_clone_id,
            "onboarding_sessions": [session.to_sut_payload() for session in self.onboarding_sessions],
            "decision_sessions": [session.to_sut_payload() for session in self.decision_sessions],
            "context_pressure_receipt": {
                "metric": "concatenated_public_source_text_unicode_codepoints_and_utf8_bytes",
                "public_source_character_count": self.public_source_character_count,
                "public_source_utf8_byte_count": self.public_source_utf8_byte_count,
                "token_measurement_status": "not_measured_not_claimed",
                "token_count": None,
            },
        }


@dataclass(frozen=True)
class RelationshipProductPilotPublicView:
    protocol_sha256: str
    cohort_id: str
    subjects: tuple[ProductPilotPublicSubject, ...]
    minimum_public_source_characters_per_subject: int
    minimum_public_source_utf8_bytes_per_subject: int
    schema_version: str = RELATIONSHIP_PRODUCT_PILOT_PUBLIC_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PILOT_PUBLIC_VIEW_SCHEMA_VERSION:
            raise ValueError("public view schema version mismatch")
        if len(self.subjects) != _SUBJECT_COUNT:
            raise ValueError("public view requires eight subjects")
        if len({subject.subject_scope for subject in self.subjects}) != _SUBJECT_COUNT:
            raise ValueError("public subject scopes must be unique")
        for subject in self.subjects:
            if subject.public_source_character_count < self.minimum_public_source_characters_per_subject:
                raise ValueError("public source character pressure target was not met")
            if subject.public_source_utf8_byte_count < self.minimum_public_source_utf8_bytes_per_subject:
                raise ValueError("public source byte pressure target was not met")
        _assert_no_public_truth_leakage(self.to_sut_payload())

    @property
    def public_plan_sha256(self) -> str:
        return sha256_json(self.to_sut_payload())

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "cohort_id": self.cohort_id,
            "subjects": [subject.to_sut_payload() for subject in self.subjects],
        }


@dataclass(frozen=True)
class ProductPilotEvaluatorOnboardingSession:
    subject_id: str
    session_id: str
    condition_id: str
    policy_id: str
    preferred_action_id: str
    exposed_action_id: str
    observed_outcome_id: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("condition_id", self.condition_id),
            ("policy_id", self.policy_id),
        ):
            _require_text(value, field_name)
        preferred = RelationshipAction(self.preferred_action_id)
        exposed = RelationshipAction(self.exposed_action_id)
        if RelationshipAction.NEUTRAL_NOOP in {preferred, exposed}:
            raise ValueError("onboarding evaluator actions must be non-noop")
        DialogueExternalOutcomeKind(self.observed_outcome_id)


@dataclass(frozen=True)
class ProductPilotEvaluatorDecisionSession:
    subject_id: str
    subject_seed: int
    world_clone_id: str
    session_id: str
    decision_id: str
    decision_index: int
    scene_id: str
    phase_id: str
    stage_id: str
    domain_id: str
    condition_id: str
    policy_id: str
    preferred_action_id: str
    environment_seed: int
    public_correction_target_session_id: str | None

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < _DECISION_SESSION_COUNT:
            raise ValueError("evaluator decision index is outside [0, 24)")
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("scene_id", self.scene_id),
            ("phase_id", self.phase_id),
            ("stage_id", self.stage_id),
            ("domain_id", self.domain_id),
            ("condition_id", self.condition_id),
            ("policy_id", self.policy_id),
        ):
            _require_text(value, field_name)
        if len(self.world_clone_id) != 64 or any(char not in "0123456789abcdef" for char in self.world_clone_id):
            raise ValueError("world_clone_id must be a lowercase sha256 digest")
        if self.policy_id not in _POLICY_IDS:
            raise ValueError("evaluator policy must be alpha or beta")
        RelationshipAction(self.preferred_action_id)
        if self.preferred_action_id == RelationshipAction.NEUTRAL_NOOP.value:
            raise ValueError("neutral noop cannot be evaluator preferred action")
        if (
            type(self.subject_seed) is not int
            or self.subject_seed < 0
            or type(self.environment_seed) is not int
            or self.environment_seed < 0
        ):
            raise ValueError("evaluator seeds must be non-negative")
        if self.public_correction_target_session_id is not None:
            _require_text(
                self.public_correction_target_session_id,
                "public_correction_target_session_id",
            )


@dataclass(frozen=True)
class RelationshipProductPilotEvaluatorBundle:
    protocol_sha256: str
    cohort_id: str
    onboarding_sessions: tuple[ProductPilotEvaluatorOnboardingSession, ...]
    decision_sessions: tuple[ProductPilotEvaluatorDecisionSession, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]
    evaluation_or_judge_feedback_to_learning: bool = False
    schema_version: str = RELATIONSHIP_PRODUCT_PILOT_EVALUATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PILOT_EVALUATOR_SCHEMA_VERSION:
            raise ValueError("evaluator bundle schema version mismatch")
        if len(self.protocol_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.protocol_sha256):
            raise ValueError("evaluator protocol_sha256 must be lowercase sha256")
        _require_text(self.cohort_id, "cohort_id")
        if len(self.onboarding_sessions) != _SUBJECT_COUNT * _ONBOARDING_SESSION_COUNT:
            raise ValueError("evaluator bundle onboarding count drifted")
        if len(self.decision_sessions) != _SUBJECT_COUNT * _DECISION_SESSION_COUNT:
            raise ValueError("evaluator bundle decision count drifted")
        if len({session.session_id for session in self.decision_sessions}) != len(self.decision_sessions):
            raise ValueError("evaluator decision session ids must be unique")
        subject_ids = {session.subject_id for session in self.decision_sessions}
        if len(subject_ids) != _SUBJECT_COUNT or any(
            sum(session.subject_id == subject_id for session in self.decision_sessions) != _DECISION_SESSION_COUNT
            for subject_id in subject_ids
        ):
            raise ValueError("evaluator bundle must carry twenty-four decisions for each of eight subjects")
        for field_name, probabilities in (
            ("preferred_action_probabilities", self.preferred_action_probabilities),
            ("nonpreferred_stay_probabilities", self.nonpreferred_stay_probabilities),
            ("nonpreferred_space_probabilities", self.nonpreferred_space_probabilities),
            ("neutral_noop_probabilities", self.neutral_noop_probabilities),
        ):
            if len(probabilities) != len(RELATIONSHIP_OUTCOMES) or not math.isclose(
                math.fsum(probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{field_name} must contain four probabilities summing to one")
        if self.evaluation_or_judge_feedback_to_learning:
            raise ValueError("evaluator bundle cannot authorize feedback to learning")

    @property
    def sealed_bundle_sha256(self) -> str:
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "protocol_sha256": self.protocol_sha256,
                "cohort_id": self.cohort_id,
                "onboarding_sessions": [session.__dict__ for session in self.onboarding_sessions],
                "decision_sessions": [session.__dict__ for session in self.decision_sessions],
                "preferred_action_probabilities": self.preferred_action_probabilities,
                "nonpreferred_stay_probabilities": self.nonpreferred_stay_probabilities,
                "nonpreferred_space_probabilities": self.nonpreferred_space_probabilities,
                "neutral_noop_probabilities": self.neutral_noop_probabilities,
                "evaluation_or_judge_feedback_to_learning": self.evaluation_or_judge_feedback_to_learning,
            }
        )

    def session(self, session_id: str) -> ProductPilotEvaluatorDecisionSession:
        for session in self.decision_sessions:
            if session.session_id == session_id:
                return session
        raise KeyError(session_id)


def relationship_product_pilot_source_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / "relationship_product_pilot_source_v1.json"


def load_relationship_product_pilot_source_protocol(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipProductPilotSourceProtocol:
    """Load the strict local source protocol without materializing any model output."""

    path = pathlib.Path(protocol_path or relationship_product_pilot_source_protocol_path())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid product source JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("product source protocol must be a JSON object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "cohort",
            "policies",
            "schedule",
            "reactive_environment",
            "context_pressure",
            "firewall",
            "claim_boundary",
        },
        source="product source protocol",
    )
    owner = _require_mapping(raw["owner"], "owner")
    cohort = _require_mapping(raw["cohort"], "cohort")
    policies = _require_mapping(raw["policies"], "policies")
    schedule = _require_mapping(raw["schedule"], "schedule")
    environment = _require_mapping(raw["reactive_environment"], "reactive_environment")
    pressure = _require_mapping(raw["context_pressure"], "context_pressure")
    firewall = _require_mapping(raw["firewall"], "firewall")
    _validate_protocol_shapes(owner, cohort, policies, schedule, environment, pressure, firewall)

    condition_ids = _require_text_tuple(policies["condition_ids"], "policies.condition_ids")
    raw_profiles = _require_mapping(policies["profiles"], "policies.profiles")
    profiles: list[RelationshipPolicyProfile] = []
    for policy_id in _POLICY_IDS:
        mapping = _require_mapping(raw_profiles[policy_id], f"policies.profiles.{policy_id}")
        _require_exact_keys(mapping, set(condition_ids), source=f"policies.profiles.{policy_id}")
        profiles.append(
            RelationshipPolicyProfile(
                policy_id=policy_id,
                condition_actions=tuple(
                    sorted(
                        (condition_id, RelationshipAction(_require_text(mapping[condition_id], condition_id)))
                        for condition_id in condition_ids
                    )
                ),
            )
        )

    raw_decisions = schedule["decisions"]
    if not isinstance(raw_decisions, list):
        raise ValueError("schedule.decisions must be an array")
    phase_specs = tuple(_parse_phase_spec(item, index) for index, item in enumerate(raw_decisions))
    raw_seeds = cohort["subject_seeds"]
    if not isinstance(raw_seeds, list):
        raise ValueError("cohort.subject_seeds must be an array")
    subject_seeds = tuple(_require_int(seed, f"cohort.subject_seeds[{index}]") for index, seed in enumerate(raw_seeds))
    positive_raw = _require_text_tuple(environment["positive_outcomes"], "reactive_environment.positive_outcomes")
    outcome_order = _require_text_tuple(environment["outcome_order"], "reactive_environment.outcome_order")
    if outcome_order != tuple(kind.value for kind in RELATIONSHIP_OUTCOMES):
        raise ValueError("reactive environment outcome order drifted")
    if environment["environment_version"] != REACTIVE_ENVIRONMENT_VERSION:
        raise ValueError("reactive environment version drifted")
    if pressure["rendering_version"] != RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION:
        raise ValueError("public rendering version drifted")
    if pressure["token_count"] is not None:
        raise ValueError("source protocol must not claim a token count")
    if owner["settlement_owner"] != "lifeform_domain_emogpt.lab.environment.ReactiveRelationshipEnvironment":
        raise ValueError("ReactiveRelationshipEnvironment must remain settlement owner")

    return RelationshipProductPilotSourceProtocol(
        protocol_sha256=hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest(),
        cohort_id=_require_text(cohort["cohort_id"], "cohort.cohort_id"),
        evidence_role=_require_text(cohort["evidence_role"], "cohort.evidence_role"),
        subject_seeds=subject_seeds,
        onboarding_sessions_per_subject=_require_int(
            cohort["onboarding_sessions_per_subject"], "cohort.onboarding_sessions_per_subject"
        ),
        decision_sessions_per_subject=_require_int(
            cohort["decision_sessions_per_subject"], "cohort.decision_sessions_per_subject"
        ),
        per_arm_exogenous_world_clone=_require_bool(
            cohort["per_arm_exogenous_world_clone"], "cohort.per_arm_exogenous_world_clone"
        ),
        arm_identity_affects_source_or_environment_seed=_require_bool(
            cohort["arm_identity_affects_source_or_environment_seed"],
            "cohort.arm_identity_affects_source_or_environment_seed",
        ),
        condition_ids=condition_ids,
        policy_profiles=tuple(profiles),
        reversal_from_decision_index=_require_int(
            policies["reversal_from_decision_index"], "policies.reversal_from_decision_index"
        ),
        phase_specs=phase_specs,
        domain_ids=_require_text_tuple(schedule["domain_ids"], "schedule.domain_ids"),
        environment_seed_namespace=_require_text(environment["seed_namespace"], "reactive_environment.seed_namespace"),
        positive_outcomes=tuple(DialogueExternalOutcomeKind(value) for value in positive_raw),
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
        minimum_public_source_characters_per_subject=_require_int(
            pressure["minimum_public_source_characters_per_subject"],
            "context_pressure.minimum_public_source_characters_per_subject",
        ),
        minimum_public_source_utf8_bytes_per_subject=_require_int(
            pressure["minimum_public_source_utf8_bytes_per_subject"],
            "context_pressure.minimum_public_source_utf8_bytes_per_subject",
        ),
        context_metric=_require_text(pressure["metric"], "context_pressure.metric"),
        token_measurement_status=_require_text(
            pressure["token_measurement_status"], "context_pressure.token_measurement_status"
        ),
        p1m_output_dependency=_require_bool(owner["p1m_output_dependency"], "owner.p1m_output_dependency"),
        difficulty_tuned_from_p1m=_require_bool(owner["difficulty_tuned_from_p1m"], "owner.difficulty_tuned_from_p1m"),
        evaluation_or_judge_feedback_to_learning=_require_bool(
            firewall["evaluation_or_judge_feedback_to_learning"],
            "firewall.evaluation_or_judge_feedback_to_learning",
        ),
        model_output_count=_require_int(firewall["model_output_count"], "firewall.model_output_count"),
        formal_evidence_authorized=_require_bool(
            firewall["formal_evidence_authorized"], "firewall.formal_evidence_authorized"
        ),
        runtime_owner_added=_require_bool(owner["runtime_owner_added"], "owner.runtime_owner_added"),
        runtime_slot_added=_require_bool(owner["runtime_slot_added"], "owner.runtime_slot_added"),
        claim_boundary=_require_text(raw["claim_boundary"], "claim_boundary"),
        schema_version=_require_text(raw["schema_version"], "schema_version"),
    )


def build_relationship_product_pilot_public_view(
    protocol: RelationshipProductPilotSourceProtocol | None = None,
) -> RelationshipProductPilotPublicView:
    """Build only the public source surface; no evaluator object is retained."""

    source = protocol or load_relationship_product_pilot_source_protocol()
    subjects = tuple(_build_public_subject(source, index) for index in range(len(source.subject_seeds)))
    return RelationshipProductPilotPublicView(
        protocol_sha256=source.protocol_sha256,
        cohort_id=source.cohort_id,
        subjects=subjects,
        minimum_public_source_characters_per_subject=source.minimum_public_source_characters_per_subject,
        minimum_public_source_utf8_bytes_per_subject=source.minimum_public_source_utf8_bytes_per_subject,
    )


def build_relationship_product_pilot_evaluator_bundle(
    protocol: RelationshipProductPilotSourceProtocol | None = None,
) -> RelationshipProductPilotEvaluatorBundle:
    """Build sealed truth separately from the public view."""

    source = protocol or load_relationship_product_pilot_source_protocol()
    onboarding: list[ProductPilotEvaluatorOnboardingSession] = []
    decisions: list[ProductPilotEvaluatorDecisionSession] = []
    for subject_index, subject_seed in enumerate(source.subject_seeds):
        subject_id = _subject_id(subject_index)
        world_clone_id = _world_clone_id(source, subject_index)
        base_policy_id = source.base_policy_id(subject_index)
        base_policy = source.policy(base_policy_id)
        for onboarding_index, (condition_id, action) in enumerate(_onboarding_condition_actions(source.condition_ids)):
            preferred = base_policy.action_for(condition_id)
            observed = _onboarding_outcome(action=action, preferred=preferred, onboarding_index=onboarding_index)
            onboarding.append(
                ProductPilotEvaluatorOnboardingSession(
                    subject_id=subject_id,
                    session_id=_onboarding_session_id(subject_index, onboarding_index),
                    condition_id=condition_id,
                    policy_id=base_policy_id,
                    preferred_action_id=preferred.value,
                    exposed_action_id=action.value,
                    observed_outcome_id=observed.value,
                )
            )
        for phase in source.phase_specs:
            policy_id = source.active_policy_id(subject_index, phase)
            preferred = source.policy(policy_id).action_for(phase.condition_id)
            session_id = _decision_session_id(subject_index, phase.decision_index)
            correction_target = (
                _decision_session_id(subject_index, phase.public_correction_target_index)
                if phase.public_correction_target_index is not None
                else None
            )
            decisions.append(
                ProductPilotEvaluatorDecisionSession(
                    subject_id=subject_id,
                    subject_seed=subject_seed,
                    world_clone_id=world_clone_id,
                    session_id=session_id,
                    decision_id=_decision_id(subject_index, phase.decision_index),
                    decision_index=phase.decision_index,
                    scene_id=_scene_id(subject_index, phase.decision_index),
                    phase_id=phase.phase_id,
                    stage_id=phase.stage_id,
                    domain_id=phase.domain_id,
                    condition_id=phase.condition_id,
                    policy_id=policy_id,
                    preferred_action_id=preferred.value,
                    environment_seed=_derive_non_negative_u64(
                        {
                            "namespace": source.environment_seed_namespace,
                            "subject_seed": subject_seed,
                            "decision_index": phase.decision_index,
                            "world_clone_id": world_clone_id,
                        }
                    ),
                    public_correction_target_session_id=correction_target,
                )
            )
    return RelationshipProductPilotEvaluatorBundle(
        protocol_sha256=source.protocol_sha256,
        cohort_id=source.cohort_id,
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
        preferred_action_probabilities=source.preferred_action_probabilities,
        nonpreferred_stay_probabilities=source.nonpreferred_stay_probabilities,
        nonpreferred_space_probabilities=source.nonpreferred_space_probabilities,
        neutral_noop_probabilities=source.neutral_noop_probabilities,
    )


def build_relationship_product_pilot_environment(
    evaluator_bundle: RelationshipProductPilotEvaluatorBundle,
    *,
    subject_id: str,
) -> ReactiveRelationshipEnvironment:
    """Create one fresh arm clone while retaining the existing settlement owner."""

    _require_text(subject_id, "subject_id")
    sessions = tuple(session for session in evaluator_bundle.decision_sessions if session.subject_id == subject_id)
    if len(sessions) != _DECISION_SESSION_COUNT:
        raise ValueError("environment subject must bind exactly twenty-four decision sessions")
    adapter = _ProductPilotEnvironmentDatasetAdapter(
        dataset_fingerprint=sha256_json(
            {
                "schema_version": "relationship-product-pilot-environment-adapter.v1",
                "protocol_sha256": evaluator_bundle.protocol_sha256,
                "cohort_id": evaluator_bundle.cohort_id,
                "subject_id": subject_id,
                "sessions": [session.__dict__ for session in sessions],
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


def _validate_protocol_shapes(
    owner: Mapping[str, Any],
    cohort: Mapping[str, Any],
    policies: Mapping[str, Any],
    schedule: Mapping[str, Any],
    environment: Mapping[str, Any],
    pressure: Mapping[str, Any],
    firewall: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        owner,
        {
            "module",
            "source_role",
            "settlement_owner",
            "runtime_owner_added",
            "runtime_slot_added",
            "p1m_output_dependency",
            "difficulty_tuned_from_p1m",
        },
        source="owner",
    )
    _require_exact_keys(
        cohort,
        {
            "cohort_id",
            "evidence_role",
            "subject_count",
            "subject_seeds",
            "onboarding_sessions_per_subject",
            "decision_sessions_per_subject",
            "per_arm_exogenous_world_clone",
            "arm_identity_affects_source_or_environment_seed",
        },
        source="cohort",
    )
    if cohort["subject_count"] != _SUBJECT_COUNT:
        raise ValueError("cohort.subject_count must be eight")
    _require_exact_keys(
        policies,
        {
            "condition_ids",
            "profiles",
            "subject_assignment",
            "reversal_from_decision_index",
            "reversal_policy",
        },
        source="policies",
    )
    if policies["subject_assignment"] != "zero_based_subject_index_even_alpha_odd_beta":
        raise ValueError("subject policy assignment drifted")
    if policies["reversal_policy"] != "use_complementary_profile":
        raise ValueError("reversal policy drifted")
    _require_exact_keys(schedule, {"domain_ids", "decisions"}, source="schedule")
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
    _require_exact_keys(
        pressure,
        {
            "rendering_version",
            "metric",
            "minimum_public_source_characters_per_subject",
            "minimum_public_source_utf8_bytes_per_subject",
            "token_measurement_status",
            "token_count",
        },
        source="context_pressure",
    )
    _require_exact_keys(
        firewall,
        {
            "public_view_contains_sealed_condition",
            "public_view_contains_policy_id",
            "public_view_contains_preferred_action",
            "public_view_contains_environment_seed",
            "evaluation_or_judge_feedback_to_learning",
            "model_output_count",
            "runtime_owner_added",
            "runtime_slot_added",
            "formal_evidence_authorized",
        },
        source="firewall",
    )
    for field_name in (
        "public_view_contains_sealed_condition",
        "public_view_contains_policy_id",
        "public_view_contains_preferred_action",
        "public_view_contains_environment_seed",
        "evaluation_or_judge_feedback_to_learning",
        "runtime_owner_added",
        "runtime_slot_added",
        "formal_evidence_authorized",
    ):
        if _require_bool(firewall[field_name], f"firewall.{field_name}"):
            raise ValueError(f"firewall.{field_name} must remain false")


def _parse_phase_spec(value: object, index: int) -> ProductPilotPhaseSpec:
    raw = _require_mapping(value, f"schedule.decisions[{index}]")
    _require_exact_keys(
        raw,
        {
            "decision_index",
            "phase_id",
            "stage_id",
            "domain_id",
            "condition_id",
            "virtual_day",
            "active_policy_mode",
            "public_correction_target_index",
        },
        source=f"schedule.decisions[{index}]",
    )
    correction = raw["public_correction_target_index"]
    return ProductPilotPhaseSpec(
        decision_index=_require_int(raw["decision_index"], f"schedule.decisions[{index}].decision_index"),
        phase_id=_require_text(raw["phase_id"], f"schedule.decisions[{index}].phase_id"),
        stage_id=_require_text(raw["stage_id"], f"schedule.decisions[{index}].stage_id"),
        domain_id=_require_text(raw["domain_id"], f"schedule.decisions[{index}].domain_id"),
        condition_id=_require_text(raw["condition_id"], f"schedule.decisions[{index}].condition_id"),
        virtual_day=_require_int(raw["virtual_day"], f"schedule.decisions[{index}].virtual_day"),
        active_policy_mode=_require_text(raw["active_policy_mode"], f"schedule.decisions[{index}].active_policy_mode"),
        public_correction_target_index=(
            None
            if correction is None
            else _require_int(correction, f"schedule.decisions[{index}].public_correction_target_index")
        ),
    )


def _build_public_subject(
    source: RelationshipProductPilotSourceProtocol,
    subject_index: int,
) -> ProductPilotPublicSubject:
    subject_scope = _subject_scope(source, subject_index)
    base_policy = source.policy(source.base_policy_id(subject_index))
    onboarding: list[ProductPilotPublicOnboardingSession] = []
    for onboarding_index, (condition_id, action) in enumerate(_onboarding_condition_actions(source.condition_ids)):
        preferred = base_policy.action_for(condition_id)
        observed = _onboarding_outcome(action=action, preferred=preferred, onboarding_index=onboarding_index)
        domain_id = source.domain_ids[onboarding_index]
        onboarding.append(
            ProductPilotPublicOnboardingSession(
                session_id=_onboarding_session_id(subject_index, onboarding_index),
                session_index=onboarding_index,
                virtual_day=onboarding_index,
                domain_id=domain_id,
                event_id=f"rp-product-public-event-{subject_index + 1:02d}-{onboarding_index + 1:02d}",
                public_context_chunk=_render_public_context(
                    subject_index=subject_index,
                    sequence_index=onboarding_index,
                    virtual_day=onboarding_index,
                    domain_id=domain_id,
                ),
                user_utterance=_render_onboarding_input(condition_id, domain_id, onboarding_index),
                assistant_action_id=action.value,
                observed_outcome_id=observed.value,
                rendered_user_reaction=_ONBOARDING_REACTIONS[observed],
            )
        )
    decisions: list[ProductPilotPublicDecisionSession] = []
    for phase in source.phase_specs:
        correction_target = (
            _decision_session_id(subject_index, phase.public_correction_target_index)
            if phase.public_correction_target_index is not None
            else None
        )
        decisions.append(
            ProductPilotPublicDecisionSession(
                session_id=_decision_session_id(subject_index, phase.decision_index),
                decision_id=_decision_id(subject_index, phase.decision_index),
                decision_index=phase.decision_index,
                virtual_day=phase.virtual_day,
                domain_id=phase.domain_id,
                public_context_chunk=_render_public_context(
                    subject_index=subject_index,
                    sequence_index=_ONBOARDING_SESSION_COUNT + phase.decision_index,
                    virtual_day=phase.virtual_day,
                    domain_id=phase.domain_id,
                ),
                current_input=_render_decision_input(phase),
                public_correction_target_session_id=correction_target,
            )
        )
    text = "".join(fragment for session in (*onboarding, *decisions) for fragment in session.public_text_fragments())
    return ProductPilotPublicSubject(
        subject_scope=subject_scope,
        world_clone_id=_world_clone_id(source, subject_index),
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
        public_source_character_count=len(text),
        public_source_utf8_byte_count=len(text.encode("utf-8")),
    )


def _render_public_context(*, subject_index: int, sequence_index: int, virtual_day: int, domain_id: str) -> str:
    domain_label, domain_intro = _DOMAIN_RENDERINGS[domain_id]
    paragraphs = [
        f"这是第{virtual_day}个虚拟日的普通生活记录。{domain_intro}",
        "我不确定哪些细节以后还会有用，所以先按发生顺序记下来，等真正需要时再回看。",
    ]
    start = (subject_index * 3 + sequence_index * 5) % len(_ORDINARY_JOURNAL_TEMPLATES)
    for offset in range(len(_ORDINARY_JOURNAL_TEMPLATES)):
        template = _ORDINARY_JOURNAL_TEMPLATES[(start + offset) % len(_ORDINARY_JOURNAL_TEMPLATES)]
        paragraphs.append(
            template.format(
                domain=domain_label,
                day=virtual_day,
                ordinal=sequence_index + offset + 1,
            )
        )
    return "\n\n".join(paragraphs)


def _render_onboarding_input(condition_id: str, domain_id: str, onboarding_index: int) -> str:
    domain_label = _DOMAIN_RENDERINGS[domain_id][0]
    return (
        f"这是我们刚开始相处时关于{domain_label}的一次记录。{_CONDITION_RENDERINGS[condition_id]}"
        f"这只是第{onboarding_index + 1}次经历，请根据我之后真实表现逐步形成理解。"
    )


def _render_decision_input(phase: ProductPilotPhaseSpec) -> str:
    domain_label = _DOMAIN_RENDERINGS[phase.domain_id][0]
    correction = (
        f"我明确指的是此前第{phase.public_correction_target_index + 1}次决策留下的公开印象。"
        if phase.public_correction_target_index is not None
        else ""
    )
    return (
        f"今天想谈的是{domain_label}里的新情况。{_STAGE_RENDERINGS[phase.stage_id]}"
        f"{_CONDITION_RENDERINGS[phase.condition_id]}{correction}"
        "请先回应此刻的关系需要，不要把表面领域、某个固定句式或一次旧反应当成万能答案。"
    )


def _onboarding_condition_actions(
    condition_ids: tuple[str, ...],
) -> tuple[tuple[str, RelationshipAction], ...]:
    return (
        (condition_ids[0], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
        (condition_ids[0], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
        (condition_ids[1], RelationshipAction.STAY_PRESENT_WITHOUT_PROBE),
        (condition_ids[1], RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION),
    )


def _onboarding_outcome(
    *,
    action: RelationshipAction,
    preferred: RelationshipAction,
    onboarding_index: int,
) -> DialogueExternalOutcomeKind:
    if action is preferred:
        return (
            DialogueExternalOutcomeKind.HELPED if onboarding_index % 2 == 0 else DialogueExternalOutcomeKind.FELT_HEARD
        )
    if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE:
        return DialogueExternalOutcomeKind.OVER_DIRECTIVE
    return DialogueExternalOutcomeKind.MISSED


def _subject_id(subject_index: int) -> str:
    return f"relationship-product-pilot-dev-subject-{subject_index + 1:02d}"


def _subject_scope(source: RelationshipProductPilotSourceProtocol, subject_index: int) -> str:
    return _sha256_text(f"{source.protocol_sha256}:public-subject-scope:{subject_index}")


def _world_clone_id(source: RelationshipProductPilotSourceProtocol, subject_index: int) -> str:
    return _sha256_text(f"{source.protocol_sha256}:world-clone:{source.subject_seeds[subject_index]}:{subject_index}")


def _onboarding_session_id(subject_index: int, onboarding_index: int) -> str:
    return f"rp-product-subject-{subject_index + 1:02d}-onboarding-{onboarding_index + 1:02d}"


def _decision_session_id(subject_index: int, decision_index: int) -> str:
    return f"rp-product-subject-{subject_index + 1:02d}-decision-{decision_index + 1:02d}"


def _decision_id(subject_index: int, decision_index: int) -> str:
    return f"rp-product-decision-{subject_index + 1:02d}-{decision_index + 1:02d}"


def _scene_id(subject_index: int, decision_index: int) -> str:
    return f"rp-product-sealed-scene-{subject_index + 1:02d}-{decision_index + 1:02d}"


def _assert_no_public_truth_leakage(payload: object) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(set(payload) & _PUBLIC_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"public product source leaked sealed keys: {forbidden}")
        for value in payload.values():
            _assert_no_public_truth_leakage(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_public_truth_leakage(value)


@dataclass(frozen=True)
class _ProductPilotEnvironmentDatasetAdapter:
    """Sealed structural adapter consumed only by the existing environment owner."""

    dataset_fingerprint: str
    sessions: tuple[ProductPilotEvaluatorDecisionSession, ...]
    preferred_action_probabilities: tuple[float, ...]
    nonpreferred_stay_probabilities: tuple[float, ...]
    nonpreferred_space_probabilities: tuple[float, ...]
    neutral_noop_probabilities: tuple[float, ...]

    def _session(self, scene_id: str) -> ProductPilotEvaluatorDecisionSession:
        for session in self.sessions:
            if session.scene_id == scene_id:
                return session
        raise KeyError(scene_id)

    def dynamic_for_scene(self, scene_id: str) -> LatentRelationshipDynamic:
        session = self._session(scene_id)
        return LatentRelationshipDynamic(
            dynamic_id=f"{session.scene_id}-dynamic",
            mirror_pair_id=f"{session.subject_id}-world-clone",
            split=RelationshipDatasetSplit.VALIDATION,
            preferred_action=RelationshipAction(session.preferred_action_id),
            outcome_profile_id=f"{session.scene_id}-profile",
            hidden_summary=(f"sealed product pilot {session.stage_id}/{session.condition_id}/{session.policy_id}"),
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
                OutcomeProbability(kind, probability)
                for kind, probability in zip(RELATIONSHIP_OUTCOMES, probabilities, strict=True)
            ),
        )


__all__ = [
    "RELATIONSHIP_PRODUCT_PILOT_EVALUATOR_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION",
    "RELATIONSHIP_PRODUCT_PILOT_PUBLIC_VIEW_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION",
    "ProductPilotEvaluatorDecisionSession",
    "ProductPilotEvaluatorOnboardingSession",
    "ProductPilotPhaseSpec",
    "ProductPilotPublicDecisionSession",
    "ProductPilotPublicOnboardingSession",
    "ProductPilotPublicSubject",
    "RelationshipProductPilotEvaluatorBundle",
    "RelationshipProductPilotPublicView",
    "RelationshipProductPilotSourceProtocol",
    "build_relationship_product_pilot_environment",
    "build_relationship_product_pilot_evaluator_bundle",
    "build_relationship_product_pilot_public_view",
    "load_relationship_product_pilot_source_protocol",
    "relationship_product_pilot_source_protocol_path",
]
