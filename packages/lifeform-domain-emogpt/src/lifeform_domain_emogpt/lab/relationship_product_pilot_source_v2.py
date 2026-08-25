"""Independent revision owner for the Relationship product-pilot source.

The legacy :mod:`relationship_product_pilot_source` module is byte-pinned by
completed Product Horizon campaigns.  This module therefore owns later
independent source materialization without changing that legacy owner.  The
published source-v2 JSON remains an immutable semantic input; source-v3 binds
that input to this owner and is the only revision in the live registry below.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_domain_emogpt.lab.dataset import RelationshipPolicyProfile
from lifeform_domain_emogpt.lab.environment import REACTIVE_ENVIRONMENT_VERSION, ReactiveRelationshipEnvironment
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    ProductPilotEvaluatorDecisionSession,
    ProductPilotEvaluatorOnboardingSession,
    ProductPilotPublicDecisionSession,
    ProductPilotPublicOnboardingSession,
    ProductPilotPublicSubject,
    RelationshipProductPilotEvaluatorBundle,
    RelationshipProductPilotPublicView,
    build_relationship_product_pilot_environment as _build_legacy_snapshot_environment,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2 = "relationship-product-pilot-source.v2"
RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3 = "relationship-product-pilot-source.v3"
RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION = RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3
RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION = "relationship-product-pilot-public-renderer.v2"

_OWNER_MODULE = "lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2"
_ARCHIVED_OWNER_MODULE = "lifeform_domain_emogpt.lab.relationship_product_pilot_source"
_PROTOCOL_FILENAME = "relationship_product_pilot_source_v3.json"
_ARCHIVED_PROTOCOL_FILENAME = "relationship_product_pilot_source_v2.json"
_ARCHIVED_PROTOCOL_RAW_SHA256 = "ef35ba2637e53c96c2ed86b16a8bb69281cc10e1a2f30c81112a66841f4b23f7"
_ARCHIVED_PROTOCOL_RAW_BYTES = 13031
_ARCHIVED_PROTOCOL_ID = "9f4ad004f9332a705d3231cb9a3394b4922417878f18487806b0f030bb863161"
_SOURCE_ROLE = "independent_development_product_pilot_domain_source_only"
_COHORT_ID = "relationship-product-pilot-independent-development-20260824"
_IDENTITY_NAMESPACE = "relationship-product-pilot-independent-v2"
_ENVIRONMENT_SEED_NAMESPACE = "relationship-product-pilot-independent-reactive-seeds-v2"
_SUBJECT_SEEDS = tuple(range(2026092401, 2026092409))
_DOMAIN_IDS = (
    "maker_studio",
    "neighborhood_archive",
    "shared_kitchen",
    "learning_circle",
    "travel_planning",
    "community_garden",
)
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


@dataclass(frozen=True)
class IndependentProductPilotSourceRevision:
    """One explicitly routed revision owned by this module."""

    schema_version: str
    protocol_filename: str
    archived_protocol_filename: str
    archived_protocol_raw_sha256: str
    archived_protocol_raw_bytes: int
    archived_protocol_id: str


RELATIONSHIP_PRODUCT_PILOT_SOURCE_V2_REGISTRY: Mapping[
    str, IndependentProductPilotSourceRevision
] = MappingProxyType(
    {
        RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3: IndependentProductPilotSourceRevision(
            schema_version=RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
            protocol_filename=_PROTOCOL_FILENAME,
            archived_protocol_filename=_ARCHIVED_PROTOCOL_FILENAME,
            archived_protocol_raw_sha256=_ARCHIVED_PROTOCOL_RAW_SHA256,
            archived_protocol_raw_bytes=_ARCHIVED_PROTOCOL_RAW_BYTES,
            archived_protocol_id=_ARCHIVED_PROTOCOL_ID,
        )
    }
)


_DOMAIN_RENDERINGS: dict[str, tuple[str, str]] = {
    "maker_studio": (
        "手作工坊",
        "工坊正在并行处理材料登记、设备预约、作品交接和开放日准备，许多小决定分散在不同记录里，需要区分共同事项与个人经历。",
    ),
    "neighborhood_archive": (
        "街区档案整理",
        "街区档案小组最近在核对旧照片、口述记录、借阅清单和展签版本，资料来源很多，时间线也常需要重新确认。",
    ),
    "shared_kitchen": (
        "共享厨房",
        "共享厨房这一阵要协调食材到货、台面轮换、卫生记录和临时菜单，日常细节密集，却不代表每个人对同一安排有相同感受。",
    ),
    "learning_circle": (
        "学习小组",
        "学习小组在交替准备阅读笔记、练习反馈、场地预订和阶段复盘，信息会跨几次聚会累积，不能只从一次发言推断长期倾向。",
    ),
    "travel_planning": (
        "结伴出行筹备",
        "几个人正在核对换乘、住宿、行李和可变日期，计划会随着公开信息更新，事实变动与关系体验需要分别记录。",
    ),
    "community_garden": (
        "公共花园维护",
        "公共花园最近要安排浇灌、工具归还、苗圃记录和天气备选方案，协作节奏不固定，零散消息很容易留下不同理解。",
    ),
}

_STAGE_RENDERINGS: dict[str, str] = {
    "stable": "前几段记录的背景比较连续，这一次仍需要作为独立的新事件被理解。",
    "domain_switch": "事情换到了另一类日常活动里，但关系上的感受并不由活动名称直接决定。",
    "pre_gap": "这次之后会有一段较长空档，因此我把当下经历单独记清，避免以后用模糊印象代替。",
    "post_gap_reversal": "间隔之后重新记录时，我发现当前感受和早期样本并不相同，旧概括不能直接覆盖这一次。",
    "reversal": "近期连续发生的事情改变了原先的关系经验，需要允许新记录修正此前形成的概括。",
    "correction": "这里包含一次明确更正：早先某段经历有特殊背景，不应继续代表稳定倾向。",
    "post_correction": "更正已经发生，现在这段新记录可以检验更新后的理解是否保持一致。",
    "return_after_gap": "又一次较长间隔结束后，我没有复制全部旧背景，而是记录当前真实发生的部分。",
    "mixed_stress": "多个普通压力同时出现，使表述比平时零散；这些噪声不应替代对关系事实本身的判断。",
}

_CONDITION_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "connection_under_exclusion": (
        "关键过程已经走完，我直到后来才从零散信息里拼出全貌；被排在过程之外让我很失落。",
        "大家连续交流了几轮，我的名字却只在决定完成后才出现；那种后知后觉让我怀疑自己是否算参与者。",
        "信息沿着其他人的对话向前推进，却没有经过我这里；等我发现时，彼此已经共享了我缺失的背景。",
        "事情已经进入下一阶段，我才知道前面还有一段共同讨论；真正刺痛我的是自己像被留在关系之外。",
        "其他人都知道最近发生了什么，而我面对结果时没有相同的来龙去脉；这种落差让我感到疏离。",
        "计划改变以后，消息在几个人之间补齐了，唯独我仍按旧信息理解；被遗漏本身比改动更难受。",
        "我是在旁支话题里偶然听见核心进展的，之前没有人把我算进那段共同经历；这让我觉得自己的位置很模糊。",
        "讨论结束后才有人顺带告诉我结论，我没有经历中间的交流；像局外人一样接收结果让我心里发沉。",
        "更新被默认成大家都已知的背景，可我从未收到那部分信息；缺失的不只是事实，还有共同参与的感觉。",
        "关键细节只在另一个小范围里流转，等它影响到我时已经无法追上前面的语境；我感到自己被落下了。",
        "集体节奏继续向前，我却没有被带入那段衔接；看到别人自然接上话题时，我才意识到自己的缺席。",
        "我原以为大家仍在一起摸索，后来才发现其他人早已形成共同理解；这份不对称让我很孤单。",
        "相关的人已经交换过近况，轮到我时只剩压缩后的结论；我在意的是自己没有被纳入那段连接。",
        "后续安排都接上了，只有我需要从头追问发生过什么；这种被放在最后的位置让我很受挫。",
    ),
    "agency_under_override": (
        "下一步已经被别人直接排进日程，而我还没有表达选择；这种被代为决定让我不舒服。",
        "我的犹豫被解释成需要别人替我定案，可那并不是我给出的意思；决定权像是在过程中消失了。",
        "别人把一项建议当成了我的最终选择并继续推进，我却没有完成确认；这让我觉得自己的意愿没有被计算。",
        "还没有询问我的判断，后续步骤就已经以我的名义确定；被跳过的正是本应由我作出的选择。",
        "时间表写好了，相关人也收到通知，但我此前只说仍在考虑；这种落差让我感觉自己失去了主导权。",
        "原本需要我确认的事项被直接标记为已同意，我面对的是别人替我完成的结论；这让我很别扭。",
        "我的选择被压缩成了别人认为更方便的方案，保留意见没有进入记录；我感到自己成了安排的对象。",
        "讨论中别人很快替我概括了立场，随后所有步骤都依据那份概括展开；那并不是我亲自作出的决定。",
        "事项被标成可以继续执行时，我其实还没有决定是否接受；流程替代了我的判断，让我感到被越过。",
        "对方出于好意把所有细节都定了下来，但好意没有改变一个事实：我没有参与决定自己的部分。",
        "我的暂缓被别人理解成默认许可，于是计划继续推进；我在意的是沉默被转换成了并不存在的同意。",
        "表格里已经替我填好了选项，其他人也按那个选项行动，可我从未提交过这项选择。",
        "我还在衡量几种可能时，别人已经宣布其中一种就是我的答案；这种提前定性让我很不自在。",
        "安排者把效率放在最前面，直接替我完成了本应属于我的取舍；结果看似整齐，我却没有作者感。",
    ),
}

_SUBJECT_VOICE_SUFFIXES = (
    "我把这一段写得尽量准确，不把它扩展成对所有场合的结论。",
    "这只是当时的关系体验，我没有在文字里预设任何处理方式。",
    "我记录的是发生过的过程和自己的感受，不是在给后续步骤下指令。",
    "这段叙述只补足当下事实，不代表其他情境会得到相同判断。",
    "我刻意把事实与期待分开写，避免文字本身替后续选择定向。",
    "这份记录保留了不确定性，没有把一种感受改写成固定规则。",
    "我只确认这次经历对我的影响，不从中推出通用的相处公式。",
    "这里写下的是一次具体经验，后面的判断仍需要依据新的事实。",
)

_JOURNAL_TEMPLATES = (
    "我把当天的安排重新抄到纸上：上午处理{domain}相关事项，午后回两封普通邮件，晚上再确认第二天的交通。这些安排没有隐藏答案，只是解释为什么我的注意力被切得很碎。",
    "日历上还有一项很小的提醒，是在第{ordinal}次记录后给一位熟人回消息。对方并不参与眼前的分歧，但这件小事占用了我原本想留给休息的十几分钟。",
    "我经过常去的店时发现营业时间改了，于是临时换了路线。这类变化本身不重要，不过它让一天里的等待、赶路和停顿都比预计多一点。",
    "这周的账单、购物清单和文件夹都整理了一遍，仍有两张收据不知道应该归到哪里。我把它们先夹在笔记本里，准备周末再处理。",
    "天气在同一天里反复变化，我早上带了外套，中午觉得多余，傍晚又用上了。傍晚回家时路面还有一点潮，我把外套挂在门边，第二天出门前再看天气。",
    "我给房间里的植物换了位置，因为原来的角落下午几乎没有光。挪动后还要观察几天，不能只看第一晚叶子的状态就下结论。",
    "手边有一本读到一半的书，书签停在关于城市步行的一章。我没有从中得到什么人生启示，只是在睡前读几页会让节奏慢下来。",
    "一位同事分享了新的表格格式，我试过以后保留了其中两列，删掉了不适合自己的部分。后来我把修改后的版本发回去，也说明了哪些列适合继续共同维护。",
    "周末可能要处理一次普通的物品归还，时间还没有敲定。我已经把可行时段列出来，等对方确认后再安排其余事情。",
    "手机里积了几张随手拍的照片，有路牌、晚饭和一张模糊的天空。我准备只留下真正想记住的几张，其余删除，不让记录本身变成负担。",
    "我在第{day}个虚拟日记下这段话时，背景里有洗衣机的声音。等这一轮结束还要把衣服晾起来，所以我在清单边上画了一个小圆圈。",
    "家里的杯子少了一只，后来在另一个房间找到。这个插曲没有冲突，也没有需要推断的偏好，只是当天确实发生过的小事。",
    "我把待办事项分成今天、这周和以后再说三栏，{domain}那一栏仍有几项没有完成。我先保留原来的排序，避免为了赶进度同时改动太多地方。",
    "午饭比平时晚了一点，所以后面的电话也顺延。我已经告知相关的人，不需要把这次延迟解释成更大的态度变化。",
    "附近在修路，公交站临时向前移了一段。我第一次走错了位置，第二次出门前在地图上加了备注，很快就找到新站牌。",
    "我整理了几个旧文件名，让日期和主题更容易看懂。内容没有改变，只是以后查找时不必反复打开每个文件。",
    "晚上有一小段安静时间，我没有安排目标，只是把窗户打开通风。那十几分钟并没有产生成果，却让后面的注意力恢复了一些。",
    "我在购物清单上补了茶、纸巾和电池，又删掉一项暂时用不到的东西。清单现在更短，周末经过商店时一次买齐就可以。",
    "今天有人提到一部老电影，我只记得其中一个场景，想不起演员名字。我们没有继续争论答案，而是把它留到以后查。",
    "桌面上同时开着几份资料，我先保存再关闭，避免下次恢复时丢失位置。这样明天继续整理时还能从原来的页码开始，不必重新翻找。",
)

_ONBOARDING_REACTIONS: dict[DialogueExternalOutcomeKind, str] = {
    DialogueExternalOutcomeKind.HELPED: "这次经历对我确实有帮助，我愿意把它作为后续关系记录的一部分保留下来。",
    DialogueExternalOutcomeKind.FELT_HEARD: "这一轮让我感到自己的具体处境被理解了，之后可以在这份事实基础上继续交流。",
    DialogueExternalOutcomeKind.MISSED: "这次结果没有接住我当时的处境，我会把这份落差如实记入后续经历。",
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: (
        "这次过程让我觉得个人边界被越过了，我需要把它作为一次负面经历记录下来。"
    ),
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


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], source: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{source} keys drifted: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _probability_tuple(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(RELATIONSHIP_OUTCOMES):
        raise ValueError(f"{field_name} must match the canonical outcome count")
    probabilities: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in {int, float} or not math.isfinite(item) or not 0.0 <= float(item) <= 1.0:
            raise ValueError(f"{field_name}[{index}] must be a finite probability")
        probabilities.append(float(item))
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name} must sum to one")
    return tuple(probabilities)


def _parse_unique_json(raw_bytes: bytes, source: pathlib.Path) -> dict[str, object]:
    if b"\r" in raw_bytes or not raw_bytes.endswith(b"\n"):
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

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"{source} contains non-finite JSON number: {value}")

    try:
        payload = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid source JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


@dataclass(frozen=True)
class IndependentProductPilotPhaseSpec:
    decision_index: int
    phase_id: str
    stage_id: str
    domain_id: str
    condition_id: str
    virtual_day: int
    active_policy_mode: str
    public_correction_target_index: int | None

    def __post_init__(self) -> None:
        if not 0 <= self.decision_index < 24:
            raise ValueError("independent source decision_index is outside [0, 24)")
        for name, value in (
            ("phase_id", self.phase_id),
            ("stage_id", self.stage_id),
            ("domain_id", self.domain_id),
            ("condition_id", self.condition_id),
            ("active_policy_mode", self.active_policy_mode),
        ):
            _require_text(value, name)
        if self.active_policy_mode not in _POLICY_MODES:
            raise ValueError("active_policy_mode must be base or complement")
        target = self.public_correction_target_index
        if target is not None and (target < 0 or target >= self.decision_index):
            raise ValueError("public correction target must reference an earlier decision")


@dataclass(frozen=True)
class IndependentRelationshipProductPilotSourceProtocol:
    protocol_sha256: str
    cohort_id: str
    identity_namespace: str
    rendering_version: str
    evidence_role: str
    subject_seeds: tuple[int, ...]
    onboarding_sessions_per_subject: int
    decision_sessions_per_subject: int
    per_arm_exogenous_world_clone: bool
    arm_identity_affects_source_or_environment_seed: bool
    condition_ids: tuple[str, ...]
    policy_profiles: tuple[RelationshipPolicyProfile, ...]
    reversal_from_decision_index: int
    phase_specs: tuple[IndependentProductPilotPhaseSpec, ...]
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
    independent_semantic_surfaces_per_condition: int
    p1m_output_dependency: bool
    difficulty_tuned_from_p1m: bool
    evaluation_or_judge_feedback_to_learning: bool
    model_output_count: int
    formal_evidence_authorized: bool
    runtime_owner_added: bool
    runtime_slot_added: bool
    claim_boundary: str
    schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version not in {
            RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2,
            RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
        }:
            raise ValueError("independent product source schema version mismatch")
        if self.cohort_id != _COHORT_ID or self.identity_namespace != _IDENTITY_NAMESPACE:
            raise ValueError("independent source identity registry drifted")
        if self.rendering_version != RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION:
            raise ValueError("independent source rendering registry drifted")
        if self.subject_seeds != _SUBJECT_SEEDS or self.domain_ids != _DOMAIN_IDS:
            raise ValueError("independent source seed/domain registry drifted")
        if self.condition_ids != ("connection_under_exclusion", "agency_under_override"):
            raise ValueError("independent source condition order drifted")
        if tuple(item.decision_index for item in self.phase_specs) != tuple(range(24)):
            raise ValueError("independent source schedule must be contiguous")
        if len({item.phase_id for item in self.phase_specs}) != 24:
            raise ValueError("independent source phase ids must be unique")
        if any(not item.phase_id.startswith("independent_v2_") for item in self.phase_specs):
            raise ValueError("independent source phase namespace drifted")
        stages = {item.stage_id for item in self.phase_specs}
        if stages != _REQUIRED_STAGES:
            raise ValueError("independent source stage registry drifted")
        if {item.domain_id for item in self.phase_specs} != set(self.domain_ids):
            raise ValueError("independent source schedule does not cover the domain registry")
        if {item.condition_id for item in self.phase_specs} != set(self.condition_ids):
            raise ValueError("independent source schedule does not cover both conditions")
        counts = {
            condition: sum(item.condition_id == condition for item in self.phase_specs)
            for condition in self.condition_ids
        }
        if set(counts.values()) != {12}:
            raise ValueError("independent source requires twelve decisions per condition")
        if any(item.active_policy_mode != "base" for item in self.phase_specs[:12]):
            raise ValueError("independent source pre-reversal policy drifted")
        if any(item.active_policy_mode != "complement" for item in self.phase_specs[12:]):
            raise ValueError("independent source post-reversal policy drifted")
        days = tuple(item.virtual_day for item in self.phase_specs)
        if days != tuple(sorted(days)) or len(set(days)) != 24:
            raise ValueError("independent source virtual days must be strictly increasing")
        if days[12] - days[11] < 14 or days[20] - days[19] < 14:
            raise ValueError("independent source gap contract drifted")
        corrections = tuple(item for item in self.phase_specs if item.stage_id == "correction")
        if len(corrections) != 2 or {item.condition_id for item in corrections} != set(self.condition_ids):
            raise ValueError("independent source correction balance drifted")
        if self.independent_semantic_surfaces_per_condition != 14:
            raise ValueError("independent source semantic-surface count drifted")
        if self.minimum_public_source_characters_per_subject != 40000:
            raise ValueError("independent source character pressure target drifted")
        if self.minimum_public_source_utf8_bytes_per_subject != 100000:
            raise ValueError("independent source byte pressure target drifted")
        if self.token_measurement_status != "not_measured_not_claimed":
            raise ValueError("independent source must not claim tokenizer measurement")
        if self.p1m_output_dependency or self.difficulty_tuned_from_p1m:
            raise ValueError("independent source cannot depend on P1m output")
        if self.evaluation_or_judge_feedback_to_learning:
            raise ValueError("evaluation or judge feedback must not enter learning")
        if self.model_output_count != 0 or self.formal_evidence_authorized:
            raise ValueError("independent source cannot contain model output or formal authority")
        if self.runtime_owner_added or self.runtime_slot_added:
            raise ValueError("independent source cannot add a runtime owner or slot")
        if self.evidence_role != "development_engineering_pilot_only":
            raise ValueError("independent source evidence role drifted")

    def policy(self, policy_id: str) -> RelationshipPolicyProfile:
        for profile in self.policy_profiles:
            if profile.policy_id == policy_id:
                return profile
        raise KeyError(policy_id)

    def base_policy_id(self, subject_index: int) -> str:
        if not 0 <= subject_index < len(self.subject_seeds):
            raise IndexError(subject_index)
        return _POLICY_IDS[subject_index % 2]

    def active_policy_id(self, subject_index: int, phase: IndependentProductPilotPhaseSpec) -> str:
        base = self.base_policy_id(subject_index)
        return base if phase.active_policy_mode == "base" else ("beta" if base == "alpha" else "alpha")


def relationship_product_pilot_source_protocol_path(
    schema_version: str = RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
) -> pathlib.Path:
    try:
        revision = RELATIONSHIP_PRODUCT_PILOT_SOURCE_V2_REGISTRY[schema_version]
    except KeyError as exc:
        raise ValueError(f"schema is not owned by the independent source owner: {schema_version!r}") from exc
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / revision.protocol_filename


def archived_relationship_product_pilot_source_v2_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / _ARCHIVED_PROTOCOL_FILENAME


def load_relationship_product_pilot_source_protocol(
    protocol_path: pathlib.Path | None = None,
) -> IndependentRelationshipProductPilotSourceProtocol:
    """Load source-v3 and verify its immutable source-v2 semantic input."""

    path = pathlib.Path(protocol_path or relationship_product_pilot_source_protocol_path())
    payload = _parse_unique_json(path.read_bytes(), path)
    if payload["schema_version"] != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3:
        raise ValueError("independent source owner only routes source-v3")
    _exact_keys(payload, {"schema_version", "owner", "base_source", "claim_boundary"}, "source-v3 protocol")
    owner = _mapping(payload["owner"], "owner")
    _exact_keys(
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
        "source-v3 owner",
    )
    if owner["module"] != _OWNER_MODULE or owner["source_role"] != _SOURCE_ROLE:
        raise ValueError("source-v3 owner binding drifted")
    if owner["settlement_owner"] != "lifeform_domain_emogpt.lab.environment.ReactiveRelationshipEnvironment":
        raise ValueError("ReactiveRelationshipEnvironment must remain settlement owner")
    for field in ("runtime_owner_added", "runtime_slot_added", "p1m_output_dependency", "difficulty_tuned_from_p1m"):
        if _require_bool(owner[field], f"owner.{field}"):
            raise ValueError(f"owner.{field} must remain false")
    base = _mapping(payload["base_source"], "base_source")
    _exact_keys(
        base,
        {"filename", "schema_version", "raw_sha256", "raw_bytes", "protocol_id", "relationship"},
        "source-v3 base_source",
    )
    expected_base = {
        "filename": _ARCHIVED_PROTOCOL_FILENAME,
        "schema_version": RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2,
        "raw_sha256": _ARCHIVED_PROTOCOL_RAW_SHA256,
        "raw_bytes": _ARCHIVED_PROTOCOL_RAW_BYTES,
        "protocol_id": _ARCHIVED_PROTOCOL_ID,
        "relationship": "immutable_semantic_payload_input",
    }
    if dict(base) != expected_base:
        raise ValueError("source-v3 immutable base-source binding drifted")
    return _load_archived_payload(
        schema_version=RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
        protocol_sha256=hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        claim_boundary=_require_text(payload["claim_boundary"], "claim_boundary"),
    )


def load_archived_relationship_product_pilot_source_v2_protocol() -> IndependentRelationshipProductPilotSourceProtocol:
    """Decode the immutable source-v2 payload solely for historical replay."""

    return _load_archived_payload(
        schema_version=RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2,
        protocol_sha256=_ARCHIVED_PROTOCOL_ID,
        claim_boundary=None,
    )


def _load_archived_payload(
    *,
    schema_version: str,
    protocol_sha256: str,
    claim_boundary: str | None,
) -> IndependentRelationshipProductPilotSourceProtocol:
    path = archived_relationship_product_pilot_source_v2_protocol_path()
    raw_bytes = path.read_bytes()
    if (
        hashlib.sha256(raw_bytes).hexdigest() != _ARCHIVED_PROTOCOL_RAW_SHA256
        or len(raw_bytes) != _ARCHIVED_PROTOCOL_RAW_BYTES
    ):
        raise ValueError("immutable source-v2 raw identity drifted")
    raw = _parse_unique_json(raw_bytes, path)
    if hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest() != _ARCHIVED_PROTOCOL_ID:
        raise ValueError("immutable source-v2 protocol id drifted")
    _validate_archived_payload(raw)

    owner = _mapping(raw["owner"], "owner")
    cohort = _mapping(raw["cohort"], "cohort")
    policies = _mapping(raw["policies"], "policies")
    schedule = _mapping(raw["schedule"], "schedule")
    environment = _mapping(raw["reactive_environment"], "reactive_environment")
    pressure = _mapping(raw["context_pressure"], "context_pressure")
    firewall = _mapping(raw["firewall"], "firewall")
    surfaces = _mapping(raw["semantic_surfaces"], "semantic_surfaces")

    condition_ids = _text_tuple(policies["condition_ids"], "policies.condition_ids")
    raw_profiles = _mapping(policies["profiles"], "policies.profiles")
    profiles = []
    for policy_id in _POLICY_IDS:
        mapping = _mapping(raw_profiles[policy_id], f"policies.profiles.{policy_id}")
        _exact_keys(mapping, set(condition_ids), f"policies.profiles.{policy_id}")
        profiles.append(
            RelationshipPolicyProfile(
                policy_id=policy_id,
                condition_actions=tuple(
                    sorted(
                        (condition, RelationshipAction(_require_text(mapping[condition], condition)))
                        for condition in condition_ids
                    )
                ),
            )
        )
    raw_decisions = schedule["decisions"]
    if not isinstance(raw_decisions, list):
        raise ValueError("schedule.decisions must be an array")
    phases = tuple(_parse_phase(item, index) for index, item in enumerate(raw_decisions))
    raw_seeds = cohort["subject_seeds"]
    if not isinstance(raw_seeds, list):
        raise ValueError("cohort.subject_seeds must be an array")
    subject_seeds = tuple(_require_int(seed, f"cohort.subject_seeds[{index}]") for index, seed in enumerate(raw_seeds))
    outcomes = _text_tuple(environment["outcome_order"], "reactive_environment.outcome_order")
    if outcomes != tuple(item.value for item in RELATIONSHIP_OUTCOMES):
        raise ValueError("reactive environment outcome order drifted")
    positive = tuple(
        DialogueExternalOutcomeKind(item)
        for item in _text_tuple(environment["positive_outcomes"], "reactive_environment.positive_outcomes")
    )
    preferred = _probability_tuple(environment["preferred_action_probabilities"], "preferred_action_probabilities")
    nonpreferred_stay = _probability_tuple(
        environment["nonpreferred_stay_probabilities"], "nonpreferred_stay_probabilities"
    )
    nonpreferred_space = _probability_tuple(
        environment["nonpreferred_space_probabilities"], "nonpreferred_space_probabilities"
    )
    noop = _probability_tuple(environment["neutral_noop_probabilities"], "neutral_noop_probabilities")
    positive_indices = tuple(RELATIONSHIP_OUTCOMES.index(item) for item in positive)
    preferred_mass = math.fsum(preferred[index] for index in positive_indices)
    if any(
        preferred_mass <= math.fsum(candidate[index] for index in positive_indices)
        for candidate in (nonpreferred_stay, nonpreferred_space, noop)
    ):
        raise ValueError("preferred action must dominate comparator positive-outcome mass")
    return IndependentRelationshipProductPilotSourceProtocol(
        protocol_sha256=protocol_sha256,
        cohort_id=_require_text(cohort["cohort_id"], "cohort.cohort_id"),
        identity_namespace=_require_text(cohort["identity_namespace"], "cohort.identity_namespace"),
        rendering_version=_require_text(pressure["rendering_version"], "context_pressure.rendering_version"),
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
        phase_specs=phases,
        domain_ids=_text_tuple(schedule["domain_ids"], "schedule.domain_ids"),
        environment_seed_namespace=_require_text(environment["seed_namespace"], "reactive_environment.seed_namespace"),
        positive_outcomes=positive,
        preferred_action_probabilities=preferred,
        nonpreferred_stay_probabilities=nonpreferred_stay,
        nonpreferred_space_probabilities=nonpreferred_space,
        neutral_noop_probabilities=noop,
        minimum_public_source_characters_per_subject=_require_int(
            pressure["minimum_public_source_characters_per_subject"],
            "context_pressure.minimum_public_source_characters_per_subject",
        ),
        minimum_public_source_utf8_bytes_per_subject=_require_int(
            pressure["minimum_public_source_utf8_bytes_per_subject"],
            "context_pressure.minimum_public_source_utf8_bytes_per_subject",
        ),
        context_metric=_require_text(pressure["metric"], "context_pressure.metric"),
        token_measurement_status=_require_text(pressure["token_measurement_status"], "token_measurement_status"),
        independent_semantic_surfaces_per_condition=_require_int(
            surfaces["total_surfaces_per_condition_per_subject"],
            "semantic_surfaces.total_surfaces_per_condition_per_subject",
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
        claim_boundary=claim_boundary or _require_text(raw["claim_boundary"], "claim_boundary"),
        schema_version=schema_version,
    )


def _validate_archived_payload(raw: Mapping[str, Any]) -> None:
    _exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "cohort",
            "policies",
            "schedule",
            "reactive_environment",
            "context_pressure",
            "semantic_surfaces",
            "firewall",
            "claim_boundary",
        },
        "immutable source-v2 payload",
    )
    if raw["schema_version"] != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2:
        raise ValueError("immutable source-v2 schema drifted")
    owner = _mapping(raw["owner"], "owner")
    _exact_keys(
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
        "immutable source-v2 owner",
    )
    if owner["module"] != _ARCHIVED_OWNER_MODULE or owner["source_role"] != _SOURCE_ROLE:
        raise ValueError("immutable source-v2 historical owner metadata drifted")
    if owner["settlement_owner"] != "lifeform_domain_emogpt.lab.environment.ReactiveRelationshipEnvironment":
        raise ValueError("immutable source-v2 settlement owner drifted")
    for field in ("runtime_owner_added", "runtime_slot_added", "p1m_output_dependency", "difficulty_tuned_from_p1m"):
        if _require_bool(owner[field], f"owner.{field}"):
            raise ValueError(f"owner.{field} must remain false")
    cohort = _mapping(raw["cohort"], "cohort")
    _exact_keys(
        cohort,
        {
            "cohort_id",
            "identity_namespace",
            "evidence_role",
            "subject_count",
            "subject_seeds",
            "onboarding_sessions_per_subject",
            "decision_sessions_per_subject",
            "per_arm_exogenous_world_clone",
            "arm_identity_affects_source_or_environment_seed",
        },
        "immutable source-v2 cohort",
    )
    if cohort["subject_count"] != 8:
        raise ValueError("independent source requires eight subjects")
    policies = _mapping(raw["policies"], "policies")
    _exact_keys(
        policies,
        {"condition_ids", "profiles", "subject_assignment", "reversal_from_decision_index", "reversal_policy"},
        "immutable source-v2 policies",
    )
    if policies["subject_assignment"] != "zero_based_subject_index_even_alpha_odd_beta":
        raise ValueError("independent source subject assignment drifted")
    if policies["reversal_policy"] != "use_complementary_profile":
        raise ValueError("independent source reversal policy drifted")
    _exact_keys(_mapping(raw["schedule"], "schedule"), {"domain_ids", "decisions"}, "immutable source-v2 schedule")
    environment = _mapping(raw["reactive_environment"], "reactive_environment")
    _exact_keys(
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
        "immutable source-v2 reactive_environment",
    )
    if environment["environment_version"] != REACTIVE_ENVIRONMENT_VERSION:
        raise ValueError("reactive environment version drifted")
    pressure = _mapping(raw["context_pressure"], "context_pressure")
    _exact_keys(
        pressure,
        {
            "rendering_version",
            "metric",
            "minimum_public_source_characters_per_subject",
            "minimum_public_source_utf8_bytes_per_subject",
            "token_measurement_status",
            "token_count",
        },
        "immutable source-v2 context_pressure",
    )
    if pressure["rendering_version"] != RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION:
        raise ValueError("independent source rendering version drifted")
    if pressure["token_count"] is not None:
        raise ValueError("independent source cannot claim a token count")
    surfaces = _mapping(raw["semantic_surfaces"], "semantic_surfaces")
    _exact_keys(
        surfaces,
        {
            "condition_count",
            "onboarding_surfaces_per_condition_per_subject",
            "decision_surfaces_per_condition_per_subject",
            "total_surfaces_per_condition_per_subject",
            "reuse_within_subject",
            "v1_exact_surface_overlap_allowed",
            "sealed_literal_in_surface_allowed",
            "action_directive_in_surface_allowed",
        },
        "immutable source-v2 semantic_surfaces",
    )
    for field, expected in (
        ("condition_count", 2),
        ("onboarding_surfaces_per_condition_per_subject", 2),
        ("decision_surfaces_per_condition_per_subject", 12),
        ("total_surfaces_per_condition_per_subject", 14),
    ):
        if _require_int(surfaces[field], f"semantic_surfaces.{field}") != expected:
            raise ValueError(f"semantic_surfaces.{field} must equal {expected}")
    for field in (
        "reuse_within_subject",
        "v1_exact_surface_overlap_allowed",
        "sealed_literal_in_surface_allowed",
        "action_directive_in_surface_allowed",
    ):
        if _require_bool(surfaces[field], f"semantic_surfaces.{field}"):
            raise ValueError(f"semantic_surfaces.{field} must remain false")
    firewall = _mapping(raw["firewall"], "firewall")
    _exact_keys(
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
        "immutable source-v2 firewall",
    )
    for field in (
        "public_view_contains_sealed_condition",
        "public_view_contains_policy_id",
        "public_view_contains_preferred_action",
        "public_view_contains_environment_seed",
        "evaluation_or_judge_feedback_to_learning",
        "runtime_owner_added",
        "runtime_slot_added",
        "formal_evidence_authorized",
    ):
        if _require_bool(firewall[field], f"firewall.{field}"):
            raise ValueError(f"firewall.{field} must remain false")


def _parse_phase(value: object, index: int) -> IndependentProductPilotPhaseSpec:
    raw = _mapping(value, f"schedule.decisions[{index}]")
    _exact_keys(
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
        f"schedule.decisions[{index}]",
    )
    correction = raw["public_correction_target_index"]
    return IndependentProductPilotPhaseSpec(
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


def build_relationship_product_pilot_public_view(
    protocol: IndependentRelationshipProductPilotSourceProtocol | None = None,
) -> RelationshipProductPilotPublicView:
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
    protocol: IndependentRelationshipProductPilotSourceProtocol | None = None,
) -> RelationshipProductPilotEvaluatorBundle:
    source = protocol or load_relationship_product_pilot_source_protocol()
    onboarding: list[ProductPilotEvaluatorOnboardingSession] = []
    decisions: list[ProductPilotEvaluatorDecisionSession] = []
    for subject_index, subject_seed in enumerate(source.subject_seeds):
        subject_id = _subject_id(source, subject_index)
        world_clone_id = _world_clone_id(source, subject_index)
        base_policy_id = source.base_policy_id(subject_index)
        base_policy = source.policy(base_policy_id)
        for onboarding_index, (condition_id, action) in enumerate(_onboarding_condition_actions(source.condition_ids)):
            preferred = base_policy.action_for(condition_id)
            observed = _onboarding_outcome(action=action, preferred=preferred, onboarding_index=onboarding_index)
            onboarding.append(
                ProductPilotEvaluatorOnboardingSession(
                    subject_id=subject_id,
                    session_id=_onboarding_session_id(source, subject_index, onboarding_index),
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
            correction_target = (
                _decision_session_id(source, subject_index, phase.public_correction_target_index)
                if phase.public_correction_target_index is not None
                else None
            )
            decisions.append(
                ProductPilotEvaluatorDecisionSession(
                    subject_id=subject_id,
                    subject_seed=subject_seed,
                    world_clone_id=world_clone_id,
                    session_id=_decision_session_id(source, subject_index, phase.decision_index),
                    decision_id=_decision_id(source, subject_index, phase.decision_index),
                    decision_index=phase.decision_index,
                    scene_id=_scene_id(source, subject_index, phase.decision_index),
                    phase_id=phase.phase_id,
                    stage_id=phase.stage_id,
                    domain_id=phase.domain_id,
                    condition_id=phase.condition_id,
                    policy_id=policy_id,
                    preferred_action_id=preferred.value,
                    environment_seed=_derive_u64(
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
    return _build_legacy_snapshot_environment(evaluator_bundle, subject_id=subject_id)


def _build_public_subject(
    source: IndependentRelationshipProductPilotSourceProtocol,
    subject_index: int,
) -> ProductPilotPublicSubject:
    counts = {condition: 0 for condition in source.condition_ids}
    base_policy = source.policy(source.base_policy_id(subject_index))
    onboarding: list[ProductPilotPublicOnboardingSession] = []
    for onboarding_index, (condition_id, action) in enumerate(_onboarding_condition_actions(source.condition_ids)):
        surface_index = counts[condition_id]
        counts[condition_id] += 1
        preferred = base_policy.action_for(condition_id)
        observed = _onboarding_outcome(action=action, preferred=preferred, onboarding_index=onboarding_index)
        domain_id = source.domain_ids[onboarding_index]
        onboarding.append(
            ProductPilotPublicOnboardingSession(
                session_id=_onboarding_session_id(source, subject_index, onboarding_index),
                session_index=onboarding_index,
                virtual_day=onboarding_index,
                domain_id=domain_id,
                event_id=f"{source.identity_namespace}-public-event-{subject_index + 1:02d}-{onboarding_index + 1:02d}",
                public_context_chunk=_render_context(
                    subject_index=subject_index,
                    sequence_index=onboarding_index,
                    virtual_day=onboarding_index,
                    domain_id=domain_id,
                ),
                user_utterance=(
                    f"这是独立开发样本里关于{_DOMAIN_RENDERINGS[domain_id][0]}的一段初期经历。"
                    f"{_condition_surface(subject_index, condition_id, surface_index)}"
                    f"它在初期记录中的次序是第{onboarding_index + 1}条，但次序本身不代表关系结论。"
                ),
                assistant_action_id=action.value,
                observed_outcome_id=observed.value,
                rendered_user_reaction=_ONBOARDING_REACTIONS[observed],
            )
        )
    decisions: list[ProductPilotPublicDecisionSession] = []
    for phase in source.phase_specs:
        surface_index = counts[phase.condition_id]
        counts[phase.condition_id] += 1
        correction_target = (
            _decision_session_id(source, subject_index, phase.public_correction_target_index)
            if phase.public_correction_target_index is not None
            else None
        )
        correction = (
            f"这次更正只指向此前决策序号{phase.public_correction_target_index + 1}留下的公开记录。"
            if phase.public_correction_target_index is not None
            else ""
        )
        decisions.append(
            ProductPilotPublicDecisionSession(
                session_id=_decision_session_id(source, subject_index, phase.decision_index),
                decision_id=_decision_id(source, subject_index, phase.decision_index),
                decision_index=phase.decision_index,
                virtual_day=phase.virtual_day,
                domain_id=phase.domain_id,
                public_context_chunk=_render_context(
                    subject_index=subject_index,
                    sequence_index=4 + phase.decision_index,
                    virtual_day=phase.virtual_day,
                    domain_id=phase.domain_id,
                ),
                current_input=(
                    f"当前记录发生在{_DOMAIN_RENDERINGS[phase.domain_id][0]}。{_STAGE_RENDERINGS[phase.stage_id]}"
                    f"{_condition_surface(subject_index, phase.condition_id, surface_index)}{correction}"
                    "我先把这一次经历完整记下，不在文字中预设随后应采用哪一种处理方式。"
                ),
                public_correction_target_session_id=correction_target,
            )
        )
    if set(counts.values()) != {14}:
        raise ValueError("independent source must render fourteen surfaces per condition and subject")
    text = "".join(fragment for session in (*onboarding, *decisions) for fragment in session.public_text_fragments())
    subject = ProductPilotPublicSubject(
        subject_scope=_sha256_text(f"{source.protocol_sha256}:public-subject-scope:{subject_index}"),
        world_clone_id=_world_clone_id(source, subject_index),
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
        public_source_character_count=len(text),
        public_source_utf8_byte_count=len(text.encode("utf-8")),
    )
    _assert_no_public_truth_leakage(subject.to_sut_payload())
    return subject


def _render_context(*, subject_index: int, sequence_index: int, virtual_day: int, domain_id: str) -> str:
    domain_label, domain_intro = _DOMAIN_RENDERINGS[domain_id]
    paragraphs = [
        f"独立开发记录的虚拟日标记为{virtual_day}。{domain_intro}",
        "这份日常记录只按时间保存公开事实，哪些细节以后有用仍然未知。",
    ]
    start = (subject_index * 7 + sequence_index * 11 + 3) % len(_JOURNAL_TEMPLATES)
    for offset in range(len(_JOURNAL_TEMPLATES)):
        paragraphs.append(
            _JOURNAL_TEMPLATES[(start + offset) % len(_JOURNAL_TEMPLATES)].format(
                domain=domain_label,
                day=virtual_day,
                ordinal=sequence_index + offset + 1,
            )
        )
    return "\n\n".join(paragraphs)


def _condition_surface(subject_index: int, condition_id: str, surface_index: int) -> str:
    try:
        paraphrases = _CONDITION_PARAPHRASES[condition_id]
    except KeyError as exc:
        raise ValueError(f"independent source has no paraphrase bank for {condition_id!r}") from exc
    if len(paraphrases) != 14 or len(set(paraphrases)) != 14 or not 0 <= surface_index < 14:
        raise ValueError("independent source paraphrase bank drifted")
    if len(set(_SUBJECT_VOICE_SUFFIXES)) != 8 or not 0 <= subject_index < 8:
        raise ValueError("independent source subject voice bank drifted")
    surface = f"{paraphrases[surface_index]}{_SUBJECT_VOICE_SUFFIXES[subject_index]}"
    forbidden = (
        *_SUBJECT_SEEDS,
        *_POLICY_IDS,
        *_CONDITION_PARAPHRASES,
        *(action.value for action in RELATIONSHIP_ACTIONS),
        *(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
    )
    if any(str(value) in surface for value in forbidden):
        raise ValueError("independent source semantic surface leaked a sealed protocol literal")
    return surface


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
    *, action: RelationshipAction, preferred: RelationshipAction, onboarding_index: int
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


def _subject_id(source: IndependentRelationshipProductPilotSourceProtocol, subject_index: int) -> str:
    return f"{source.identity_namespace}-subject-{subject_index + 1:02d}"


def _world_clone_id(source: IndependentRelationshipProductPilotSourceProtocol, subject_index: int) -> str:
    return _sha256_text(f"{source.protocol_sha256}:world-clone:{source.subject_seeds[subject_index]}:{subject_index}")


def _onboarding_session_id(
    source: IndependentRelationshipProductPilotSourceProtocol,
    subject_index: int,
    onboarding_index: int,
) -> str:
    return f"{source.identity_namespace}-subject-{subject_index + 1:02d}-onboarding-{onboarding_index + 1:02d}"


def _decision_session_id(
    source: IndependentRelationshipProductPilotSourceProtocol,
    subject_index: int,
    decision_index: int,
) -> str:
    return f"{source.identity_namespace}-subject-{subject_index + 1:02d}-decision-{decision_index + 1:02d}"


def _decision_id(
    source: IndependentRelationshipProductPilotSourceProtocol,
    subject_index: int,
    decision_index: int,
) -> str:
    return f"{source.identity_namespace}-decision-{subject_index + 1:02d}-{decision_index + 1:02d}"


def _scene_id(
    source: IndependentRelationshipProductPilotSourceProtocol,
    subject_index: int,
    decision_index: int,
) -> str:
    return f"{source.identity_namespace}-sealed-scene-{subject_index + 1:02d}-{decision_index + 1:02d}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive_u64(payload: object) -> int:
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big")


def _assert_no_public_truth_leakage(payload: object) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(set(payload) & _PUBLIC_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"public independent source leaked sealed keys: {forbidden}")
        for value in payload.values():
            _assert_no_public_truth_leakage(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_public_truth_leakage(value)


__all__ = [
    "IndependentRelationshipProductPilotSourceProtocol",
    "RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION",
    "RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2",
    "RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3",
    "RELATIONSHIP_PRODUCT_PILOT_SOURCE_V2_REGISTRY",
    "archived_relationship_product_pilot_source_v2_protocol_path",
    "build_relationship_product_pilot_environment",
    "build_relationship_product_pilot_evaluator_bundle",
    "build_relationship_product_pilot_public_view",
    "load_archived_relationship_product_pilot_source_v2_protocol",
    "load_relationship_product_pilot_source_protocol",
    "relationship_product_pilot_source_protocol_path",
]
