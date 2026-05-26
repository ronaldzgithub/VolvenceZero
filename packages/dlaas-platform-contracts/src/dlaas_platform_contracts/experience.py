"""Cross-app ``ExperienceLoop`` typed contracts.

These types define the product-level surface shared by every DLaaS-aware
app that wants to participate in a unified ``brief -> receipt ->
reflection`` loop (digital-employee marketing, coread / volvence-press
reader feedback, family-memorial bake jobs, capability-exchange
learning runs, ...).

Design stance (carried from `docs/specs/dlaas-platform.md`):

* These specs live in ``dlaas-platform-contracts`` because they describe
  the wire format that every app emits. They do not import ``vz-*``
  beyond ``volvence_zero.dialogue_trace`` (already a contracts-tier
  dependency) and **do not** introduce new kernel sinks. ``vz-*`` keeps
  its kernel-level vocabulary (``observe`` / ``feedback`` / ``corpus``)
  and never learns the words "campaign / memorial / capability".

* SHADOW rollout: the typed specs sit alongside the existing
  free-form ``CLASS_NOTE`` path. The dispatcher accepts the new
  ``ObservationType.EXPERIENCE_*`` values but still routes through the
  same ``submit_reviewed_knowledge_event`` /
  ``submit_tool_result`` sinks, so no kernel changes are required.

* Per-domain product semantics live on
  :class:`ExperienceDomainBinding`. Each app picks ONE binding (e.g.
  ``marketing``, ``reader_dialogue``, ``memorial``, ``capability``)
  and routes its product events through the matching brief / receipt /
  reflection envelopes. The binding is pure data — no orchestration —
  so the platform stays envelope-generic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final


# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

EXPERIENCE_BRIEF_SCHEMA_VERSION: Final = "experience.brief.v1"
EXPERIENCE_RECEIPT_SCHEMA_VERSION: Final = "experience.receipt.v1"
EXPERIENCE_REFLECTION_SCHEMA_VERSION: Final = "experience.reflection.v1"

_VALID_AUTONOMY_LEVELS: Final[frozenset[str]] = frozenset(
    {"read_only", "draft_only", "approved_write", "quota_limited_write"}
)


def _required_str(data: Mapping[str, Any], key: str, owner: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string when set")
    return value


def _string_tuple(value: Any, owner: str, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{owner}.{key} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{owner}.{key} entries must be non-empty strings")
        out.append(item)
    return tuple(out)


# ---------------------------------------------------------------------------
# Budget hint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceBudgetHint:
    """Optional budget hint carried inside an ``ExperienceBriefSpec``.

    The platform records this verbatim. Actual budget enforcement lives
    in the app-side quota envelope (digital-employee
    ``MarketingQuotaEnvelope``); the hint is just the value the agent
    will see on the brief side.
    """

    currency: str = "USD"
    daily_cap_cents: int = 0
    lifetime_cap_cents: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.currency, str) or len(self.currency.strip()) < 3:
            raise ValueError(
                "ExperienceBudgetHint.currency must be a >=3 char string"
            )
        for name, value in (
            ("daily_cap_cents", self.daily_cap_cents),
            ("lifetime_cap_cents", self.lifetime_cap_cents),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"ExperienceBudgetHint.{name} must be a non-negative int"
                )

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any] | None
    ) -> "ExperienceBudgetHint":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("ExperienceBudgetHint payload must be a JSON object")
        return cls(
            currency=str(data.get("currency", "USD") or "USD"),
            daily_cap_cents=int(data.get("daily_cap_cents", 0) or 0),
            lifetime_cap_cents=int(data.get("lifetime_cap_cents", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "daily_cap_cents": self.daily_cap_cents,
            "lifetime_cap_cents": self.lifetime_cap_cents,
        }


# ---------------------------------------------------------------------------
# Brief / Receipt / Reflection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceBriefSpec:
    """A typed product-level brief.

    ``domain`` keys into one of the registered
    :class:`ExperienceDomainBinding` entries (e.g. ``"marketing"``,
    ``"reader_dialogue"``, ``"memorial"``, ``"capability"``).
    ``experience_id`` is the per-domain handle (campaign_id,
    dialogue_session_id, memorial_id, capability_id) — anything stable
    enough to fold receipts and reflections back into the same
    experience.
    """

    domain: str
    goal: str
    kpi: str
    schema_version: str = EXPERIENCE_BRIEF_SCHEMA_VERSION
    target_url: str = ""
    channels: tuple[str, ...] = ()
    audience: str = ""
    geo: tuple[str, ...] = ()
    budget_hint: ExperienceBudgetHint = field(default_factory=ExperienceBudgetHint)
    autonomy_level: str = "draft_only"
    boundaries: tuple[str, ...] = ()
    experience_id: str = ""
    domain_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIENCE_BRIEF_SCHEMA_VERSION:
            raise ValueError(
                f"ExperienceBriefSpec.schema_version must be "
                f"{EXPERIENCE_BRIEF_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("ExperienceBriefSpec.domain must be a non-empty string")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("ExperienceBriefSpec.goal must be a non-empty string")
        if not isinstance(self.kpi, str) or not self.kpi.strip():
            raise ValueError("ExperienceBriefSpec.kpi must be a non-empty string")
        if self.autonomy_level not in _VALID_AUTONOMY_LEVELS:
            allowed = ", ".join(sorted(_VALID_AUTONOMY_LEVELS))
            raise ValueError(
                f"ExperienceBriefSpec.autonomy_level must be one of: {allowed}; "
                f"got {self.autonomy_level!r}"
            )
        if not isinstance(self.domain_payload, Mapping):
            raise ValueError(
                "ExperienceBriefSpec.domain_payload must be a Mapping"
            )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ExperienceBriefSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ExperienceBriefSpec payload must be a JSON object")
        return cls(
            schema_version=str(
                data.get("schema_version", EXPERIENCE_BRIEF_SCHEMA_VERSION)
                or EXPERIENCE_BRIEF_SCHEMA_VERSION
            ),
            domain=_required_str(data, "domain", "ExperienceBriefSpec"),
            goal=_required_str(data, "goal", "ExperienceBriefSpec"),
            kpi=_required_str(data, "kpi", "ExperienceBriefSpec"),
            target_url=_optional_str(data, "target_url"),
            channels=_string_tuple(data.get("channels"), "ExperienceBriefSpec", "channels"),
            audience=_optional_str(data, "audience"),
            geo=_string_tuple(data.get("geo"), "ExperienceBriefSpec", "geo"),
            budget_hint=ExperienceBudgetHint.from_json(data.get("budget_hint")),
            autonomy_level=str(
                data.get("autonomy_level", "draft_only") or "draft_only"
            ),
            boundaries=_string_tuple(
                data.get("boundaries"), "ExperienceBriefSpec", "boundaries"
            ),
            experience_id=_optional_str(data, "experience_id"),
            domain_payload=dict(data.get("domain_payload") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "goal": self.goal,
            "kpi": self.kpi,
            "target_url": self.target_url,
            "channels": list(self.channels),
            "audience": self.audience,
            "geo": list(self.geo),
            "budget_hint": self.budget_hint.to_json(),
            "autonomy_level": self.autonomy_level,
            "boundaries": list(self.boundaries),
            "experience_id": self.experience_id,
            "domain_payload": dict(self.domain_payload),
        }


@dataclass(frozen=True)
class ExperienceReceiptSpec:
    """One observable outcome attached to an experience.

    ``event_kind`` is a free-form domain label like
    ``"metric_observed"``, ``"receipt_observed"``,
    ``"bake_completed"``. The platform does not interpret it; it is
    audit / reflection material.
    """

    domain: str
    experience_id: str
    event_kind: str
    summary: str
    schema_version: str = EXPERIENCE_RECEIPT_SCHEMA_VERSION
    detail: str = ""
    tool_name: str = ""
    artifact_refs: tuple[str, ...] = ()
    confidence: float = 0.8
    domain_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIENCE_RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"ExperienceReceiptSpec.schema_version must be "
                f"{EXPERIENCE_RECEIPT_SCHEMA_VERSION!r}; got "
                f"{self.schema_version!r}"
            )
        for name, value in (
            ("domain", self.domain),
            ("experience_id", self.experience_id),
            ("event_kind", self.event_kind),
            ("summary", self.summary),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ExperienceReceiptSpec.{name} must be a non-empty string"
                )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "ExperienceReceiptSpec.confidence must be in [0.0, 1.0]"
            )
        if not isinstance(self.domain_payload, Mapping):
            raise ValueError(
                "ExperienceReceiptSpec.domain_payload must be a Mapping"
            )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ExperienceReceiptSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ExperienceReceiptSpec payload must be a JSON object")
        return cls(
            schema_version=str(
                data.get("schema_version", EXPERIENCE_RECEIPT_SCHEMA_VERSION)
                or EXPERIENCE_RECEIPT_SCHEMA_VERSION
            ),
            domain=_required_str(data, "domain", "ExperienceReceiptSpec"),
            experience_id=_required_str(
                data, "experience_id", "ExperienceReceiptSpec"
            ),
            event_kind=_required_str(data, "event_kind", "ExperienceReceiptSpec"),
            summary=_required_str(data, "summary", "ExperienceReceiptSpec"),
            detail=_optional_str(data, "detail"),
            tool_name=_optional_str(data, "tool_name"),
            artifact_refs=_string_tuple(
                data.get("artifact_refs"),
                "ExperienceReceiptSpec",
                "artifact_refs",
            ),
            confidence=float(data.get("confidence", 0.8) or 0.8),
            domain_payload=dict(data.get("domain_payload") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "experience_id": self.experience_id,
            "event_kind": self.event_kind,
            "summary": self.summary,
            "detail": self.detail,
            "tool_name": self.tool_name,
            "artifact_refs": list(self.artifact_refs),
            "confidence": self.confidence,
            "domain_payload": dict(self.domain_payload),
        }


@dataclass(frozen=True)
class ExperienceReflectionSpec:
    """A completed-experience retrospective.

    Apps emit this once after receipts + feedback are in. The detail
    text is the reviewed corpus body that the app may then upload via
    its own ``training/corpus`` path; this contract only declares the
    envelope shape, not the storage.
    """

    domain: str
    experience_id: str
    title: str
    summary: str
    schema_version: str = EXPERIENCE_REFLECTION_SCHEMA_VERSION
    detail: str = ""
    receipt_count: int = 0
    feedback_count: int = 0
    corpus_ref: str = ""
    domain_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIENCE_REFLECTION_SCHEMA_VERSION:
            raise ValueError(
                f"ExperienceReflectionSpec.schema_version must be "
                f"{EXPERIENCE_REFLECTION_SCHEMA_VERSION!r}; got "
                f"{self.schema_version!r}"
            )
        for name, value in (
            ("domain", self.domain),
            ("experience_id", self.experience_id),
            ("title", self.title),
            ("summary", self.summary),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ExperienceReflectionSpec.{name} must be a non-empty string"
                )
        for name, value in (
            ("receipt_count", self.receipt_count),
            ("feedback_count", self.feedback_count),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"ExperienceReflectionSpec.{name} must be a non-negative int"
                )
        if not isinstance(self.domain_payload, Mapping):
            raise ValueError(
                "ExperienceReflectionSpec.domain_payload must be a Mapping"
            )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ExperienceReflectionSpec":
        if not isinstance(data, Mapping):
            raise ValueError(
                "ExperienceReflectionSpec payload must be a JSON object"
            )
        return cls(
            schema_version=str(
                data.get("schema_version", EXPERIENCE_REFLECTION_SCHEMA_VERSION)
                or EXPERIENCE_REFLECTION_SCHEMA_VERSION
            ),
            domain=_required_str(data, "domain", "ExperienceReflectionSpec"),
            experience_id=_required_str(
                data, "experience_id", "ExperienceReflectionSpec"
            ),
            title=_required_str(data, "title", "ExperienceReflectionSpec"),
            summary=_required_str(data, "summary", "ExperienceReflectionSpec"),
            detail=_optional_str(data, "detail"),
            receipt_count=int(data.get("receipt_count", 0) or 0),
            feedback_count=int(data.get("feedback_count", 0) or 0),
            corpus_ref=_optional_str(data, "corpus_ref"),
            domain_payload=dict(data.get("domain_payload") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "experience_id": self.experience_id,
            "title": self.title,
            "summary": self.summary,
            "detail": self.detail,
            "receipt_count": self.receipt_count,
            "feedback_count": self.feedback_count,
            "corpus_ref": self.corpus_ref,
            "domain_payload": dict(self.domain_payload),
        }


# ---------------------------------------------------------------------------
# Domain binding registry (pure data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceDomainBinding:
    """Per-domain product binding.

    Pure data. No orchestration. Apps consult the binding to know
    which ``InboxItem.kind`` to use, which ``Turn.toolName`` prefix
    marks a receipt, and how to label reflections. Adding a domain
    here is the *only* place the contracts wheel needs to learn about
    a new product surface.
    """

    domain: str
    brief_kind: str
    receipt_tool_prefix: str
    reflection_title_prefix: str
    panel_title: str = ""
    receipt_fallback_tool_name: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("domain", self.domain),
            ("brief_kind", self.brief_kind),
            ("receipt_tool_prefix", self.receipt_tool_prefix),
            ("reflection_title_prefix", self.reflection_title_prefix),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ExperienceDomainBinding.{name} must be a non-empty string"
                )

    def experience_event_id(
        self, *, event: str, scope_id: str, monotonic_ms: int
    ) -> str:
        """Deterministic event id used in observation envelopes.

        Mirrors the BFF helper ``experience_observation_label`` so the
        same id can be reconstructed on Python and TS sides.
        """
        return f"{self.domain}:{event}:{scope_id}:{monotonic_ms}"

    def experience_source_label(self, *, event: str) -> str:
        return f"digital-employee.{self.domain}.{event}"


MARKETING_EXPERIENCE_BINDING: Final[ExperienceDomainBinding] = ExperienceDomainBinding(
    domain="marketing",
    brief_kind="marketing.brief",
    receipt_tool_prefix="marketing.",
    reflection_title_prefix="Campaign retrospective",
    panel_title="Marketing brief",
    receipt_fallback_tool_name="marketing.receipt",
)

READER_DIALOGUE_EXPERIENCE_BINDING: Final[ExperienceDomainBinding] = (
    ExperienceDomainBinding(
        domain="reader_dialogue",
        brief_kind="reader_dialogue.brief",
        receipt_tool_prefix="reader_dialogue.",
        reflection_title_prefix="Reader dialogue retrospective",
        panel_title="Reader dialogue",
        receipt_fallback_tool_name="reader_dialogue.receipt",
    )
)

MEMORIAL_EXPERIENCE_BINDING: Final[ExperienceDomainBinding] = ExperienceDomainBinding(
    domain="memorial",
    brief_kind="memorial.brief",
    receipt_tool_prefix="memorial.",
    reflection_title_prefix="Memorial bake retrospective",
    panel_title="Memorial",
    receipt_fallback_tool_name="memorial.receipt",
)

CAPABILITY_EXPERIENCE_BINDING: Final[ExperienceDomainBinding] = (
    ExperienceDomainBinding(
        domain="capability",
        brief_kind="capability.brief",
        receipt_tool_prefix="capability.",
        reflection_title_prefix="Capability learning retrospective",
        panel_title="Capability",
        receipt_fallback_tool_name="capability.receipt",
    )
)


_BUILTIN_BINDINGS: Final[tuple[ExperienceDomainBinding, ...]] = (
    MARKETING_EXPERIENCE_BINDING,
    READER_DIALOGUE_EXPERIENCE_BINDING,
    MEMORIAL_EXPERIENCE_BINDING,
    CAPABILITY_EXPERIENCE_BINDING,
)


def builtin_experience_domain_bindings() -> tuple[ExperienceDomainBinding, ...]:
    """Return the bindings registered at the platform-contracts tier.

    Apps may layer additional bindings on top (digital-employee adds
    persona-specific ones); the contracts wheel only commits to the
    four listed here so downstream code can rely on them being
    schema-stable.
    """
    return _BUILTIN_BINDINGS


def find_binding_for_brief_kind(
    brief_kind: str,
) -> ExperienceDomainBinding | None:
    """Look up the binding for a known ``InboxItem.kind`` / brief kind.

    Returns ``None`` for unknown kinds; callers decide whether to
    treat that as a contract error (digital-employee marketing flow)
    or to fall through to a generic experience handler.
    """
    for binding in _BUILTIN_BINDINGS:
        if binding.brief_kind == brief_kind:
            return binding
    return None


__all__ = [
    "CAPABILITY_EXPERIENCE_BINDING",
    "EXPERIENCE_BRIEF_SCHEMA_VERSION",
    "EXPERIENCE_RECEIPT_SCHEMA_VERSION",
    "EXPERIENCE_REFLECTION_SCHEMA_VERSION",
    "ExperienceBriefSpec",
    "ExperienceBudgetHint",
    "ExperienceDomainBinding",
    "ExperienceReceiptSpec",
    "ExperienceReflectionSpec",
    "MARKETING_EXPERIENCE_BINDING",
    "MEMORIAL_EXPERIENCE_BINDING",
    "READER_DIALOGUE_EXPERIENCE_BINDING",
    "builtin_experience_domain_bindings",
    "find_binding_for_brief_kind",
]
