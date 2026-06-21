"""Multi-angle bake contracts.

A *bake* turns one set of **raw materials** (a work, a corpus, a set of
reviewed sources) into one or more grounded personas, each cut at a
distinct **angle**:

* ``author``      — the real creator/author as a primary-source figure.
* ``interpreter`` — a 诠释者 / narrator-commentator persona that explains
                    and contextualises the work in the third person.
* ``character``   — an in-world 角色; one persona per named character.

A single :class:`BakeRequest` therefore fans out into N
:class:`BakeAngleJob` units that share the same raw materials but route
to different verticals (figure vs character) and produce one template
each. Monitoring is per-angle; the run aggregate rolls the angle states
up into a single :class:`BakeRunStatus`.

Ownership boundary (R8 / R12 / R15): this contract is **platform
orchestration state**. It records what was requested, how far each
angle got, and pointers to the produced artifacts (template id, figure
bundle id). It never mirrors cognition and never becomes a second owner
of the baked bundle — the bundle stays in the figure/character bundle
store, and governance of the persona stays in the persona lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BakeAngleKind(str, Enum):
    """Perspective a persona is cut from shared raw materials."""

    AUTHOR = "author"
    INTERPRETER = "interpreter"
    CHARACTER = "character"


class BakeRunStatus(str, Enum):
    """Aggregate state of a multi-angle bake run."""

    QUEUED = "queued"
    RUNNING = "running"
    PARTIAL = "partial"  # terminal: some angles done, others failed
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BakeAngleStatus(str, Enum):
    """Per-angle bake progress.

    The forward stages mirror the proven family-memorial pipeline so a
    shared monitor UI can render the same progress bar for every app.
    """

    QUEUED = "queued"
    STAGING = "staging"
    CLEANING = "cleaning"
    VERIFYING = "verifying"
    BAKING = "baking"
    REGISTERING = "registering"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Forward order of per-angle stages (terminal states excluded).
ANGLE_STAGE_ORDER: tuple[BakeAngleStatus, ...] = (
    BakeAngleStatus.QUEUED,
    BakeAngleStatus.STAGING,
    BakeAngleStatus.CLEANING,
    BakeAngleStatus.VERIFYING,
    BakeAngleStatus.BAKING,
    BakeAngleStatus.REGISTERING,
    BakeAngleStatus.DONE,
)

_TERMINAL_ANGLE_STATES: frozenset[BakeAngleStatus] = frozenset(
    {
        BakeAngleStatus.DONE,
        BakeAngleStatus.FAILED,
        BakeAngleStatus.CANCELLED,
    }
)

_TERMINAL_RUN_STATES: frozenset[BakeRunStatus] = frozenset(
    {
        BakeRunStatus.PARTIAL,
        BakeRunStatus.DONE,
        BakeRunStatus.FAILED,
        BakeRunStatus.CANCELLED,
    }
)

VALID_RAW_MATERIAL_KINDS: tuple[str, ...] = ("text", "asset_ref", "uri")
VALID_CORPUS_MODES: tuple[str, ...] = ("synthetic", "curated")


class BakeContractError(ValueError):
    """Raised when a bake request/state violates the contract."""


def angle_stage_index(status: BakeAngleStatus) -> int:
    """Position of ``status`` on the forward pipeline.

    Terminal failure/cancel states return ``-1``; ``done`` returns the
    last index. Useful for rendering a progress fraction.
    """

    try:
        return ANGLE_STAGE_ORDER.index(status)
    except ValueError:
        return -1


def is_terminal_angle_status(status: BakeAngleStatus) -> bool:
    return status in _TERMINAL_ANGLE_STATES


def is_terminal_run_status(status: BakeRunStatus) -> bool:
    return status in _TERMINAL_RUN_STATES


def rollup_run_status(angle_statuses: Sequence[BakeAngleStatus]) -> BakeRunStatus:
    """Derive the run status from its angle states (SSOT for rollup).

    Rules:

    * no angles                                  → ``failed`` (invalid).
    * any angle still non-terminal               → ``running``.
    * all cancelled                              → ``cancelled``.
    * all done                                   → ``done``.
    * all terminal, every non-cancelled failed   → ``failed``.
    * all terminal, mix of done and failed       → ``partial``.
    """

    statuses = list(angle_statuses)
    if not statuses:
        return BakeRunStatus.FAILED
    if any(not is_terminal_angle_status(s) for s in statuses):
        return BakeRunStatus.RUNNING
    if all(s is BakeAngleStatus.CANCELLED for s in statuses):
        return BakeRunStatus.CANCELLED
    done = [s for s in statuses if s is BakeAngleStatus.DONE]
    failed = [s for s in statuses if s is BakeAngleStatus.FAILED]
    if done and not failed:
        return BakeRunStatus.DONE
    if failed and not done:
        return BakeRunStatus.FAILED
    return BakeRunStatus.PARTIAL


@dataclass(frozen=True)
class RawMaterial:
    """One unit of shared input routed to one or more angles.

    ``angle_slugs`` empty means the material feeds every angle. A
    non-empty tuple restricts it (e.g. a character's dialogue lines only
    feed that character's angle).
    """

    kind: str  # one of VALID_RAW_MATERIAL_KINDS
    ref: str = ""  # asset id / uri when kind != "text"
    text: str = ""  # inline text when kind == "text"
    media_kind: str = ""  # optional hint: "transcript" | "prose" | ...
    angle_slugs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RawMaterial":
        if not isinstance(data, Mapping):
            raise BakeContractError("RawMaterial must be a JSON object")
        kind = str(data.get("kind", "") or "").strip()
        if kind not in VALID_RAW_MATERIAL_KINDS:
            allowed = ", ".join(VALID_RAW_MATERIAL_KINDS)
            raise BakeContractError(
                f"RawMaterial.kind must be one of: {allowed} (got {kind!r})"
            )
        text = str(data.get("text", "") or "")
        ref = str(data.get("ref", "") or "")
        if kind == "text" and not text.strip():
            raise BakeContractError("RawMaterial.kind='text' requires non-empty text")
        if kind != "text" and not ref.strip():
            raise BakeContractError(
                f"RawMaterial.kind={kind!r} requires a non-empty ref"
            )
        raw_slugs = data.get("angle_slugs") or ()
        if not isinstance(raw_slugs, (list, tuple)):
            raise BakeContractError("RawMaterial.angle_slugs must be an array")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise BakeContractError("RawMaterial.metadata must be an object")
        return cls(
            kind=kind,
            ref=ref,
            text=text,
            media_kind=str(data.get("media_kind", "") or ""),
            angle_slugs=tuple(str(s) for s in raw_slugs),
            metadata=dict(metadata),
        )

    def feeds_angle(self, slug: str) -> bool:
        return not self.angle_slugs or slug in self.angle_slugs

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "text": self.text,
            "media_kind": self.media_kind,
            "angle_slugs": list(self.angle_slugs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BakeAngle:
    """One persona to cut from the shared raw materials."""

    kind: BakeAngleKind
    slug: str
    display_name: str = ""
    profile_overrides: Mapping[str, Any] = field(default_factory=dict)
    style_prior: Mapping[str, Any] = field(default_factory=dict)
    boundary_priors: tuple[str, ...] = ()
    target_contexts: tuple[str, ...] = ()
    time_window: str = ""
    notes: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BakeAngle":
        if not isinstance(data, Mapping):
            raise BakeContractError("BakeAngle must be a JSON object")
        try:
            kind = BakeAngleKind(str(data.get("kind", "") or ""))
        except ValueError as exc:
            allowed = ", ".join(k.value for k in BakeAngleKind)
            raise BakeContractError(
                f"BakeAngle.kind must be one of: {allowed}"
            ) from exc
        slug = str(data.get("slug", "") or "").strip()
        if not slug:
            raise BakeContractError("BakeAngle.slug must be non-empty")
        profile_overrides = data.get("profile_overrides") or {}
        if not isinstance(profile_overrides, Mapping):
            raise BakeContractError("BakeAngle.profile_overrides must be an object")
        style_prior = data.get("style_prior") or {}
        if not isinstance(style_prior, Mapping):
            raise BakeContractError("BakeAngle.style_prior must be an object")
        boundary = data.get("boundary_priors") or ()
        if not isinstance(boundary, (list, tuple)):
            raise BakeContractError("BakeAngle.boundary_priors must be an array")
        contexts = data.get("target_contexts") or ()
        if not isinstance(contexts, (list, tuple)):
            raise BakeContractError("BakeAngle.target_contexts must be an array")
        return cls(
            kind=kind,
            slug=slug,
            display_name=str(data.get("display_name", "") or ""),
            profile_overrides=dict(profile_overrides),
            style_prior=dict(style_prior),
            boundary_priors=tuple(str(b) for b in boundary),
            target_contexts=tuple(str(c) for c in contexts),
            time_window=str(data.get("time_window", "") or ""),
            notes=str(data.get("notes", "") or ""),
        )

    @property
    def angle_id(self) -> str:
        """Stable identity of an angle within a run (kind:slug)."""

        return f"{self.kind.value}:{self.slug}"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "slug": self.slug,
            "display_name": self.display_name,
            "profile_overrides": dict(self.profile_overrides),
            "style_prior": dict(self.style_prior),
            "boundary_priors": list(self.boundary_priors),
            "target_contexts": list(self.target_contexts),
            "time_window": self.time_window,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BakeRequest:
    """Submit one multi-angle bake from shared raw materials."""

    source_ref: str
    angles: tuple[BakeAngle, ...]
    raw_materials: tuple[RawMaterial, ...] = ()
    shared_profile: Mapping[str, Any] = field(default_factory=dict)
    corpus_mode: str = "synthetic"
    runtime_template_id: str = ""
    tenant_id: str = ""
    app_id: str = ""
    notes: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BakeRequest":
        if not isinstance(data, Mapping):
            raise BakeContractError("BakeRequest must be a JSON object")
        source_ref = str(data.get("source_ref", "") or "").strip()
        if not source_ref:
            raise BakeContractError("BakeRequest.source_ref must be non-empty")
        raw_angles = data.get("angles") or ()
        if not isinstance(raw_angles, (list, tuple)) or not raw_angles:
            raise BakeContractError("BakeRequest.angles must be a non-empty array")
        angles = tuple(BakeAngle.from_json(a) for a in raw_angles)
        seen: set[str] = set()
        for angle in angles:
            if angle.angle_id in seen:
                raise BakeContractError(
                    f"duplicate angle {angle.angle_id!r}; (kind, slug) must be unique"
                )
            seen.add(angle.angle_id)
        raw_materials = data.get("raw_materials") or ()
        if not isinstance(raw_materials, (list, tuple)):
            raise BakeContractError("BakeRequest.raw_materials must be an array")
        materials = tuple(RawMaterial.from_json(m) for m in raw_materials)
        corpus_mode = str(data.get("corpus_mode", "synthetic") or "synthetic")
        if corpus_mode not in VALID_CORPUS_MODES:
            allowed = ", ".join(VALID_CORPUS_MODES)
            raise BakeContractError(
                f"BakeRequest.corpus_mode must be one of: {allowed}"
            )
        shared_profile = data.get("shared_profile") or {}
        if not isinstance(shared_profile, Mapping):
            raise BakeContractError("BakeRequest.shared_profile must be an object")
        return cls(
            source_ref=source_ref,
            angles=angles,
            raw_materials=materials,
            shared_profile=dict(shared_profile),
            corpus_mode=corpus_mode,
            runtime_template_id=str(data.get("runtime_template_id", "") or ""),
            tenant_id=str(data.get("tenant_id", "") or ""),
            app_id=str(data.get("app_id", "") or ""),
            notes=str(data.get("notes", "") or ""),
        )

    def materials_for(self, angle: BakeAngle) -> tuple[RawMaterial, ...]:
        return tuple(m for m in self.raw_materials if m.feeds_angle(angle.slug))

    def to_json(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "angles": [a.to_json() for a in self.angles],
            "raw_materials": [m.to_json() for m in self.raw_materials],
            "shared_profile": dict(self.shared_profile),
            "corpus_mode": self.corpus_mode,
            "runtime_template_id": self.runtime_template_id,
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BakeAngleResult:
    """Pointers to the artifacts produced for one finished angle."""

    angle_kind: BakeAngleKind
    angle_slug: str
    template_id: str = ""
    figure_artifact_id: str = ""
    bundle_id: str = ""
    lifecycle_stage: str = ""
    integrity_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "angle_kind": self.angle_kind.value,
            "angle_slug": self.angle_slug,
            "template_id": self.template_id,
            "figure_artifact_id": self.figure_artifact_id,
            "bundle_id": self.bundle_id,
            "lifecycle_stage": self.lifecycle_stage,
            "integrity_hash": self.integrity_hash,
        }


@dataclass(frozen=True)
class BakeAngleJob:
    """Per-angle bake unit + its current observable state."""

    angle_job_id: str
    run_id: str
    angle: BakeAngle
    status: BakeAngleStatus = BakeAngleStatus.QUEUED
    result: BakeAngleResult | None = None
    error: str = ""
    log_tail: str = ""
    updated_at_ms: int = 0

    def with_status(
        self,
        status: BakeAngleStatus,
        *,
        log_tail: str | None = None,
        error: str | None = None,
        result: BakeAngleResult | None = None,
        updated_at_ms: int | None = None,
    ) -> "BakeAngleJob":
        from dataclasses import replace

        changes: dict[str, Any] = {"status": status}
        if log_tail is not None:
            changes["log_tail"] = log_tail
        if error is not None:
            changes["error"] = error
        if result is not None:
            changes["result"] = result
        if updated_at_ms is not None:
            changes["updated_at_ms"] = updated_at_ms
        return replace(self, **changes)

    @property
    def progress_fraction(self) -> float:
        if self.status is BakeAngleStatus.DONE:
            return 1.0
        if self.status in (BakeAngleStatus.FAILED, BakeAngleStatus.CANCELLED):
            return 0.0
        idx = angle_stage_index(self.status)
        last = len(ANGLE_STAGE_ORDER) - 1
        return round(idx / last, 3) if last > 0 and idx >= 0 else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "angle_job_id": self.angle_job_id,
            "run_id": self.run_id,
            "angle": self.angle.to_json(),
            "status": self.status.value,
            "progress": self.progress_fraction,
            "result": self.result.to_json() if self.result else None,
            "error": self.error,
            "log_tail": self.log_tail,
            "updated_at_ms": self.updated_at_ms,
        }


@dataclass(frozen=True)
class BakeRun:
    """Aggregate of a multi-angle bake run keyed by ``run_id``."""

    run_id: str
    source_ref: str
    status: BakeRunStatus = BakeRunStatus.QUEUED
    tenant_id: str = ""
    app_id: str = ""
    corpus_mode: str = "synthetic"
    runtime_template_id: str = ""
    notes: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_json(
        self, *, jobs: Sequence[BakeAngleJob] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "source_ref": self.source_ref,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "corpus_mode": self.corpus_mode,
            "runtime_template_id": self.runtime_template_id,
            "notes": self.notes,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }
        if jobs is not None:
            payload["angles"] = [j.to_json() for j in jobs]
        return payload


@dataclass(frozen=True)
class BakeRunEvent:
    """One progress event emitted to the SSE monitor stream."""

    event_kind: str  # "run" | "angle" | "done" | "error"
    run_id: str
    run_status: BakeRunStatus
    angle_id: str = ""
    angle_status: BakeAngleStatus | None = None
    message: str = ""
    recorded_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "run_id": self.run_id,
            "run_status": self.run_status.value,
            "angle_id": self.angle_id,
            "angle_status": (
                self.angle_status.value if self.angle_status else ""
            ),
            "message": self.message,
            "recorded_at_ms": self.recorded_at_ms,
        }


__all__ = [
    "ANGLE_STAGE_ORDER",
    "BakeAngle",
    "BakeAngleJob",
    "BakeAngleKind",
    "BakeAngleResult",
    "BakeAngleStatus",
    "BakeContractError",
    "BakeRequest",
    "BakeRun",
    "BakeRunEvent",
    "BakeRunStatus",
    "RawMaterial",
    "VALID_CORPUS_MODES",
    "VALID_RAW_MATERIAL_KINDS",
    "angle_stage_index",
    "is_terminal_angle_status",
    "is_terminal_run_status",
    "rollup_run_status",
]
