"""Typed dataclass specs for DLaaS control-plane resources.

Slice 3 / 4 of the rollout introduces persistent, multi-tenant state
(tenants / shells / assets / templates / contracts / focus_persons /
identity_links). Every wire-format shape lives in this module as a
frozen dataclass with explicit ``from_json`` / ``to_json`` symmetry.

Design rules carried from ``docs/specs/dlaas-platform.md``:

* Resources are **identifiers + immutable snapshot fields**. Mutable
  state (turn counters, regime, vitals, ...) belongs to the kernel
  facade and is read via vz-contracts snapshot types — never mirrored
  here. Resources here are governance metadata only.
* IDs are public, secrets are not. ``TenantSpec.api_secret`` is only
  populated immediately after creation; reads return an empty
  string. The registry persists the SHA-256 hash, never the plaintext.
* Status enums are typed (``TemplateStatus`` / ``ContractStatus``).
  No string magic at the wire boundary — invalid values are rejected
  at parse time so downstream logic always sees a known value.
* ``runtime_template_id`` on :class:`TemplateSpec` must resolve to a
  registered ``lifeform-service.verticals`` entry at activation /
  adoption time. Validation lives in
  ``dlaas-platform-registry`` because it requires reading the
  vertical registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dlaas_platform_contracts.plugins import PluginManifest


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantSpec:
    """Public tenant record.

    The ``api_secret`` field is populated **only** in the response of
    ``POST /dlaas/tenants``; subsequent ``GET`` calls always return
    ``""`` for it. The registry persists ``api_secret_hash`` (SHA-256
    of plaintext) and rejects all auth attempts that do not match.
    """

    tenant_id: str
    tenant_name: str
    contact_email: str
    business_type: str = "generic"
    billing_plan: str = "pay_as_you_go"
    quota: Mapping[str, Any] = field(default_factory=dict)
    api_key: str = ""
    api_secret: str = ""  # plaintext, only on create response
    created_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "TenantSpec":
        if not isinstance(data, Mapping):
            raise ValueError("TenantSpec payload must be a JSON object")
        tenant_id = str(data.get("tenant_id", "") or "")
        tenant_name = str(data.get("tenant_name", "") or "")
        contact_email = str(data.get("contact_email", "") or "")
        if not tenant_name.strip():
            raise ValueError("TenantSpec.tenant_name must be non-empty")
        if not contact_email.strip():
            raise ValueError("TenantSpec.contact_email must be non-empty")
        return cls(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            contact_email=contact_email,
            business_type=str(data.get("business_type", "generic") or "generic"),
            billing_plan=str(data.get("billing_plan", "pay_as_you_go") or "pay_as_you_go"),
            quota=dict(data.get("quota") or {}),
            api_key=str(data.get("api_key", "") or ""),
            api_secret=str(data.get("api_secret", "") or ""),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self, *, include_secret: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "contact_email": self.contact_email,
            "business_type": self.business_type,
            "billing_plan": self.billing_plan,
            "quota": dict(self.quota),
            "api_key": self.api_key,
            "created_at_ms": self.created_at_ms,
        }
        if include_secret:
            body["api_secret"] = self.api_secret
        return body


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


class ShellKind(str, Enum):
    """Whether a shell is for production deployment or training studio.

    Adoption only accepts ``DEPLOYMENT`` shells. ``STUDIO`` shells are
    used for control-plane training workflows (audience analysis,
    exam runs) and never receive end-user traffic.
    """

    DEPLOYMENT = "deployment"
    STUDIO = "studio"


@dataclass(frozen=True)
class ShellSpec:
    """Deployment / studio shell descriptor.

    ``embodiment`` mirrors the four-Kind affordance descriptor schema
    from ``lifeform-affordance``: ``perception`` / ``expression`` /
    ``action`` lists + ``constraints``. The platform stores it as
    a snapshot dict because shell capabilities evolve independently
    of the affordance registry's runtime state.
    """

    shell_id: str
    tenant_id: str
    shell_kind: ShellKind
    shell_type: str = "generic"
    display_name: str = ""
    embodiment: Mapping[str, Any] = field(default_factory=dict)
    channel: Mapping[str, Any] = field(default_factory=dict)
    scene_meta: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ShellSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ShellSpec payload must be a JSON object")
        shell_id = str(data.get("shell_id", "") or "")
        tenant_id = str(data.get("tenant_id", "") or "")
        if not shell_id.strip():
            raise ValueError("ShellSpec.shell_id must be non-empty")
        if not tenant_id.strip():
            raise ValueError("ShellSpec.tenant_id must be non-empty")
        kind_raw = str(data.get("shell_kind", "deployment") or "deployment")
        try:
            shell_kind = ShellKind(kind_raw.lower())
        except ValueError as exc:
            allowed = ", ".join(k.value for k in ShellKind)
            raise ValueError(
                f"ShellSpec.shell_kind must be one of: {allowed}"
            ) from exc
        return cls(
            shell_id=shell_id,
            tenant_id=tenant_id,
            shell_kind=shell_kind,
            shell_type=str(data.get("shell_type", "generic") or "generic"),
            display_name=str(data.get("display_name", "") or ""),
            embodiment=dict(data.get("embodiment") or {}),
            channel=dict(data.get("channel") or {}),
            scene_meta=dict(data.get("scene_meta") or {}),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "shell_id": self.shell_id,
            "tenant_id": self.tenant_id,
            "shell_kind": self.shell_kind.value,
            "shell_type": self.shell_type,
            "display_name": self.display_name,
            "embodiment": dict(self.embodiment),
            "channel": dict(self.channel),
            "scene_meta": dict(self.scene_meta),
            "created_at_ms": self.created_at_ms,
        }


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetSpec:
    """Tenant-owned content reference (chat logs, persona kits, manuals, …).

    Assets are referenced by ID from templates; the actual bytes live
    at ``uri`` (S3, HTTP, local file, …). The platform never streams
    asset content through the registry — it only persists metadata.
    Activation pulls the ``uri`` and feeds it to
    ``lifeform-ingestion`` (Slice 3.4).
    """

    asset_id: str
    tenant_id: str
    asset_type: str
    title: str = ""
    uri: str = ""
    mime_type: str = ""
    language: str = ""
    source_meta: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AssetSpec":
        if not isinstance(data, Mapping):
            raise ValueError("AssetSpec payload must be a JSON object")
        tenant_id = str(data.get("tenant_id", "") or "")
        asset_type = str(data.get("asset_type", "") or "")
        if not tenant_id.strip():
            raise ValueError("AssetSpec.tenant_id must be non-empty")
        if not asset_type.strip():
            raise ValueError("AssetSpec.asset_type must be non-empty")
        return cls(
            asset_id=str(data.get("asset_id", "") or ""),
            tenant_id=tenant_id,
            asset_type=asset_type,
            title=str(data.get("title", "") or ""),
            uri=str(data.get("uri", "") or ""),
            mime_type=str(data.get("mime_type", "") or ""),
            language=str(data.get("language", "") or ""),
            source_meta=dict(data.get("source_meta") or {}),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "tenant_id": self.tenant_id,
            "asset_type": self.asset_type,
            "title": self.title,
            "uri": self.uri,
            "mime_type": self.mime_type,
            "language": self.language,
            "source_meta": dict(self.source_meta),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class TemplateAssetLinkSpec:
    """Links an :class:`AssetSpec` to a template (and version)."""

    template_id: str
    asset_id: str
    template_version: int = 1
    role: str = "training_material"
    link_meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], *, template_id: str
    ) -> "TemplateAssetLinkSpec":
        if not isinstance(data, Mapping):
            raise ValueError("TemplateAssetLinkSpec payload must be a JSON object")
        asset_id = str(data.get("asset_id", "") or "")
        if not asset_id.strip():
            raise ValueError("TemplateAssetLinkSpec.asset_id must be non-empty")
        return cls(
            template_id=template_id,
            asset_id=asset_id,
            template_version=int(data.get("template_version", 1) or 1),
            role=str(data.get("role", "training_material") or "training_material"),
            link_meta=dict(data.get("link_meta") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "asset_id": self.asset_id,
            "template_version": self.template_version,
            "role": self.role,
            "link_meta": dict(self.link_meta),
        }


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class TemplateStatus(str, Enum):
    """Template lifecycle.

    * ``DRAFT`` — editable, not adoptable.
    * ``PUBLISHED`` — adoptable iff readiness gate passed.
    * ``DEPRECATED`` — kept for existing records, not recommended.
    """

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


class TemplateActivationStatus(str, Enum):
    """Activation lifecycle (Slice 3.4)."""

    UNACTIVATED = "unactivated"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    ACTIVATION_FAILED = "activation_failed"


class CitationPolicy(str, Enum):
    """How strictly the runtime enforces L3 grounded-decoding.

    Mirrors the ``score_threshold`` / ``cosine_floor`` knobs the
    figure vertical's ``GroundedDecoder`` consumes; the policy is
    the templating-side dial that governs whether the runtime
    refuses unsupported assertions, marks them, or passes through.
    """

    REQUIRED = "required"
    PREFERRED = "preferred"
    DISABLED = "disabled"


class CoveragePolicy(str, Enum):
    """How the runtime treats out-of-scope queries.

    Mirrors :class:`lifeform_expression.CoveragePolicy`; redeclared
    here so the platform tier does not import the lifeform layer.
    Renaming or reordering values here MUST be reflected on the
    expression-side enum (a contract test in P4.1 keeps the two
    aligned).
    """

    STRICT_REFUSE = "strict_refuse"
    SOFT_DISCLAIM = "soft_disclaim"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class TemplateSpec:
    """Template = persona + seed bundle.

    ``runtime_template_id`` bridges the control-plane template to a
    registered ``lifeform-service.verticals`` entry. Readiness gating
    requires it to be non-empty.

    The four ``figure_*`` fields below are **optional** — they are
    only populated for templates that carry a real-person figure
    artifact bundle compiled by ``lifeform-domain-figure``:

    * ``figure_artifact_id`` — opaque id of the registered
      :class:`lifeform_domain_figure.FigureArtifactBundle`. The
      runtime adopt path resolves the id to the actual frozen
      bundle via the figure-vertical loader.
    * ``citation_policy`` — :class:`CitationPolicy` controlling
      L3 enforcement.
    * ``coverage_policy`` — :class:`CoveragePolicy` controlling
      L4 enforcement.
    * ``figure_time_window`` — optional ``"window_id"`` selecting
      a :class:`TimeWindowedView` on the underlying profile. Empty
      string means "no window selected".

    These fields default to safe-no-op values so existing templates
    that do not opt into the figure vertical keep behaving exactly
    as before (additive-by-default discipline; matches the
    DLaaS-rollout slice 5.4 substrate streaming carve-out).
    """

    template_id: str
    tenant_id: str
    template_name: str
    domain: str = "generic"
    description: str = ""
    runtime_template_id: str = ""
    status: TemplateStatus = TemplateStatus.DRAFT
    current_version: int = 1
    activation_status: TemplateActivationStatus = TemplateActivationStatus.UNACTIVATED
    base_persona: Mapping[str, Any] = field(default_factory=dict)
    persona_spec: Mapping[str, Any] = field(default_factory=dict)
    seed_config: Mapping[str, Any] = field(default_factory=dict)
    activation_stats: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0
    figure_artifact_id: str = ""
    citation_policy: CitationPolicy = CitationPolicy.DISABLED
    coverage_policy: CoveragePolicy = CoveragePolicy.PASSTHROUGH
    figure_time_window: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "TemplateSpec":
        if not isinstance(data, Mapping):
            raise ValueError("TemplateSpec payload must be a JSON object")
        tenant_id = str(data.get("tenant_id", "") or "")
        template_name = str(data.get("template_name", "") or "")
        if not tenant_id.strip():
            raise ValueError("TemplateSpec.tenant_id must be non-empty")
        if not template_name.strip():
            raise ValueError("TemplateSpec.template_name must be non-empty")
        status_raw = str(data.get("status", "draft") or "draft").lower()
        try:
            status = TemplateStatus(status_raw)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in TemplateStatus)
            raise ValueError(
                f"TemplateSpec.status must be one of: {allowed}"
            ) from exc
        activation_raw = str(
            data.get("activation_status", "unactivated") or "unactivated"
        ).lower()
        try:
            activation = TemplateActivationStatus(activation_raw)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in TemplateActivationStatus)
            raise ValueError(
                f"TemplateSpec.activation_status must be one of: {allowed}"
            ) from exc
        citation_raw = str(
            data.get("citation_policy", CitationPolicy.DISABLED.value)
            or CitationPolicy.DISABLED.value
        ).lower()
        try:
            citation_policy = CitationPolicy(citation_raw)
        except ValueError as exc:
            allowed = ", ".join(p.value for p in CitationPolicy)
            raise ValueError(
                f"TemplateSpec.citation_policy must be one of: {allowed}"
            ) from exc
        coverage_raw = str(
            data.get("coverage_policy", CoveragePolicy.PASSTHROUGH.value)
            or CoveragePolicy.PASSTHROUGH.value
        ).lower()
        try:
            coverage_policy = CoveragePolicy(coverage_raw)
        except ValueError as exc:
            allowed = ", ".join(p.value for p in CoveragePolicy)
            raise ValueError(
                f"TemplateSpec.coverage_policy must be one of: {allowed}"
            ) from exc
        return cls(
            template_id=str(data.get("template_id", "") or ""),
            tenant_id=tenant_id,
            template_name=template_name,
            domain=str(data.get("domain", "generic") or "generic"),
            description=str(data.get("description", "") or ""),
            runtime_template_id=str(data.get("runtime_template_id", "") or ""),
            status=status,
            current_version=int(data.get("current_version", 1) or 1),
            activation_status=activation,
            base_persona=dict(data.get("base_persona") or {}),
            persona_spec=dict(data.get("persona_spec") or {}),
            seed_config=dict(data.get("seed_config") or {}),
            activation_stats=dict(data.get("activation_stats") or {}),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
            figure_artifact_id=str(data.get("figure_artifact_id", "") or ""),
            citation_policy=citation_policy,
            coverage_policy=coverage_policy,
            figure_time_window=str(data.get("figure_time_window", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "tenant_id": self.tenant_id,
            "template_name": self.template_name,
            "domain": self.domain,
            "description": self.description,
            "runtime_template_id": self.runtime_template_id,
            "status": self.status.value,
            "current_version": self.current_version,
            "activation_status": self.activation_status.value,
            "base_persona": dict(self.base_persona),
            "persona_spec": dict(self.persona_spec),
            "seed_config": dict(self.seed_config),
            "activation_stats": dict(self.activation_stats),
            "created_at_ms": self.created_at_ms,
            "figure_artifact_id": self.figure_artifact_id,
            "citation_policy": self.citation_policy.value,
            "coverage_policy": self.coverage_policy.value,
            "figure_time_window": self.figure_time_window,
        }

    @property
    def has_figure_artifact(self) -> bool:
        """Whether this template binds a figure artifact bundle."""
        return bool(self.figure_artifact_id.strip())


@dataclass(frozen=True)
class TemplateVersionSpec:
    """Immutable snapshot of a template at a specific version.

    Created on every ``PATCH /dlaas/templates/{id}`` and on explicit
    ``snapshot`` calls. The platform NEVER mutates an existing
    version — adoption pins a contract to one ``(template_id,
    template_version)`` pair forever.
    """

    version_id: str
    template_id: str
    version_number: int
    snapshot: Mapping[str, Any]
    version_note: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "template_id": self.template_id,
            "version_number": self.version_number,
            "snapshot": dict(self.snapshot),
            "version_note": self.version_note,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Result of ``GET /dlaas/templates/{id}/readiness``.

    ``ready`` is ``True`` only when ``missing`` is empty AND
    ``activation_status == ACTIVATED`` AND ``has_runtime_template_id``
    is True. Templates that fail readiness cannot be moved to
    ``PUBLISHED`` status.
    """

    template_id: str
    ready: bool
    missing: tuple[str, ...] = ()
    activation_status: TemplateActivationStatus = TemplateActivationStatus.UNACTIVATED
    has_runtime_template_id: bool = False
    world_nodes: int = 0
    self_nodes: int = 0
    l2_cards: int = 0
    snapshot_summary: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "ready": self.ready,
            "missing": list(self.missing),
            "activation_status": self.activation_status.value,
            "has_runtime_template_id": self.has_runtime_template_id,
            "world_nodes": self.world_nodes,
            "self_nodes": self.self_nodes,
            "l2_cards": self.l2_cards,
            "snapshot_summary": dict(self.snapshot_summary),
        }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class ContractStatus(str, Enum):
    """Contract lifecycle (mirrors DLaaS public taxonomy)."""

    CREATED = "created"
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    FAILED = "failed"


@dataclass(frozen=True)
class ContractSpec:
    """Binding between tenant × template × shell × tool policy × ai_id.

    ``ai_id`` is empty until adoption succeeds; afterwards it is
    immutable for the life of the contract. ``tool_policy_snapshot``
    is the frozen ``AffordanceRegistry`` capability allowlist computed
    at adoption time.
    """

    contract_id: str
    tenant_id: str
    template_id: str
    template_version: int
    shell_id: str
    ai_id: str = ""
    owner_user_id: str = ""
    engine_tools: Mapping[str, Any] = field(default_factory=dict)
    tool_policy_snapshot: Mapping[str, Any] = field(default_factory=dict)
    service_contract: Mapping[str, Any] = field(default_factory=dict)
    contract_status: ContractStatus = ContractStatus.CREATED
    created_at_ms: int = 0
    # debt #PluginFoundation: per-contract plugin manifests resolved
    # from the application bundles approved by this contract's
    # tenant. Empty tuple on legacy contracts created before the
    # plugin foundation rolled out — :meth:`from_json` accepts a
    # missing ``plugins`` key as ``()`` so existing DB rows
    # deserialise unchanged.
    plugins: tuple[PluginManifest, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ContractSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ContractSpec payload must be a JSON object")
        for key in ("tenant_id", "template_id", "shell_id"):
            if not str(data.get(key, "") or "").strip():
                raise ValueError(f"ContractSpec.{key} must be non-empty")
        status_raw = str(
            data.get("contract_status", "created") or "created"
        ).lower()
        try:
            status = ContractStatus(status_raw)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in ContractStatus)
            raise ValueError(
                f"ContractSpec.contract_status must be one of: {allowed}"
            ) from exc
        plugins_raw = data.get("plugins") or ()
        if not isinstance(plugins_raw, (list, tuple)):
            raise ValueError("ContractSpec.plugins must be a list of objects")
        plugins = tuple(
            PluginManifest.from_json(item) for item in plugins_raw
        )
        return cls(
            contract_id=str(data.get("contract_id", "") or ""),
            tenant_id=str(data["tenant_id"]),
            template_id=str(data["template_id"]),
            template_version=int(data.get("template_version", 1) or 1),
            shell_id=str(data["shell_id"]),
            ai_id=str(data.get("ai_id", "") or ""),
            owner_user_id=str(data.get("owner_user_id", "") or ""),
            engine_tools=dict(data.get("engine_tools") or {}),
            tool_policy_snapshot=dict(data.get("tool_policy_snapshot") or {}),
            service_contract=dict(data.get("service_contract") or {}),
            contract_status=status,
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
            plugins=plugins,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "tenant_id": self.tenant_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "shell_id": self.shell_id,
            "ai_id": self.ai_id,
            "owner_user_id": self.owner_user_id,
            "engine_tools": dict(self.engine_tools),
            "tool_policy_snapshot": dict(self.tool_policy_snapshot),
            "service_contract": dict(self.service_contract),
            "contract_status": self.contract_status.value,
            "created_at_ms": self.created_at_ms,
            "plugins": [plugin.to_json() for plugin in self.plugins],
        }


# ---------------------------------------------------------------------------
# Focus Person + Identity Link
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FocusPersonSpec:
    """Canonical "person of interest" record bound to one ai_id.

    The platform persists the identifier + display fields. Cognitive
    state (belief / preference / role) is owned by
    ``vz-cognition.social_cognition.*`` and read via snapshots. Writes
    on a focus person flow through ``submit_profile_event`` so the
    platform never becomes a second owner.
    """

    person_id: str
    contract_id: str
    name: str = ""
    role: str = "user"
    relationship_to_owner: str = ""
    age: int | None = None
    initial_profile: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], *, contract_id: str
    ) -> "FocusPersonSpec":
        if not isinstance(data, Mapping):
            raise ValueError("FocusPersonSpec payload must be a JSON object")
        person_id = str(data.get("person_id", "") or "")
        if not person_id.strip():
            raise ValueError("FocusPersonSpec.person_id must be non-empty")
        age_raw = data.get("age")
        age: int | None = None
        if age_raw is not None:
            try:
                age = int(age_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "FocusPersonSpec.age must be an integer if present"
                ) from exc
        return cls(
            person_id=person_id,
            contract_id=contract_id,
            name=str(data.get("name", "") or ""),
            role=str(data.get("role", "user") or "user"),
            relationship_to_owner=str(
                data.get("relationship_to_owner", "") or ""
            ),
            age=age,
            initial_profile=dict(data.get("initial_profile") or {}),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "contract_id": self.contract_id,
            "name": self.name,
            "role": self.role,
            "relationship_to_owner": self.relationship_to_owner,
            "age": self.age,
            "initial_profile": dict(self.initial_profile),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class IdentityLinkSpec:
    """Maps a per-channel ID to a canonical ``end_user_ref``.

    ``UserIdentity.scope_key`` consumers can compose
    ``f"{tenant_id}/{ai_id}/{canonical_end_user_ref}"`` without
    touching ``vz-memory`` schema.
    """

    ai_id: str
    channel_type: str
    channel_ref: str
    canonical_end_user_ref: str
    link_meta: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], *, ai_id: str
    ) -> "IdentityLinkSpec":
        if not isinstance(data, Mapping):
            raise ValueError("IdentityLinkSpec payload must be a JSON object")
        for key in ("channel_type", "channel_ref", "canonical_end_user_ref"):
            if not str(data.get(key, "") or "").strip():
                raise ValueError(f"IdentityLinkSpec.{key} must be non-empty")
        return cls(
            ai_id=ai_id,
            channel_type=str(data["channel_type"]),
            channel_ref=str(data["channel_ref"]),
            canonical_end_user_ref=str(data["canonical_end_user_ref"]),
            link_meta=dict(data.get("link_meta") or {}),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "channel_type": self.channel_type,
            "channel_ref": self.channel_ref,
            "canonical_end_user_ref": self.canonical_end_user_ref,
            "link_meta": dict(self.link_meta),
            "created_at_ms": self.created_at_ms,
        }


# ---------------------------------------------------------------------------
# Handoff (Slice 5)
# ---------------------------------------------------------------------------


class HandoffStatus(str, Enum):
    OPEN = "open"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class HandoffTicketSpec:
    """One handoff queue entry; created when ops detects an escalation."""

    ticket_id: str
    ai_id: str
    contract_id: str
    end_user_ref: str
    session_id: str = ""
    trigger_reason: str = ""
    trigger_details: Mapping[str, Any] = field(default_factory=dict)
    confidence_aggregate: float = 0.0
    recent_response_ids: tuple[str, ...] = ()
    status: HandoffStatus = HandoffStatus.OPEN
    operator_id: str = ""
    human_reply: str = ""
    resolution_notes: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "end_user_ref": self.end_user_ref,
            "session_id": self.session_id,
            "trigger_reason": self.trigger_reason,
            "trigger_details": dict(self.trigger_details),
            "confidence_aggregate": self.confidence_aggregate,
            "recent_response_ids": list(self.recent_response_ids),
            "status": self.status.value,
            "operator_id": self.operator_id,
            "human_reply": self.human_reply,
            "resolution_notes": self.resolution_notes,
            "created_at_ms": self.created_at_ms,
        }


__all__ = [
    "AssetSpec",
    "ContractSpec",
    "ContractStatus",
    "FocusPersonSpec",
    "HandoffStatus",
    "HandoffTicketSpec",
    "IdentityLinkSpec",
    "ReadinessReport",
    "ShellKind",
    "ShellSpec",
    "TemplateActivationStatus",
    "TemplateAssetLinkSpec",
    "TemplateSpec",
    "TemplateStatus",
    "TemplateVersionSpec",
    "TenantSpec",
]
