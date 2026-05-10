"""Browser-chat template adapter for the character vertical.

Implements :class:`lifeform_service.templates.VerticalTemplateAdapter`
so the chat UI can list / load / save :class:`LifeformTemplate`s
without ``lifeform-service`` ever importing the character vertical
directly. The adapter glues four existing pieces together:

* :func:`build_zhang_wuji_lifeform` — vertical default factory.
* :func:`give_birth` — reincarnate a saved template.
* :func:`save_lifeform_template` — write a new template.
* :class:`LifeformTemplate` — the on-disk schema.

R8 posture (the constraint that drives the design here):

* The adapter is the **only** thing the service layer sees. It
  treats it as opaque (a Protocol). When a future vertical (e.g.
  ``lifeform-domain-coding``) wants to support templates, it ships
  its own adapter with its own template type — no edits to
  ``lifeform-service`` are required.
* The :class:`TemplateContext.payload` we round-trip carries the
  reviewed :class:`CharacterSoulProfile` (and optionally an
  ``evolved_profile``). Memory state is **not** stashed in the
  context: at save time we read it fresh from
  ``session.brain_session.memory_store`` so we capture the lived
  state up to "now", not a stale snapshot taken at session
  creation.
* Save-as-template defaults to ``include_memory=True`` to mirror
  ``examples/train_zhang_wuji_template.py`` — that script is the
  canonical reference for what a saved template looks like, and
  diverging from it would create two parallel save shapes.

What this is NOT:

* It is **not** a DLaaS ``tpl_*`` registry adapter. The DLaaS
  control-plane template store is a separate SSOT (multi-tenant,
  contract-bound, billing-aware); this adapter exclusively serves
  the local browser-chat / studio lane.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from lifeform_core import Lifeform, LifeformConfig, LifeformSession
from lifeform_service.templates import (
    TemplateContext,
    TemplateMetadata,
    VerticalTemplateAdapter,
)

from lifeform_domain_character.lifeform_builder import (
    build_zhang_wuji_lifeform,
)
from lifeform_domain_character.profile import CharacterSoulProfile
from lifeform_domain_character.profiles import build_zhang_wuji_profile
from lifeform_domain_character.template import (
    IncompatibleTemplateVersion,
    LifeformTemplate,
)
from lifeform_domain_character.template_load import RebirthBundle, give_birth
from lifeform_domain_character.template_save import (
    save_lifeform_template,
    vitals_drive_levels_from_session,
)


if TYPE_CHECKING:
    from volvence_zero.memory import IdentityProvider, MemoryStore
    from volvence_zero.substrate import OpenWeightResidualRuntime


@dataclass(frozen=True)
class _CharacterContextPayload:
    """Typed view of the opaque ``TemplateContext.payload`` we attach.

    Stored as a frozen dataclass so contract violations (missing
    fields) fail loudly at attribute-access time rather than via
    ``dict.get(default=None)`` silent drift.
    """

    profile: CharacterSoulProfile
    evolved_profile: CharacterSoulProfile | None
    source_template_id: str | None
    source_arc_id: str | None


class CharacterTemplateAdapter(VerticalTemplateAdapter):
    """Concrete adapter for the character vertical.

    One adapter instance is created per :class:`VerticalSpec`. The
    adapter is stateless apart from the optional response-synthesizer
    factory (the vertical's verticals.py wires both).
    """

    def __init__(
        self,
        *,
        default_profile_factory: Any = build_zhang_wuji_profile,
        response_synthesizer_factory: Any = None,
        semantic_proposal_runtime_factory: Any = None,
    ) -> None:
        self._default_profile_factory = default_profile_factory
        self._response_synthesizer_factory = response_synthesizer_factory
        self._semantic_proposal_runtime_factory = (
            semantic_proposal_runtime_factory
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_templates(self, root_dir: pathlib.Path) -> tuple[TemplateMetadata, ...]:
        if not root_dir.exists():
            return ()
        if not root_dir.is_dir():
            raise NotADirectoryError(
                f"CharacterTemplateAdapter: templates root_dir={root_dir} "
                "exists but is not a directory"
            )
        out: list[TemplateMetadata] = []
        for path in sorted(root_dir.glob("*.json")):
            metadata = self._read_metadata(path)
            if metadata is not None:
                out.append(metadata)
        return tuple(out)

    def _read_metadata(self, path: pathlib.Path) -> TemplateMetadata | None:
        """Pull the manifest from a template file without inflating it.

        Reading just the manifest is ~10x cheaper than full JSON
        parse on large templates. We still ``json.loads`` the whole
        file because canonical Python json has no streaming-key
        primitive, but we do not call ``LifeformTemplate.from_json_bytes``
        (which validates the entire schema) — for a listing the
        manifest plus profile.character_name is enough. Verticals
        that need stronger guarantees can opt-in to full validation
        later.
        """
        try:
            raw = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # Corrupted file in the templates dir: fail loudly so the
            # operator sees it. We re-raise rather than silently skip
            # because a hidden corrupt file would cause confusing UI
            # behavior ("template missing from dropdown").
            raise
        if not isinstance(raw, dict):
            raise ValueError(
                f"CharacterTemplateAdapter: {path} top-level JSON must be a dict"
            )
        manifest = raw.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(
                f"CharacterTemplateAdapter: {path} missing 'manifest' object"
            )
        profile = raw.get("profile") or {}
        display_name = (
            str(profile.get("character_name") or manifest.get("template_id") or path.stem)
        )
        description_parts: list[str] = []
        source_title = profile.get("source_title")
        if isinstance(source_title, str) and source_title.strip():
            description_parts.append(source_title.strip())
        version = profile.get("version")
        if isinstance(version, str) and version.strip():
            description_parts.append(f"v={version.strip()}")
        description = " | ".join(description_parts)
        return TemplateMetadata(
            template_id=str(manifest.get("template_id") or path.stem),
            display_name=display_name,
            description=description,
            source_arc_id=manifest.get("source_arc_id"),
            replay_provenance=str(manifest.get("replay_provenance") or ""),
            created_at_utc=str(manifest.get("created_at_utc") or ""),
            integrity_hash=str(manifest.get("integrity_hash") or ""),
            file_path=str(path),
        )

    # ------------------------------------------------------------------
    # Build (default)
    # ------------------------------------------------------------------

    def build_default_session_context(
        self,
        *,
        runtime: "OpenWeightResidualRuntime | None",
        identity_provider: "IdentityProvider | None",
        memory_scope_root_dir: str | None,
        alpha_enabled: bool,
    ) -> tuple[Lifeform, TemplateContext]:
        profile = self._default_profile_factory()
        synthesizer = self._build_synthesizer(
            runtime, repair_alpha_enabled=alpha_enabled
        )
        semantic_runtime = self._build_semantic_runtime(runtime)
        config = self._build_config(
            alpha_enabled=alpha_enabled,
            memory_scope_root_dir=memory_scope_root_dir,
        )
        bundle = build_zhang_wuji_lifeform(
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=semantic_runtime,
            identity_provider=identity_provider if alpha_enabled else None,
        )
        return bundle.lifeform, _wrap_payload(
            _CharacterContextPayload(
                profile=profile,
                evolved_profile=None,
                source_template_id=None,
                source_arc_id=None,
            )
        )

    # ------------------------------------------------------------------
    # Build (from template)
    # ------------------------------------------------------------------

    def build_session_context_from_template(
        self,
        *,
        root_dir: pathlib.Path,
        template_id: str,
        runtime: "OpenWeightResidualRuntime | None",
        identity_provider: "IdentityProvider | None",
        memory_scope_root_dir: str | None,
        alpha_enabled: bool,
    ) -> tuple[Lifeform, TemplateContext]:
        template_path = self._resolve_template_path(root_dir, template_id)
        synthesizer = self._build_synthesizer(
            runtime, repair_alpha_enabled=alpha_enabled
        )
        semantic_runtime = self._build_semantic_runtime(runtime)
        config = self._build_config(
            alpha_enabled=alpha_enabled,
            memory_scope_root_dir=memory_scope_root_dir,
        )
        # Alpha mode + saved template: keep the template's profile /
        # drives / evolved profile as the trained base, but skip the
        # template's frozen memory checkpoint so the per-user
        # filesystem store accumulates on top instead of inheriting
        # another user's lived memories. Non-alpha keeps the full
        # checkpoint for studio-style replay continuity.
        bundle: RebirthBundle = give_birth(
            template_path,
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=semantic_runtime,
            identity_provider=identity_provider if alpha_enabled else None,
            skip_memory_restore=alpha_enabled,
        )
        payload = _CharacterContextPayload(
            profile=bundle.profile,
            evolved_profile=bundle.template.evolved_profile,
            source_template_id=bundle.template.manifest.template_id,
            source_arc_id=bundle.template.manifest.source_arc_id,
        )
        return bundle.lifeform, _wrap_payload(payload)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_session_as_template(
        self,
        *,
        session: LifeformSession,
        context: TemplateContext,
        root_dir: pathlib.Path,
        template_id: str,
        replay_provenance: str,
        include_memory: bool,
        overwrite_existing: bool,
    ) -> TemplateMetadata:
        payload = _unwrap_payload(context)
        if not template_id.strip():
            raise ValueError("save_session_as_template: template_id is empty")
        if "/" in template_id or "\\" in template_id or template_id in {".", ".."}:
            raise ValueError(
                f"save_session_as_template: template_id={template_id!r} "
                "must not contain path separators or be relative-path tokens"
            )
        root_dir.mkdir(parents=True, exist_ok=True)
        memory_store: "MemoryStore | None" = None
        if include_memory:
            memory_store = _extract_memory_store(session)
        vitals_levels = vitals_drive_levels_from_session(session)
        provenance = replay_provenance.strip() or _default_provenance(
            payload, template_id
        )
        save_result = save_lifeform_template(
            profile=payload.profile,
            evolved_profile=payload.evolved_profile,
            template_id=template_id,
            output_dir=root_dir,
            memory_store=memory_store,
            vitals_drive_levels=vitals_levels,
            replay_report=None,
            source_arc_id=payload.source_arc_id,
            replay_provenance=provenance,
            overwrite_existing=overwrite_existing,
        )
        return TemplateMetadata(
            template_id=save_result.template.manifest.template_id,
            display_name=save_result.template.profile.character_name,
            description=save_result.template.profile.source_title,
            source_arc_id=save_result.template.manifest.source_arc_id,
            replay_provenance=save_result.template.manifest.replay_provenance,
            created_at_utc=save_result.template.manifest.created_at_utc,
            integrity_hash=save_result.template.manifest.integrity_hash,
            file_path=str(save_result.template_path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_template_path(
        self, root_dir: pathlib.Path, template_id: str
    ) -> pathlib.Path:
        if not template_id.strip():
            raise ValueError("template_id is empty")
        if "/" in template_id or "\\" in template_id:
            raise ValueError(
                f"template_id={template_id!r} must not contain path separators"
            )
        path = root_dir / f"{template_id}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Template {template_id!r} not found under {root_dir}"
            )
        return path

    def _build_synthesizer(
        self,
        runtime: "OpenWeightResidualRuntime | None",
        *,
        repair_alpha_enabled: bool,
    ) -> Any:
        if self._response_synthesizer_factory is None:
            return None
        return self._response_synthesizer_factory(
            runtime, repair_alpha_enabled=repair_alpha_enabled
        )

    def _build_semantic_runtime(
        self, runtime: "OpenWeightResidualRuntime | None"
    ) -> Any:
        if self._semantic_proposal_runtime_factory is None:
            return None
        return self._semantic_proposal_runtime_factory(runtime)

    def _build_config(
        self,
        *,
        alpha_enabled: bool,
        memory_scope_root_dir: str | None,
    ) -> LifeformConfig | None:
        # Only mint a fresh LifeformConfig in alpha mode where we need
        # to thread BrainConfig.memory_scope_root_dir down. Non-alpha
        # uses the lifeform builder's default (a stock LifeformConfig)
        # so we do not silently override the vertical's defaults.
        if not alpha_enabled:
            return None
        from volvence_zero.brain import BrainConfig  # local import: optional dep

        return LifeformConfig(
            brain_config=BrainConfig(memory_scope_root_dir=memory_scope_root_dir)
        )


def _wrap_payload(payload: _CharacterContextPayload) -> TemplateContext:
    return TemplateContext(payload={"_character_payload": payload})


def _unwrap_payload(context: TemplateContext) -> _CharacterContextPayload:
    inner = context.payload.get("_character_payload")
    if not isinstance(inner, _CharacterContextPayload):
        raise TypeError(
            "TemplateContext was not produced by CharacterTemplateAdapter "
            f"(got payload keys={sorted(context.payload.keys())!r}). "
            "This usually means a vertical adapter mismatch — the session "
            "was created by one adapter and saved by another."
        )
    return inner


def _extract_memory_store(session: LifeformSession) -> Any:
    """Pull the live :class:`MemoryStore` from a running session.

    Access path: ``LifeformSession.brain_session`` (public on
    lifeform-core) → ``BrainSession.runner`` (public on vz-runtime)
    → ``AgentSessionRunner.memory_store`` (public property on
    vz-runtime). All three are non-private; we deliberately do
    **not** reach into ``_memory_store`` private fields. If any of
    these accessors disappears in a future kernel revision, this
    raises ``AttributeError`` and we update both ends together
    rather than silently producing memory-less templates.
    """
    brain_session = session.brain_session
    runner = brain_session.runner
    return runner.memory_store


def _default_provenance(
    payload: _CharacterContextPayload, template_id: str
) -> str:
    parts = [f"browser-chat-save:{template_id}"]
    if payload.source_template_id:
        parts.append(f"derived_from={payload.source_template_id}")
    if payload.source_arc_id:
        parts.append(f"arc={payload.source_arc_id}")
    return " ".join(parts)


def build_character_template_adapter(
    *,
    default_profile_factory: Any = build_zhang_wuji_profile,
    response_synthesizer_factory: Any = None,
    semantic_proposal_runtime_factory: Any = None,
) -> CharacterTemplateAdapter:
    """Convenience constructor.

    Verticals that ship multiple character profiles (e.g. a future
    ``character-multi`` vertical) can instantiate one adapter per
    default profile. Today only the 张无忌 vertical uses it.
    """
    return CharacterTemplateAdapter(
        default_profile_factory=default_profile_factory,
        response_synthesizer_factory=response_synthesizer_factory,
        semantic_proposal_runtime_factory=semantic_proposal_runtime_factory,
    )


__all__ = [
    "CharacterTemplateAdapter",
    "IncompatibleTemplateVersion",
    "LifeformTemplate",
    "build_character_template_adapter",
]
