"""Browser-chat template surface (vertical-agnostic).

This module defines the **service-side** template-management contract:
the typed metadata the chat UI sees, an opaque per-session
``TemplateContext`` the service round-trips on behalf of the vertical,
and the :class:`VerticalTemplateAdapter` Protocol that vertical wheels
implement to plug their own template I/O into the browser-chat surface.

R8 posture (read this before touching):

* The template format itself is owned by the vertical wheel
  (e.g. ``LifeformTemplate`` lives in ``lifeform-domain-character``
  because it carries a typed ``CharacterSoulProfile``). Putting the
  adapter Protocol here means ``lifeform-service`` does **not** need
  to import any specific ``lifeform-domain-*`` wheel — verticals
  inject their adapter on the :class:`VerticalSpec` and the service
  treats it as opaque.
* ``TemplateContext`` is the service-side equivalent of a snapshot:
  the vertical produces it on session creation, the service stashes
  it on ``_SessionEntry``, and hands it back when the user clicks
  "Save as Template". The service never inspects ``payload``.
* DLaaS multi-tenant ``tpl_*`` resources are a **different SSOT**
  (``dlaas-platform-registry.TemplateStore``). This module is for the
  dev / studio browser-chat lane only and never crosses that
  boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable


if TYPE_CHECKING:
    from lifeform_core import Lifeform, LifeformSession
    from volvence_zero.memory import IdentityProvider
    from volvence_zero.substrate import OpenWeightResidualRuntime


@dataclass(frozen=True)
class TemplateMetadata:
    """User-facing template summary for the chat UI.

    Fields are deliberately serialisable primitives — the UI only
    needs to render a dropdown plus a tooltip/description. Verticals
    that carry additional structured fields can extend by stashing
    them in :class:`TemplateContext.payload` instead.
    """

    template_id: str
    display_name: str
    description: str
    source_arc_id: str | None
    replay_provenance: str
    created_at_utc: str
    integrity_hash: str
    file_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "display_name": self.display_name,
            "description": self.description,
            "source_arc_id": self.source_arc_id,
            "replay_provenance": self.replay_provenance,
            "created_at_utc": self.created_at_utc,
            "integrity_hash": self.integrity_hash,
            "file_path": self.file_path,
        }


@dataclass(frozen=True)
class TemplateContext:
    """Opaque per-session payload the vertical attaches at session
    creation time and consumes back on save-as-template.

    The service treats ``payload`` as a black box — only the vertical
    that produced it knows the shape. Mapping[str, Any] is the right
    type because it is structurally equivalent to a snapshot value
    (frozen dataclass field on a service-side _SessionEntry record).

    A session that was created **without** a template adapter (legacy
    ``factory`` / ``alpha_factory`` path) carries ``None`` instead of
    a ``TemplateContext``; in that case ``save-as-template`` returns
    ``503 templates_not_supported``.
    """

    payload: Mapping[str, Any]


@runtime_checkable
class VerticalTemplateAdapter(Protocol):
    """Vertical-side template I/O bridge.

    A vertical that wants to participate in the browser-chat template
    surface implements this Protocol and registers an instance on its
    :class:`VerticalSpec.template_adapter` field. Verticals without
    template support leave that field ``None``; the service then
    advertises ``templates_supported=False`` and refuses save calls.

    All four methods are synchronous on purpose — they run on the
    aiohttp event loop and the underlying I/O (JSON file read /
    write, MemoryStore checkpoint) is fast enough that switching to
    async would just add complexity. If a vertical's loader becomes
    expensive, route the heavy work through ``run_in_executor`` at
    the implementation level rather than changing this Protocol.
    """

    def list_templates(self, root_dir: Path) -> tuple[TemplateMetadata, ...]:
        """Return every template available under ``root_dir``.

        Implementations should fail loudly on parse errors (R8: a
        corrupted template file is a contract breach, not a silent
        skip). Empty directory returns an empty tuple.
        """

    def build_default_session_context(
        self,
        *,
        runtime: "OpenWeightResidualRuntime | None",
        identity_provider: "IdentityProvider | None",
        memory_scope_root_dir: str | None,
        alpha_enabled: bool,
    ) -> "tuple[Lifeform, TemplateContext]":
        """Build a Lifeform from the vertical's default profile.

        Equivalent to the vertical's ``factory(runtime)`` /
        ``alpha_factory(runtime, identity, root)`` path, except it
        also returns a :class:`TemplateContext` so the resulting
        session can be saved as a new template later.

        The returned ``TemplateContext.payload`` typically captures
        the profile that was used; the actual ``MemoryStore`` is
        read off ``session.brain_session.memory_store`` at save time.
        """

    def build_session_context_from_template(
        self,
        *,
        root_dir: Path,
        template_id: str,
        runtime: "OpenWeightResidualRuntime | None",
        identity_provider: "IdentityProvider | None",
        memory_scope_root_dir: str | None,
        alpha_enabled: bool,
    ) -> "tuple[Lifeform, TemplateContext]":
        """Build a Lifeform by reincarnating a saved template.

        ``alpha_enabled=True`` typically maps to
        ``give_birth(skip_memory_restore=True)`` so per-user
        accumulated memory layers on top of the template's profile
        and drives instead of inheriting another user's frozen
        memory snapshot (see ``template_load.py``'s alpha-path
        comment).
        """

    def save_session_as_template(
        self,
        *,
        session: "LifeformSession",
        context: TemplateContext,
        root_dir: Path,
        template_id: str,
        replay_provenance: str,
        include_memory: bool,
        overwrite_existing: bool,
    ) -> TemplateMetadata:
        """Capture the current session's lived state as a new template.

        Returns the metadata for the freshly written template.
        Implementations must validate ``template_id`` (non-empty,
        filesystem-safe) and respect ``overwrite_existing``.
        """


__all__ = [
    "TemplateContext",
    "TemplateMetadata",
    "VerticalTemplateAdapter",
]
