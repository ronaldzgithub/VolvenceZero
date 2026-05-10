"""Multi-vertical registry for the browser-chat service.

The dev / browser-chat lane needs to host **every** discovered
vertical at once (companion / zhang_wuji / einstein / coding /
growth_advisor / companion-cold) so the chat UI can pick which
one a new session is bound to. This module centralises the
``{vertical_name -> VerticalSpec}`` map plus the "which one is
the default for sessions that don't declare a vertical?" pointer.

Why a small typed wrapper (instead of a bare ``dict``):

1. A frozen dataclass gives us one canonical place to validate
   ``default_name in by_name`` and "default supports alpha when
   alpha is enabled" — both invariants that previously only held
   implicitly in :func:`lifeform_service.app.create_app`.
2. The UI surface (``GET /v1/verticals``) needs the same shape
   from many call sites; centralising :meth:`VerticalRegistry.summary_for_ui`
   keeps the wire format SSOT.
3. Per-session vertical lookup (``manager.create_session(vertical=...)``)
   becomes ``registry.require(name)``, fail-loud, no defensive
   ``getattr(...)`` chains.

R8 posture:

* The registry is the single owner of "which verticals are
  available to this service process and which is the default".
* ``SessionManager`` consumes the registry through it; it never
  caches a ``VerticalSpec`` reference outside ``_SessionEntry``
  (where the binding is per-session-immutable, not a competing
  registry).
* DLaaS production paths (``dlaas-platform-launcher.InstanceManager``)
  bypass the registry by constructing ``SessionManager`` with
  legacy ``lifeform_factory + alpha_lifeform_factory`` args. The
  back-compat shim in ``SessionManager.__init__`` synthesises a
  one-entry registry in that case so the rest of the class stays
  uniform — DLaaS multi-instance multi-vertical wiring continues
  to live in the launcher, not in this registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lifeform_service.verticals import VerticalSpec


@dataclass(frozen=True)
class VerticalRegistry:
    """Service-level vertical registry.

    Constructed once at ``create_app`` time. ``alpha_enabled`` is a
    capability gate: when True, vertical selection from the chat UI
    or HTTP layer must reject verticals without an
    ``alpha_factory``. The registry itself does not enforce this on
    construction (we still allow non-alpha verticals to be listed
    so the UI can show them as disabled / informational); the
    enforcement lives at session-creation time in
    :meth:`SessionManager.create_session`.
    """

    default_name: str
    by_name: Mapping[str, VerticalSpec]
    alpha_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.by_name:
            raise ValueError("VerticalRegistry needs a non-empty by_name map")
        if self.default_name not in self.by_name:
            raise ValueError(
                f"VerticalRegistry default_name={self.default_name!r} not in "
                f"by_name keys {sorted(self.by_name.keys())!r}"
            )
        default_spec = self.by_name[self.default_name]
        if self.alpha_enabled and default_spec.alpha_factory is None:
            raise ValueError(
                f"VerticalRegistry alpha_enabled=True but default "
                f"vertical {self.default_name!r} has no alpha_factory; "
                "pick a different default or disable alpha"
            )

    @classmethod
    def single(
        cls,
        spec: VerticalSpec,
        *,
        alpha_enabled: bool = False,
    ) -> "VerticalRegistry":
        """Construct a one-entry registry from a single VerticalSpec.

        Used by ``create_app(vertical=...)`` back-compat path and
        by :class:`SessionManager`'s legacy ctor shim.
        """
        return cls(
            default_name=spec.name,
            by_name={spec.name: spec},
            alpha_enabled=alpha_enabled,
        )

    @classmethod
    def from_mapping(
        cls,
        verticals: Mapping[str, VerticalSpec] | Iterable[VerticalSpec],
        *,
        default_name: str,
        alpha_enabled: bool = False,
    ) -> "VerticalRegistry":
        if isinstance(verticals, Mapping):
            by_name = dict(verticals)
        else:
            by_name = {spec.name: spec for spec in verticals}
        return cls(
            default_name=default_name,
            by_name=by_name,
            alpha_enabled=alpha_enabled,
        )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def default(self) -> VerticalSpec:
        return self.by_name[self.default_name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.by_name.keys())

    def get(self, name: str) -> VerticalSpec | None:
        return self.by_name.get(name)

    def require(self, name: str) -> VerticalSpec:
        """Return the spec for ``name`` or raise :class:`UnknownVerticalError`.

        Used by ``create_session(vertical=...)`` so a typo'd vertical
        name fails loudly with a typed exception the route layer can
        map to HTTP 422 — never silently fall back to the default
        (would mask client bugs).
        """
        spec = self.by_name.get(name)
        if spec is None:
            raise UnknownVerticalError(
                f"vertical {name!r} not registered; available: "
                f"{sorted(self.by_name.keys())!r}"
            )
        return spec

    def is_alpha_capable(self, name: str) -> bool:
        spec = self.by_name.get(name)
        return spec is not None and spec.alpha_factory is not None

    def supports_templates(self, name: str) -> bool:
        spec = self.by_name.get(name)
        return spec is not None and spec.template_adapter is not None

    def summary_for_ui(self) -> tuple[dict[str, Any], ...]:
        """Render the registry as a JSON-friendly tuple.

        Wire format consumed by ``GET /v1/verticals`` and the chat
        UI's vertical dropdown. Each entry advertises capability
        flags so the UI can grey out alpha-incompatible options
        when alpha mode is on, and highlight which verticals
        support save-as-template.
        """
        out: list[dict[str, Any]] = []
        for name, spec in self.by_name.items():
            out.append(
                {
                    "name": name,
                    "is_default": name == self.default_name,
                    "alpha_supported": spec.alpha_factory is not None,
                    "templates_supported": spec.template_adapter is not None,
                    "has_temporal_bootstrap": spec.has_temporal_bootstrap,
                    "has_regime_bootstrap": spec.has_regime_bootstrap,
                    "bootstraps_dir": spec.bootstraps_dir,
                    "scenarios_dir": spec.scenarios_dir,
                    "template_subdir": spec.template_subdir or name,
                }
            )
        return tuple(out)


class UnknownVerticalError(LookupError):
    """Raised when a request references a vertical name that is not
    registered in the service's :class:`VerticalRegistry`. The route
    layer maps this to HTTP 422 ``unknown_vertical`` so the client
    can surface a typo / wrong-deployment error.
    """


class VerticalNotAlphaCapableError(ValueError):
    """Raised when a session is requested for a vertical that has no
    ``alpha_factory`` while the service is running in alpha mode.
    The route layer maps this to HTTP 422 ``vertical_not_alpha_capable``.
    """


__all__ = [
    "UnknownVerticalError",
    "VerticalNotAlphaCapableError",
    "VerticalRegistry",
]
