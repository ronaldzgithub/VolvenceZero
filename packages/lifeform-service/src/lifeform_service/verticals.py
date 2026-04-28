"""Vertical registry \u2014 maps vertical name to a Lifeform factory.

The service layer must not import any specific ``lifeform-domain-*``
wheel directly in route code (that would couple the service to a single
vertical). Instead we keep the import locally here, so when more verticals
are added (``lifeform-domain-coding``, ``lifeform-domain-customer-service``,
\u2026) only this module changes.

A vertical entry is a small dataclass with:

* ``name`` \u2014 stable string identifier shipped in API responses.
* ``factory`` \u2014 zero-arg callable returning a ``Lifeform``. Both axes'
  bootstraps are pre-wired through the vertical's own factory; the
  service does not assemble bootstraps itself.
* ``has_temporal_bootstrap`` / ``has_regime_bootstrap`` \u2014 advertised
  capability flags so ``GET /v1/info`` can tell the client whether the
  vertical ships pre-trained calibration.
* ``bootstraps_dir`` / ``scenarios_dir`` \u2014 paths shown in ``/v1/info``
  for diagnostics; ``None`` when the vertical does not ship them.

If the importing wheel is missing (e.g. someone ran ``pip install
lifeform-service`` without ``lifeform-domain-emogpt``), the registry
catches the ImportError and skips that vertical rather than failing
service start. This keeps a future ``lifeform-service`` install useful
even with vertical-only optional deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from lifeform_core import Lifeform


@dataclass(frozen=True)
class VerticalSpec:
    name: str
    factory: Callable[[], Lifeform]
    has_temporal_bootstrap: bool
    has_regime_bootstrap: bool
    bootstraps_dir: str | None = None
    scenarios_dir: str | None = None


def _try_companion() -> VerticalSpec | None:
    try:
        from lifeform_domain_emogpt import (
            bootstraps_dir,
            build_companion_lifeform,
            scenarios_dir,
        )
    except ImportError:
        return None
    bdir = bootstraps_dir()
    sdir = scenarios_dir()
    return VerticalSpec(
        name="companion",
        factory=build_companion_lifeform,
        has_temporal_bootstrap=(bdir / "companion-temporal.snap").is_file(),
        has_regime_bootstrap=(bdir / "companion-regime.bs").is_file(),
        bootstraps_dir=str(bdir) if bdir.is_dir() else None,
        scenarios_dir=str(sdir) if sdir.is_dir() else None,
    )


def _try_uncalibrated_companion() -> VerticalSpec | None:
    """Bare companion vertical without the pre-trained bootstraps.

    Always reachable as ``companion-cold`` for ablation runs, even when
    the bootstrap files are intentionally missing.
    """
    try:
        from lifeform_domain_emogpt import build_companion_lifeform
    except ImportError:
        return None
    return VerticalSpec(
        name="companion-cold",
        factory=lambda: build_companion_lifeform(
            use_temporal_bootstrap=False, use_regime_bootstrap=False
        ),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )


_BUILDERS = (_try_companion, _try_uncalibrated_companion)


def discover_verticals() -> dict[str, VerticalSpec]:
    """Return every vertical that successfully imports in this environment."""
    out: dict[str, VerticalSpec] = {}
    for builder in _BUILDERS:
        spec = builder()
        if spec is not None:
            out[spec.name] = spec
    return out


def default_vertical_name() -> str:
    """First vertical that loaded successfully wins as default.

    In normal installs that is ``companion``. In a kernel-only install
    (lifeform-service without lifeform-domain-emogpt) we raise rather than
    silently fall through to no default.
    """
    discovered = discover_verticals()
    if not discovered:
        raise RuntimeError(
            "No lifeform-domain-* wheel is installed alongside lifeform-service. "
            "Install lifeform-domain-emogpt (or another vertical) before serving."
        )
    return next(iter(discovered.keys()))
