"""Generic disk-backed resolver for any figure's baked bundle.

This is the slug-generic sibling of
:mod:`lifeform_service.einstein_resolver`. Where the Einstein resolver
is pinned to ``figure_id="einstein"``, this module resolves the baked
:class:`FigureArtifactBundle` for an arbitrary figure ``slug`` so the
generic ``figure-companion`` / ``figure-full`` verticals (D-press-1 /
D25) can wake non-Einstein figures — volvence-press / novel-worlds
author companions, coread historical figures, family / myriad personas
— against their own boundary instead of falling back to
``einstein-full``.

Why a thin resolver and not direct ``load_figure_bundle`` calls from
:mod:`lifeform_service.verticals`:

* It is the **only** module besides :mod:`einstein_resolver` that
  imports ``lifeform_domain_figure.bundle_io`` (deferred to call time);
  route code never touches the pickle path, keeping the DLaaS allowlist
  invariant intact (platform tier never imports the domain wheel).
* It keeps each ``_try_figure_*`` factory short and lets smoke tests
  inject a tmp_path root.

Environment variables (all optional):

* ``FIGURE_BUNDLE_ROOT`` — directory containing ``<figure_id>/<bundle_id>/``
  directories. Defaults to the ``data/figure_bundles`` checkout-relative
  path (same root the Einstein bake scripts use).
* ``<SLUG_ENV>`` — the env var (name supplied by the caller, e.g.
  ``FIGURE_COMPANION_SLUG``) holding the figure ``slug`` / ``figure_id``
  to resolve. When unset, no disk bundle is resolved and the caller
  uses its documented placeholder fallback.
* ``FIGURE_REQUIRE_REAL_BUNDLE`` — ``1`` to fail-loud when a slug is
  configured but no bundle is reachable on disk (production / staging
  mode). Defaults to ``0`` so a fresh checkout still starts.

Fail-loud is the default for **schema / integrity** errors regardless
of the optional flag (the underlying ``load_figure_bundle`` raises on a
present-but-broken bundle). The flag only controls the
"slug configured but bundle simply missing on disk" branch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


_LOG = logging.getLogger("lifeform_service.figure_resolver")

_DEFAULT_BUNDLE_ROOT = "data/figure_bundles"


@dataclass(frozen=True)
class FigureBundleResolution:
    """Outcome of resolving a figure's baked bundle from disk.

    ``bundle`` is the loaded :class:`FigureArtifactBundle` typed as
    ``object`` so this module does not force ``lifeform_domain_figure``
    onto every service-layer importer; callers cast at the use site.

    ``source`` is one of:

    * ``"disk"`` — ``bundle`` was loaded from
      ``<bundle_root>/<figure_id>/<bundle_id>/``.
    * ``"unbound"`` — no slug was configured; the caller should use its
      placeholder fallback (the per-ai_id bundle is bound later by the
      DLaaS adopt path).
    * ``"missing"`` — a slug WAS configured but no bundle was found on
      disk AND ``FIGURE_REQUIRE_REAL_BUNDLE`` was not set.

    ``figure_id`` is the resolved slug (empty in the ``unbound`` case);
    ``bundle_id`` is non-empty only in the ``disk`` case.
    """

    bundle: object | None
    figure_id: str
    bundle_id: str
    source: str
    bundle_root: Path


def resolve_figure_bundle(*, slug_env: str) -> FigureBundleResolution:
    """Resolve the figure bundle named by ``os.environ[slug_env]``.

    See the module docstring for env-var semantics. Returns a
    :class:`FigureBundleResolution`; the caller decides what to do with
    a ``None`` bundle (typical pattern: build a placeholder lifeform
    that the adopt path rebinds per ai_id).
    """

    bundle_root = _env_path("FIGURE_BUNDLE_ROOT") or Path(_DEFAULT_BUNDLE_ROOT)
    slug = _env_str(slug_env)
    require_real = _env_bool("FIGURE_REQUIRE_REAL_BUNDLE")

    if not slug:
        # No figure pinned for this vertical's process-default; the real
        # bundle arrives via the DLaaS adopt path per ai_id.
        return FigureBundleResolution(
            bundle=None,
            figure_id="",
            bundle_id="",
            source="unbound",
            bundle_root=bundle_root,
        )

    figure_root = bundle_root / slug
    if not figure_root.is_dir():
        return _missing_disk_bundle(
            slug=slug,
            bundle_root=bundle_root,
            require_real=require_real,
            reason=f"no figure directory at {figure_root}",
        )

    from lifeform_domain_figure.bundle_io import (
        list_figure_bundles,
        load_figure_bundle,
    )

    manifests = list_figure_bundles(root_dir=bundle_root, figure_id=slug)
    if not manifests:
        return _missing_disk_bundle(
            slug=slug,
            bundle_root=bundle_root,
            require_real=require_real,
            reason=f"no manifests under {figure_root}",
        )

    chosen = manifests[-1]
    bundle = load_figure_bundle(
        root_dir=bundle_root,
        bundle_id=chosen.bundle_id,
        figure_id=slug,
    )
    _LOG.info(
        "resolve_figure_bundle: loaded disk bundle figure_id=%s id=%s "
        "created_at=%s lora_present=%s root=%s",
        slug,
        chosen.bundle_id,
        chosen.created_at_iso,
        chosen.lora_present,
        bundle_root,
    )
    return FigureBundleResolution(
        bundle=bundle,
        figure_id=slug,
        bundle_id=chosen.bundle_id,
        source="disk",
        bundle_root=bundle_root,
    )


def _missing_disk_bundle(
    *,
    slug: str,
    bundle_root: Path,
    require_real: bool,
    reason: str,
) -> FigureBundleResolution:
    if require_real:
        raise FileNotFoundError(
            "FIGURE_REQUIRE_REAL_BUNDLE=1 but resolver found no bundle on "
            f"disk for slug={slug!r}: {reason} (root={bundle_root}). Bake "
            "the figure with `python -m lifeform_domain_figure.cli "
            "bake-bundle --figure <slug> ...` first, or unset the flag to "
            "allow the placeholder fallback."
        )
    _LOG.warning(
        "resolve_figure_bundle: slug=%s configured but %s; caller will use "
        "its placeholder fallback. Set FIGURE_REQUIRE_REAL_BUNDLE=1 in "
        "production to fail-loud instead.",
        slug,
        reason,
    )
    return FigureBundleResolution(
        bundle=None,
        figure_id=slug,
        bundle_id="",
        source="missing",
        bundle_root=bundle_root,
    )


def _env_str(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_path(name: str) -> Path | None:
    raw = _env_str(name)
    return Path(raw) if raw else None


def _env_bool(name: str) -> bool:
    return _env_str(name).lower() in {"1", "true", "yes", "on"}


__all__ = (
    "FigureBundleResolution",
    "resolve_figure_bundle",
)
