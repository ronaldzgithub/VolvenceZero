"""Disk-backed resolver for the Einstein figure bundle.

Bridges the persisted F4 / F5 / F6 bake output (under
``<root>/einstein/<bundle_id>/{bundle.pickle,manifest.json}``,
written by :mod:`lifeform_domain_figure.bundle_io`) into the
service-layer Einstein verticals so the chat UI exercises the real
Wave K artefacts instead of the synthetic placeholder corpus.

Why a thin resolver and not direct ``load_figure_bundle`` calls from
:mod:`lifeform_service.verticals`:

* The verticals module already encodes the wiring matrix for every
  installed lifeform (companion / coding / zhang_wuji / einstein-*);
  resolving the on-disk bundle is a single responsibility that does
  not belong inline. Pulling it out keeps each ``_try_einstein_*``
  factory short and lets the smoke tests inject a tmp_path root.
* The resolver is the **only** module in this wheel that imports
  ``lifeform_domain_figure.bundle_io``; route code never touches the
  pickle path. That keeps the DLaaS allowlist invariant intact
  (platform tier never imports the domain wheel).

Environment variables (all optional):

* ``EINSTEIN_BUNDLE_ROOT`` -- directory containing
  ``einstein/<bundle_id>/`` directories. Defaults to the
  ``data/figure_bundles`` checkout-relative path produced by
  ``figure_demo_einstein.sh`` / ``figure_collect_einstein.sh``.
* ``EINSTEIN_BUNDLE_ID`` -- pin a specific bundle id. When unset the
  resolver picks the manifest with the newest ``created_at_iso``.
* ``EINSTEIN_REQUIRE_REAL_BUNDLE`` -- ``1`` to fail-loud when no disk
  bundle is reachable (production / staging mode). Defaults to ``0``
  so a fresh checkout that has not yet run the bake pipeline still
  starts (resolver falls back to ``synthetic_einstein_corpus()``).

Fail-loud is the default for **schema / integrity** errors regardless
of the optional flag: a present-but-broken bundle never silently
falls through to the synthetic path. The flag only controls the
"bundle simply missing on disk" branch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


_LOG = logging.getLogger("lifeform_service.einstein_resolver")


_DEFAULT_BUNDLE_ROOT = "data/figure_bundles"
_DEFAULT_FIGURE_ID = "einstein"


@dataclass(frozen=True)
class EinsteinBundleResolution:
    """Outcome of resolving the Einstein figure bundle.

    ``bundle`` is the loaded :class:`FigureArtifactBundle` typed as
    ``object`` so this module does not force ``lifeform_domain_figure``
    onto every service-layer importer; callers cast at the use site.

    ``source`` is one of:

    * ``"disk"`` -- ``bundle`` was loaded from
      ``<bundle_root>/einstein/<bundle_id>/``.
    * ``"synthetic"`` -- no disk bundle found AND
      ``EINSTEIN_REQUIRE_REAL_BUNDLE`` was not set; the resolver
      handed back ``None`` so the caller can compile a fresh bundle
      from ``synthetic_einstein_corpus()``.

    ``bundle_id`` is empty in the synthetic case; non-empty otherwise.
    """

    bundle: object | None
    bundle_id: str
    source: str
    bundle_root: Path | None


def resolve_einstein_bundle() -> EinsteinBundleResolution:
    """Resolve the Einstein figure bundle from disk (or fall back).

    See module docstring for environment variable semantics. Returns
    an :class:`EinsteinBundleResolution`; the caller decides what to
    do with a ``None`` bundle (typical pattern: build a synthetic
    one through ``build_einstein_lifeform`` which already encodes
    the fallback wiring).
    """

    bundle_root = _env_path("EINSTEIN_BUNDLE_ROOT") or Path(_DEFAULT_BUNDLE_ROOT)
    pinned_id = _env_str("EINSTEIN_BUNDLE_ID")
    require_real = _env_bool("EINSTEIN_REQUIRE_REAL_BUNDLE")

    figure_root = bundle_root / _DEFAULT_FIGURE_ID
    if not figure_root.is_dir():
        return _missing_disk_bundle(
            bundle_root=bundle_root,
            require_real=require_real,
            reason=f"no figure directory at {figure_root}",
        )

    try:
        from lifeform_domain_figure.bundle_io import (
            list_figure_bundles,
            load_figure_bundle,
        )
    except ImportError as exc:
        if require_real:
            raise RuntimeError(
                "EINSTEIN_REQUIRE_REAL_BUNDLE=1 but "
                "lifeform-domain-figure is not importable: "
                f"{exc}"
            ) from exc
        _LOG.warning(
            "resolve_einstein_bundle: lifeform-domain-figure not "
            "importable (%s); falling back to synthetic.",
            exc,
        )
        return EinsteinBundleResolution(
            bundle=None,
            bundle_id="",
            source="synthetic",
            bundle_root=bundle_root,
        )

    manifests = list_figure_bundles(
        root_dir=bundle_root, figure_id=_DEFAULT_FIGURE_ID
    )
    if not manifests:
        return _missing_disk_bundle(
            bundle_root=bundle_root,
            require_real=require_real,
            reason=f"no manifests under {figure_root}",
        )

    if pinned_id:
        match = next(
            (m for m in manifests if m.bundle_id == pinned_id), None
        )
        if match is None:
            raise FileNotFoundError(
                f"EINSTEIN_BUNDLE_ID={pinned_id!r} is set but no "
                f"matching manifest exists under {figure_root}. "
                f"Available: "
                f"{', '.join(m.bundle_id for m in manifests)}"
            )
        chosen = match
    else:
        chosen = manifests[-1]

    bundle = load_figure_bundle(
        root_dir=bundle_root,
        bundle_id=chosen.bundle_id,
        figure_id=_DEFAULT_FIGURE_ID,
    )
    _LOG.info(
        "resolve_einstein_bundle: loaded disk bundle id=%s "
        "created_at=%s lora_present=%s steering_present=%s "
        "root=%s",
        chosen.bundle_id,
        chosen.created_at_iso,
        chosen.lora_present,
        chosen.steering_present,
        bundle_root,
    )
    return EinsteinBundleResolution(
        bundle=bundle,
        bundle_id=chosen.bundle_id,
        source="disk",
        bundle_root=bundle_root,
    )


def _missing_disk_bundle(
    *,
    bundle_root: Path,
    require_real: bool,
    reason: str,
) -> EinsteinBundleResolution:
    if require_real:
        raise FileNotFoundError(
            "EINSTEIN_REQUIRE_REAL_BUNDLE=1 but resolver found no "
            f"Einstein bundle on disk: {reason} (root={bundle_root}). "
            "Run scripts/figure_collect_einstein.sh + "
            "`python -m lifeform_domain_figure.cli bake-bundle "
            "--figure einstein` first, or unset the flag to allow "
            "the synthetic fallback."
        )
    _LOG.warning(
        "resolve_einstein_bundle: %s; falling back to synthetic "
        "corpus. Set EINSTEIN_REQUIRE_REAL_BUNDLE=1 in production "
        "to fail-loud instead.",
        reason,
    )
    return EinsteinBundleResolution(
        bundle=None,
        bundle_id="",
        source="synthetic",
        bundle_root=bundle_root,
    )


def _env_str(name: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw


def _env_path(name: str) -> Path | None:
    raw = _env_str(name)
    return Path(raw) if raw else None


def _env_bool(name: str) -> bool:
    return _env_str(name).lower() in {"1", "true", "yes", "on"}


__all__ = (
    "EinsteinBundleResolution",
    "resolve_einstein_bundle",
)
