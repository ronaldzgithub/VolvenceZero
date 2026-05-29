"""Portable resolution of saved PEFT adapter checkpoint directories.

The figure-vertical bake job (:class:`PEFTLoRABakeBackend`) saves the
trained LoRA A/B matrices via ``peft.save_pretrained`` under a
checkpoint root and records the **absolute** path on
:attr:`FigureLoRAArtifact.peft_checkpoint_dir`. That absolute path is
brittle: a service started in a different working directory (or on a
different machine / container image) than the bake job cannot find the
checkpoint, so the adopt path silently degraded to the LayerNorm-eaten
projected-summary hook.

This module centralises the checkpoint root and a re-rooting resolver
so the *logical* key of a checkpoint (``<figure_id>/<plan_hash>`` under
a ``peft-checkpoints`` directory) stays stable across processes while
the physical root is configurable via ``VZ_PEFT_CHECKPOINT_ROOT``.

Design (R8 / no-swallow-errors):

* The resolver never silently swallows a missing checkpoint into an
  empty string. An empty input maps to an empty output (the documented
  "no real checkpoint, use the hook fallback" signal); a non-empty
  input that cannot be located is returned best-effort so the
  downstream :meth:`activate_peft_adapter` raises a loud
  ``FileNotFoundError`` pointing at the missing path.
"""

from __future__ import annotations

import os
import pathlib

CHECKPOINT_ROOT_ENV = "VZ_PEFT_CHECKPOINT_ROOT"

# Logical anchor segment. Both the bake-side default cache root and the
# re-rooting resolver key off this directory name so a checkpoint baked
# under ``<cwd>/.local/peft-checkpoints/<figure>/<hash>`` can be located
# under a different physical root that ends in the same anchor.
CHECKPOINT_ANCHOR = "peft-checkpoints"

_DEFAULT_ROOT = pathlib.Path(".local") / CHECKPOINT_ANCHOR


def peft_checkpoint_root() -> pathlib.Path:
    """Return the configured PEFT checkpoint root.

    Honours the ``VZ_PEFT_CHECKPOINT_ROOT`` environment variable; falls
    back to ``<cwd>/.local/peft-checkpoints``. The bake job and the
    runtime adopt path both read this so a deployment can point both at
    a shared mount with a single variable.
    """

    env = os.environ.get(CHECKPOINT_ROOT_ENV, "").strip()
    if env:
        return pathlib.Path(env)
    return _DEFAULT_ROOT


def _reroot_under(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path | None:
    """Re-root ``path`` under ``root`` using the stable anchor tail.

    Returns ``<root>/<tail>`` where ``tail`` is the portion of ``path``
    after its last :data:`CHECKPOINT_ANCHOR` segment, or ``None`` when
    ``path`` has no anchor segment (so the caller can fall back).
    """

    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == CHECKPOINT_ANCHOR:
            tail = parts[index + 1 :]
            if tail:
                return root.joinpath(*tail)
            return None
    return None


def resolve_peft_checkpoint_dir(stored: str) -> str:
    """Resolve a stored checkpoint path to an existing absolute path.

    Resolution order:

    1. Empty / blank input -> ``""`` (no checkpoint; hook fallback).
    2. The stored path itself if it is an existing directory.
    3. A relative stored path joined under :func:`peft_checkpoint_root`.
    4. The stored path re-rooted under :func:`peft_checkpoint_root`
       using the stable ``peft-checkpoints/<figure>/<hash>`` tail.
    5. Otherwise the stored value verbatim, so
       :meth:`activate_peft_adapter` fails loud with the path it could
       not find (never a silent degrade to the hook path).
    """

    if not stored or not stored.strip():
        return ""
    path = pathlib.Path(stored)
    if path.is_dir():
        return str(path.resolve())
    root = peft_checkpoint_root()
    if not path.is_absolute():
        candidate = root / path
        if candidate.is_dir():
            return str(candidate.resolve())
    rerooted = _reroot_under(path, root)
    if rerooted is not None and rerooted.is_dir():
        return str(rerooted.resolve())
    return str(path)


__all__ = [
    "CHECKPOINT_ANCHOR",
    "CHECKPOINT_ROOT_ENV",
    "peft_checkpoint_root",
    "resolve_peft_checkpoint_dir",
]
