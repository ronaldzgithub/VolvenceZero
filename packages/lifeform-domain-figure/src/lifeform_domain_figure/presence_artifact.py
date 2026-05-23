"""L0 visual presence layer for the figure vertical.

Adds a frozen, hash-addressed :class:`FigurePresenceArtifact` to the
figure compilation pipeline. The artifact captures the *metadata* a
deploy-side rendering plane (``apps/presence-service``) needs to render
a figure's web digital-human + lipsync stream:

* a stable persona handle (``persona_id_at_presence_service``) the
  rendering plane already knows,
* the reference image / voice clone / motion model URIs (opaque to
  this wheel; the rendering plane resolves them),
* a hashed likeness consent token + license label so audit can replay
  who agreed to what,
* the engine subset that should be considered usable for this figure.

Importantly, **no model weights or pixels live in this artifact**. The
figure wheel is the source of truth for *who* the figure is and *what*
permissions exist; the rendering plane is the source of truth for
*how* the bytes are produced. This keeps R2 (vz-* diff = 0) intact and
keeps the rendering tech swappable on its own clock.

Position in the fidelity ladder (see
``docs/specs/figure-vertical.md``):

* L0 表象保真 — appearance / lipsync (this artifact)
* L1 语气保真 — voice text-style prior + persona LoRA
* L2 立场保真 — steering + persona LoRA
* L3 引证保真 — retrieval + grounded decoder
* L4 不知拒答 — coverage map + scope refuser

Default is :class:`FigureArtifactBundle.presence is None`, which keeps
the bundle integrity hash byte-stable for legacy bundles. When a
non-None presence artifact is attached, a deterministic fingerprint is
folded into the bundle hash so a presence revoke or re-bake produces a
fresh ``bundle_id`` (R15 byte-level rollback contract).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


SCHEMA_VERSION = 1


PresenceEngineId = Literal["server-photoreal", "client-3d"]
"""Stable string ids matching ``apps/presence-service`` PresenceEngineId."""


@dataclass(frozen=True)
class FigurePresenceArtifact:
    """Frozen metadata bundle for the L0 visual presence layer.

    Construct only via :func:`build_figure_presence_artifact` so the
    integrity hash is computed in one place.
    """

    schema_version: int
    presence_artifact_id: str
    figure_id: str
    persona_id_at_presence_service: str
    reference_image_uri: str
    voice_clone_id: str
    motion_model_id: str
    supported_engines: tuple[PresenceEngineId, ...]
    likeness_consent_token_hash: str
    license_label: str
    irreversible_likeness: bool
    integrity_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "FigurePresenceArtifact.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.presence_artifact_id.strip():
            raise ValueError(
                "FigurePresenceArtifact.presence_artifact_id must be non-empty"
            )
        if not self.figure_id.strip():
            raise ValueError(
                "FigurePresenceArtifact.figure_id must be non-empty"
            )
        if not self.persona_id_at_presence_service.strip():
            raise ValueError(
                "FigurePresenceArtifact.persona_id_at_presence_service must be non-empty"
            )
        if not self.reference_image_uri.strip():
            raise ValueError(
                "FigurePresenceArtifact.reference_image_uri must be non-empty"
            )
        if not self.license_label.strip():
            raise ValueError(
                "FigurePresenceArtifact.license_label must be non-empty"
            )
        if not self.likeness_consent_token_hash.strip():
            raise ValueError(
                "FigurePresenceArtifact.likeness_consent_token_hash must be non-empty; "
                "the rendering plane requires a hashed consent receipt for audit replay."
            )
        if len(self.supported_engines) == 0:
            raise ValueError(
                "FigurePresenceArtifact.supported_engines must declare at least one engine"
            )
        for engine in self.supported_engines:
            if engine not in ("server-photoreal", "client-3d"):
                raise ValueError(
                    f"FigurePresenceArtifact.supported_engines: unknown engine {engine!r}"
                )
        if not self.integrity_hash.strip():
            raise ValueError(
                "FigurePresenceArtifact.integrity_hash must be non-empty"
            )


def compute_presence_integrity_hash(
    *,
    figure_id: str,
    persona_id_at_presence_service: str,
    reference_image_uri: str,
    voice_clone_id: str,
    motion_model_id: str,
    supported_engines: tuple[PresenceEngineId, ...],
    likeness_consent_token_hash: str,
    license_label: str,
    irreversible_likeness: bool,
) -> str:
    """Deterministic SHA-256 over the presence artifact's identity fields.

    The same inputs produce the same hash byte-for-byte (R15). This hash
    is also the value that gets folded into
    :class:`FigureArtifactBundle.integrity_hash` when a non-None
    presence artifact is attached, so any change to the consent token,
    license, or reference image yields a fresh ``bundle_id`` and is
    rollback-addressable.
    """

    payload: tuple[object, ...] = (
        SCHEMA_VERSION,
        figure_id,
        persona_id_at_presence_service,
        reference_image_uri,
        voice_clone_id,
        motion_model_id,
        tuple(sorted(supported_engines)),
        likeness_consent_token_hash,
        license_label,
        irreversible_likeness,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def hash_consent_token(plain_token: str) -> str:
    """Hash a plain consent token before storing it in the artifact.

    Mirrors ``apps/presence-service`` `crypto.sha256Hex`. The plain
    token is held only by the operator/legal system that issued it; the
    figure bundle and the rendering plane both store the hash.
    """

    if not plain_token or len(plain_token) < 8:
        raise ValueError(
            "hash_consent_token: plain_token must be at least 8 chars; "
            "presence consent receipts should not be empty or trivial"
        )
    return hashlib.sha256(plain_token.encode("utf-8")).hexdigest()


def presence_artifact_id_from_hash(figure_id: str, integrity_hash: str) -> str:
    """Compose a stable, human-readable presence-artifact id."""

    return f"figure-presence:{figure_id}:{integrity_hash[:16]}"


def build_figure_presence_artifact(
    *,
    figure_id: str,
    persona_id_at_presence_service: str,
    reference_image_uri: str,
    voice_clone_id: str | None,
    motion_model_id: str | None,
    supported_engines: tuple[PresenceEngineId, ...],
    likeness_consent_token: str,
    license_label: str,
    irreversible_likeness: bool = True,
) -> FigurePresenceArtifact:
    """Construct a :class:`FigurePresenceArtifact` with derived hashes.

    The plain ``likeness_consent_token`` is hashed before being stored;
    callers are expected to retain the plain token in the operator /
    legal system that issued it (the figure wheel never sees it again).

    This is intentionally **metadata-only**: no neural training, no
    model weight production. The actual rendering models live behind
    ``apps/presence-service`` and evolve on their own clock. Bake-time
    semantics (rolling out a new model) are handled by re-baking the
    presence artifact with an updated ``motion_model_id`` and replaying
    consent — which yields a different ``presence_artifact_id`` and a
    different bundle hash, so the rollback contract holds.
    """

    voice_id = (voice_clone_id or "").strip()
    motion_id = (motion_model_id or "").strip()
    consent_hash = hash_consent_token(likeness_consent_token)
    integrity_hash = compute_presence_integrity_hash(
        figure_id=figure_id,
        persona_id_at_presence_service=persona_id_at_presence_service,
        reference_image_uri=reference_image_uri,
        voice_clone_id=voice_id,
        motion_model_id=motion_id,
        supported_engines=supported_engines,
        likeness_consent_token_hash=consent_hash,
        license_label=license_label,
        irreversible_likeness=irreversible_likeness,
    )
    return FigurePresenceArtifact(
        schema_version=SCHEMA_VERSION,
        presence_artifact_id=presence_artifact_id_from_hash(
            figure_id, integrity_hash
        ),
        figure_id=figure_id,
        persona_id_at_presence_service=persona_id_at_presence_service,
        reference_image_uri=reference_image_uri,
        voice_clone_id=voice_id,
        motion_model_id=motion_id,
        supported_engines=tuple(supported_engines),
        likeness_consent_token_hash=consent_hash,
        license_label=license_label,
        irreversible_likeness=irreversible_likeness,
        integrity_hash=integrity_hash,
    )


def presence_fingerprint(presence: FigurePresenceArtifact | None) -> str:
    """Fingerprint folded into :class:`FigureArtifactBundle.integrity_hash`.

    Returns ``""`` when ``presence is None`` so legacy bundles stay
    byte-stable. Returns the artifact's ``integrity_hash`` otherwise.
    """

    if presence is None:
        return ""
    return presence.integrity_hash


__all__ = [
    "SCHEMA_VERSION",
    "FigurePresenceArtifact",
    "PresenceEngineId",
    "build_figure_presence_artifact",
    "compute_presence_integrity_hash",
    "hash_consent_token",
    "presence_artifact_id_from_hash",
    "presence_fingerprint",
]
