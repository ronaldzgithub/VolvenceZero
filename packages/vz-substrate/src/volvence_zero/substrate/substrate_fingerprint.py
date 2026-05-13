"""Substrate compatibility fingerprint (debt #47 / F-C SHADOW).

A frozen, content-addressed identifier for a substrate model
configuration. Used by:

* :class:`lifeform_domain_figure.FigureArtifactBundle` —
  ``compatible_substrates`` field declares which substrate
  fingerprints a baked LoRA / steering bundle can run on
  (R15 byte-level rollback contract).
* :class:`lifeform_domain_growth_advisor.GrowthAdvisorProfile` —
  ``validated_substrates`` field declares which substrate
  fingerprints the reviewed profile has been validated against
  (advisory; runtime warns on mismatch but does not fail).
* :class:`companion_bench.RunRecord` — optional
  ``sut_substrate_fingerprint`` so the public leaderboard can
  group results by SUT substrate.

See:

* ``docs/specs/substrate-upgrade-protocol.md``
* ``docs/moving forward/cross-cutting-foundation-packet.md`` §2.3
* ``docs/known-debts.md`` #47
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SubstrateFingerprint:
    """Identifies a substrate model + version + weight hash.

    * ``model_id`` — HuggingFace-style model id (e.g.
      ``"Qwen/Qwen2.5-1.5B-Instruct"``) or a synthetic backend label
      (e.g. ``"synthetic-tiny-gpt2"``).
    * ``version`` — model release version (``"v2.5"`` /
      ``"legacy"`` for migration shim).
    * ``weights_sha256`` — SHA-256 of the substrate ``state_dict``;
      this is the primary key for "is this exactly the same weights?"
      Use ``"legacy"`` as a sentinel for old bundles that predate the
      fingerprint field (migration shim).
    """

    model_id: str
    version: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("SubstrateFingerprint.model_id must be non-empty")
        if not self.version.strip():
            raise ValueError("SubstrateFingerprint.version must be non-empty")
        if not self.weights_sha256.strip():
            raise ValueError(
                "SubstrateFingerprint.weights_sha256 must be non-empty "
                "(use 'legacy' for migration shim)"
            )

    def to_short_id(self) -> str:
        """Compact identifier for log lines / leaderboard display."""

        if self.weights_sha256 == "legacy":
            return f"{self.model_id}@{self.version}#legacy"
        return f"{self.model_id}@{self.version}#{self.weights_sha256[:8]}"

    def is_legacy(self) -> bool:
        """Whether this is the migration-shim sentinel."""

        return self.weights_sha256 == "legacy"


# Migration shim: any bundle saved before #47 land defaults to this.
LEGACY_FINGERPRINT = SubstrateFingerprint(
    model_id="tinygpt2",
    version="legacy",
    weights_sha256="legacy",
)


def fingerprint_set_sha256(fingerprints: tuple[SubstrateFingerprint, ...]) -> str:
    """Deterministic SHA-256 over a tuple of fingerprints.

    Used by :func:`compute_bundle_integrity_hash` to fold
    compatibility into the bundle hash. Empty input returns ``""``
    so existing bundles (without ``compatible_substrates``) keep
    byte-stable hashes.
    """

    if not fingerprints:
        return ""
    payload = tuple(
        (fp.model_id, fp.version, fp.weights_sha256) for fp in fingerprints
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = (
    "LEGACY_FINGERPRINT",
    "SubstrateFingerprint",
    "fingerprint_set_sha256",
)
