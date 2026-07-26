"""Per-turn carrier attestation for state-delivery evidence.

Owner of the three audit fingerprints defined in
``docs/specs/state-kv-identification-evidence.md`` §审计标签. They exist so
the claim "this relationship state did not reach the model through the
prompt" is checkable by a third party instead of asserted:

- ``prompt_fingerprint`` — canonical digest of the exact chat messages sent
  to the substrate (carrier C1 + C2). Two turns with equal fingerprints
  received byte-identical prompts.
- ``decode_fingerprint`` — digest of the sampling-layer configuration
  (carrier C5). Reported, not closed: decode config is snapshot-derived, so
  two users can differ here even with an identical prompt. A verdict may
  only claim "the only difference is internal state" when these match too.

Both are emitted on every LLM turn, not only in experiment arms: an
attestation that appears only when convenient proves nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from volvence_zero.agent.response import GenerationConstraints

__all__ = [
    "AUDIT_FINGERPRINT_LENGTH",
    "prompt_fingerprint",
    "decode_fingerprint",
]

# 64 bits of digest: collision-free for comparing arms within a run, short
# enough to read in a rationale tag.
AUDIT_FINGERPRINT_LENGTH = 16


def _digest(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
        :AUDIT_FINGERPRINT_LENGTH
    ]


def prompt_fingerprint(*, messages: Sequence[tuple[str, str]]) -> str:
    """Fingerprint the assembled chat messages.

    Must be computed on the same object handed to ``runtime.generate()``.
    Recomputing a supposedly-equivalent prompt would attest to something
    that was never sent.
    """

    return _digest([[role, content] for role, content in messages])


def decode_fingerprint(
    *,
    constraints: "GenerationConstraints | None",
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Fingerprint the sampling-layer configuration for this turn.

    Covers the snapshot-derived decode surface (``GenerationConstraints``)
    plus the two synthesizer-level knobs. Control code / control scale are
    deliberately excluded: those are model-layer carriers (C4), not
    sampling configuration, and folding them in here would let a
    latent-carrier difference masquerade as a decode difference.
    """

    payload: dict[str, object] = {
        "temperature": round(float(temperature), 6),
        "max_new_tokens": int(max_new_tokens),
    }
    if constraints is None:
        payload["constraints"] = None
    else:
        payload["constraints"] = {
            "response_mode": constraints.response_mode,
            "answer_depth_limit": constraints.answer_depth_limit,
            "citation_mode": constraints.citation_mode,
            "max_questions": constraints.max_questions,
            "required_disclaimer_phrases": list(
                constraints.required_disclaimer_phrases
            ),
            "ordering_bias": list(constraints.ordering_bias),
            "continuum_target_position": round(
                float(constraints.continuum_target_position), 6
            ),
            "ordering_driver": constraints.ordering_driver,
            "decoding_profile": constraints.decoding_profile,
            "question_budget": constraints.question_budget,
        }
    return _digest(payload)
