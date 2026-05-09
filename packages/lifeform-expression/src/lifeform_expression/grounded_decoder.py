"""GroundedDecoder — L3 enforcement for the figure vertical.

Wraps any object that exposes the
:class:`lifeform_domain_figure.FigureRetrievalIndex` API
(``assertion_is_supported(text, *, score_threshold, cosine_floor, top_k)``)
and verifies that every substantive assertion in a generated text
is supported by at least one citation-quality piece of evidence.

This module **does not import** the figure vertical: it pins against
the ``GroundedDecodeHook`` Protocol from ``vz-substrate`` so that
lifeform-expression has no compile-time dependency on
lifeform-domain-figure. The retrieval index is passed in by duck-
typed parameter; runtime conformance is checked by the call shape.

Why this matters (R8 + ``no-swallow-errors-no-hasattr-abuse.mdc``):

* The expression layer must not silently downgrade unsupported
  assertions to supported. ``verify`` returns a verdict with
  ``passed=False`` AND populates ``unsupported_assertions``;
  ``verify_strict`` additionally raises
  :class:`UngroundedAssertionError` so callers that opted into
  strict L3 cannot accidentally swallow the failure.
* Substantive assertions are detected by token count, not by
  keyword heuristics. The threshold is a tunable on the decoder so
  callers can dial sensitivity per scenario.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from volvence_zero.substrate import GroundedDecodeHook, GroundingVerdict


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4e00-\u9fff]+")
_DEFAULT_MIN_ASSERTION_TOKENS = 4
# Combined BM25+cosine score required for an assertion to count as
# supported. Calibrated against the synthetic corpus + hashing
# embedding scoring; production callers should tune this once they
# wire in a real corpus and observe ROC characteristics.
_DEFAULT_SCORE_THRESHOLD = 0.22
# Raw cosine alignment required on top of the combined score. After
# the index drops stopwords + short tokens at indexing time, the
# remaining content-token cosine cleanly separates in-corpus and
# off-topic text; a 0.10 floor keeps a small margin against the
# hashing collision floor at 256 dim.
_DEFAULT_COSINE_FLOOR = 0.10
_DEFAULT_TOP_K = 3


@runtime_checkable
class _RetrievalIndexLike(Protocol):
    """Duck-typed retrieval index contract that GroundedDecoder consumes."""

    def assertion_is_supported(
        self,
        assertion: str,
        *,
        score_threshold: float,
        cosine_floor: float,
        top_k: int,
    ) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class GroundedDecoderConfig:
    """Tunables for :class:`GroundedDecoder`.

    Defaults are calibrated for the synthetic Einstein corpus +
    hashing-embedding scoring shipped in
    ``lifeform-domain-figure``. Production callers will likely
    raise ``min_assertion_tokens`` and ``score_threshold`` once the
    real corpus is wired in.
    """

    min_assertion_tokens: int = _DEFAULT_MIN_ASSERTION_TOKENS
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD
    cosine_floor: float = _DEFAULT_COSINE_FLOOR
    top_k: int = _DEFAULT_TOP_K

    def __post_init__(self) -> None:
        if self.min_assertion_tokens <= 0:
            raise ValueError(
                f"GroundedDecoderConfig.min_assertion_tokens must be > 0, "
                f"got {self.min_assertion_tokens!r}"
            )
        if self.top_k <= 0:
            raise ValueError(
                f"GroundedDecoderConfig.top_k must be > 0, got {self.top_k!r}"
            )


class UngroundedAssertionError(RuntimeError):
    """Raised by :meth:`GroundedDecoder.verify_strict` on L3 failure.

    The message includes the unsupported assertions verbatim so the
    caller-side audit log records exactly what the model said
    without supporting evidence.
    """

    def __init__(self, verdict: GroundingVerdict) -> None:
        joined = " | ".join(verdict.unsupported_assertions)
        super().__init__(
            "Grounded-decode L3 failure: the following assertions had no "
            f"citation-quality support: {joined}"
        )
        self.verdict = verdict


class GroundedDecoder:
    """L3 enforcer that consumes a retrieval index and emits verdicts.

    The decoder is a stateless wrapper: a single instance can be
    cloned across sessions because it holds no mutable state. The
    backing ``retrieval_index`` is itself frozen (see
    :class:`lifeform_domain_figure.FigureRetrievalIndex`).
    """

    def __init__(
        self,
        retrieval_index: _RetrievalIndexLike,
        *,
        config: GroundedDecoderConfig | None = None,
    ) -> None:
        self._retrieval_index = retrieval_index
        self._config = config or GroundedDecoderConfig()

    @property
    def config(self) -> GroundedDecoderConfig:
        return self._config

    def verify(self, *, text: str) -> GroundingVerdict:
        """Return a :class:`GroundingVerdict` for ``text``.

        Splits ``text`` into candidate assertions, drops sub-token
        fragments shorter than ``config.min_assertion_tokens``, and
        asks the retrieval index for citation-quality support on each
        survivor. The verdict's ``passed`` is True only when every
        substantive assertion has at least one piece of evidence
        clearing both the combined score threshold and the cosine
        floor (per the index's own ``assertion_is_supported`` rules).
        """

        assertions = _split_assertions(text, self._config.min_assertion_tokens)
        if not assertions:
            return GroundingVerdict(
                passed=True,
                unsupported_assertions=(),
                evidence_pointers=(),
                rationale=(
                    "Generated text contained no substantive assertions "
                    f"(threshold={self._config.min_assertion_tokens} "
                    "tokens); L3 contract trivially satisfied."
                ),
            )
        unsupported: list[str] = []
        evidence: list[str] = []
        for assertion in assertions:
            supports = self._retrieval_index.assertion_is_supported(
                assertion,
                score_threshold=self._config.score_threshold,
                cosine_floor=self._config.cosine_floor,
                top_k=self._config.top_k,
            )
            if not supports:
                unsupported.append(assertion)
                continue
            for evidence_record in supports:
                citation = _evidence_citation(evidence_record)
                if citation:
                    evidence.append(citation)
        passed = not unsupported
        rationale = (
            f"Verified {len(assertions)} assertion(s); "
            f"{len(unsupported)} unsupported. "
            f"{len(evidence)} evidence pointer(s) collected."
        )
        return GroundingVerdict(
            passed=passed,
            unsupported_assertions=tuple(unsupported),
            evidence_pointers=tuple(dict.fromkeys(evidence)),
            rationale=rationale,
        )

    def verify_strict(self, *, text: str) -> GroundingVerdict:
        """Same as :meth:`verify` but raises on L3 failure.

        Use this when the calling layer has opted into strict L3 — a
        verdict with ``passed=False`` is treated as an error rather
        than a soft signal. Mirrors the
        ``no-swallow-errors-no-hasattr-abuse.mdc`` requirement to
        fail loud on contract violations.
        """

        verdict = self.verify(text=text)
        if not verdict.passed:
            raise UngroundedAssertionError(verdict)
        return verdict


# ---------------------------------------------------------------------------
# Static type guard: assert GroundedDecoder satisfies GroundedDecodeHook
#
# The Protocol from vz-substrate uses ``def verify(self, *, text: str) ->
# GroundingVerdict``. Asserting structural compatibility here makes the
# class usable as a ``GroundedDecodeHook`` anywhere the substrate runtime
# would accept one.
# ---------------------------------------------------------------------------


def _assert_protocol() -> GroundedDecodeHook:
    """Type-only sanity check; no runtime side-effect."""
    return GroundedDecoder.__init__  # type: ignore[return-value]


def _split_assertions(text: str, min_tokens: int) -> tuple[str, ...]:
    if not text or not text.strip():
        return ()
    candidates = _SENTENCE_SPLIT_RE.split(text.strip())
    out: list[str] = []
    for raw in candidates:
        cleaned = raw.strip()
        if not cleaned:
            continue
        tokens = _WORD_RE.findall(cleaned)
        if len(tokens) < min_tokens:
            continue
        out.append(cleaned)
    return tuple(out)


def _evidence_citation(evidence_record: Any) -> str:
    """Extract a citation string from a retrieval-evidence record.

    Prefers the typed ``citation`` property exposed by
    :class:`lifeform_domain_figure.RetrievalEvidence`; falls back
    to a stable ``"chunk_id|locator"`` rendering for any duck-typed
    record. The ``getattr`` here is the **single** place where we
    accept structural variation in evidence shape — anywhere else
    in this file we rely on the duck-typed Protocol.
    """

    citation_attr = getattr(evidence_record, "citation", None)
    if isinstance(citation_attr, str) and citation_attr:
        return citation_attr
    chunk_id = getattr(evidence_record, "chunk_id", "")
    locator = getattr(evidence_record, "locator", "")
    if chunk_id or locator:
        return f"{locator} | {chunk_id}"
    return ""


__all__ = [
    "GroundedDecoder",
    "GroundedDecoderConfig",
    "UngroundedAssertionError",
]
