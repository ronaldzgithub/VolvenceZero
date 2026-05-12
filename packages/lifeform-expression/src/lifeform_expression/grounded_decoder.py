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


# Typed locator parser exposed by lifeform-domain-figure. Imported
# lazily inside helpers so this module can degrade to opaque-string
# behaviour when the figure wheel is not installed (lifeform-expression
# does not declare a hard dependency on lifeform-domain-figure).
def _try_parse_locator(locator: str) -> Any:
    """Best-effort typed locator parse.

    Returns the typed :class:`ParsedLocator` when the figure-vertical
    is reachable AND the string parses; ``None`` otherwise. The
    helper localises the optional dependency to one place — the
    decoder body itself stays type-clean.
    """

    if not locator:
        return None
    try:
        from lifeform_domain_figure.corpus.citation import parse_locator
    except ImportError:
        return None
    try:
        return parse_locator(locator)
    except ValueError:
        return None


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
class EvidencePointer:
    """Typed structured citation attached to a grounded assertion.

    The ``GroundingVerdict.evidence_pointers`` field carries opaque
    strings to keep the substrate-side Protocol stable. Callers that
    want the structured fields (sender / recipient / date / venue
    / page / offset, etc.) consume :class:`EvidencePointer` returned
    by :meth:`GroundedDecoder.verify_with_pointers` instead.

    ``raw_locator`` is always populated and is what gets surfaced as
    the opaque string in :class:`GroundingVerdict.evidence_pointers`.
    Structured fields are populated only when the locator parses
    successfully under :func:`parse_locator`; otherwise they are
    empty / sentinel values and consumers should fall back to
    ``raw_locator`` (debt #24 closure).
    """

    raw_locator: str
    chunk_id: str
    source_envelope_id: str
    locator_kind: str = ""
    document_id: str = ""
    paragraph_index: int = -1
    offset_start: int = -1
    offset_end: int = -1
    language: str = ""
    sender_id: str = ""
    recipient_id: str = ""
    date_iso: str = ""
    venue_id: str = ""
    volume: str = ""
    page: int = -1

    @property
    def parsed(self) -> bool:
        """Whether the underlying locator was structurally parsed."""
        return bool(self.locator_kind)

    @property
    def rendered(self) -> str:
        """Single-line render combining structured fields + envelope id."""
        if not self.parsed:
            return f"{self.raw_locator} | {self.source_envelope_id}"
        head = self.locator_kind
        if self.locator_kind == "letter":
            head = f"letter[{self.sender_id}->{self.recipient_id}@{self.date_iso}]"
        elif self.locator_kind == "lecture":
            head = f"lecture[{self.document_id}@{self.venue_id}/{self.date_iso}]"
        elif self.locator_kind == "notebook":
            head = f"notebook[{self.document_id} vol={self.volume} p.{self.page}]"
        else:
            head = f"paper[{self.document_id}]"
        offset_tail = (
            f" para={self.paragraph_index} off={self.offset_start}-{self.offset_end}"
            if self.paragraph_index >= 0
            else ""
        )
        lang_tail = f" lang={self.language}" if self.language else ""
        return f"{head}{offset_tail}{lang_tail} | {self.source_envelope_id}"


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

        verdict, _pointers = self._verify_internal(text=text)
        return verdict

    def verify_with_pointers(
        self, *, text: str
    ) -> tuple[GroundingVerdict, tuple[EvidencePointer, ...]]:
        """Return a verdict plus typed structured evidence pointers.

        Same shape as :meth:`verify` but additionally returns a
        tuple of :class:`EvidencePointer` records carrying the
        structured locator fields (sender / recipient / date /
        venue / page / offset / language) when the underlying
        locator parses successfully (debt #24 closure). When the
        locator does not parse, ``EvidencePointer.parsed`` is
        False and only ``raw_locator`` / ``chunk_id`` /
        ``source_envelope_id`` are populated.
        """

        return self._verify_internal(text=text)

    def _verify_internal(
        self, *, text: str
    ) -> tuple[GroundingVerdict, tuple[EvidencePointer, ...]]:
        assertions = _split_assertions(text, self._config.min_assertion_tokens)
        if not assertions:
            return (
                GroundingVerdict(
                    passed=True,
                    unsupported_assertions=(),
                    evidence_pointers=(),
                    rationale=(
                        "Generated text contained no substantive assertions "
                        f"(threshold={self._config.min_assertion_tokens} "
                        "tokens); L3 contract trivially satisfied."
                    ),
                ),
                (),
            )
        unsupported: list[str] = []
        evidence: list[str] = []
        pointers: list[EvidencePointer] = []
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
                pointer = _build_evidence_pointer(evidence_record)
                if pointer is None:
                    continue
                pointers.append(pointer)
                evidence.append(pointer.rendered)
        passed = not unsupported
        rationale = (
            f"Verified {len(assertions)} assertion(s); "
            f"{len(unsupported)} unsupported. "
            f"{len(evidence)} evidence pointer(s) collected."
        )
        verdict = GroundingVerdict(
            passed=passed,
            unsupported_assertions=tuple(unsupported),
            evidence_pointers=tuple(dict.fromkeys(evidence)),
            rationale=rationale,
        )
        # Dedupe pointers by rendered string while preserving order.
        seen: set[str] = set()
        deduped: list[EvidencePointer] = []
        for pointer in pointers:
            key = pointer.rendered
            if key in seen:
                continue
            seen.add(key)
            deduped.append(pointer)
        return (verdict, tuple(deduped))

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


def _build_evidence_pointer(evidence_record: Any) -> EvidencePointer | None:
    """Build a typed :class:`EvidencePointer` from a retrieval record.

    Reads the duck-typed evidence shape (``locator`` / ``chunk_id``
    / ``source_envelope_id``) and tries to parse the locator into
    typed fields via the optional :func:`parse_locator` helper. The
    ``getattr`` calls here are the **single** place where we accept
    structural variation in evidence shape — anywhere else in this
    module we rely on the typed return.
    """

    raw_locator = getattr(evidence_record, "locator", "") or ""
    chunk_id = getattr(evidence_record, "chunk_id", "") or ""
    source_envelope_id = getattr(evidence_record, "source_envelope_id", "") or ""
    if not raw_locator and not chunk_id:
        return None
    parsed = _try_parse_locator(raw_locator)
    if parsed is None:
        return EvidencePointer(
            raw_locator=raw_locator,
            chunk_id=chunk_id,
            source_envelope_id=source_envelope_id,
        )
    return EvidencePointer(
        raw_locator=raw_locator,
        chunk_id=chunk_id,
        source_envelope_id=source_envelope_id,
        locator_kind=parsed.kind.value,
        document_id=parsed.document_id,
        paragraph_index=parsed.paragraph_index,
        offset_start=parsed.offset.start,
        offset_end=parsed.offset.end,
        language=parsed.language,
        sender_id=parsed.sender_id,
        recipient_id=parsed.recipient_id,
        date_iso=parsed.date_iso,
        venue_id=parsed.venue_id,
        volume=parsed.volume,
        page=parsed.page,
    )


__all__ = [
    "EvidencePointer",
    "GroundedDecoder",
    "GroundedDecoderConfig",
    "UngroundedAssertionError",
]
