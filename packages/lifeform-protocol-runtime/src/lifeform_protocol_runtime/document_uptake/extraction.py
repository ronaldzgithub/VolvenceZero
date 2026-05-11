"""LLM-driven structured extraction (packet 2.3).

Pipeline:

    chunks: tuple[DocumentChunk, ...]
        ↓ (per chunk)
    LLM JSON-mode call (identity / boundary / strategy prompts)
        ↓
    typed parse → partial protocol fields
        ↓ (cross-chunk merge)
    BehaviorProtocol → BehaviorProtocolCandidate

The LLM client is duck-typed via :class:`LlmJsonClient` to keep
this module testable without a real network. The
``lifeform-openai-compat`` wheel will provide the production
implementation that calls OpenAI Chat Completions in JSON mode.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Protocol

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    KnowledgeSeed,
    ProtocolProvenance,
    ProtocolSourceKind,
    ReviewStatus,
    SignatureCase,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)

from lifeform_protocol_runtime.document_uptake.ingestion import DocumentChunk
from lifeform_protocol_runtime.document_uptake.prompts import (
    BOUNDARY_SYSTEM_PROMPT,
    BOUNDARY_USER_TEMPLATE,
    IDENTITY_SYSTEM_PROMPT,
    IDENTITY_USER_TEMPLATE,
    STRATEGY_SYSTEM_PROMPT,
    STRATEGY_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Duck-typed LLM client
# ---------------------------------------------------------------------------


class LlmJsonClient(Protocol):
    """Minimal interface for an LLM client returning JSON-mode responses.

    Production implementation: thin wrapper around
    ``lifeform-openai-compat`` that calls OpenAI Chat Completions
    with ``response_format={"type": "json_object"}``.

    Test implementation: ``MockLlmJsonClient`` returns canned
    responses keyed by (system_prompt_hash, user_prompt_hash).
    """

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Return the JSON object the LLM produced for this prompt pair.

        Implementations MUST validate that the LLM output is a
        well-formed JSON object and raise on failure (no silent
        partial parsing).
        """


# ---------------------------------------------------------------------------
# Signal source mapping (string → typed enum)
# ---------------------------------------------------------------------------


_SIGNAL_SOURCE_BY_NAME: dict[str, BehaviorProtocolSignalSource] = {
    member.value: member for member in BehaviorProtocolSignalSource
}


def _signal_source_or_default(
    raw: str,
) -> BehaviorProtocolSignalSource:
    """Map a string to a ``BehaviorProtocolSignalSource``.

    LLM output is expected to use the canonical lowercase enum
    value (e.g. "boundary_violation_fired"). Unknown strings fall
    back to ``BOUNDARY_VIOLATION_FIRED`` as the safest default
    (any unknown trigger is conservatively treated as a boundary
    event); this is logged in the candidate's ``review_evidence``.
    """

    if raw in _SIGNAL_SOURCE_BY_NAME:
        return _SIGNAL_SOURCE_BY_NAME[raw]
    return BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED


_SEVERITY_BY_NAME: dict[str, BoundarySeverity] = {
    "soft_remind": BoundarySeverity.SOFT_REMIND,
    "hard_block": BoundarySeverity.HARD_BLOCK,
    "escalate_human": BoundarySeverity.ESCALATE_HUMAN,
}


def _severity_or_default(raw: str) -> BoundarySeverity:
    return _SEVERITY_BY_NAME.get(raw, BoundarySeverity.HARD_BLOCK)


# ---------------------------------------------------------------------------
# Per-chunk merge accumulator
# ---------------------------------------------------------------------------


@dataclass
class _ExtractionAccumulator:
    advisor_name: str = ""
    description: str = ""
    identity_traits: list[str] = None
    regime_compatibility: list[str] = None
    boundaries: list[BoundaryContract] = None
    strategies: list[StrategyPrior] = None
    knowledge_seeds: list[KnowledgeSeed] = None
    cases: list[SignatureCase] = None
    chunks_seen: int = 0
    extractor_warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.identity_traits is None:
            self.identity_traits = []
        if self.regime_compatibility is None:
            self.regime_compatibility = []
        if self.boundaries is None:
            self.boundaries = []
        if self.strategies is None:
            self.strategies = []
        if self.knowledge_seeds is None:
            self.knowledge_seeds = []
        if self.cases is None:
            self.cases = []
        if self.extractor_warnings is None:
            self.extractor_warnings = []

    # ------------------------------------------------------------------
    # Merge logic — last-non-empty-wins for scalar fields, dedup-by-id
    # for collections (so duplicate chunks don't double-count).
    # ------------------------------------------------------------------

    def merge_identity(self, payload: dict) -> None:
        if isinstance(payload.get("advisor_name"), str):
            name = payload["advisor_name"].strip()
            if name and not self.advisor_name:
                self.advisor_name = name
        if isinstance(payload.get("description"), str):
            desc = payload["description"].strip()
            if desc and not self.description:
                self.description = desc
        traits = payload.get("identity_traits") or []
        for trait in traits:
            if isinstance(trait, str) and trait.strip():
                t = trait.strip()
                if t not in self.identity_traits:
                    self.identity_traits.append(t)
        regimes = payload.get("regime_compatibility") or []
        for regime in regimes:
            if isinstance(regime, str) and regime.strip():
                r = regime.strip()
                if r not in self.regime_compatibility:
                    self.regime_compatibility.append(r)

    def merge_boundaries(self, payload: dict) -> None:
        seen_ids = {b.boundary_id for b in self.boundaries}
        items = payload.get("boundaries") or []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            bid = (raw.get("boundary_id") or "").strip()
            if not bid or bid in seen_ids:
                continue
            description = (raw.get("description") or "").strip() or bid
            triggers = tuple(
                t for t in (raw.get("trigger_reasons") or [])
                if isinstance(t, str) and t.strip()
            )
            if not triggers:
                self.extractor_warnings.append(
                    f"boundary {bid!r} has no trigger_reasons; "
                    "review required."
                )
                continue
            blocked = tuple(
                t for t in (raw.get("blocked_topics") or [])
                if isinstance(t, str)
            )
            try:
                contract = BoundaryContract(
                    boundary_id=bid,
                    description=description,
                    trigger_reasons=triggers,
                    blocked_topics=blocked,
                    refer_out_required=bool(raw.get("refer_out_required", False)),
                    severity=_severity_or_default(raw.get("severity", "hard_block")),
                )
            except ValueError as exc:
                self.extractor_warnings.append(
                    f"boundary {bid!r} rejected by schema: {exc}"
                )
                continue
            self.boundaries.append(contract)
            seen_ids.add(bid)

    def merge_strategies(self, payload: dict) -> None:
        seen_strategy_ids = {s.rule_id for s in self.strategies}
        seen_seed_ids = {s.seed_id for s in self.knowledge_seeds}
        seen_case_ids = {c.case_id for c in self.cases}
        for raw in payload.get("strategies") or []:
            if not isinstance(raw, dict):
                continue
            rid = (raw.get("rule_id") or "").strip()
            if not rid or rid in seen_strategy_ids:
                continue
            ordering = tuple(
                s for s in (raw.get("recommended_ordering") or [])
                if isinstance(s, str) and s.strip()
            )
            if not ordering:
                self.extractor_warnings.append(
                    f"strategy {rid!r} has no recommended_ordering; "
                    "review required."
                )
                continue
            try:
                strategy = StrategyPrior(
                    rule_id=rid,
                    problem_pattern=(raw.get("problem_pattern") or "").strip()
                    or rid,
                    recommended_ordering=ordering,
                    recommended_pacing=(raw.get("recommended_pacing") or "").strip()
                    or "moderate",
                    avoid_patterns=tuple(
                        p for p in (raw.get("avoid_patterns") or [])
                        if isinstance(p, str)
                    ),
                    applicability_phase=tuple(
                        p for p in (raw.get("applicability_phase") or [])
                        if isinstance(p, str)
                    ),
                )
            except ValueError as exc:
                self.extractor_warnings.append(
                    f"strategy {rid!r} rejected by schema: {exc}"
                )
                continue
            self.strategies.append(strategy)
            seen_strategy_ids.add(rid)
        for raw in payload.get("knowledge_seeds") or []:
            if not isinstance(raw, dict):
                continue
            sid = (raw.get("seed_id") or "").strip()
            if not sid or sid in seen_seed_ids:
                continue
            topic = (raw.get("topic") or "").strip()
            domain = (raw.get("domain") or topic or "general").strip()
            summary = (raw.get("summary") or "").strip()
            try:
                seed = KnowledgeSeed(
                    seed_id=sid,
                    domain=domain,
                    title=(raw.get("title") or topic or sid).strip(),
                    summary=summary,
                    snippet=(raw.get("snippet") or summary or sid).strip(),
                    evidence_locator=(raw.get("evidence_locator") or "uptake").strip(),
                    confidence=_clamp_unit(raw.get("confidence", 0.7)),
                    topic_tags=tuple(
                        t for t in (raw.get("topic_tags") or [])
                        if isinstance(t, str)
                    ),
                    jurisdiction_tags=tuple(
                        t for t in (raw.get("jurisdiction_tags") or [])
                        if isinstance(t, str)
                    ),
                )
            except ValueError as exc:
                self.extractor_warnings.append(
                    f"knowledge_seed {sid!r} rejected: {exc}"
                )
                continue
            self.knowledge_seeds.append(seed)
            seen_seed_ids.add(sid)
        for raw in payload.get("cases") or []:
            if not isinstance(raw, dict):
                continue
            cid = (raw.get("case_id") or "").strip()
            if not cid or cid in seen_case_ids:
                continue
            domain = (raw.get("domain") or "general").strip()
            problem_pattern = (
                raw.get("problem_pattern")
                or raw.get("title")
                or cid
            ).strip()
            user_state_pattern = (raw.get("user_state_pattern") or "").strip()
            outcome_label = (
                raw.get("outcome_label")
                or raw.get("lesson")
                or "outcome_unspecified"
            ).strip()
            description = (
                raw.get("description")
                or raw.get("transcript_summary")
                or raw.get("lesson")
                or cid
            ).strip()
            ordering = tuple(
                step for step in (raw.get("intervention_ordering") or [])
                if isinstance(step, str) and step.strip()
            )
            if not ordering:
                # Synthesize a single-step ordering so the schema validator
                # accepts the case; reviewers can refine later.
                ordering = (
                    f"reflect:{cid}",
                )
            try:
                case = SignatureCase(
                    case_id=cid,
                    domain=domain,
                    problem_pattern=problem_pattern,
                    user_state_pattern=user_state_pattern,
                    risk_markers=tuple(
                        m for m in (raw.get("risk_markers") or [])
                        if isinstance(m, str)
                    ),
                    track_tags=tuple(
                        m for m in (raw.get("track_tags") or [])
                        if isinstance(m, str)
                    ),
                    regime_tags=tuple(
                        m for m in (raw.get("regime_tags") or [])
                        if isinstance(m, str)
                    ),
                    intervention_ordering=ordering,
                    outcome_label=outcome_label,
                    confidence=_clamp_unit(raw.get("confidence", 0.7)),
                    description=description,
                )
            except ValueError as exc:
                self.extractor_warnings.append(
                    f"case {cid!r} rejected: {exc}"
                )
                continue
            self.cases.append(case)
            seen_case_ids.add(cid)


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------


def extract_protocol_candidate(
    chunks: tuple[DocumentChunk, ...],
    *,
    llm_client: LlmJsonClient,
    source_locator: str,
    source_kind: ProtocolSourceKind = ProtocolSourceKind.PDF_UPTAKE,
    extractor_id: str = "lifeform-protocol-runtime/document-uptake",
    protocol_id_seed: str | None = None,
) -> BehaviorProtocolCandidate:
    """Extract a ``BehaviorProtocolCandidate`` from document chunks.

    Walks every chunk through three LLM passes (identity /
    boundary / strategy), merges via :class:`_ExtractionAccumulator`,
    and assembles a ``BehaviorProtocol`` + ``BehaviorProtocolCandidate``.

    Args:
        chunks: chunked document text (from
            :func:`.ingestion.chunk_document`).
        llm_client: duck-typed JSON-mode LLM client.
        source_locator: human / machine-readable source path
            (PDF path / URL). Stored verbatim in both the
            inner protocol's ``source_locator`` and the
            candidate's ``provenance.source_locator`` (the
            schema enforces consistency).
        source_kind: ``PDF_UPTAKE`` (default) or
            ``MARKDOWN_UPTAKE`` etc.
        extractor_id: identifier of the extraction component for
            audit. Defaults to this module's canonical id.
        protocol_id_seed: optional explicit ``protocol_id`` for
            the resulting ``BehaviorProtocol``. Defaults to
            ``"uptake:{slug(source_locator)}"``.

    Returns:
        ``BehaviorProtocolCandidate`` with ``requires_review=True``
        and review_evidence populated with extractor warnings +
        chunk count + per-family counts. Inner protocol's
        ``review_status`` is ``DRAFT``.

    Raises:
        ValueError if extraction yielded zero strategies AND
            zero boundaries (no actionable content extracted).
    """

    if not chunks:
        raise ValueError("extract_protocol_candidate requires non-empty chunks")

    acc = _ExtractionAccumulator()

    for chunk in chunks:
        acc.chunks_seen += 1
        identity_payload = llm_client.complete_json(
            system_prompt=IDENTITY_SYSTEM_PROMPT,
            user_prompt=IDENTITY_USER_TEMPLATE.format(chunk_text=chunk.text),
        )
        if not isinstance(identity_payload, dict):
            acc.extractor_warnings.append(
                f"chunk {chunk.chunk_index}: identity LLM returned non-dict"
            )
        else:
            acc.merge_identity(identity_payload)

        boundary_payload = llm_client.complete_json(
            system_prompt=BOUNDARY_SYSTEM_PROMPT,
            user_prompt=BOUNDARY_USER_TEMPLATE.format(chunk_text=chunk.text),
        )
        if not isinstance(boundary_payload, dict):
            acc.extractor_warnings.append(
                f"chunk {chunk.chunk_index}: boundary LLM returned non-dict"
            )
        else:
            acc.merge_boundaries(boundary_payload)

        strategy_payload = llm_client.complete_json(
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            user_prompt=STRATEGY_USER_TEMPLATE.format(chunk_text=chunk.text),
        )
        if not isinstance(strategy_payload, dict):
            acc.extractor_warnings.append(
                f"chunk {chunk.chunk_index}: strategy LLM returned non-dict"
            )
        else:
            acc.merge_strategies(strategy_payload)

    if (
        not acc.boundaries
        and not acc.strategies
        and not acc.knowledge_seeds
        and not acc.cases
    ):
        raise ValueError(
            "extract_protocol_candidate produced zero boundaries, "
            "zero strategies, zero knowledge seeds, and zero cases; "
            "document yielded no actionable content"
        )

    advisor_name = acc.advisor_name or "uptake-extracted"
    description = acc.description or "Extracted via DocumentUptake"
    protocol_id = protocol_id_seed or _slug_protocol_id(source_locator)

    success_signals, failure_signals = _synthesize_pe_signals(acc)

    inner = BehaviorProtocol(
        protocol_id=protocol_id,
        version="0.1.0",
        advisor_name=advisor_name,
        description=description,
        source_kind=source_kind,
        source_locator=source_locator,
        identity_assertion=IdentityAssertion(
            requires_self_traits=tuple(acc.identity_traits),
            forbidden_self_traits=(),
            required_regime_compatibility=tuple(acc.regime_compatibility),
        ),
        boundary_contracts=tuple(acc.boundaries),
        activation_conditions=ActivationConditions(),
        strategy_priors=tuple(acc.strategies),
        knowledge_seeds=tuple(acc.knowledge_seeds),
        signature_cases=tuple(acc.cases),
        temporal_arc=TemporalArc(),
        success_signals=success_signals,
        failure_signals=failure_signals,
        review_status=ReviewStatus.DRAFT,
    )

    confidence = _estimate_confidence(acc)

    provenance = ProtocolProvenance(
        source_kind=source_kind,
        source_locator=source_locator,
        extracted_at_iso=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        extractor_id=extractor_id,
        confidence=confidence,
    )

    review_evidence = (
        f"chunks_seen={acc.chunks_seen}",
        f"boundaries={len(acc.boundaries)}",
        f"strategies={len(acc.strategies)}",
        f"knowledge_seeds={len(acc.knowledge_seeds)}",
        f"cases={len(acc.cases)}",
        f"identity_traits={len(acc.identity_traits)}",
        *(f"warn:{w}" for w in acc.extractor_warnings),
    )

    return BehaviorProtocolCandidate(
        protocol=inner,
        provenance=provenance,
        requires_review=True,
        review_evidence=review_evidence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_unit(value) -> float:
    """Clamp anything plausible to ``[0, 1]``; default to 0.7 on bad input."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.7
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _slug_protocol_id(source_locator: str) -> str:
    """Derive a stable protocol_id from the source locator path."""
    base = source_locator.replace("\\", "/").rsplit("/", 1)[-1]
    base = base.rsplit(".", 1)[0]
    slug = "".join(
        c if c.isalnum() else "-" for c in base.lower()
    ).strip("-")
    if not slug:
        slug = "uptake-extracted"
    return f"uptake:{slug}"


def _synthesize_pe_signals(
    acc: _ExtractionAccumulator,
) -> tuple[tuple[SuccessSignal, ...], tuple[FailureSignal, ...]]:
    """Derive minimal success/failure signals from extraction shape.

    Every BehaviorProtocol must declare at least one success and
    one failure signal (schema invariant) unless ``legacy_fixture``.
    For a fresh DocumentUptake we synthesize generic signals tied
    to rupture state — the canonical "things went wrong" PE
    channel — so the protocol passes schema validation. Reviewers
    can refine these in a later revision.
    """

    success = (
        SuccessSignal(
            signal_id="uptake-success-no-rupture",
            description=(
                "No rupture observed during interactions guided by "
                "this protocol (synthesized by DocumentUptake; "
                "review and refine before promoting beyond SHADOW)."
            ),
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
        ),
    )
    failure = (
        FailureSignal(
            signal_id="uptake-failure-rupture-fired",
            description=(
                "Rupture fired during interactions guided by this "
                "protocol. Synthesized by DocumentUptake; review "
                "and refine before promoting beyond SHADOW."
            ),
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
        ),
    )
    return success, failure


def _estimate_confidence(acc: _ExtractionAccumulator) -> float:
    """Heuristic confidence in the extracted candidate.

    Combines (a) presence of an identity, (b) at least 1 boundary,
    (c) at least 1 strategy, (d) absence of extractor warnings.
    """

    score = 0.0
    if acc.advisor_name:
        score += 0.25
    if acc.boundaries:
        score += 0.25
    if acc.strategies:
        score += 0.25
    if not acc.extractor_warnings:
        score += 0.25
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Test helper: deterministic mock client
# ---------------------------------------------------------------------------


class MockLlmJsonClient:
    """Deterministic mock client driven by a per-prompt-family canned response.

    Construct with a dict mapping family name (``"identity"`` /
    ``"boundary"`` / ``"strategy"``) to either:

    * a single response dict (returned for every chunk), OR
    * a callable ``f(chunk_text: str) -> dict`` for chunk-aware
      responses.

    Unknown system prompts fall back to ``{}``. This client makes
    no network calls — ideal for unit tests.
    """

    def __init__(
        self,
        *,
        identity: dict | None = None,
        boundary: dict | None = None,
        strategy: dict | None = None,
        identity_fn=None,
        boundary_fn=None,
        strategy_fn=None,
    ) -> None:
        self._identity = identity
        self._boundary = boundary
        self._strategy = strategy
        self._identity_fn = identity_fn
        self._boundary_fn = boundary_fn
        self._strategy_fn = strategy_fn

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        if system_prompt == IDENTITY_SYSTEM_PROMPT:
            if self._identity_fn is not None:
                return self._identity_fn(user_prompt)
            return self._identity or {}
        if system_prompt == BOUNDARY_SYSTEM_PROMPT:
            if self._boundary_fn is not None:
                return self._boundary_fn(user_prompt)
            return self._boundary or {}
        if system_prompt == STRATEGY_SYSTEM_PROMPT:
            if self._strategy_fn is not None:
                return self._strategy_fn(user_prompt)
            return self._strategy or {}
        return {}


__all__ = [
    "LlmJsonClient",
    "MockLlmJsonClient",
    "extract_protocol_candidate",
]
