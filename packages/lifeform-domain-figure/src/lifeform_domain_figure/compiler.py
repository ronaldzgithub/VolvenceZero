"""Compile reviewed figure profiles into canonical lifeform inputs + bundle.

Two responsibilities:

1. Compile a :class:`HistoricalFigureProfile` into the existing
   application-tier inputs (``DomainExperiencePackage`` + ``VitalsBootstrap``)
   so the figure vertical does NOT introduce a new kernel owner — knowledge
   / cases / playbook / boundaries flow into the same domain_knowledge /
   case_memory / strategy_playbook / boundary_policy owners that the
   character vertical uses.

2. Assemble the F2.x retrieval / coverage / style artifacts plus the
   compiled application package into one frozen
   :class:`FigureArtifactBundle` that runtime + DLaaS layers consume.

Steering (F5) and persona LoRA (F6) artifacts attach to the bundle in
later packets via :func:`replace_bundle_with_steering` /
:func:`replace_bundle_with_lora` (kept here so all bundle-shape
mutations live in one module).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace as _replace

from lifeform_core import DriveSpec, VitalsBootstrap
from lifeform_ingestion.envelope import IngestionEnvelope
from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_figure.corpus.dedupe import compute_dedup_report
from lifeform_domain_figure.corpus.provenance import (
    SourceProvenance,
    fingerprint_provenance,
)
from lifeform_domain_figure.coverage_map import (
    FigureCoverageMap,
    build_figure_coverage_map,
)
from lifeform_domain_figure.metadata.records import MetadataDigest
from lifeform_domain_figure.figure_artifact import (
    FigureArtifactBundle,
    SCHEMA_VERSION,
    bundle_id_from_hash,
    compute_bundle_integrity_hash,
)
from lifeform_domain_figure.presence_artifact import (
    FigurePresenceArtifact,
    presence_fingerprint,
)
from lifeform_domain_figure.profile import (
    FigureBoundaryPrior,
    FigureKnowledgeSeed,
    FigureSignatureCase,
    FigureStrategyPrior,
    HistoricalFigureProfile,
)
from lifeform_domain_figure.retrieval_index import (
    FigureRetrievalIndex,
    build_figure_retrieval_index,
)
from lifeform_domain_figure.style_prior import (
    FigureStylePrior,
    build_figure_style_prior,
)
from lifeform_domain_figure.verification import (
    VerificationGateError,
    VerificationLedger,
    assert_all_provenances_pass,
)


_PACKAGE_PREFIX = "lifeform-figure"


def build_figure_package(profile: HistoricalFigureProfile) -> DomainExperiencePackage:
    """Compile a reviewed figure profile into a DomainExperiencePackage.

    The package is data only. It enters the kernel through the
    existing ``domain_knowledge``, ``case_memory``,
    ``strategy_playbook``, and ``boundary_policy`` owners; no figure-
    specific runtime owner is introduced (R8).
    """

    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=f"{_PACKAGE_PREFIX}:{profile.profile_id}",
            version=profile.version,
            display_name=f"{profile.figure_name} figure bootstrap",
            domain_ids=_domain_ids(profile),
            target_contexts=profile.target_contexts,
            evidence_level="reviewed-primary-source",
            owner="lifeform-domain-figure",
            description=(
                f"Reviewed real-person figure bootstrap for "
                f"{profile.figure_name} (lifespan "
                f"{profile.figure_lifespan[0]}-{profile.figure_lifespan[1]}). "
                f"{profile.description}"
            ),
        ),
        knowledge_records=tuple(
            _knowledge_record(profile, seed) for seed in profile.knowledge_seeds
        ),
        case_records=tuple(_case_record(case) for case in profile.signature_cases),
        playbook_rules=tuple(
            _playbook_rule(rule) for rule in profile.strategy_priors
        ),
        boundary_hints=tuple(
            _boundary_hint(boundary) for boundary in profile.boundary_priors
        ),
    )


def build_figure_vitals_bootstrap(
    profile: HistoricalFigureProfile,
    *,
    proactive_pe_threshold: float = 1.0,
    proactive_followup_priority: float = 0.55,
    proactive_cooldown_ticks: int = 60,
) -> VitalsBootstrap:
    """Compile figure drive priors into a VitalsBootstrap."""

    return VitalsBootstrap(
        schema_version=1,
        drives=tuple(
            DriveSpec(
                name=drive.name,
                target=drive.target,
                homeostatic_band=drive.homeostatic_band,
                decay_per_tick=drive.decay_per_tick,
                pe_weight=drive.pe_weight,
                initial_level=drive.initial_level,
                recharge_per_turn=drive.recharge_per_turn,
                recharge_per_regime=dict(drive.recharge_per_regime),
            )
            for drive in profile.drive_priors
        ),
        proactive_pe_threshold=proactive_pe_threshold,
        proactive_followup_priority=proactive_followup_priority,
        proactive_cooldown_ticks=proactive_cooldown_ticks,
    )


@dataclass(frozen=True)
class FigureBundleInputs:
    """Typed inputs for :func:`build_figure_artifact_bundle`.

    Keeps the function signature stable as P5 / P6 add steering /
    LoRA artifacts: callers populate the optional fields when those
    artifacts have been baked, and the builder re-uses the same hash
    discipline.

    L2 verification gate (debt #28 L2 first batch):

    * ``provenance_records`` — tuple of :class:`SourceProvenance`
      records the curator declares cover every shipped source. The
      gate iterates these (not the envelopes) because provenance is
      the audit-bearing record.
    * ``verification_ledger`` — content-addressable
      :class:`VerificationLedger` of past verifier verdicts.
    * ``require_verification_pass`` — when ``True``, the builder
      calls :func:`assert_all_provenances_pass` before assembling
      anything; ``False`` (default) leaves all existing callers
      unaffected.
    """

    profile: HistoricalFigureProfile
    envelopes: tuple[IngestionEnvelope, ...]
    extra_style_terms: tuple[str, ...] = ()
    time_window_id: str | None = None
    steering: object | None = None
    lora: object | None = None
    provenance_records: tuple[SourceProvenance, ...] = ()
    verification_ledger: VerificationLedger | None = None
    require_verification_pass: bool = False
    metadata_digest: MetadataDigest | None = None


def build_figure_artifact_bundle(
    inputs: FigureBundleInputs,
) -> FigureArtifactBundle:
    """Assemble a complete :class:`FigureArtifactBundle` from inputs.

    Steps:

    0. (L2 gate, debt #28) When
       ``inputs.require_verification_pass`` is ``True``, refuse to
       proceed unless every supplied :class:`SourceProvenance` has
       all-PASS verdicts in :class:`VerificationLedger` for every
       :data:`IMPLEMENTED_CHECK_KINDS`. Raises
       :class:`VerificationGateError` with the full failure manifest.
    1. Apply the optional time window selection on the profile.
    2. Build the :class:`DomainExperiencePackage` + ``VitalsBootstrap``.
    3. Build the F2 artifacts (retrieval / coverage / style) from the
       supplied envelopes.
    4. Compute the deterministic integrity hash and bundle id.
    """

    if inputs.require_verification_pass:
        if inputs.verification_ledger is None:
            raise VerificationGateError(
                "build_figure_artifact_bundle: require_verification_pass=True "
                "but verification_ledger is None; cannot consult ledger."
            )
        if not inputs.provenance_records:
            raise VerificationGateError(
                "build_figure_artifact_bundle: require_verification_pass=True "
                "but provenance_records is empty; nothing to gate."
            )
        assert_all_provenances_pass(
            inputs.provenance_records, inputs.verification_ledger
        )
    selected = inputs.profile.select_window(inputs.time_window_id)
    domain_package = build_figure_package(selected)
    vitals_bootstrap = build_figure_vitals_bootstrap(selected)
    # debt #24: dedupe canonical chunks before retrieval index so
    # cross-envelope duplicate boilerplate (e.g., the curator header
    # note that lands on every synthetic source) does not over-weight
    # BM25 corpus statistics. Only the canonical (highest-trust
    # source kind) instance per byte-identical text survives.
    dedup_report = compute_dedup_report(inputs.envelopes)
    retrieval_index = build_figure_retrieval_index(
        figure_id=selected.profile_id,
        envelopes=inputs.envelopes,
        chunk_id_allowlist=dedup_report.canonical_chunk_ids,
    )
    coverage_map = build_figure_coverage_map(
        figure_id=selected.profile_id,
        profile=selected,
        retrieval_index=retrieval_index,
    )
    style_prior = build_figure_style_prior(
        figure_id=selected.profile_id,
        envelopes=inputs.envelopes,
        extra_terms=inputs.extra_style_terms,
    )
    version_window = _resolve_version_window(selected, inputs.time_window_id)
    steering_integrity = _artifact_integrity(inputs.steering)
    lora_integrity = _artifact_integrity(inputs.lora)
    metadata_digest_fingerprint = (
        inputs.metadata_digest.fingerprint if inputs.metadata_digest is not None else ""
    )
    provenance_fingerprint = (
        fingerprint_provenance(inputs.provenance_records)
        if inputs.provenance_records
        else ""
    )
    integrity_hash = compute_bundle_integrity_hash(
        figure_id=selected.profile_id,
        profile_version=selected.version,
        version_window=version_window,
        retrieval_integrity=retrieval_index.integrity_hash,
        coverage_integrity=coverage_map.integrity_hash,
        style_integrity=style_prior.integrity_hash,
        steering_integrity=steering_integrity,
        lora_integrity=lora_integrity,
        metadata_digest_fingerprint=metadata_digest_fingerprint,
        provenance_fingerprint=provenance_fingerprint,
    )
    return FigureArtifactBundle(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id_from_hash(selected.profile_id, integrity_hash),
        figure_id=selected.profile_id,
        profile_version=selected.version,
        version_window=version_window,
        profile=selected,
        domain_package=domain_package,
        vitals_bootstrap=vitals_bootstrap,
        retrieval_index=retrieval_index,
        coverage_map=coverage_map,
        style_prior=style_prior,
        steering=inputs.steering,
        lora=inputs.lora,
        integrity_hash=integrity_hash,
        metadata_digest_fingerprint=metadata_digest_fingerprint,
        provenance_fingerprint=provenance_fingerprint,
    )


def attach_steering_to_bundle(
    bundle: FigureArtifactBundle,
    *,
    steering: object,
    steering_integrity: str,
) -> FigureArtifactBundle:
    """Return a new bundle with the steering artifact attached.

    Mutating helper for the F5 path: the gated steering artifact is
    produced offline and re-attached here so the integrity hash
    reflects the bundle's full identity (R15: any attribute change
    yields a fresh bundle id).
    """

    new_integrity = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity=steering_integrity,
        lora_integrity=_artifact_integrity(bundle.lora),
        metadata_digest_fingerprint=bundle.metadata_digest_fingerprint,
        provenance_fingerprint=bundle.provenance_fingerprint,
        presence_integrity=presence_fingerprint(bundle.presence),
    )
    return _replace(
        bundle,
        steering=steering,
        bundle_id=bundle_id_from_hash(bundle.figure_id, new_integrity),
        integrity_hash=new_integrity,
    )


def attach_lora_to_bundle(
    bundle: FigureArtifactBundle,
    *,
    lora: object,
    lora_integrity: str,
) -> FigureArtifactBundle:
    """Return a new bundle with the LoRA artifact attached (F6 path)."""

    new_integrity = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity=_artifact_integrity(bundle.steering),
        lora_integrity=lora_integrity,
        metadata_digest_fingerprint=bundle.metadata_digest_fingerprint,
        provenance_fingerprint=bundle.provenance_fingerprint,
        presence_integrity=presence_fingerprint(bundle.presence),
    )
    return _replace(
        bundle,
        lora=lora,
        bundle_id=bundle_id_from_hash(bundle.figure_id, new_integrity),
        integrity_hash=new_integrity,
    )


def attach_presence_to_bundle(
    bundle: FigureArtifactBundle,
    *,
    presence: FigurePresenceArtifact,
) -> FigureArtifactBundle:
    """Return a new bundle with the L0 presence artifact attached.

    This is the rare-heavy path equivalent for L0: the presence
    artifact is produced offline by the rendering plane (after a real
    operator has signed the consent receipt and a reference image /
    voice clone / motion model has been registered with
    ``apps/presence-service``). Re-attaching here folds the presence
    fingerprint into the bundle's ``integrity_hash`` so a likeness
    re-bake or consent revoke yields a fresh ``bundle_id``
    (R15 byte-level rollback contract).
    """

    if presence.figure_id != bundle.figure_id:
        raise ValueError(
            "attach_presence_to_bundle: presence.figure_id "
            f"{presence.figure_id!r} does not match bundle.figure_id "
            f"{bundle.figure_id!r}"
        )
    new_integrity = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity=_artifact_integrity(bundle.steering),
        lora_integrity=_artifact_integrity(bundle.lora),
        metadata_digest_fingerprint=bundle.metadata_digest_fingerprint,
        provenance_fingerprint=bundle.provenance_fingerprint,
        presence_integrity=presence.integrity_hash,
    )
    return _replace(
        bundle,
        presence=presence,
        bundle_id=bundle_id_from_hash(bundle.figure_id, new_integrity),
        integrity_hash=new_integrity,
    )


def _resolve_version_window(
    profile: HistoricalFigureProfile,
    time_window_id: str | None,
) -> tuple[int, int]:
    if not time_window_id:
        return (0, 0)
    for view in profile.time_windows:
        if view.window_id == time_window_id:
            return (view.year_start, view.year_end)
    return (0, 0)


def _artifact_integrity(artifact: object | None) -> str:
    """Return a stable integrity hash for an optional artifact slot.

    Steering / LoRA artifacts (F5 / F6) are typed but not yet
    importable here; we read ``integrity_hash`` if present and fall
    back to a sentinel otherwise. This is the **single** ``getattr``
    in this module — it is justified because the artifact types
    arrive in later packets and must remain forward-compatible.
    """

    if artifact is None:
        return "absent"
    integrity = getattr(artifact, "integrity_hash", None)
    if not isinstance(integrity, str) or not integrity:
        return hashlib.sha256(repr(artifact).encode("utf-8")).hexdigest()
    return integrity


def _domain_ids(profile: HistoricalFigureProfile) -> tuple[str, ...]:
    ordered: list[str] = list(profile.domain_coverage_seed)
    for domain in tuple(seed.domain for seed in profile.knowledge_seeds) + tuple(
        case.domain for case in profile.signature_cases
    ):
        if domain not in ordered:
            ordered.append(domain)
    if not ordered:
        ordered.append(f"figure:{profile.profile_id}")
    return tuple(ordered)


def _knowledge_record(
    profile: HistoricalFigureProfile,
    seed: FigureKnowledgeSeed,
) -> DomainKnowledgeRecord:
    return DomainKnowledgeRecord(
        record_id=f"rid-figure:{profile.profile_id}:knowledge:{seed.seed_id}",
        domain=seed.domain,
        topic_tags=seed.topic_tags,
        jurisdiction_tags=("real-person-figure",),
        # Map to the closest existing KnowledgeSourceType: reviewed
        # paraphrases of primary-source material are reviewer-curated
        # secondary records, which is exactly what REVIEWED_ARTICLE
        # encodes in the application-tier enum.
        source_type="reviewed-article",
        title=seed.title,
        locator=seed.evidence_locator,
        summary=seed.summary,
        snippet=seed.snippet,
        freshness_label="reviewed",
        confidence=seed.confidence,
        evidence_strength=seed.evidence_strength,
        url=profile.source_uri,
    )


def _case_record(case: FigureSignatureCase) -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id=f"rid-figure:case:{case.case_id}",
        domain=case.domain,
        problem_pattern=case.problem_pattern,
        user_state_pattern=case.user_state_pattern,
        risk_markers=case.risk_markers,
        track_tags=case.track_tags,
        regime_tags=case.regime_tags,
        intervention_ordering=case.intervention_ordering,
        outcome_label=case.outcome_label,
        delayed_signal_count=1,
        escalation_observed=case.escalation_observed,
        repair_observed=case.repair_observed,
        confidence=case.confidence,
        relevance_score=case.relevance_score,
        description=case.description,
        reconstruction_source="reviewed-figure-profile",
    )


def _playbook_rule(rule: FigureStrategyPrior) -> PlaybookRule:
    return PlaybookRule(
        rule_id=f"rid-figure:playbook:{rule.rule_id}",
        problem_pattern=rule.problem_pattern,
        recommended_regime=rule.recommended_regime,
        recommended_ordering=rule.recommended_ordering,
        recommended_pacing=rule.recommended_pacing,
        avoid_patterns=rule.avoid_patterns,
        knowledge_weight_hint=rule.knowledge_weight_hint,
        experience_weight_hint=rule.experience_weight_hint,
        applicability_scope=rule.applicability_scope,
        confidence=rule.confidence,
        description=rule.description,
    )


def _boundary_hint(boundary: FigureBoundaryPrior) -> BoundaryPriorHint:
    return BoundaryPriorHint(
        hint_id=f"rid-figure:boundary:{boundary.boundary_id}",
        regime_id=boundary.regime_id,
        trigger_reasons=boundary.trigger_reasons,
        answer_depth_limit_hint=boundary.answer_depth_limit_hint,
        clarification_required=boundary.clarification_required,
        refer_out_required=boundary.refer_out_required,
        blocked_topics=boundary.blocked_topics,
        required_disclaimers=boundary.required_disclaimers,
        confidence=boundary.confidence,
        description=boundary.description,
    )


__all__ = [
    "FigureBundleInputs",
    "attach_lora_to_bundle",
    "attach_presence_to_bundle",
    "attach_steering_to_bundle",
    "build_figure_artifact_bundle",
    "build_figure_package",
    "build_figure_vitals_bootstrap",
]
