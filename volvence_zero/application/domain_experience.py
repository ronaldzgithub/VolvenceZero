from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    ApplicationRareHeavyCheckpoint,
    ApplicationRareHeavyState,
    BoundaryPolicyPriorUpdate,
    BoundaryPriorHint,
    CaseMemoryPriorUpdate,
    DomainKnowledgePriorUpdate,
    EvidenceStrength,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    KnowledgeSourceType,
    PlaybookRule,
    ReviewedKnowledgeCandidate,
    StrategyPlaybookPriorUpdate,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)


ALLOWED_RISK_MARKERS = frozenset({"risk-low", "risk-medium", "risk-high", "risk-critical"})
ALLOWED_TRACK_TAGS = frozenset({"world", "self", "shared"})


@dataclass(frozen=True)
class DomainExperienceManifest:
    package_id: str
    version: str
    display_name: str
    domain_ids: tuple[str, ...]
    target_contexts: tuple[str, ...]
    evidence_level: str
    owner: str
    description: str


@dataclass(frozen=True)
class DomainScenario:
    scenario_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DomainExperienceEvaluationScenario:
    scenario_id: str
    domain: str
    prompt: str
    expected_capabilities: tuple[str, ...]
    risk_markers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DomainExperiencePackage:
    manifest: DomainExperienceManifest
    scenarios: tuple[DomainScenario, ...] = ()
    knowledge_records: tuple[DomainKnowledgeRecord, ...] = ()
    case_records: tuple[CaseMemoryRecord, ...] = ()
    playbook_rules: tuple[PlaybookRule, ...] = ()
    boundary_hints: tuple[BoundaryPriorHint, ...] = ()
    reviewed_knowledge_candidates: tuple[ReviewedKnowledgeCandidate, ...] = ()
    domain_template_biases: tuple[tuple[str, float], ...] = ()
    evaluation_scenarios: tuple[DomainExperienceEvaluationScenario, ...] = ()


@dataclass(frozen=True)
class DomainExperienceValidationIssue:
    issue_id: str
    severity: str
    location: str
    description: str


@dataclass(frozen=True)
class DomainExperienceValidationReport:
    package_id: str
    valid: bool
    issues: tuple[DomainExperienceValidationIssue, ...]
    description: str


@dataclass(frozen=True)
class CompiledDomainExperiencePackage:
    package: DomainExperiencePackage
    validation_report: DomainExperienceValidationReport
    knowledge_records: tuple[DomainKnowledgeRecord, ...]
    case_records: tuple[CaseMemoryRecord, ...]
    application_prior_update: ApplicationPriorUpdate
    rare_heavy_checkpoint: ApplicationRareHeavyCheckpoint
    evaluation_scenarios: tuple[DomainExperienceEvaluationScenario, ...]
    description: str


@dataclass(frozen=True)
class DomainExperienceApplicationReport:
    package_ids: tuple[str, ...]
    applied_knowledge_count: int
    applied_case_count: int
    imported_playbook_count: int
    imported_boundary_hint_count: int
    imported_reviewed_knowledge_count: int
    operations: tuple[str, ...]
    description: str


def validate_domain_experience_package(
    package: DomainExperiencePackage,
) -> DomainExperienceValidationReport:
    issues: list[DomainExperienceValidationIssue] = []
    package_id = package.manifest.package_id
    _require_non_empty(issues, package_id, "manifest.package_id", package.manifest.package_id)
    _require_non_empty(issues, package_id, "manifest.version", package.manifest.version)
    _require_non_empty(issues, package_id, "manifest.owner", package.manifest.owner)
    if not package.manifest.domain_ids:
        issues.append(
            _issue(
                package_id,
                "error",
                "manifest.domain_ids",
                "Domain experience package must declare at least one domain.",
            )
        )
    if not package.boundary_hints:
        issues.append(
            _issue(
                package_id,
                "error",
                "boundary_hints",
                "Domain experience package must include at least one boundary hint.",
            )
        )
    _check_duplicate_values(
        issues,
        package_id,
        "knowledge_records.record_id",
        tuple(record.record_id for record in package.knowledge_records),
    )
    _check_duplicate_values(
        issues,
        package_id,
        "case_records.case_id",
        tuple(record.case_id for record in package.case_records),
    )
    _check_duplicate_values(
        issues,
        package_id,
        "playbook_rules.rule_id",
        tuple(rule.rule_id for rule in package.playbook_rules),
    )
    _check_duplicate_values(
        issues,
        package_id,
        "boundary_hints.hint_id",
        tuple(hint.hint_id for hint in package.boundary_hints),
    )
    _check_duplicate_values(
        issues,
        package_id,
        "scenarios.scenario_id",
        tuple(scenario.scenario_id for scenario in package.scenarios),
    )
    for record in package.knowledge_records:
        _validate_knowledge_record(issues, package_id, record)
    for record in package.case_records:
        _validate_case_record(issues, package_id, record)
    for rule in package.playbook_rules:
        _validate_playbook_rule(issues, package_id, rule)
    for hint in package.boundary_hints:
        _validate_boundary_hint(issues, package_id, hint)
    for candidate in package.reviewed_knowledge_candidates:
        _validate_reviewed_candidate(issues, package_id, candidate)
    for scenario in package.scenarios:
        _validate_scenario(issues, package_id, scenario)
    for scenario in package.evaluation_scenarios:
        _validate_evaluation_scenario(issues, package_id, scenario)
    for domain_id, bias in package.domain_template_biases:
        _require_non_empty(issues, package_id, "domain_template_biases.domain_id", domain_id)
        if bias < 0.0 or bias > 1.0:
            issues.append(
                _issue(
                    package_id,
                    "error",
                    f"domain_template_biases.{domain_id}",
                    "Domain template bias must be within [0.0, 1.0].",
                )
            )
    valid = not any(issue.severity == "error" for issue in issues)
    return DomainExperienceValidationReport(
        package_id=package_id,
        valid=valid,
        issues=tuple(issues),
        description=(
            f"Domain experience package {package_id} validation "
            f"{'passed' if valid else 'failed'} with {len(issues)} issue(s)."
        ),
    )


def compile_domain_experience_package(
    package: DomainExperiencePackage,
) -> CompiledDomainExperiencePackage:
    report = validate_domain_experience_package(package)
    if not report.valid:
        issue_summary = "; ".join(issue.description for issue in report.issues if issue.severity == "error")
        raise ValueError(f"Invalid domain experience package {package.manifest.package_id}: {issue_summary}")
    package_id = package.manifest.package_id
    case_updates = tuple(
        CaseMemoryPriorUpdate(
            update_id=f"{package_id}:case:{record.case_id}",
            target=f"domain_experience.{package_id}.case_memory.{record.case_id}",
            record=record,
            confidence=record.confidence,
            description=f"Domain experience package {package_id} seeds case {record.case_id}.",
        )
        for record in package.case_records
    )
    knowledge_updates = tuple(
        DomainKnowledgePriorUpdate(
            update_id=f"{package_id}:knowledge:{record.record_id}",
            target=f"domain_experience.{package_id}.domain_knowledge.{record.record_id}",
            record=record,
            confidence=record.confidence,
            description=f"Domain experience package {package_id} seeds knowledge {record.record_id}.",
            source_kind=KnowledgeSourceKind.RARE_HEAVY_IMPORT,
            review_status=KnowledgeReviewStatus.APPROVED,
            citation_ids=(record.locator,),
        )
        for record in package.knowledge_records
    )
    playbook_updates = tuple(
        StrategyPlaybookPriorUpdate(
            update_id=f"{package_id}:playbook:{rule.rule_id}",
            target=f"domain_experience.{package_id}.strategy_playbook.{rule.rule_id}",
            rule=rule,
            confidence=rule.confidence,
            description=f"Domain experience package {package_id} seeds playbook rule {rule.rule_id}.",
        )
        for rule in package.playbook_rules
    )
    boundary_updates = tuple(
        BoundaryPolicyPriorUpdate(
            update_id=f"{package_id}:boundary:{hint.hint_id}",
            target=f"domain_experience.{package_id}.boundary_policy.{hint.hint_id}",
            hint=hint,
            confidence=hint.confidence,
            description=f"Domain experience package {package_id} seeds boundary hint {hint.hint_id}.",
        )
        for hint in package.boundary_hints
    )
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id=f"domain-experience:{package_id}:{package.manifest.version}",
        case_memory_updates=case_updates,
        strategy_playbook_updates=playbook_updates,
        boundary_policy_updates=boundary_updates,
        domain_knowledge_updates=knowledge_updates,
        description=(
            f"Domain experience package {package_id} compiled into {len(case_updates)} case, "
            f"{len(knowledge_updates)} knowledge, {len(playbook_updates)} playbook, and "
            f"{len(boundary_updates)} boundary update(s)."
        ),
    )
    checkpoint = ApplicationRareHeavyCheckpoint(
        checkpoint_id=f"domain-experience:{package_id}:{package.manifest.version}",
        domain_template_biases=package.domain_template_biases,
        case_clusters=(),
        distilled_playbook_rules=package.playbook_rules,
        boundary_prior_hints=package.boundary_hints,
        reviewed_knowledge_candidates=package.reviewed_knowledge_candidates,
        continuum_profile_id=None,
        retrieval_readout_checkpoint=None,
        description=(
            f"Rare-heavy checkpoint compiled from domain experience package {package_id}."
        ),
    )
    return CompiledDomainExperiencePackage(
        package=package,
        validation_report=report,
        knowledge_records=package.knowledge_records,
        case_records=package.case_records,
        application_prior_update=prior_update,
        rare_heavy_checkpoint=checkpoint,
        evaluation_scenarios=package.evaluation_scenarios,
        description=(
            f"Compiled domain experience package {package_id} with "
            f"{len(package.knowledge_records)} knowledge record(s), {len(package.case_records)} case record(s), "
            f"{len(package.playbook_rules)} playbook rule(s), and {len(package.boundary_hints)} boundary hint(s)."
        ),
    )


def compile_domain_experience_packages(
    packages: tuple[DomainExperiencePackage, ...],
) -> tuple[CompiledDomainExperiencePackage, ...]:
    return tuple(compile_domain_experience_package(package) for package in packages)


def apply_compiled_domain_experience_packages(
    *,
    compiled_packages: tuple[CompiledDomainExperiencePackage, ...],
    domain_knowledge_store: ApplicationDomainKnowledgeStore,
    case_memory_store: ApplicationCaseMemoryStore,
    application_rare_heavy_state: ApplicationRareHeavyState,
    persist: bool = False,
) -> DomainExperienceApplicationReport:
    package_ids: list[str] = []
    knowledge_count = 0
    case_count = 0
    playbook_count = 0
    boundary_count = 0
    reviewed_count = 0
    operations: list[str] = []
    merged_checkpoint = application_rare_heavy_state.export_rare_heavy_state(
        checkpoint_id="domain-experience:pre-merge"
    )
    for compiled in compiled_packages:
        package_ids.append(compiled.package.manifest.package_id)
        domain_knowledge_store.upsert_records(compiled.knowledge_records)
        case_memory_store.upsert_records(compiled.case_records)
        knowledge_count += len(compiled.knowledge_records)
        case_count += len(compiled.case_records)
        operations.append(f"domain-experience:knowledge-upsert:{compiled.package.manifest.package_id}")
        operations.append(f"domain-experience:case-upsert:{compiled.package.manifest.package_id}")
        merged_checkpoint = _merge_rare_heavy_checkpoints(
            base=merged_checkpoint,
            update=compiled.rare_heavy_checkpoint,
        )
        playbook_count += len(compiled.rare_heavy_checkpoint.distilled_playbook_rules)
        boundary_count += len(compiled.rare_heavy_checkpoint.boundary_prior_hints)
        reviewed_count += len(compiled.rare_heavy_checkpoint.reviewed_knowledge_candidates)
    if compiled_packages:
        operations.extend(application_rare_heavy_state.import_rare_heavy_state(merged_checkpoint))
    if persist:
        if domain_knowledge_store.save_to_backend():
            operations.append("domain-experience:domain-knowledge-persist")
        if case_memory_store.save_to_backend():
            operations.append("domain-experience:case-memory-persist")
    return DomainExperienceApplicationReport(
        package_ids=tuple(package_ids),
        applied_knowledge_count=knowledge_count,
        applied_case_count=case_count,
        imported_playbook_count=playbook_count,
        imported_boundary_hint_count=boundary_count,
        imported_reviewed_knowledge_count=reviewed_count,
        operations=tuple(operations),
        description=(
            f"Applied {len(compiled_packages)} domain experience package(s): "
            f"{knowledge_count} knowledge record(s), {case_count} case record(s), "
            f"{playbook_count} playbook rule(s), and {boundary_count} boundary hint(s)."
        ),
    )


def apply_domain_experience_packages(
    *,
    packages: tuple[DomainExperiencePackage, ...],
    domain_knowledge_store: ApplicationDomainKnowledgeStore,
    case_memory_store: ApplicationCaseMemoryStore,
    application_rare_heavy_state: ApplicationRareHeavyState,
    persist: bool = False,
) -> DomainExperienceApplicationReport:
    return apply_compiled_domain_experience_packages(
        compiled_packages=compile_domain_experience_packages(packages),
        domain_knowledge_store=domain_knowledge_store,
        case_memory_store=case_memory_store,
        application_rare_heavy_state=application_rare_heavy_state,
        persist=persist,
    )


def _merge_rare_heavy_checkpoints(
    *,
    base: ApplicationRareHeavyCheckpoint,
    update: ApplicationRareHeavyCheckpoint,
) -> ApplicationRareHeavyCheckpoint:
    domain_biases = dict(base.domain_template_biases)
    domain_biases.update(dict(update.domain_template_biases))
    case_clusters = {
        cluster.cluster_id: cluster
        for cluster in base.case_clusters
    }
    case_clusters.update({cluster.cluster_id: cluster for cluster in update.case_clusters})
    playbook_by_pattern = {rule.problem_pattern: rule for rule in base.distilled_playbook_rules}
    for rule in update.distilled_playbook_rules:
        existing = playbook_by_pattern.get(rule.problem_pattern)
        if existing is None or rule.confidence >= existing.confidence:
            playbook_by_pattern[rule.problem_pattern] = rule
    boundary_by_key = {
        (hint.regime_id, hint.trigger_reasons): hint
        for hint in base.boundary_prior_hints
    }
    for hint in update.boundary_prior_hints:
        key = (hint.regime_id, hint.trigger_reasons)
        existing = boundary_by_key.get(key)
        if existing is None or hint.confidence >= existing.confidence:
            boundary_by_key[key] = hint
    reviewed = {
        candidate.candidate_id: candidate
        for candidate in base.reviewed_knowledge_candidates
    }
    reviewed.update({candidate.candidate_id: candidate for candidate in update.reviewed_knowledge_candidates})
    retrieval_checkpoint = (
        update.retrieval_readout_checkpoint
        if update.retrieval_readout_checkpoint is not None
        else base.retrieval_readout_checkpoint
    )
    return ApplicationRareHeavyCheckpoint(
        checkpoint_id=update.checkpoint_id,
        domain_template_biases=tuple(sorted(domain_biases.items())),
        case_clusters=tuple(sorted(case_clusters.values(), key=lambda cluster: cluster.cluster_id)),
        distilled_playbook_rules=tuple(
            sorted(playbook_by_pattern.values(), key=lambda rule: (rule.problem_pattern, rule.rule_id))
        ),
        boundary_prior_hints=tuple(
            sorted(
                boundary_by_key.values(),
                key=lambda hint: (hint.regime_id or "", ",".join(hint.trigger_reasons), hint.hint_id),
            )
        ),
        reviewed_knowledge_candidates=tuple(
            sorted(reviewed.values(), key=lambda candidate: candidate.candidate_id)
        ),
        continuum_profile_id=update.continuum_profile_id or base.continuum_profile_id,
        retrieval_readout_checkpoint=retrieval_checkpoint,
        description=(
            f"Merged domain experience checkpoint {update.checkpoint_id} with prior rare-heavy application state."
        ),
    )


def _issue(
    package_id: str,
    severity: str,
    location: str,
    description: str,
) -> DomainExperienceValidationIssue:
    return DomainExperienceValidationIssue(
        issue_id=f"{package_id}:{location}:{severity}",
        severity=severity,
        location=location,
        description=description,
    )


def _require_non_empty(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    location: str,
    value: str,
) -> None:
    if value:
        return
    issues.append(_issue(package_id, "error", location, "Required text field must be non-empty."))


def _check_duplicate_values(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    location: str,
    values: tuple[str, ...],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    for value in sorted(duplicates):
        issues.append(_issue(package_id, "error", location, f"Duplicate value {value} is not allowed."))


def _validate_knowledge_record(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    record: DomainKnowledgeRecord,
) -> None:
    _require_non_empty(issues, package_id, f"knowledge_records.{record.record_id}.record_id", record.record_id)
    _require_non_empty(issues, package_id, f"knowledge_records.{record.record_id}.domain", record.domain)
    _require_non_empty(issues, package_id, f"knowledge_records.{record.record_id}.locator", record.locator)
    if record.confidence < 0.0 or record.confidence > 1.0:
        issues.append(
            _issue(
                package_id,
                "error",
                f"knowledge_records.{record.record_id}.confidence",
                "Knowledge confidence must be within [0.0, 1.0].",
            )
        )
    if record.evidence_strength not in {item.value for item in EvidenceStrength}:
        issues.append(
            _issue(
                package_id,
                "error",
                f"knowledge_records.{record.record_id}.evidence_strength",
                "Knowledge evidence strength must match EvidenceStrength.",
            )
        )
    if record.source_type not in {item.value for item in KnowledgeSourceType}:
        issues.append(
            _issue(
                package_id,
                "error",
                f"knowledge_records.{record.record_id}.source_type",
                "Knowledge source type must match KnowledgeSourceType.",
            )
        )


def _validate_case_record(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    record: CaseMemoryRecord,
) -> None:
    _require_non_empty(issues, package_id, f"case_records.{record.case_id}.case_id", record.case_id)
    _require_non_empty(issues, package_id, f"case_records.{record.case_id}.domain", record.domain)
    _validate_risk_markers(issues, package_id, f"case_records.{record.case_id}.risk_markers", record.risk_markers)
    _validate_track_tags(issues, package_id, f"case_records.{record.case_id}.track_tags", record.track_tags)
    if not record.intervention_ordering:
        issues.append(
            _issue(
                package_id,
                "error",
                f"case_records.{record.case_id}.intervention_ordering",
                "Case memory record must include at least one intervention step.",
            )
        )
    if record.confidence < 0.0 or record.confidence > 1.0 or record.relevance_score < 0.0 or record.relevance_score > 1.0:
        issues.append(
            _issue(
                package_id,
                "error",
                f"case_records.{record.case_id}.scores",
                "Case confidence and relevance score must be within [0.0, 1.0].",
            )
        )


def _validate_playbook_rule(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    rule: PlaybookRule,
) -> None:
    _require_non_empty(issues, package_id, f"playbook_rules.{rule.rule_id}.rule_id", rule.rule_id)
    _require_non_empty(issues, package_id, f"playbook_rules.{rule.rule_id}.problem_pattern", rule.problem_pattern)
    if not rule.recommended_ordering:
        issues.append(
            _issue(
                package_id,
                "error",
                f"playbook_rules.{rule.rule_id}.recommended_ordering",
                "Playbook rule must include recommended ordering.",
            )
        )
    if rule.confidence < 0.0 or rule.confidence > 1.0:
        issues.append(
            _issue(
                package_id,
                "error",
                f"playbook_rules.{rule.rule_id}.confidence",
                "Playbook confidence must be within [0.0, 1.0].",
            )
        )


def _validate_boundary_hint(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    hint: BoundaryPriorHint,
) -> None:
    _require_non_empty(issues, package_id, f"boundary_hints.{hint.hint_id}.hint_id", hint.hint_id)
    if not hint.trigger_reasons:
        issues.append(
            _issue(
                package_id,
                "error",
                f"boundary_hints.{hint.hint_id}.trigger_reasons",
                "Boundary hint must include at least one trigger reason.",
            )
        )
    if hint.confidence < 0.0 or hint.confidence > 1.0:
        issues.append(
            _issue(
                package_id,
                "error",
                f"boundary_hints.{hint.hint_id}.confidence",
                "Boundary confidence must be within [0.0, 1.0].",
            )
        )


def _validate_reviewed_candidate(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    candidate: ReviewedKnowledgeCandidate,
) -> None:
    if candidate.source_kind not in {
        KnowledgeSourceKind.EXTERNAL_IMPORT,
        KnowledgeSourceKind.RARE_HEAVY_IMPORT,
    }:
        issues.append(
            _issue(
                package_id,
                "error",
                f"reviewed_knowledge_candidates.{candidate.candidate_id}.source_kind",
                "Reviewed package knowledge must come from external or rare-heavy import.",
            )
        )
    if candidate.review_status is not KnowledgeReviewStatus.APPROVED:
        issues.append(
            _issue(
                package_id,
                "error",
                f"reviewed_knowledge_candidates.{candidate.candidate_id}.review_status",
                "Reviewed package knowledge must be approved before import.",
            )
        )
    if candidate.confidence < 0.0 or candidate.confidence > 1.0:
        issues.append(
            _issue(
                package_id,
                "error",
                f"reviewed_knowledge_candidates.{candidate.candidate_id}.confidence",
                "Reviewed knowledge confidence must be within [0.0, 1.0].",
            )
        )


def _validate_scenario(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    scenario: DomainScenario,
) -> None:
    _require_non_empty(issues, package_id, f"scenarios.{scenario.scenario_id}.scenario_id", scenario.scenario_id)
    _require_non_empty(issues, package_id, f"scenarios.{scenario.scenario_id}.domain", scenario.domain)
    _validate_risk_markers(issues, package_id, f"scenarios.{scenario.scenario_id}.risk_markers", scenario.risk_markers)
    _validate_track_tags(issues, package_id, f"scenarios.{scenario.scenario_id}.track_tags", scenario.track_tags)


def _validate_evaluation_scenario(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    scenario: DomainExperienceEvaluationScenario,
) -> None:
    _require_non_empty(
        issues,
        package_id,
        f"evaluation_scenarios.{scenario.scenario_id}.scenario_id",
        scenario.scenario_id,
    )
    _require_non_empty(issues, package_id, f"evaluation_scenarios.{scenario.scenario_id}.prompt", scenario.prompt)
    _validate_risk_markers(
        issues,
        package_id,
        f"evaluation_scenarios.{scenario.scenario_id}.risk_markers",
        scenario.risk_markers,
    )


def _validate_risk_markers(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    location: str,
    markers: tuple[str, ...],
) -> None:
    for marker in markers:
        if marker in ALLOWED_RISK_MARKERS:
            continue
        issues.append(
            _issue(
                package_id,
                "error",
                location,
                f"Risk marker {marker} is not supported by the domain experience contract.",
            )
        )


def _validate_track_tags(
    issues: list[DomainExperienceValidationIssue],
    package_id: str,
    location: str,
    tags: tuple[str, ...],
) -> None:
    for tag in tags:
        if tag in ALLOWED_TRACK_TAGS:
            continue
        issues.append(
            _issue(
                package_id,
                "error",
                location,
                f"Track tag {tag} is not supported by the domain experience contract.",
            )
        )
