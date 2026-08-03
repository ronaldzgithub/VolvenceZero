"""Read-only parsers for transcripts, promotion evidence and campaign plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml

from .config import ForgePaths
from .foundation import ForgeError, canonical_json, sha256_text


VERDICT_GATE_FIELDS = (
    "causal_gates",
    "mechanism_gates",
    "selector_gates",
    "signal_gates",
    "acceptance_gates",
    "gates",
)
PASSING_VERDICTS = frozenset({"causal-supported", "supported", "pass", "passed", "promoted"})
FAILING_VERDICTS = frozenset({"not-supported", "blocked", "fail", "failed", "rejected"})
BENCH_TURN_AVERAGE_THRESHOLD = 3.0
BENCH_ARC_AXIS_THRESHOLD = 60.0
LIVE_DIALOGUE_OUTCOME_SCHEMA_VERSION = "lifeform-live-dialogue-outcome.v1"
LIVE_DIALOGUE_OUTCOME_PRIVACY_PROFILE = "typed-metadata-only.v1"
_T = TypeVar("_T")


class SourceParseError(ForgeError):
    """Raised when a public source artifact violates its declared shape."""


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str
    source_kind: str
    locator: str
    excerpt: str
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "locator": self.locator,
            "excerpt": self.excerpt,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class TranscriptSource:
    source_id: str
    path: Path
    tool_sequence: tuple[str, ...]
    error_refs: tuple[EvidenceRef, ...]
    turn_statuses: tuple[str, ...]
    repeated_tool_runs: tuple[tuple[str, int], ...]

    def analysis_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": "transcript",
            "path": str(self.path),
            "tool_sequence": list(self.tool_sequence[-80:]),
            "turn_statuses": list(self.turn_statuses),
            "repeated_tool_runs": [list(item) for item in self.repeated_tool_runs],
            "errors": [ref.as_dict() for ref in self.error_refs],
        }


@dataclass(frozen=True)
class VerdictSource:
    source_id: str
    path: Path
    verdict: str | None
    failed_gate_refs: tuple[EvidenceRef, ...]
    passing_behaviors: tuple[str, ...]
    report_excerpt: str

    def analysis_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": "promotion_verdict",
            "path": str(self.path),
            "verdict": self.verdict,
            "failed_gates": [ref.as_dict() for ref in self.failed_gate_refs],
            "passing_behaviors": list(self.passing_behaviors),
            "report_excerpt": self.report_excerpt,
        }


@dataclass(frozen=True)
class PlanSource:
    source_id: str
    path: Path
    name: str
    overview: str
    todo_summary: tuple[str, ...]

    def analysis_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": "plan",
            "path": str(self.path),
            "name": self.name,
            "overview": self.overview,
            "todo_summary": list(self.todo_summary),
        }


@dataclass(frozen=True)
class BenchBundleSource:
    source_id: str
    path: Path
    arc_id: str
    scenario_id: str
    family: str
    failure_refs: tuple[EvidenceRef, ...]
    passing_behaviors: tuple[str, ...]

    def analysis_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": "bench_bundle",
            "path": str(self.path),
            "arc_id": self.arc_id,
            "scenario_id": self.scenario_id,
            "family": self.family,
            "failures": [ref.as_dict() for ref in self.failure_refs],
            "passing_behaviors": list(self.passing_behaviors),
        }


@dataclass(frozen=True)
class LiveDialogueOutcomeSource:
    source_id: str
    path: Path
    artifact_id: str
    recorded_at_iso: str
    outcome_kind: str
    evidence_source: str
    confidence: float
    consuming_turn_index: int
    action_turn_index: int
    action_context: dict[str, Any] | None
    service_version: str
    policy_version: str
    observation_ref: EvidenceRef

    def analysis_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": "live_dialogue_outcome",
            "path_sha256": sha256_text(str(self.path.resolve())),
            "artifact_id": self.artifact_id,
            "recorded_at_iso": self.recorded_at_iso,
            "outcome_kind": self.outcome_kind,
            "evidence_source": self.evidence_source,
            "confidence": self.confidence,
            "consuming_turn_index": self.consuming_turn_index,
            "action_turn_index": self.action_turn_index,
            "action_context": self.action_context,
            "service_version": self.service_version,
            "policy_version": self.policy_version,
            "observation": self.observation_ref.as_dict(),
        }


@dataclass(frozen=True)
class SourceBundle:
    transcripts: tuple[TranscriptSource, ...]
    verdicts: tuple[VerdictSource, ...]
    plans: tuple[PlanSource, ...]
    bench_bundles: tuple[BenchBundleSource, ...] = ()
    live_dialogue_outcomes: tuple[LiveDialogueOutcomeSource, ...] = ()
    evidence_since: str | None = None


def parse_transcript(path: Path) -> TranscriptSource:
    tool_sequence: list[str] = []
    error_refs: list[EvidenceRef] = []
    statuses: list[str] = []
    source_id = _source_id("transcript", path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise SourceParseError(f"Cannot read transcript {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceParseError(f"Invalid transcript JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise SourceParseError(f"Transcript record must be an object at {path}:{line_number}")

        record_type = record.get("type")
        if record_type == "turn_ended":
            status = record.get("status")
            if not isinstance(status, str) or not status:
                raise SourceParseError(f"turn_ended.status must be a string at {path}:{line_number}")
            statuses.append(status)
            if status == "error":
                error = record.get("error")
                excerpt = _bounded_text(error, context=f"{path}:{line_number}.error")
                error_refs.append(_evidence_ref(source_id, "transcript", f"line:{line_number}", excerpt))

        message = record.get("message")
        if message is None:
            continue
        if not isinstance(message, dict):
            raise SourceParseError(f"message must be an object at {path}:{line_number}")
        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, list):
            raise SourceParseError(f"message.content must be a list at {path}:{line_number}")
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise SourceParseError(f"content block must be an object at {path}:{line_number}:{block_index}")
            block_type = block.get("type")
            if block_type == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise SourceParseError(f"tool_use.name must be a string at {path}:{line_number}:{block_index}")
                tool_sequence.append(name)
            elif block_type == "tool_result" and block.get("is_error") is True:
                excerpt = _bounded_text(
                    block.get("content"), context=f"{path}:{line_number}:{block_index}.content"
                )
                error_refs.append(
                    _evidence_ref(
                        source_id,
                        "transcript",
                        f"line:{line_number}/content:{block_index}",
                        excerpt,
                    )
                )
    return TranscriptSource(
        source_id=source_id,
        path=path,
        tool_sequence=tuple(tool_sequence),
        error_refs=tuple(error_refs),
        turn_statuses=tuple(statuses),
        repeated_tool_runs=_repeated_runs(tool_sequence),
    )


def parse_verdict(path: Path) -> VerdictSource:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceParseError(f"Missing promotion verdict: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceParseError(f"Invalid promotion verdict JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SourceParseError(f"Promotion verdict must be an object: {path}")
    source_id = _source_id("promotion_verdict", path)
    failed: list[EvidenceRef] = []
    passing: list[str] = []
    for field in VERDICT_GATE_FIELDS:
        if field not in raw:
            continue
        for gate_name, passed in _named_boolean_gates(raw[field], context=f"{path}:{field}"):
            locator = f"{field}.{gate_name}"
            if passed:
                passing.append(locator)
            else:
                failed.append(_evidence_ref(source_id, "promotion_verdict", locator, f"{locator}=false"))

    promotion_allowed = raw.get("promotion_allowed")
    if promotion_allowed is False:
        failed.append(
            _evidence_ref(
                source_id,
                "promotion_verdict",
                "promotion_allowed",
                "promotion_allowed=false",
            )
        )
    elif promotion_allowed is True:
        passing.append("promotion_allowed=true")
    elif promotion_allowed is not None:
        raise SourceParseError(f"promotion_allowed must be boolean in {path}")

    verdict_value = raw.get("verdict")
    if verdict_value is not None and not isinstance(verdict_value, str):
        raise SourceParseError(f"verdict must be a string in {path}")
    if verdict_value in FAILING_VERDICTS:
        failed.append(
            _evidence_ref(source_id, "promotion_verdict", "verdict", f"verdict={verdict_value}")
        )
    elif verdict_value in PASSING_VERDICTS:
        passing.append(f"verdict={verdict_value}")

    report_path = path.with_name("report.md")
    report_excerpt = ""
    if report_path.exists():
        try:
            report_excerpt = _normalize_excerpt(report_path.read_text(encoding="utf-8"), 1200)
        except UnicodeDecodeError as exc:
            raise SourceParseError(f"Cannot decode report {report_path}: {exc}") from exc
    return VerdictSource(
        source_id=source_id,
        path=path,
        verdict=verdict_value,
        failed_gate_refs=tuple(_deduplicate_refs(failed)),
        passing_behaviors=tuple(sorted(set(passing))),
        report_excerpt=report_excerpt,
    )


def parse_plan(path: Path) -> PlanSource:
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise SourceParseError(f"Cannot read plan {path}: {exc}") from exc
    if not text.startswith("---\n"):
        return _parse_legacy_markdown_plan(path, text)
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise SourceParseError(f"Plan frontmatter is not terminated: {path}")
    try:
        metadata = yaml.safe_load(text[4:closing])
    except yaml.YAMLError as exc:
        raise SourceParseError(f"Invalid plan YAML {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SourceParseError(f"Plan frontmatter must be a mapping: {path}")
    name = metadata.get("name")
    overview = metadata.get("overview")
    if not isinstance(name, str) or not name.strip():
        raise SourceParseError(f"Plan name must be a non-empty string: {path}")
    if not isinstance(overview, str) or not overview.strip():
        raise SourceParseError(f"Plan overview must be a non-empty string: {path}")
    todos_raw = metadata.get("todos", [])
    if not isinstance(todos_raw, list):
        raise SourceParseError(f"Plan todos must be a list: {path}")
    todos: list[str] = []
    for index, item in enumerate(todos_raw):
        if not isinstance(item, dict):
            raise SourceParseError(f"Plan todo[{index}] must be a mapping: {path}")
        content = item.get("content")
        status = item.get("status")
        if not isinstance(content, str) or not isinstance(status, str):
            raise SourceParseError(f"Plan todo[{index}] needs string content/status: {path}")
        todos.append(f"{status}: {content}")
    return PlanSource(
        source_id=_source_id("plan", path),
        path=path,
        name=name.strip(),
        overview=_normalize_excerpt(overview, 1200),
        todo_summary=tuple(todos),
    )


def _parse_legacy_markdown_plan(path: Path, text: str) -> PlanSource:
    """Parse the repository's one pre-frontmatter plan format explicitly."""

    lines = text.splitlines()
    heading = next((line[2:].strip() for line in lines if line.startswith("# ") and line[2:].strip()), None)
    if heading is None:
        raise SourceParseError(f"Legacy plan needs a level-one heading: {path}")
    body = "\n".join(line for line in lines if not line.startswith("# ")).strip()
    if not body:
        raise SourceParseError(f"Legacy plan needs non-empty narrative context: {path}")
    return PlanSource(
        source_id=_source_id("plan", path),
        path=path,
        name=heading,
        overview=_normalize_excerpt(body, 1200),
        todo_summary=(),
    )


def parse_bench_bundle(path: Path) -> BenchBundleSource:
    """Parse a judged Companion Bench arc into bounded, turn-level evidence."""

    raw = _read_json_object(path, context="Companion Bench bundle")
    arc = _required_mapping(raw, "arc", path)
    arc_id = _required_string(arc, "arc_id", path)
    scenario_id = _required_string(arc, "scenario_id", path)
    family = _required_string(arc, "family", path)
    turns = _bench_turn_index(arc, path)
    failures: list[EvidenceRef] = []
    passing: set[str] = set()

    perturn = _required_mapping(raw, "perturn_rubric", path)
    turn_scores = perturn.get("turn_scores")
    if not isinstance(turn_scores, list):
        raise SourceParseError(f"perturn_rubric.turn_scores must be a list: {path}")
    criterion_values: dict[str, list[float]] = {}
    low_rows: list[tuple[float, int, int, dict[str, float]]] = []
    for index, row in enumerate(turn_scores):
        if not isinstance(row, dict):
            raise SourceParseError(f"turn_scores[{index}] must be a mapping: {path}")
        session_index = _required_integer(row, "session_index", path)
        turn_index = _required_integer(row, "turn_index", path)
        scores_raw = row.get("scores")
        if not isinstance(scores_raw, dict) or not scores_raw:
            raise SourceParseError(f"turn_scores[{index}].scores must be a non-empty mapping: {path}")
        scores: dict[str, float] = {}
        for criterion, value in scores_raw.items():
            if not isinstance(criterion, str) or not criterion:
                raise SourceParseError(f"turn_scores[{index}] has an invalid criterion name: {path}")
            numeric = _finite_number(value, f"turn_scores[{index}].scores.{criterion}", path)
            if not 0.0 <= numeric <= 5.0:
                raise SourceParseError(f"turn score must be in [0, 5] at {path}:{index}")
            scores[criterion] = numeric
            criterion_values.setdefault(criterion, []).append(numeric)
        average = _finite_number(row.get("average"), f"turn_scores[{index}].average", path)
        recomputed = sum(scores.values()) / len(scores)
        if abs(average - recomputed) > 1e-6:
            raise SourceParseError(
                f"turn_scores[{index}].average does not match criterion scores: {path}"
            )
        if average < BENCH_TURN_AVERAGE_THRESHOLD:
            low_rows.append((average, session_index, turn_index, scores))

    source_id = _source_id("bench_bundle", path)
    for average, session_index, turn_index, scores in sorted(
        low_rows, key=lambda item: (item[0], item[1], item[2])
    )[:12]:
        turn = turns.get((session_index, turn_index))
        if turn is None:
            raise SourceParseError(
                f"per-turn score has no matching dialogue turn session={session_index} "
                f"turn={turn_index}: {path}"
            )
        excerpt = canonical_json(
            {
                "average": average,
                "scores": scores,
                "user_text": _normalize_excerpt(_required_string(turn, "user_text", path), 500),
                "assistant_text": _normalize_excerpt(
                    _required_string(turn, "assistant_text", path), 500
                ),
            }
        )
        failures.append(
            _evidence_ref(
                source_id,
                "bench_bundle",
                f"arc:{arc_id}/session:{session_index}/turn:{turn_index}",
                excerpt,
            )
        )
    for criterion, values in criterion_values.items():
        if values and sum(values) / len(values) >= BENCH_TURN_AVERAGE_THRESHOLD:
            passing.add(f"perturn_rubric.{criterion}")

    disqualifier = _required_mapping(raw, "disqualifier_report", path)
    any_triggered = disqualifier.get("any_triggered")
    if not isinstance(any_triggered, bool):
        raise SourceParseError(f"disqualifier_report.any_triggered must be boolean: {path}")
    results = disqualifier.get("results")
    if not isinstance(results, list):
        raise SourceParseError(f"disqualifier_report.results must be a list: {path}")
    triggered_count = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise SourceParseError(f"disqualifier_report.results[{index}] must be a mapping: {path}")
        kind = _required_string(result, "kind", path)
        triggered = result.get("triggered")
        if not isinstance(triggered, bool):
            raise SourceParseError(
                f"disqualifier_report.results[{index}].triggered must be boolean: {path}"
            )
        if triggered:
            triggered_count += 1
            detail = _required_string(result, "detail", path)
            failures.append(
                _evidence_ref(
                    source_id,
                    "bench_bundle",
                    f"arc:{arc_id}/disqualifier:{kind}",
                    detail,
                )
            )
        else:
            passing.add(f"disqualifier.{kind}=false")
    if any_triggered != (triggered_count > 0):
        raise SourceParseError(f"disqualifier_report.any_triggered is inconsistent: {path}")

    arc_scores = _required_mapping(_required_mapping(raw, "arc_axis_scores", path), "scores", path)
    for axis, value in arc_scores.items():
        if not isinstance(axis, str) or not axis:
            raise SourceParseError(f"arc_axis_scores has an invalid axis name: {path}")
        numeric = _finite_number(value, f"arc_axis_scores.scores.{axis}", path)
        if not 0.0 <= numeric <= 100.0:
            raise SourceParseError(f"arc axis score must be in [0, 100] at {path}:{axis}")
        if numeric < BENCH_ARC_AXIS_THRESHOLD:
            failures.append(
                _evidence_ref(
                    source_id,
                    "bench_bundle",
                    f"arc:{arc_id}/axis:{axis}",
                    f"arc_axis_scores.{axis}={numeric:.3f}",
                )
            )
        else:
            passing.add(f"arc_axis_scores.{axis}")

    return BenchBundleSource(
        source_id=source_id,
        path=path,
        arc_id=arc_id,
        scenario_id=scenario_id,
        family=family,
        failure_refs=tuple(_deduplicate_refs(failures)),
        passing_behaviors=tuple(sorted(passing)),
    )


def parse_arc_failure_log(path: Path) -> tuple[BenchBundleSource, ...]:
    """Parse transport/runtime failures that prevented a bench arc bundle."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise SourceParseError(f"Cannot read arc failure log {path}: {exc}") from exc
    sources: list[BenchBundleSource] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceParseError(f"Invalid arc failure JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SourceParseError(f"Arc failure row must be a mapping at {path}:{line_number}")
        scenario_id = _required_string(raw, "scenario_id", path)
        stage = _required_string(raw, "stage", path)
        exception_type = _required_string(raw, "exception_type", path)
        exception = _required_string(raw, "exception", path)
        source_id = f"bench_bundle:{sha256_text(f'{path.resolve()}:{line_number}')[:16]}"
        reference = _evidence_ref(
            source_id,
            "bench_bundle",
            f"arc_failure:{line_number}/stage:{stage}",
            canonical_json(
                {
                    "exception": _normalize_excerpt(exception, 800),
                    "exception_type": exception_type,
                    "stage": stage,
                }
            ),
        )
        sources.append(
            BenchBundleSource(
                source_id=source_id,
                path=path,
                arc_id=f"failed:{scenario_id}:{line_number}",
                scenario_id=scenario_id,
                family="arc_failure",
                failure_refs=(reference,),
                passing_behaviors=(),
            )
        )
    return tuple(sources)


def parse_live_dialogue_outcome(path: Path) -> LiveDialogueOutcomeSource:
    """Parse one privacy-bounded service outcome without classifying failure."""

    raw = _read_json_object(path, context="live dialogue outcome")
    expected_fields = {
        "schema_version",
        "artifact_id",
        "recorded_at_iso",
        "subject_scope_sha256",
        "session_scope_sha256",
        "source_evidence_sha256",
        "outcome_kind",
        "evidence_source",
        "confidence",
        "consuming_turn_index",
        "action_turn_index",
        "action_context",
        "service_version",
        "policy_version",
        "privacy_profile",
        "content_sha256",
    }
    if set(raw) != expected_fields:
        raise SourceParseError(
            f"live dialogue outcome has unexpected schema fields: {path}"
        )
    if raw["schema_version"] != LIVE_DIALOGUE_OUTCOME_SCHEMA_VERSION:
        raise SourceParseError(f"unsupported live dialogue outcome schema: {path}")
    if raw["privacy_profile"] != LIVE_DIALOGUE_OUTCOME_PRIVACY_PROFILE:
        raise SourceParseError(f"unsupported live dialogue outcome privacy profile: {path}")
    stored_digest = _required_sha256(raw, "content_sha256", path)
    content_body = {key: value for key, value in raw.items() if key != "content_sha256"}
    computed_digest = sha256_text(_canonical_ascii_json(content_body))
    if stored_digest != computed_digest:
        raise SourceParseError(f"live dialogue outcome content_sha256 mismatch: {path}")

    recorded_at_iso = _required_string(raw, "recorded_at_iso", path)
    parse_evidence_timestamp(recorded_at_iso)
    source_evidence_sha256 = _required_sha256(raw, "source_evidence_sha256", path)
    artifact_id = _required_string(raw, "artifact_id", path)
    if artifact_id != f"live-dialogue-outcome:{source_evidence_sha256[:24]}":
        raise SourceParseError(f"live dialogue outcome artifact_id is not content-bound: {path}")
    _required_sha256(raw, "subject_scope_sha256", path)
    _required_sha256(raw, "session_scope_sha256", path)
    outcome_kind = _required_string(raw, "outcome_kind", path)
    evidence_source = _required_string(raw, "evidence_source", path)
    confidence = _finite_number(raw.get("confidence"), "confidence", path)
    if not 0.0 <= confidence <= 1.0:
        raise SourceParseError(f"live dialogue outcome confidence must be in [0, 1]: {path}")
    consuming_turn_index = _required_integer(raw, "consuming_turn_index", path)
    action_turn_index = _required_integer(raw, "action_turn_index", path)
    if consuming_turn_index < 0:
        raise SourceParseError(f"consuming_turn_index must be non-negative: {path}")
    if action_turn_index < -1:
        raise SourceParseError(f"action_turn_index must be non-negative or -1: {path}")
    action_context = _parse_live_action_context(
        raw.get("action_context"),
        action_turn_index=action_turn_index,
        path=path,
    )
    service_version = _required_string(raw, "service_version", path)
    policy_version = _required_string(raw, "policy_version", path)
    source_id = f"live_dialogue_outcome:{source_evidence_sha256[:16]}"
    observation = canonical_json(
        {
            "outcome_kind": outcome_kind,
            "evidence_source": evidence_source,
            "confidence": confidence,
            "consuming_turn_index": consuming_turn_index,
            "action_turn_index": action_turn_index,
            "action_context": action_context,
            "service_version": service_version,
            "policy_version": policy_version,
        }
    )
    return LiveDialogueOutcomeSource(
        source_id=source_id,
        path=path,
        artifact_id=artifact_id,
        recorded_at_iso=recorded_at_iso,
        outcome_kind=outcome_kind,
        evidence_source=evidence_source,
        confidence=confidence,
        consuming_turn_index=consuming_turn_index,
        action_turn_index=action_turn_index,
        action_context=action_context,
        service_version=service_version,
        policy_version=policy_version,
        observation_ref=_evidence_ref(
            source_id,
            "live_dialogue_outcome",
            f"artifact:{artifact_id}/action-turn:{action_turn_index}",
            observation,
        ),
    )


def load_source_bundle(
    paths: ForgePaths,
    *,
    max_transcripts: int | None = None,
    max_verdicts: int | None = None,
    max_plans: int | None = None,
    verdict_root: Path | None = None,
    bench_root: Path | None = None,
    max_bench_bundles: int | None = None,
    live_outcome_root: Path | None = None,
    max_live_outcomes: int | None = None,
    since: datetime | None = None,
) -> SourceBundle:
    if since is not None and since.tzinfo is None:
        raise SourceParseError("evidence since timestamp must include a timezone")
    transcript_paths = sorted(paths.transcripts_root.rglob("*.jsonl")) if paths.transcripts_root.exists() else []
    verdict_base = (verdict_root or paths.artifacts_root).resolve()
    verdict_paths = sorted(verdict_base.rglob("promotion_verdict.json")) if verdict_base.exists() else []
    plan_paths = sorted(paths.plans_root.glob("*.plan.md")) if paths.plans_root.exists() else []
    bench_base = (bench_root or paths.artifacts_root).resolve()
    bench_paths = sorted(bench_base.rglob("*.bundle.json")) if bench_base.exists() else []
    arc_failure_paths = sorted(bench_base.rglob("arc_failure.jsonl")) if bench_base.exists() else []
    live_outcome_paths: list[Path] = []
    if live_outcome_root is not None:
        resolved_live_root = live_outcome_root.expanduser().resolve()
        if not resolved_live_root.is_dir():
            raise SourceParseError(
                f"live outcome root must be an existing directory: {resolved_live_root}"
            )
        live_outcome_paths = sorted(resolved_live_root.rglob("*.json"))
    transcript_paths = _modified_since(transcript_paths, since)
    verdict_paths = _modified_since(verdict_paths, since)
    plan_paths = _modified_since(plan_paths, since)
    bench_paths = _modified_since(bench_paths, since)
    arc_failure_paths = _modified_since(arc_failure_paths, since)
    transcript_paths = _limit(transcript_paths, max_transcripts)
    verdict_paths = _limit(verdict_paths, max_verdicts)
    plan_paths = _limit(plan_paths, max_plans)
    bench_sources: list[BenchBundleSource] = [parse_bench_bundle(path) for path in bench_paths]
    for path in arc_failure_paths:
        bench_sources.extend(parse_arc_failure_log(path))
    bench_sources = _limit(bench_sources, max_bench_bundles)
    live_outcomes = [parse_live_dialogue_outcome(path) for path in live_outcome_paths]
    live_source_ids = [source.source_id for source in live_outcomes]
    if len(live_source_ids) != len(set(live_source_ids)):
        raise SourceParseError("live outcome root contains duplicate source evidence ids")
    if since is not None:
        live_outcomes = [
            source
            for source in live_outcomes
            if parse_evidence_timestamp(source.recorded_at_iso) >= since.astimezone(timezone.utc)
        ]
    live_outcomes = _limit(live_outcomes, max_live_outcomes)
    return SourceBundle(
        transcripts=tuple(parse_transcript(path) for path in transcript_paths),
        verdicts=tuple(parse_verdict(path) for path in verdict_paths),
        plans=tuple(parse_plan(path) for path in plan_paths),
        bench_bundles=tuple(bench_sources),
        live_dialogue_outcomes=tuple(live_outcomes),
        evidence_since=_render_timestamp(since) if since is not None else None,
    )


def parse_evidence_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceParseError(f"Invalid evidence timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise SourceParseError("Evidence timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def latest_applied_timestamp(ledger_path: Path) -> datetime:
    if not ledger_path.exists():
        raise SourceParseError(f"Missing Forge ledger: {ledger_path}")
    timestamps: list[datetime] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceParseError(f"Invalid ledger JSON at {ledger_path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise SourceParseError(f"Ledger event must be a mapping at {ledger_path}:{line_number}")
        if event.get("event") != "proposal_decision" or event.get("decision") != "applied":
            continue
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str):
            raise SourceParseError(f"Applied ledger event lacks timestamp at {ledger_path}:{line_number}")
        timestamps.append(parse_evidence_timestamp(timestamp))
    if not timestamps:
        raise SourceParseError("--evidence-since-ledger requires at least one applied proposal event")
    return max(timestamps)


def _limit(values: list[_T], maximum: int | None) -> list[_T]:
    if maximum is None:
        return values
    if maximum < 0:
        raise SourceParseError("Source limits must be non-negative")
    return values[:maximum]


def _modified_since(paths: list[Path], since: datetime | None) -> list[Path]:
    if since is None:
        return paths
    threshold = since.timestamp()
    selected: list[Path] = []
    for path in paths:
        try:
            modified = path.stat().st_mtime
        except OSError as exc:
            raise SourceParseError(f"Cannot stat evidence source {path}: {exc}") from exc
        if modified >= threshold:
            selected.append(path)
    return selected


def _render_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceParseError(f"Missing {context}: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SourceParseError(f"Cannot decode {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SourceParseError(f"Invalid {context} JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SourceParseError(f"{context} must be a mapping: {path}")
    return raw


def _required_mapping(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise SourceParseError(f"{key} must be a mapping: {path}")
    return value


def _required_string(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SourceParseError(f"{key} must be a non-empty string: {path}")
    return value.strip()


def _required_integer(raw: dict[str, Any], key: str, path: Path) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SourceParseError(f"{key} must be an integer: {path}")
    return value


def _required_sha256(raw: dict[str, Any], key: str, path: Path) -> str:
    value = _required_string(raw, key, path)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SourceParseError(f"{key} must be a lowercase SHA-256 digest: {path}")
    return value


def _parse_live_action_context(
    value: Any,
    *,
    action_turn_index: int,
    path: Path,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SourceParseError(f"action_context must be a mapping or null: {path}")
    expected = {
        "turn_index",
        "scene_id_sha256",
        "trigger_kind",
        "active_regime",
        "active_abstract_action",
        "prediction_error_magnitude",
        "open_loop_count",
        "commitment_count",
        "elapsed_at_tick",
    }
    if set(value) != expected:
        raise SourceParseError(f"action_context has unexpected schema fields: {path}")
    turn_index = _required_integer(value, "turn_index", path)
    if turn_index != action_turn_index:
        raise SourceParseError(f"action_context.turn_index must equal action_turn_index: {path}")
    _required_sha256(value, "scene_id_sha256", path)
    _required_string(value, "trigger_kind", path)
    for key in ("active_regime", "active_abstract_action"):
        optional = value.get(key)
        if optional is not None and (not isinstance(optional, str) or not optional.strip()):
            raise SourceParseError(f"action_context.{key} must be null or a non-empty string: {path}")
    _finite_number(
        value.get("prediction_error_magnitude"),
        "action_context.prediction_error_magnitude",
        path,
    )
    for key in ("open_loop_count", "commitment_count", "elapsed_at_tick"):
        if _required_integer(value, key, path) < 0:
            raise SourceParseError(f"action_context.{key} must be non-negative: {path}")
    return dict(value)


def _canonical_ascii_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _finite_number(value: Any, context: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SourceParseError(f"{context} must be numeric: {path}")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise SourceParseError(f"{context} must be finite: {path}")
    return numeric


def _bench_turn_index(arc: dict[str, Any], path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    sessions = arc.get("sessions")
    if not isinstance(sessions, list):
        raise SourceParseError(f"arc.sessions must be a list: {path}")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for session_position, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise SourceParseError(f"arc.sessions[{session_position}] must be a mapping: {path}")
        turns = session.get("turns")
        if not isinstance(turns, list):
            raise SourceParseError(f"arc.sessions[{session_position}].turns must be a list: {path}")
        for turn_position, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise SourceParseError(
                    f"arc.sessions[{session_position}].turns[{turn_position}] must be a mapping: {path}"
                )
            identity = (
                _required_integer(turn, "session_index", path),
                _required_integer(turn, "turn_index", path),
            )
            if identity in indexed:
                raise SourceParseError(f"Duplicate bench turn identity {identity}: {path}")
            indexed[identity] = turn
    return indexed


def _source_id(kind: str, path: Path) -> str:
    return f"{kind}:{sha256_text(str(path.resolve()))[:16]}"


def _evidence_ref(source_id: str, source_kind: str, locator: str, excerpt: str) -> EvidenceRef:
    return EvidenceRef(
        source_id=source_id,
        source_kind=source_kind,
        locator=locator,
        excerpt=_normalize_excerpt(excerpt, 1200),
        digest=sha256_text(f"{source_id}\n{locator}\n{excerpt}"),
    )


def _bounded_text(value: Any, *, context: str) -> str:
    if isinstance(value, str):
        return _normalize_excerpt(value, 1200)
    if isinstance(value, list):
        parts: list[str] = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                raise SourceParseError(f"Unsupported error content at {context}[{index}]")
        return _normalize_excerpt("\n".join(parts), 1200)
    if value is None:
        return "explicit error status without an error message"
    raise SourceParseError(f"Unsupported error content at {context}: {type(value).__name__}")


def _normalize_excerpt(value: str, maximum: int) -> str:
    normalized = " ".join(value.split())
    return normalized[:maximum]


def _repeated_runs(sequence: list[str]) -> tuple[tuple[str, int], ...]:
    if not sequence:
        return ()
    runs: list[tuple[str, int]] = []
    current = sequence[0]
    count = 1
    for name in sequence[1:]:
        if name == current:
            count += 1
        else:
            if count > 1:
                runs.append((current, count))
            current = name
            count = 1
    if count > 1:
        runs.append((current, count))
    return tuple(runs)


def _named_boolean_gates(value: Any, *, context: str) -> Iterable[tuple[str, bool]]:
    if isinstance(value, dict):
        for name, passed in value.items():
            if not isinstance(name, str) or not isinstance(passed, bool):
                raise SourceParseError(f"{context} dict must map string gate names to booleans")
            yield name, passed
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, list) or len(item) < 2:
                raise SourceParseError(f"{context}[{index}] must be [name, passed, ...]")
            name, passed = item[0], item[1]
            if not isinstance(name, str) or not isinstance(passed, bool):
                raise SourceParseError(f"{context}[{index}] has invalid name/passed fields")
            yield name, passed
        return
    raise SourceParseError(f"{context} must be a mapping or list of gate rows")


def _deduplicate_refs(refs: Iterable[EvidenceRef]) -> list[EvidenceRef]:
    by_locator: dict[str, EvidenceRef] = {}
    for ref in refs:
        by_locator.setdefault(ref.locator, ref)
    return list(by_locator.values())


def source_bundle_digest(bundle: SourceBundle) -> str:
    payload = {
        "transcripts": [source.analysis_record() for source in bundle.transcripts],
        "verdicts": [source.analysis_record() for source in bundle.verdicts],
        "plans": [source.analysis_record() for source in bundle.plans],
        "bench_bundles": [source.analysis_record() for source in bundle.bench_bundles],
        "live_dialogue_outcomes": [
            source.analysis_record() for source in bundle.live_dialogue_outcomes
        ],
        "evidence_since": bundle.evidence_since,
    }
    return sha256_text(canonical_json(payload))
