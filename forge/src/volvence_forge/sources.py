"""Read-only parsers for transcripts, promotion evidence and campaign plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
class SourceBundle:
    transcripts: tuple[TranscriptSource, ...]
    verdicts: tuple[VerdictSource, ...]
    plans: tuple[PlanSource, ...]


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
        raise SourceParseError(f"Plan is missing YAML frontmatter: {path}")
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


def load_source_bundle(
    paths: ForgePaths,
    *,
    max_transcripts: int | None = None,
    max_verdicts: int | None = None,
    max_plans: int | None = None,
    verdict_root: Path | None = None,
) -> SourceBundle:
    transcript_paths = sorted(paths.transcripts_root.rglob("*.jsonl")) if paths.transcripts_root.exists() else []
    verdict_base = (verdict_root or paths.artifacts_root).resolve()
    verdict_paths = sorted(verdict_base.rglob("promotion_verdict.json")) if verdict_base.exists() else []
    plan_paths = sorted(paths.plans_root.glob("*.plan.md")) if paths.plans_root.exists() else []
    transcript_paths = _limit(transcript_paths, max_transcripts)
    verdict_paths = _limit(verdict_paths, max_verdicts)
    plan_paths = _limit(plan_paths, max_plans)
    return SourceBundle(
        transcripts=tuple(parse_transcript(path) for path in transcript_paths),
        verdicts=tuple(parse_verdict(path) for path in verdict_paths),
        plans=tuple(parse_plan(path) for path in plan_paths),
    )


def _limit(paths: list[Path], maximum: int | None) -> list[Path]:
    if maximum is None:
        return paths
    if maximum < 0:
        raise SourceParseError("Source limits must be non-negative")
    return paths[:maximum]


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
    }
    return sha256_text(canonical_json(payload))
