"""Loop-external, fail-closed proposal validation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import yaml

from .config import EditableSurfaceEntry, ForgeConfig, ForgeConfigError
from .foundation import (
    BackendError,
    ForgeError,
    PromptStore,
    SchemaContractError,
    SchemaStore,
    StructuredBackend,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_text,
    utc_now,
)


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def __call__(self, argv: tuple[str, ...], *, cwd: Path, timeout: int) -> CommandOutcome: ...


@dataclass(frozen=True)
class ValidationResult:
    status: str
    report_path: Path
    checks: tuple[dict[str, str], ...]


def validate_proposal(
    *,
    config: ForgeConfig,
    proposal_dir: Path,
    relevance_backend: StructuredBackend | None,
    report_path: Path | None = None,
    command_runner: CommandRunner | None = None,
) -> ValidationResult:
    proposal_dir = proposal_dir.resolve()
    patch_path = proposal_dir / "patch.diff"
    manifesto_path = proposal_dir / "manifesto.json"
    pattern_path = proposal_dir / "failure_pattern.json"
    try:
        patch = patch_path.read_text(encoding="utf-8")
        manifesto_text = manifesto_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ForgeError(f"Incomplete proposal bundle {proposal_dir}: missing {exc.filename}") from exc
    manifesto = read_json(manifesto_path)
    pattern = read_json(pattern_path)
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    schema_store.validate(manifesto, "proposal_manifesto.schema.json")
    schema_store.validate(pattern, "failure_pattern.schema.json")
    checks: list[dict[str, str]] = []
    entry = config.editable_entry_for(manifesto["target"])

    _check(
        checks,
        "bundle-pattern-consistency",
        manifesto["pattern_id"] == pattern["pattern_id"],
        f"manifesto={manifesto['pattern_id']}, frozen_pattern={pattern['pattern_id']}",
    )
    paths_ok, paths_detail, patch_targets = _validate_patch_paths(patch, manifesto["target"], config)
    _check(checks, "editable-surface", paths_ok, paths_detail)

    target_path: Path | None = None
    current = ""
    preimage_ok = False
    try:
        target_path = config.resolve_target(manifesto["target"], must_exist=True)
        current = target_path.read_text(encoding="utf-8")
        current_digest = sha256_text(current)
        preimage_ok = current_digest == manifesto["target_preimage_sha256"]
        preimage_detail = f"expected={manifesto['target_preimage_sha256']}, current={current_digest}"
    except (ForgeConfigError, FileNotFoundError, UnicodeDecodeError) as exc:
        preimage_detail = str(exc)
    _check(checks, "target-preimage", preimage_ok, preimage_detail)

    clean_apply, clean_detail = _git_apply_check(config.paths.repo_root, patch)
    _check(checks, "git-apply-check", clean_apply, clean_detail)
    append_only = not any(line.startswith("-") and not line.startswith("---") for line in patch.splitlines())
    _check(
        checks,
        "phase1-append-only",
        append_only,
        "no deleted hunk lines" if append_only else "phase 1 blocks every deleted hunk line",
    )
    if patch_targets and manifesto["target"].endswith(".mdc"):
        _check(
            checks,
            "mdc-hard-constraint-preservation",
            append_only,
            "append-only structure preserves every existing .mdc line",
        )

    relevance_ok, relevance_detail = _relevance_check(
        config=config,
        backend=relevance_backend,
        manifesto=manifesto,
        pattern=pattern,
        patch=patch,
    )
    _check(checks, "targeted-relevance-held-in", relevance_ok, relevance_detail)

    runtime_gate_evidence: dict[str, Any] | None = None
    if entry is not None and entry.requires_offline_gate:
        candidate, rollback_resilience, materialize_detail = _materialize_candidate(
            repo_root=config.paths.repo_root,
            target=manifesto["target"],
            current=current,
            patch=patch,
        )
        _check(
            checks,
            "runtime-candidate-materialization-and-rollback",
            candidate is not None and rollback_resilience,
            materialize_detail,
        )
        runtime_ok, runtime_detail, runtime_gate_evidence = _runtime_suite_check(
            config=config,
            entry=entry,
            backend=relevance_backend,
            pattern=pattern,
            target=manifesto["target"],
            baseline=current,
            candidate=candidate,
            rollback_resilience=rollback_resilience,
        )
        _check(checks, "runtime-frozen-suite-evaluation", runtime_ok, runtime_detail)

    runner = command_runner or _run_command
    _run_command_group(
        checks,
        group_name="static",
        commands=config.validation.static_commands,
        config=config,
        runner=runner,
    )
    if entry is not None and entry.validation is not None:
        _run_command_group(
            checks,
            group_name=f"component-held-in:{entry.component}",
            commands=entry.validation.held_in_commands,
            config=config,
            runner=runner,
        )
        _run_command_group(
            checks,
            group_name=f"component-held-out:{entry.component}",
            commands=entry.validation.held_out_commands,
            config=config,
            runner=runner,
        )
    _run_command_group(
        checks,
        group_name="held-in",
        commands=config.validation.held_in_commands,
        config=config,
        runner=runner,
    )
    _run_command_group(
        checks,
        group_name="held-out",
        commands=config.validation.held_out_commands,
        config=config,
        runner=runner,
    )

    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "BLOCK"
    report = {
        "schema_version": "forge-validation-report.v2",
        "proposal_id": manifesto["proposal_id"],
        "component": entry.component if entry is not None else "unresolved",
        "runtime_gate_evidence": runtime_gate_evidence,
        "status": status,
        "patch_sha256": sha256_text(patch),
        "manifesto_sha256": sha256_text(manifesto_text),
        "checks": checks,
        "validated_at": utc_now(),
    }
    schema_store.validate(report, "validation_report.schema.json")
    destination = (report_path or proposal_dir / "validation.json").resolve()
    atomic_write_json(destination, report)
    return ValidationResult(status=status, report_path=destination, checks=tuple(checks))


def _validate_patch_paths(
    patch: str, manifesto_target: str, config: ForgeConfig
) -> tuple[bool, str, tuple[str, ...]]:
    old_paths: list[str] = []
    new_paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("--- "):
            old_paths.append(line[4:].split("\t", 1)[0])
        elif line.startswith("+++ "):
            new_paths.append(line[4:].split("\t", 1)[0])
    if len(old_paths) != 1 or len(new_paths) != 1:
        return False, f"expected one old/new path pair, got old={old_paths}, new={new_paths}", ()
    if old_paths[0] == "/dev/null" or new_paths[0] == "/dev/null":
        return False, "phase 1 does not allow file creation or deletion", ()
    expected_old = f"a/{manifesto_target}"
    expected_new = f"b/{manifesto_target}"
    if old_paths[0] != expected_old or new_paths[0] != expected_new:
        return False, f"patch paths {old_paths[0]!r}/{new_paths[0]!r} do not match {manifesto_target!r}", ()
    try:
        entry = config.editable_entry_for(manifesto_target)
    except ForgeConfigError as exc:
        return False, str(exc), ()
    if entry is None:
        return False, f"target is protected or absent from editable surface: {manifesto_target}", ()
    return True, f"single target {manifesto_target} in component {entry.component}", (manifesto_target,)


def _git_apply_check(repo_root: Path, patch: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ("git", "apply", "--check", "--recount", "-"),
            cwd=repo_root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, f"git executable unavailable: {exc}"
    except subprocess.TimeoutExpired:
        return False, "git apply --check timed out"
    detail = _command_detail(completed.stdout, completed.stderr)
    return completed.returncode == 0, detail or f"returncode={completed.returncode}"


def _relevance_check(
    *,
    config: ForgeConfig,
    backend: StructuredBackend | None,
    manifesto: dict[str, Any],
    pattern: dict[str, Any],
    patch: str,
) -> tuple[bool, str]:
    if backend is None:
        return False, "relevance judge is required and unavailable"
    prompt_store = PromptStore(config.paths.forge_root / "prompts")
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    try:
        response = backend.complete_json(
            system=prompt_store.render("relevance_judge.system.md"),
            user=prompt_store.render(
                "relevance_judge.user.md",
                manifesto=json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True),
                patch=patch,
                failure_pattern=json.dumps(pattern, ensure_ascii=False, indent=2, sort_keys=True),
            ),
            schema=schema_store.load("relevance_judgment.schema.json"),
        )
        schema_store.validate(response, "relevance_judgment.schema.json")
    except (BackendError, SchemaContractError) as exc:
        return False, f"relevance judge failed closed: {exc}"
    passed = response["relevant"] and response["evidence_alignment"] and response["preservation_assessment"]
    return bool(passed), response["reason"]


def _materialize_candidate(
    *,
    repo_root: Path,
    target: str,
    current: str,
    patch: str,
) -> tuple[str | None, bool, str]:
    if not current:
        return None, False, "target preimage was unavailable"
    with tempfile.TemporaryDirectory(prefix="forge-candidate-") as temporary:
        sandbox = Path(temporary)
        sandbox_target = sandbox / target
        sandbox_target.parent.mkdir(parents=True, exist_ok=True)
        sandbox_target.write_text(current, encoding="utf-8")
        applied, apply_detail = _sandbox_git_apply(sandbox, patch, reverse=False, check_only=False)
        if not applied:
            return None, False, f"candidate apply failed in sandbox: {apply_detail}"
        candidate = sandbox_target.read_text(encoding="utf-8")
        reverse_check, check_detail = _sandbox_git_apply(
            sandbox, patch, reverse=True, check_only=True
        )
        if not reverse_check:
            return candidate, False, f"reverse patch check failed: {check_detail}"
        reversed_ok, reverse_detail = _sandbox_git_apply(
            sandbox, patch, reverse=True, check_only=False
        )
        if not reversed_ok:
            return candidate, False, f"reverse patch failed: {reverse_detail}"
        restored = sandbox_target.read_text(encoding="utf-8")
        resilient = restored == current
        detail = (
            f"sandbox candidate and byte-identical reverse drill passed under {repo_root} policy"
            if resilient
            else "reverse patch did not restore the byte-identical target preimage"
        )
        return candidate, resilient, detail


def _sandbox_git_apply(
    root: Path,
    patch: str,
    *,
    reverse: bool,
    check_only: bool,
) -> tuple[bool, str]:
    argv = ["git", "apply"]
    if reverse:
        argv.append("--reverse")
    if check_only:
        argv.append("--check")
    argv.extend(("--recount", "-"))
    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, f"git executable unavailable: {exc}"
    except subprocess.TimeoutExpired:
        return False, "git apply timed out in candidate sandbox"
    detail = _command_detail(completed.stdout, completed.stderr)
    return completed.returncode == 0, detail or f"returncode={completed.returncode}"


def _runtime_suite_check(
    *,
    config: ForgeConfig,
    entry: EditableSurfaceEntry,
    backend: StructuredBackend | None,
    pattern: dict[str, Any],
    target: str,
    baseline: str,
    candidate: str | None,
    rollback_resilience: bool,
) -> tuple[bool, str, dict[str, Any] | None]:
    if entry.validation is None:
        return False, "offline-gated component lacks validation policy", None
    if backend is None:
        return False, "runtime suite judge is required and unavailable", None
    if candidate is None:
        return False, "candidate could not be materialized", None
    try:
        suite_relative, suite_path = _resolve_frozen_suite(config, entry, target)
        suite_text = suite_path.read_text(encoding="utf-8")
        test_ids = _frozen_suite_test_ids(suite_text, suite_path)
    except (ForgeConfigError, FileNotFoundError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return False, f"frozen suite is unavailable or invalid: {exc}", None
    prompt_store = PromptStore(config.paths.forge_root / "prompts")
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    try:
        response = backend.complete_json(
            system=prompt_store.render("runtime_suite_judge.system.md"),
            user=prompt_store.render(
                "runtime_suite_judge.user.md",
                failure_pattern=json.dumps(pattern, ensure_ascii=False, indent=2, sort_keys=True),
                frozen_suite=suite_text,
                baseline_asset=baseline,
                candidate_asset=candidate,
            ),
            schema=schema_store.load("runtime_suite_judgment.schema.json"),
        )
        schema_store.validate(response, "runtime_suite_judgment.schema.json")
    except (BackendError, SchemaContractError) as exc:
        return False, f"runtime suite judge failed closed: {exc}", None
    known = set(test_ids)
    baseline_passed = tuple(response["baseline_passed_test_ids"])
    candidate_passed = tuple(response["candidate_passed_test_ids"])
    unknown = (set(baseline_passed) | set(candidate_passed)) - known
    if unknown:
        return False, f"runtime suite judge returned unknown test IDs: {sorted(unknown)}", None
    baseline_rate = len(baseline_passed) / len(test_ids)
    candidate_rate = len(candidate_passed) / len(test_ids)
    evidence = {
        "component": entry.component,
        "target": target,
        "frozen_suite": suite_relative,
        "frozen_suite_sha256": sha256_text(suite_text),
        "evaluated_test_ids": list(test_ids),
        "baseline_passed_test_ids": list(baseline_passed),
        "candidate_passed_test_ids": list(candidate_passed),
        "baseline_pass_rate": baseline_rate,
        "candidate_pass_rate": candidate_rate,
        "validation_delta": candidate_rate - baseline_rate,
        "capacity_cost": 0.1,
        "contract_integrity": config.is_read_only(suite_relative),
        "rollback_resilience": rollback_resilience,
        "judge": {"backend": backend.backend_name, "model": backend.model_name},
    }
    return True, response["reason"], evidence


def _resolve_frozen_suite(
    config: ForgeConfig,
    entry: EditableSurfaceEntry,
    target: str,
) -> tuple[str, Path]:
    assert entry.validation is not None
    target_path = PurePosixPath(config.normalize_relative_path(target))
    suite_relative = target_path.with_name("test_suite.yaml").as_posix()
    if not PurePosixPath(suite_relative).match(entry.validation.frozen_suite):
        raise ForgeConfigError(
            f"Sibling suite {suite_relative} does not match frozen suite policy "
            f"{entry.validation.frozen_suite}"
        )
    if not config.is_read_only(suite_relative):
        raise ForgeConfigError(f"Frozen suite is not protected by read-only policy: {suite_relative}")
    return suite_relative, config.resolve_target(suite_relative, must_exist=True)


def _frozen_suite_test_ids(text: str, path: Path) -> tuple[str, ...]:
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ForgeConfigError(f"Frozen suite must be a mapping: {path}")
    identifiers: list[str] = []
    routing = raw.get("routing_tests")
    if not isinstance(routing, list):
        raise ForgeConfigError(f"Frozen suite routing_tests must be a list: {path}")
    for index, case in enumerate(routing):
        if not isinstance(case, dict) or not isinstance(case.get("test_id"), str):
            raise ForgeConfigError(f"routing_tests[{index}] lacks test_id: {path}")
        identifiers.append(case["test_id"])
    llm = raw.get("llm_evaluation")
    coherence = llm.get("semantic_coherence") if isinstance(llm, dict) else None
    if not isinstance(coherence, list):
        raise ForgeConfigError(f"Frozen suite semantic_coherence must be a list: {path}")
    for index, case in enumerate(coherence):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ForgeConfigError(f"semantic_coherence[{index}] lacks case_id: {path}")
        identifiers.append(case["case_id"])
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ForgeConfigError(f"Frozen suite test IDs must be non-empty and unique: {path}")
    return tuple(identifiers)


def _run_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
) -> CommandOutcome:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandOutcome(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _run_command_group(
    checks: list[dict[str, str]],
    *,
    group_name: str,
    commands: tuple[tuple[str, ...], ...],
    config: ForgeConfig,
    runner: CommandRunner,
) -> None:
    for index, argv in enumerate(commands):
        name = f"{group_name}:{index}:{' '.join(argv)}"
        try:
            outcome = runner(
                argv,
                cwd=config.paths.repo_root,
                timeout=config.validation.command_timeout_seconds,
            )
            passed = outcome.returncode == 0
            detail = _command_detail(outcome.stdout, outcome.stderr) or f"returncode={outcome.returncode}"
        except FileNotFoundError as exc:
            passed = False
            detail = f"command unavailable: {exc}"
        except subprocess.TimeoutExpired:
            passed = False
            detail = f"command exceeded {config.validation.command_timeout_seconds}s timeout"
        _check(checks, name, passed, detail)


def _check(checks: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "BLOCK", "detail": detail[:4000]})


def _command_detail(stdout: str, stderr: str) -> str:
    rendered = canonical_json({"stdout": stdout[-1800:], "stderr": stderr[-1800:]})
    return rendered if stdout or stderr else ""
