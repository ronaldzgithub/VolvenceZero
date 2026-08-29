"""Demand-driven, approval-gated Codex research-topic discovery."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import jsonschema

from .config import ForgeConfig
from .foundation import (
    BackendError,
    ForgeError,
    PromptStore,
    ReplayStructuredBackend,
    SchemaStore,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    utc_now,
)
from .research_control import (
    ResearchRequestResult,
    submit_research_request,
    validate_research_request,
)
from .research_opportunity import (
    ResolvedResearchBinding,
    resolve_registered_task_binding,
)
from .research_promotion import validate_research_task

SCHEMA_NAME = "research_discovery.schema.json"
RESPONSE_SCHEMA_NAME = "research_discovery_response.schema.json"
PROMPT_REVISION = "demand-topic-discovery.v1"

DEMAND_VERSION = "forge-volvence-research-demand.v1"
RUN_VERSION = "forge-research-discovery-run.v1"
PROPOSAL_VERSION = "forge-research-topic-proposal.v1"
BINDING_VERSION = "forge-research-demand-binding.v1"

DISCOVERY_LOOP_OWNER = "forge:demand-discovery-loop.v1"
_DISCOVERY_ROOT = "research_discovery"


class ResearchDiscoveryError(ForgeError):
    """Raised when a demand discovery or exact binding is unsafe."""


class ResearchDiscoveryBackend(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        cwd: Path,
    ) -> dict[str, Any]: ...


@dataclass
class ReplayResearchDiscoveryBackend:
    """Replay adapter for deterministic discovery tests and offline drills."""

    delegate: ReplayStructuredBackend

    @classmethod
    def from_path(cls, path: Path) -> ReplayResearchDiscoveryBackend:
        return cls(ReplayStructuredBackend.from_path(path))

    @property
    def backend_name(self) -> str:
        return "replay"

    @property
    def model_name(self) -> str:
        return self.delegate.model_name

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        cwd: Path,
    ) -> dict[str, Any]:
        del cwd
        return self.delegate.complete_json(system=system, user=user, schema=schema)


@dataclass(frozen=True)
class CodexNativeResearchDiscoveryBackend:
    """One-turn, read-only Codex SDK backend using the operator's saved login."""

    model_name: str
    codex_bin: Path | None = None
    backend_name: str = "codex_sdk"

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        cwd: Path,
    ) -> dict[str, Any]:
        if not self.model_name.strip():
            raise BackendError("Codex discovery requires an explicit model")
        executable = None
        if self.codex_bin is not None:
            executable = _resolve_executable(self.codex_bin, context="Codex executable")
        try:
            from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
            from openai_codex.errors import CodexError
        except ImportError as exc:
            raise BackendError(
                "Install the openai-codex SDK to use --backend codex_sdk"
            ) from exc

        try:
            sdk_config = CodexConfig(
                codex_bin=str(executable) if executable is not None else None,
                cwd=str(cwd),
                client_name="volvence_forge_discovery",
                client_title="Volvence Forge Research Discovery",
            )
            with Codex(sdk_config) as codex:
                account = codex.account()
                if account.account is None:
                    raise BackendError(
                        "Codex Native is not logged in; authenticate Codex before discovery"
                    )
                thread = codex.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    base_instructions=system,
                    cwd=str(cwd),
                    ephemeral=True,
                    model=self.model_name,
                    sandbox=Sandbox.read_only,
                    service_name="volvence_forge_discovery",
                )
                result = thread.run(
                    user,
                    model=self.model_name,
                    output_schema=schema,
                    sandbox=Sandbox.read_only,
                )
        except BackendError:
            raise
        except (CodexError, OSError, RuntimeError, ValueError) as exc:
            raise BackendError(f"Codex Native discovery failed: {exc}") from exc

        if result.final_response is None:
            raise BackendError("Codex Native discovery returned no final response")
        try:
            payload = json.loads(result.final_response)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"Codex Native discovery returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise BackendError("Codex Native discovery response must be a JSON object")
        try:
            jsonschema.Draft202012Validator(schema).validate(payload)
        except jsonschema.ValidationError as exc:
            raise BackendError(
                f"Codex Native discovery response violates schema: {exc.message}"
            ) from exc
        return payload


@dataclass(frozen=True)
class CorpusSnapshot:
    tree_sha256: str
    file_count: int
    total_bytes: int
    files: tuple[dict[str, Any], ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "tree_sha256": self.tree_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [dict(item) for item in self.files],
        }


@dataclass(frozen=True)
class ResearchDiscoveryResult:
    run_id: str
    run_path: Path
    proposal_paths: tuple[Path, ...]
    reused: bool


@dataclass(frozen=True)
class ResearchDemandSealResult:
    demand_id: str
    demand_path: Path
    reused: bool


@dataclass(frozen=True)
class ResearchDemandBindingResult:
    binding_id: str
    binding_path: Path
    decision: str


@dataclass(frozen=True)
class ResearchBoundRequestResult:
    request_id: str
    request_path: Path
    reused: bool


def seal_research_demand(
    *,
    config: ForgeConfig,
    draft_path: Path,
    output_path: Path | None = None,
) -> ResearchDemandSealResult:
    """Seal a human-authored Demand draft into the repository Demand inbox."""

    resolved_draft = _resolve_repo_file(
        config,
        draft_path,
        context="ResearchDemand draft",
    )
    draft = read_json(resolved_draft)
    if draft.get("schema_version") != DEMAND_VERSION:
        raise ResearchDiscoveryError(
            f"ResearchDemand draft must declare {DEMAND_VERSION}"
        )
    demand = dict(draft)
    supplied_identity = demand.pop("demand_id", None)
    identity = _artifact_id("research-demand", demand, "demand_id")
    if supplied_identity is not None and supplied_identity != identity:
        raise ResearchDiscoveryError(
            "ResearchDemand draft demand_id does not match its canonical payload"
        )
    demand["demand_id"] = identity
    _validate_payload(config, demand, expected_version=DEMAND_VERSION)
    for source_root in demand["discovery"]["source_roots"]:
        _resolve_repo_source(config, source_root, context="demand source root")
    for evidence in demand["evidence"]:
        _verify_content_ref(config, evidence, context="demand evidence")

    digest = identity.partition(":")[2]
    destination = output_path or (
        config.paths.repo_root / "research" / "demands" / f"{digest}.json"
    )
    target = destination if destination.is_absolute() else config.paths.repo_root / destination
    target = target.expanduser().resolve(strict=False)
    demand_root = (
        config.paths.repo_root / "research" / "demands"
    ).resolve(strict=False)
    if not target.is_relative_to(demand_root) or target.suffix != ".json":
        raise ResearchDiscoveryError(
            "sealed ResearchDemand must be a JSON file below research/demands"
        )
    if target.is_symlink():
        raise ResearchDiscoveryError(f"ResearchDemand output may not be a symlink: {target}")
    if target.exists():
        existing = validate_research_demand(config=config, demand_path=target)
        if _identity_body(existing, "demand_id") != _identity_body(
            demand, "demand_id"
        ):
            raise ResearchDiscoveryError(
                f"refusing to overwrite a different ResearchDemand: {target}"
            )
        return ResearchDemandSealResult(
            demand_id=str(existing["demand_id"]),
            demand_path=target,
            reused=True,
        )
    _write_create_only_json(target, demand)
    validate_research_demand(config=config, demand_path=target)
    return ResearchDemandSealResult(
        demand_id=identity,
        demand_path=target,
        reused=False,
    )


def validate_research_demand(
    *,
    config: ForgeConfig,
    demand_path: Path,
) -> dict[str, Any]:
    """Validate a content-addressed Volvence-owned research demand."""

    resolved = _resolve_repo_file(config, demand_path, context="ResearchDemand")
    demand = _load_versioned_artifact(
        config=config,
        path=resolved,
        expected_version=DEMAND_VERSION,
        identity_field="demand_id",
        identity_prefix="research-demand",
    )
    for source_root in demand["discovery"]["source_roots"]:
        _resolve_repo_source(config, source_root, context="demand source root")
    for evidence in demand["evidence"]:
        _verify_content_ref(config, evidence, context="demand evidence")
    return demand


def build_research_corpus(
    *,
    config: ForgeConfig,
    demand: Mapping[str, Any],
) -> CorpusSnapshot:
    """Freeze the exact regular-file corpus permitted by one Demand."""

    discovery = demand["discovery"]
    max_files = int(discovery["max_source_files"])
    max_bytes = int(discovery["max_source_bytes"])
    files: dict[str, Path] = {}
    for source_root in discovery["source_roots"]:
        root = _resolve_repo_source(config, source_root, context="demand source root")
        candidates = (root,) if root.is_file() else tuple(sorted(root.rglob("*")))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ResearchDiscoveryError(
                    f"demand corpus may not contain symlinks: {candidate}"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ResearchDiscoveryError(
                    f"demand corpus contains a non-regular path: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(config.paths.repo_root):
                raise ResearchDiscoveryError(
                    f"demand corpus escapes repository root: {candidate}"
                )
            locator = resolved.relative_to(config.paths.repo_root).as_posix()
            files[locator] = resolved
            if len(files) > max_files:
                raise ResearchDiscoveryError(
                    f"demand corpus exceeds max_source_files={max_files}"
                )

    if not files:
        raise ResearchDiscoveryError("demand corpus contains no regular files")

    frozen: list[dict[str, Any]] = []
    total_bytes = 0
    for locator, path in sorted(files.items()):
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ResearchDiscoveryError(
                f"cannot read demand corpus file {path}: {exc}"
            ) from exc
        total_bytes += len(content)
        if total_bytes > max_bytes:
            raise ResearchDiscoveryError(
                f"demand corpus exceeds max_source_bytes={max_bytes}"
            )
        frozen.append(
            {
                "locator": locator,
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
        )
    tree_sha256 = sha256_text(canonical_json(frozen))
    return CorpusSnapshot(
        tree_sha256=tree_sha256,
        file_count=len(frozen),
        total_bytes=total_bytes,
        files=tuple(frozen),
    )


def discover_research_topics(
    *,
    config: ForgeConfig,
    demand_path: Path,
    backend: ResearchDiscoveryBackend,
) -> ResearchDiscoveryResult:
    """Serialize model consumption, then run or reuse one exact discovery."""

    lock_path = (
        config.paths.artifacts_root / _DISCOVERY_ROOT / ".discover.lock"
    )
    with _exclusive_lock(lock_path, context="research discovery"):
        return _discover_research_topics_unlocked(
            config=config,
            demand_path=demand_path,
            backend=backend,
        )


def _discover_research_topics_unlocked(
    *,
    config: ForgeConfig,
    demand_path: Path,
    backend: ResearchDiscoveryBackend,
) -> ResearchDiscoveryResult:
    """Run or reuse one exact, bounded Codex discovery for a Demand."""

    resolved_demand = _resolve_repo_file(config, demand_path, context="ResearchDemand")
    demand = validate_research_demand(config=config, demand_path=resolved_demand)
    if demand["status"] != "OPEN":
        raise ResearchDiscoveryError(
            f"only OPEN demands may run discovery, got {demand['status']!r}"
        )
    if backend.backend_name not in {"codex_sdk", "replay"}:
        raise ResearchDiscoveryError(
            f"unsupported research discovery backend: {backend.backend_name!r}"
        )
    corpus = build_research_corpus(config=config, demand=demand)
    demand_ref = _content_ref(config, resolved_demand, context="ResearchDemand")
    run_key_payload = {
        "demand": {"artifact_id": demand["demand_id"], "artifact": demand_ref},
        "corpus_sha256": corpus.tree_sha256,
        "backend": backend.backend_name,
        "model": backend.model_name,
        "prompt_revision": PROMPT_REVISION,
        "max_topics": demand["discovery"]["max_topics"],
    }
    run_key = sha256_text(canonical_json(run_key_payload))
    run_id = f"research-discovery-run:{run_key}"
    run_root = _run_root(config, demand=demand, run_key=run_key)
    run_path = run_root / "run.json"
    if run_path.exists():
        run = validate_research_discovery_run(config=config, run_path=run_path)
        _verify_reused_run(
            run=run,
            run_id=run_id,
            demand=demand,
            demand_ref=demand_ref,
            corpus=corpus,
            backend=backend,
        )
        proposal_paths: list[Path] = []
        for item in run["proposals"]:
            _, proposal_path = _verify_identified_ref(
                config,
                item,
                context="reused TopicProposal",
                expected_version=PROPOSAL_VERSION,
                identity_field="proposal_id",
                identity_prefix="research-topic-proposal",
            )
            validate_research_topic_proposal(
                config=config,
                proposal_path=proposal_path,
            )
            proposal_paths.append(proposal_path)
        proposals = tuple(proposal_paths)
        return ResearchDiscoveryResult(
            run_id=run_id,
            run_path=run_path,
            proposal_paths=proposals,
            reused=True,
        )

    response_schema = SchemaStore(config.paths.forge_root / "schemas").load(
        RESPONSE_SCHEMA_NAME
    )
    prompts = PromptStore(config.paths.forge_root / "prompts")
    system_prompt = prompts.render("research_discovery.system.md")
    user_prompt = prompts.render(
        "research_discovery.user.md",
        max_topics=str(demand["discovery"]["max_topics"]),
        demand_json=json.dumps(demand, ensure_ascii=False, indent=2, sort_keys=True),
        corpus_json=json.dumps(
            corpus.to_jsonable(), ensure_ascii=False, indent=2, sort_keys=True
        ),
    )
    with _materialized_enclosure(config=config, corpus=corpus) as enclosure:
        response = backend.complete_json(
            system=system_prompt,
            user=user_prompt,
            schema=response_schema,
            cwd=enclosure,
        )
    SchemaStore(config.paths.forge_root / "schemas").validate(
        response, RESPONSE_SCHEMA_NAME
    )
    topics = response["topics"]
    max_topics = int(demand["discovery"]["max_topics"])
    if len(topics) > max_topics:
        raise ResearchDiscoveryError(
            f"discovery returned {len(topics)} topics above max_topics={max_topics}"
        )
    _validate_topic_response(topics=topics, corpus=corpus)

    created_at = utc_now()
    proposal_refs: list[dict[str, Any]] = []
    proposal_paths: list[Path] = []
    for topic in topics:
        source_refs = topic["source_refs"]
        topic_body = {key: value for key, value in topic.items() if key != "source_refs"}
        proposal: dict[str, Any] = {
            "schema_version": PROPOSAL_VERSION,
            "demand": {"artifact_id": demand["demand_id"], "artifact": demand_ref},
            "discovery_run_id": run_id,
            "corpus_sha256": corpus.tree_sha256,
            "capability_axes": list(demand["capability_axes"]),
            "topic": topic_body,
            "source_refs": source_refs,
            "binding_status": "UNBOUND",
            "authority": _discovery_authority(),
            "created_at": created_at,
        }
        proposal["proposal_id"] = _artifact_id(
            "research-topic-proposal", proposal, "proposal_id"
        )
        _validate_payload(config, proposal, expected_version=PROPOSAL_VERSION)
        digest = str(proposal["proposal_id"]).partition(":")[2]
        proposal_path = run_root / "topics" / f"{digest}.json"
        _write_immutable_artifact(
            config=config,
            destination=proposal_path,
            payload=proposal,
            expected_version=PROPOSAL_VERSION,
            identity_field="proposal_id",
        )
        proposal_paths.append(proposal_path)
        proposal_refs.append(
            {
                "artifact_id": proposal["proposal_id"],
                "artifact": _content_ref(
                    config, proposal_path, context="TopicProposal"
                ),
            }
        )

    run: dict[str, Any] = {
        "schema_version": RUN_VERSION,
        "run_id": run_id,
        "run_key": run_key,
        "demand": {"artifact_id": demand["demand_id"], "artifact": demand_ref},
        "corpus": corpus.to_jsonable(),
        "execution": {
            "backend": backend.backend_name,
            "model": backend.model_name,
            "prompt_revision": PROMPT_REVISION,
            "sandbox": "read_only",
            "approval_mode": "deny_all",
            "turn_limit": 1,
        },
        "proposals": proposal_refs,
        "authority": _discovery_authority(),
        "created_at": created_at,
    }
    _validate_payload(config, run, expected_version=RUN_VERSION)
    _write_immutable_artifact(
        config=config,
        destination=run_path,
        payload=run,
        expected_version=RUN_VERSION,
        identity_field="run_id",
    )
    return ResearchDiscoveryResult(
        run_id=run_id,
        run_path=run_path,
        proposal_paths=tuple(proposal_paths),
        reused=False,
    )


def validate_research_discovery_run(
    *,
    config: ForgeConfig,
    run_path: Path,
) -> dict[str, Any]:
    return _load_versioned_artifact(
        config=config,
        path=_resolve_discovery_file(config, run_path, context="DiscoveryRun"),
        expected_version=RUN_VERSION,
        identity_field="run_id",
        identity_prefix="research-discovery-run",
        identity_is_run_key=True,
    )


def validate_research_topic_proposal(
    *,
    config: ForgeConfig,
    proposal_path: Path,
) -> dict[str, Any]:
    proposal = _load_versioned_artifact(
        config=config,
        path=_resolve_discovery_file(config, proposal_path, context="TopicProposal"),
        expected_version=PROPOSAL_VERSION,
        identity_field="proposal_id",
        identity_prefix="research-topic-proposal",
    )
    demand, demand_path = _verify_identified_ref(
        config,
        proposal["demand"],
        context="TopicProposal Demand",
        expected_version=DEMAND_VERSION,
        identity_field="demand_id",
        identity_prefix="research-demand",
        require_repo_file=True,
    )
    if tuple(proposal["capability_axes"]) != tuple(demand["capability_axes"]):
        raise ResearchDiscoveryError(
            "TopicProposal capability axes do not match its exact Demand"
        )
    validate_research_demand(config=config, demand_path=demand_path)
    for source_ref in proposal["source_refs"]:
        _verify_content_ref(
            config,
            source_ref,
            context="TopicProposal frozen source",
        )
    run_path = _resolve_discovery_file(
        config,
        _resolve_discovery_file(config, proposal_path, context="TopicProposal")
        .parent.parent
        / "run.json",
        context="TopicProposal DiscoveryRun",
    )
    run = validate_research_discovery_run(config=config, run_path=run_path)
    if (
        proposal["discovery_run_id"] != run["run_id"]
        or proposal["corpus_sha256"] != run["corpus"]["tree_sha256"]
    ):
        raise ResearchDiscoveryError(
            "TopicProposal does not match its canonical DiscoveryRun"
        )
    proposal_ref = _content_ref(config, _resolve_discovery_file(
        config, proposal_path, context="TopicProposal"
    ), context="TopicProposal")
    expected_ref = {
        "artifact_id": proposal["proposal_id"],
        "artifact": proposal_ref,
    }
    if expected_ref not in run["proposals"]:
        raise ResearchDiscoveryError(
            "TopicProposal is not registered by its canonical DiscoveryRun"
        )
    return proposal


def review_research_topic(
    *,
    config: ForgeConfig,
    demand_path: Path,
    proposal_path: Path,
    mapping_id: str,
    reviewed_by: str,
    reason: str,
    decision: str,
    registry_path: Path | None = None,
) -> ResearchDemandBindingResult:
    """Create one named-human exact TopicProposal-to-Task binding decision."""

    normalized_decision = decision.upper()
    if normalized_decision not in {"APPROVE", "REJECT"}:
        raise ResearchDiscoveryError("binding decision must be APPROVE or REJECT")
    if not reviewed_by.strip() or not reason.strip():
        raise ResearchDiscoveryError("binding review requires a named human and reason")
    resolved_demand = _resolve_repo_file(config, demand_path, context="ResearchDemand")
    resolved_proposal = _resolve_discovery_file(
        config, proposal_path, context="TopicProposal"
    )
    demand = validate_research_demand(config=config, demand_path=resolved_demand)
    proposal = validate_research_topic_proposal(
        config=config, proposal_path=resolved_proposal
    )
    demand_ref = _content_ref(config, resolved_demand, context="ResearchDemand")
    if proposal["demand"] != {
        "artifact_id": demand["demand_id"],
        "artifact": demand_ref,
    }:
        raise ResearchDiscoveryError(
            "TopicProposal is not bound to the submitted exact Demand"
        )
    requested_mapping = demand["routing"]["requested_mapping_id"]
    if requested_mapping is not None and requested_mapping != mapping_id:
        raise ResearchDiscoveryError(
            "binding mapping_id does not match the Demand requested_mapping_id"
        )
    resolved_binding, resolved_registry = resolve_registered_task_binding(
        config=config,
        mapping_id=mapping_id,
        identity_key=str(proposal["proposal_id"]),
        registry_path=registry_path,
    )
    _validate_demand_task_alignment(
        config=config,
        demand=demand,
        binding=resolved_binding,
    )
    binding: dict[str, Any] = {
        "schema_version": BINDING_VERSION,
        "demand": {"artifact_id": demand["demand_id"], "artifact": demand_ref},
        "proposal": {
            "artifact_id": proposal["proposal_id"],
            "artifact": _content_ref(
                config, resolved_proposal, context="TopicProposal"
            ),
        },
        "registry": _content_ref(
            config, resolved_registry, context="research task registry"
        ),
        "mapping": _mapping_payload(resolved_binding),
        "decision": normalized_decision,
        "review": {"reviewed_by": reviewed_by.strip(), "reason": reason.strip()},
        "authority": {
            "topic_submission_to_a0_authorized": normalized_decision == "APPROVE",
            "human_a0_required": True,
            "research_start_authorized": False,
            "formal_validation_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    binding["binding_id"] = _artifact_id(
        "research-demand-binding", binding, "binding_id"
    )
    _validate_payload(config, binding, expected_version=BINDING_VERSION)
    proposal_digest = str(proposal["proposal_id"]).partition(":")[2]
    binding_root = (
        resolved_proposal.parent.parent / "bindings" / proposal_digest
    )
    existing = sorted(binding_root.glob("*.json")) if binding_root.exists() else []
    if existing:
        if len(existing) != 1:
            raise ResearchDiscoveryError(
                "TopicProposal has multiple binding decisions; refusing ambiguous review"
            )
        prior = validate_research_demand_binding(
            config=config, binding_path=existing[0]
        )
        if _identity_body(prior, "binding_id") != _identity_body(
            binding, "binding_id"
        ):
            raise ResearchDiscoveryError(
                "TopicProposal already has a different immutable binding decision"
            )
        return ResearchDemandBindingResult(
            binding_id=str(prior["binding_id"]),
            binding_path=existing[0],
            decision=str(prior["decision"]),
        )
    digest = str(binding["binding_id"]).partition(":")[2]
    binding_path = binding_root / f"{digest}.json"
    _write_immutable_artifact(
        config=config,
        destination=binding_path,
        payload=binding,
        expected_version=BINDING_VERSION,
        identity_field="binding_id",
    )
    return ResearchDemandBindingResult(
        binding_id=str(binding["binding_id"]),
        binding_path=binding_path,
        decision=normalized_decision,
    )


def validate_research_demand_binding(
    *,
    config: ForgeConfig,
    binding_path: Path,
) -> dict[str, Any]:
    binding = _load_versioned_artifact(
        config=config,
        path=_resolve_discovery_file(
            config, binding_path, context="ResearchDemand Binding"
        ),
        expected_version=BINDING_VERSION,
        identity_field="binding_id",
        identity_prefix="research-demand-binding",
    )
    demand, demand_path = _verify_identified_ref(
        config,
        binding["demand"],
        context="Binding Demand",
        expected_version=DEMAND_VERSION,
        identity_field="demand_id",
        identity_prefix="research-demand",
        require_repo_file=True,
    )
    proposal, proposal_path = _verify_identified_ref(
        config,
        binding["proposal"],
        context="Binding TopicProposal",
        expected_version=PROPOSAL_VERSION,
        identity_field="proposal_id",
        identity_prefix="research-topic-proposal",
    )
    if proposal["demand"] != binding["demand"]:
        raise ResearchDiscoveryError(
            "Binding Proposal does not preserve the exact Demand"
        )
    validate_research_demand(config=config, demand_path=demand_path)
    validate_research_topic_proposal(
        config=config,
        proposal_path=proposal_path,
    )
    registry_path = _verify_content_ref(
        config, binding["registry"], context="Binding task registry"
    )
    resolved, actual_registry = resolve_registered_task_binding(
        config=config,
        mapping_id=str(binding["mapping"]["mapping_id"]),
        identity_key=str(proposal["proposal_id"]),
        registry_path=registry_path,
    )
    if actual_registry != registry_path or binding["mapping"] != _mapping_payload(
        resolved
    ):
        raise ResearchDiscoveryError(
            "Binding no longer matches its exact registered task mapping"
        )
    _validate_demand_task_alignment(
        config=config,
        demand=demand,
        binding=resolved,
    )
    return binding


def submit_bound_topic_for_a0(
    *,
    config: ForgeConfig,
    binding_path: Path,
) -> ResearchBoundRequestResult:
    """Submit one approved binding to the existing A0 control plane."""

    resolved_binding_path = _resolve_discovery_file(
        config, binding_path, context="ResearchDemand Binding"
    )
    binding_artifact = validate_research_demand_binding(
        config=config, binding_path=resolved_binding_path
    )
    if binding_artifact["decision"] != "APPROVE":
        raise ResearchDiscoveryError(
            "only an APPROVE DemandBinding may be submitted to A0"
        )
    demand_path = _verify_content_ref(
        config, binding_artifact["demand"]["artifact"], context="Binding Demand"
    )
    proposal_path = _verify_content_ref(
        config,
        binding_artifact["proposal"]["artifact"],
        context="Binding TopicProposal",
    )
    proposal = validate_research_topic_proposal(
        config=config, proposal_path=proposal_path
    )
    registry_path = _verify_content_ref(
        config,
        binding_artifact["registry"],
        context="Binding task registry",
    )
    resolved, _ = resolve_registered_task_binding(
        config=config,
        mapping_id=str(binding_artifact["mapping"]["mapping_id"]),
        identity_key=str(proposal["proposal_id"]),
        registry_path=registry_path,
    )
    evidence_paths = (demand_path, proposal_path, resolved_binding_path)
    reason = _binding_submission_reason(binding_artifact)
    existing = _find_bound_request(
        config=config,
        binding=binding_artifact,
        resolved=resolved,
        evidence_paths=evidence_paths,
        reason=reason,
    )
    if existing is not None:
        return ResearchBoundRequestResult(
            request_id=existing.request_id,
            request_path=existing.request_path,
            reused=True,
        )
    created = submit_research_request(
        config=config,
        task_manifest_path=resolved.task_manifest,
        task_project_path=resolved.task_project,
        praxist_executable=resolved.praxist_executable,
        run_dir=resolved.run_dir,
        requested_by=DISCOVERY_LOOP_OWNER,
        reason=reason,
        trigger_kind="typed_signal",
        evidence_paths=evidence_paths,
        config_file=resolved.config_file,
        agent_system=resolved.agent_system,
        runtime=resolved.runtime,
        codex_native=resolved.codex_native,
        model_provider=resolved.model_provider,
        model=resolved.model,
        strategy=resolved.strategy,
        cohort=resolved.cohort,
        generations=resolved.generations,
        startup_timeout_seconds=resolved.startup_timeout_seconds,
    )
    return ResearchBoundRequestResult(
        request_id=created.request_id,
        request_path=created.request_path,
        reused=False,
    )


def _find_bound_request(
    *,
    config: ForgeConfig,
    binding: Mapping[str, Any],
    resolved: ResolvedResearchBinding,
    evidence_paths: Sequence[Path],
    reason: str,
) -> ResearchRequestResult | None:
    expected_evidence = [
        _content_ref(config, path, context="bound Request evidence")
        for path in evidence_paths
    ]
    expected_trigger = {
        "kind": "typed_signal",
        "submitted_by": DISCOVERY_LOOP_OWNER,
        "rationale": reason,
        "evidence": expected_evidence,
    }
    matches: list[ResearchRequestResult] = []
    control_root = config.paths.artifacts_root / "research_control"
    for request_path in sorted(control_root.glob("*/*/request.json")):
        request = validate_research_request(
            config=config,
            request_path=request_path,
            verify_bindings=False,
        )
        if (
            request["trigger"] == expected_trigger
            and request["task_id"] == resolved.task_id
            and request["launch"]["run_dir"] == str(resolved.run_dir)
            and request["launch"]["run_id"] == resolved.run_dir.name
        ):
            matches.append(
                ResearchRequestResult(
                    request_id=str(request["request_id"]),
                    request_path=request_path.resolve(),
                )
            )
    if len(matches) > 1:
        raise ResearchDiscoveryError(
            f"multiple ResearchRequests match DemandBinding {binding['binding_id']}"
        )
    return matches[0] if matches else None


def _validate_demand_task_alignment(
    *,
    config: ForgeConfig,
    demand: Mapping[str, Any],
    binding: ResolvedResearchBinding,
) -> None:
    task = validate_research_task(config=config, task_path=binding.task_manifest)
    if task["claim_id"] != demand["claim_id"]:
        raise ResearchDiscoveryError(
            "Demand claim_id does not match the registered ResearchTask"
        )
    if task["owner"] != demand["owner"] or binding.owner != demand["owner"]:
        raise ResearchDiscoveryError(
            "Demand owner does not match the registered ResearchTask"
        )
    demand_axes = set(demand["capability_axes"])
    task_axes = set(task["capability_axes"])
    if not demand_axes.issubset(task_axes):
        raise ResearchDiscoveryError(
            "Demand capability axes are not covered by the registered ResearchTask"
        )


def _verify_reused_run(
    *,
    run: Mapping[str, Any],
    run_id: str,
    demand: Mapping[str, Any],
    demand_ref: Mapping[str, Any],
    corpus: CorpusSnapshot,
    backend: ResearchDiscoveryBackend,
) -> None:
    expected = {
        "run_id": run_id,
        "demand": {"artifact_id": demand["demand_id"], "artifact": demand_ref},
        "corpus": corpus.to_jsonable(),
        "execution": {
            "backend": backend.backend_name,
            "model": backend.model_name,
            "prompt_revision": PROMPT_REVISION,
            "sandbox": "read_only",
            "approval_mode": "deny_all",
            "turn_limit": 1,
        },
    }
    for key, value in expected.items():
        if run[key] != value:
            raise ResearchDiscoveryError(
                f"reused DiscoveryRun exact input mismatch at {key}"
            )


def _validate_topic_response(
    *,
    topics: Sequence[Mapping[str, Any]],
    corpus: CorpusSnapshot,
) -> None:
    allowed = {
        (str(item["locator"]), str(item["sha256"])) for item in corpus.files
    }
    identities: set[str] = set()
    for index, topic in enumerate(topics):
        identity = sha256_text(canonical_json(topic))
        if identity in identities:
            raise ResearchDiscoveryError(
                f"discovery returned a duplicate topic at index {index}"
            )
        identities.add(identity)
        for field in (
            "success_signals",
            "falsification_signals",
            "source_refs",
            "caveats",
        ):
            values = topic[field]
            canonical_values = [canonical_json(value) for value in values]
            if len(canonical_values) != len(set(canonical_values)):
                raise ResearchDiscoveryError(
                    f"discovery returned duplicate values in {field} at topic index {index}"
                )
        for source in topic["source_refs"]:
            key = (str(source["locator"]), str(source["sha256"]))
            if key not in allowed:
                raise ResearchDiscoveryError(
                    "TopicProposal cites a source outside the frozen corpus or with a stale digest"
                )


@contextlib.contextmanager
def _materialized_enclosure(
    *,
    config: ForgeConfig,
    corpus: CorpusSnapshot,
):
    with tempfile.TemporaryDirectory(prefix="forge-research-discovery-") as raw:
        enclosure = Path(raw)
        for item in corpus.files:
            locator = str(item["locator"])
            source = _resolve_repo_locator(
                config, locator, context="frozen corpus source"
            )
            try:
                content = source.read_bytes()
            except OSError as exc:
                raise ResearchDiscoveryError(
                    f"cannot materialize frozen corpus source {source}: {exc}"
                ) from exc
            if sha256_bytes(content) != item["sha256"] or len(content) != item["bytes"]:
                raise ResearchDiscoveryError(
                    f"frozen corpus source changed before discovery: {locator}"
                )
            target = enclosure / Path(*PurePosixPath(locator).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        yield enclosure


def _run_root(
    config: ForgeConfig,
    *,
    demand: Mapping[str, Any],
    run_key: str,
) -> Path:
    demand_digest = str(demand["demand_id"]).partition(":")[2]
    return (
        config.paths.artifacts_root
        / _DISCOVERY_ROOT
        / demand_digest
        / "runs"
        / run_key
    )


def _mapping_payload(binding: ResolvedResearchBinding) -> dict[str, Any]:
    return {
        "mapping_id": binding.mapping_id,
        "binding_sha256": binding.binding_sha256,
        "task_id": binding.task_id,
        "owner": binding.owner,
        "capability_axes": list(binding.capability_axes),
    }


def _binding_submission_reason(binding: Mapping[str, Any]) -> str:
    return (
        f"Approved DemandBinding {binding['binding_id']} mapped by "
        f"{binding['mapping']['mapping_id']}."
    )


def _discovery_authority() -> dict[str, bool]:
    return {
        "discovery_only": True,
        "human_topic_binding_required": True,
        "human_a0_required": True,
        "research_start_authorized": False,
        "formal_validation_performed": False,
        "production_promotion_authorized": False,
        "runtime_wiring_changed": False,
        "evaluation_is_learning_source": False,
    }


def _verify_identified_ref(
    config: ForgeConfig,
    value: Mapping[str, Any],
    *,
    context: str,
    expected_version: str,
    identity_field: str,
    identity_prefix: str,
    require_repo_file: bool = False,
) -> tuple[dict[str, Any], Path]:
    path = _verify_content_ref(config, value["artifact"], context=context)
    if require_repo_file and not path.is_relative_to(config.paths.repo_root):
        raise ResearchDiscoveryError(f"{context} must be inside the repository")
    artifact = _load_versioned_artifact(
        config=config,
        path=path,
        expected_version=expected_version,
        identity_field=identity_field,
        identity_prefix=identity_prefix,
    )
    if artifact[identity_field] != value["artifact_id"]:
        raise ResearchDiscoveryError(f"{context} artifact id mismatch")
    return artifact, path


def _load_versioned_artifact(
    *,
    config: ForgeConfig,
    path: Path,
    expected_version: str,
    identity_field: str,
    identity_prefix: str,
    identity_is_run_key: bool = False,
) -> dict[str, Any]:
    artifact = read_json(path)
    _validate_payload(config, artifact, expected_version=expected_version)
    if identity_is_run_key:
        expected = f"{identity_prefix}:{artifact['run_key']}"
    else:
        expected = _artifact_id(identity_prefix, artifact, identity_field)
    if artifact[identity_field] != expected:
        raise ResearchDiscoveryError(
            f"{expected_version} identity does not match its canonical payload"
        )
    return artifact


def _validate_payload(
    config: ForgeConfig,
    payload: Mapping[str, Any],
    *,
    expected_version: str,
) -> None:
    SchemaStore(config.paths.forge_root / "schemas").validate(
        dict(payload), SCHEMA_NAME
    )
    if payload.get("schema_version") != expected_version:
        raise ResearchDiscoveryError(
            f"expected schema_version {expected_version!r}, got {payload.get('schema_version')!r}"
        )


def _artifact_id(
    prefix: str,
    payload: Mapping[str, Any],
    identity_field: str,
) -> str:
    return f"{prefix}:{sha256_text(canonical_json(_identity_body(payload, identity_field)))}"


def _identity_body(
    payload: Mapping[str, Any],
    identity_field: str,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {identity_field, "created_at"}
    }


def _write_immutable_artifact(
    *,
    config: ForgeConfig,
    destination: Path,
    payload: dict[str, Any],
    expected_version: str,
    identity_field: str,
) -> None:
    root = (config.paths.artifacts_root / _DISCOVERY_ROOT).resolve(strict=False)
    target = destination.resolve(strict=False)
    if not target.is_relative_to(root):
        raise ResearchDiscoveryError(
            "research discovery artifacts may only be written below artifacts/research_discovery"
        )
    if target.exists():
        existing = read_json(target)
        _validate_payload(config, existing, expected_version=expected_version)
        if existing[identity_field] != payload[identity_field]:
            raise ResearchDiscoveryError(
                f"refusing to overwrite another immutable artifact: {target}"
            )
        if _identity_body(existing, identity_field) != _identity_body(
            payload, identity_field
        ):
            raise ResearchDiscoveryError(
                f"refusing to overwrite changed immutable artifact: {target}"
            )
        return
    _write_create_only_json(target, payload)


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ResearchDiscoveryError(
            f"refusing to overwrite create-only artifact: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


@contextlib.contextmanager
def _exclusive_lock(path: Path, *, context: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Forge targets POSIX hosts.
        raise ResearchDiscoveryError(f"{context} requires POSIX file locking") from exc
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ResearchDiscoveryError(f"cannot open {context} lock {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _resolve_repo_source(
    config: ForgeConfig,
    locator: str,
    *,
    context: str,
) -> Path:
    path = _resolve_repo_locator(config, locator, context=context)
    if not path.is_file() and not path.is_dir():
        raise ResearchDiscoveryError(f"{context} must be a file or directory: {path}")
    return path


def _resolve_repo_locator(
    config: ForgeConfig,
    locator: str,
    *,
    context: str,
) -> Path:
    relative = _safe_relative(locator, context=context)
    candidate = config.paths.repo_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ResearchDiscoveryError(f"{context} may not traverse a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchDiscoveryError(f"missing {context}: {candidate}") from exc
    if not resolved.is_relative_to(config.paths.repo_root):
        raise ResearchDiscoveryError(f"{context} escapes repository root: {locator!r}")
    return resolved


def _resolve_repo_file(
    config: ForgeConfig,
    path: Path,
    *,
    context: str,
) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        try:
            resolved = expanded.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ResearchDiscoveryError(f"missing {context}: {expanded}") from exc
        if not resolved.is_relative_to(config.paths.repo_root):
            raise ResearchDiscoveryError(f"{context} must be inside the repository")
        locator = resolved.relative_to(config.paths.repo_root).as_posix()
        resolved = _resolve_repo_locator(config, locator, context=context)
    else:
        resolved = _resolve_repo_locator(config, path.as_posix(), context=context)
    if not resolved.is_file():
        raise ResearchDiscoveryError(f"{context} must be a regular file: {resolved}")
    return resolved


def _resolve_discovery_file(
    config: ForgeConfig,
    path: Path,
    *,
    context: str,
) -> Path:
    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else config.paths.repo_root / expanded
    if candidate.is_symlink():
        raise ResearchDiscoveryError(f"{context} may not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchDiscoveryError(f"missing {context}: {candidate}") from exc
    root = (config.paths.artifacts_root / _DISCOVERY_ROOT).resolve(strict=False)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ResearchDiscoveryError(
            f"{context} must be a regular file below artifacts/research_discovery"
        )
    return resolved


def _verify_content_ref(
    config: ForgeConfig,
    value: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    locator = str(value["locator"])
    raw = Path(locator).expanduser()
    if raw.is_absolute():
        if raw.is_symlink():
            raise ResearchDiscoveryError(f"{context} may not be a symlink: {raw}")
        try:
            path = raw.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ResearchDiscoveryError(f"missing {context}: {raw}") from exc
    else:
        path = _resolve_repo_locator(config, locator, context=context)
    if not path.is_file():
        raise ResearchDiscoveryError(f"{context} must be a regular file: {path}")
    try:
        actual = sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ResearchDiscoveryError(f"cannot read {context} {path}: {exc}") from exc
    if actual != value["sha256"]:
        raise ResearchDiscoveryError(f"{context} digest mismatch")
    return path


def _content_ref(
    config: ForgeConfig,
    path: Path,
    *,
    context: str,
) -> dict[str, str]:
    if path.is_symlink():
        raise ResearchDiscoveryError(f"{context} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        content = resolved.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ResearchDiscoveryError(f"cannot read {context} {path}: {exc}") from exc
    locator = (
        resolved.relative_to(config.paths.repo_root).as_posix()
        if resolved.is_relative_to(config.paths.repo_root)
        else str(resolved)
    )
    return {"locator": locator, "sha256": sha256_bytes(content)}


def _safe_relative(locator: str, *, context: str) -> PurePosixPath:
    relative = PurePosixPath(locator)
    if (
        not locator
        or "\\" in locator
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ResearchDiscoveryError(f"unsafe {context} locator: {locator!r}")
    return relative


def _resolve_executable(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BackendError(f"{context} may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BackendError(f"missing {context}: {expanded}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise BackendError(f"{context} must be an executable file: {resolved}")
    return resolved


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise ResearchDiscoveryError(
            f"cannot open discovery artifact directory for fsync {path}: {exc}"
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ResearchDiscoveryError(
            f"cannot fsync discovery artifact directory {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


__all__ = [
    "BINDING_VERSION",
    "CodexNativeResearchDiscoveryBackend",
    "CorpusSnapshot",
    "DEMAND_VERSION",
    "PROPOSAL_VERSION",
    "PROMPT_REVISION",
    "RUN_VERSION",
    "ReplayResearchDiscoveryBackend",
    "ResearchDemandBindingResult",
    "ResearchDiscoveryBackend",
    "ResearchDiscoveryError",
    "ResearchDiscoveryResult",
    "build_research_corpus",
    "discover_research_topics",
    "review_research_topic",
    "submit_bound_topic_for_a0",
    "validate_research_demand",
    "validate_research_demand_binding",
    "validate_research_discovery_run",
    "validate_research_topic_proposal",
]
