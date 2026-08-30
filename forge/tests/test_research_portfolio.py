from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import canonical_json, sha256_bytes, sha256_text
from volvence_forge.research_portfolio import (
    ResearchPortfolioError,
    inspect_research_portfolio,
    run_managed_research_loop_once,
    run_research_portfolio_once,
    seal_research_portfolio,
    validate_research_portfolio,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class NoCallBackend:
    backend_name = "replay"
    model_name = "must-not-be-called"

    def complete_json(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("portfolio with no runnable study may not call discovery")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _identity(prefix: str, payload: dict[str, Any], field: str) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {field, "created_at"}
    }
    return f"{prefix}:{sha256_text(canonical_json(body))}"


def _ref(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "locator": path.relative_to(root).as_posix(),
        "sha256": sha256_bytes(path.read_bytes()),
    }


def _authority() -> dict[str, bool]:
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


def _study_authority() -> dict[str, bool]:
    return {
        "registration_only": True,
        "human_topic_binding_required": True,
        "human_a0_required": True,
        "human_outcome_decision_required": True,
        "research_start_authorized": False,
        "production_promotion_authorized": False,
        "runtime_wiring_changed": False,
        "evaluation_is_learning_source": False,
    }


def _config(tmp_path: Path) -> ForgeConfig:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge/editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge/schemas", forge / "schemas")
    return ForgeConfig.load(
        ForgePaths.discover(repo_root=repo, transcripts_root=repo / "transcripts")
    )


def _demand(
    config: ForgeConfig,
    *,
    study_id: str,
    owner: str,
    axes: list[str],
) -> Path:
    source = config.paths.repo_root / "research" / "sources" / f"{study_id}.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"Frozen source for {study_id}.\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": "forge-volvence-research-demand.v1",
        "claim_id": f"claim:{study_id}",
        "title": f"Research {study_id}",
        "objective": f"Resolve the falsifiable {study_id} research gap.",
        "owner": owner,
        "capability_axes": axes,
        "need": {
            "current_gap": "The current mechanism lacks decisive evidence.",
            "required_outcome": "Produce a bounded falsifiable result.",
            "success_criteria": ["The preregistered primary gate passes."],
            "falsification_criteria": ["Matched controls explain the result."],
            "protected_boundaries": [
                "Evaluation is not a learning source.",
                "No production wiring changes are authorized.",
            ],
        },
        "evidence": [_ref(source, root=config.paths.repo_root)],
        "discovery": {
            "source_roots": [source.relative_to(config.paths.repo_root).as_posix()],
            "max_source_files": 2,
            "max_source_bytes": 4096,
            "max_topics": 1,
        },
        "routing": {"requested_mapping_id": None},
        "status": "OPEN",
        "authority": _authority(),
        "created_at": "2026-08-30T00:00:00Z",
    }
    payload["demand_id"] = _identity("research-demand", payload, "demand_id")
    return _write_json(
        config.paths.repo_root / "research" / "demands" / f"{study_id}.json",
        payload,
    )


def _portfolio(config: ForgeConfig) -> Path:
    first = _demand(
        config,
        study_id="reader_validity",
        owner="vz-cognition",
        axes=["readable", "steerable"],
    )
    second = _demand(
        config,
        study_id="substrate_authority",
        owner="vz-substrate",
        axes=["steerable"],
    )
    payload: dict[str, Any] = {
        "schema_version": "forge-research-portfolio.v1",
        "title": "Four-axis research portfolio",
        "objective": "Resolve upstream four-axis uncertainty in dependency order.",
        "owner": "volvence-research-program",
        "scheduling": {
            "ordering_strategy": "dependency_then_priority",
            "max_active_runs_global": 1,
            "unknown_active_run_policy": "BLOCK",
            "dependency_gate": "NAMED_HUMAN_OUTCOME",
            "resume_policy": "completed_generation",
            "lanes": [{"name": "steering_foundation", "max_active_runs": 1}],
        },
        "studies": [
            {
                "study_id": "reader_validity",
                "title": "Reader validity",
                "objective": "Establish cross-view causal reader validity.",
                "claim_id": "claim:reader_validity",
                "owner": "vz-cognition",
                "capability_axes": ["readable", "steerable"],
                "priority": 10,
                "depends_on": [],
                "concurrency_lane": "steering_foundation",
                "demand": {
                    "artifact_id": json.loads(first.read_text())["demand_id"],
                    "artifact": _ref(first, root=config.paths.repo_root),
                },
                "mapping_id": None,
                "task_id": None,
                "readiness": "NEEDS_TASK_DESIGN",
                "required_completion_decision": "PROCEED",
                "authority": _study_authority(),
            },
            {
                "study_id": "substrate_authority",
                "title": "Substrate authority",
                "objective": "Measure bounded substrate control authority.",
                "claim_id": "claim:substrate_authority",
                "owner": "vz-substrate",
                "capability_axes": ["steerable"],
                "priority": 20,
                "depends_on": ["reader_validity"],
                "concurrency_lane": "steering_foundation",
                "demand": {
                    "artifact_id": json.loads(second.read_text())["demand_id"],
                    "artifact": _ref(second, root=config.paths.repo_root),
                },
                "mapping_id": None,
                "task_id": None,
                "readiness": "NEEDS_TASK_DESIGN",
                "required_completion_decision": "PROCEED",
                "authority": _study_authority(),
            },
        ],
        "authority": {
            "portfolio_scheduling_only": True,
            "automatic_human_gates_authorized": False,
            "automatic_candidate_import_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": "2026-08-30T00:00:00Z",
    }
    payload["portfolio_id"] = _identity(
        "research-portfolio", payload, "portfolio_id"
    )
    return _write_json(
        config.paths.repo_root / "research" / "portfolios" / "four_axes.json",
        payload,
    )


def test_portfolio_registers_dag_and_blocks_downstream_without_outcome(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _portfolio(config)

    portfolio = validate_research_portfolio(config=config, portfolio_path=path)
    assert len(portfolio["studies"]) == 2

    status = inspect_research_portfolio(config=config, portfolio_path=path)
    assert [(item.study_id, item.state) for item in status.studies] == [
        ("reader_validity", "NEEDS_TASK_DESIGN"),
        ("substrate_authority", "WAITING_FOR_DEPENDENCIES"),
    ]

    result = run_research_portfolio_once(
        config=config,
        portfolio_path=path,
        backend=NoCallBackend(),
    )
    assert result.eligible_study_ids == ()
    assert result.loop.demand_count == 0


def test_managed_loop_blocks_registered_ineligible_demands_but_keeps_unregistered(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _portfolio(config)
    unregistered_path = _demand(
        config,
        study_id="unregistered_paused",
        owner="vz-memory",
        axes=["appendable"],
    )
    unregistered = json.loads(unregistered_path.read_text(encoding="utf-8"))
    unregistered["status"] = "PAUSED"
    unregistered["demand_id"] = _identity(
        "research-demand",
        unregistered,
        "demand_id",
    )
    _write_json(unregistered_path, unregistered)

    result = run_managed_research_loop_once(
        config=config,
        backend=NoCallBackend(),
    )

    assert len(result.portfolio_statuses) == 1
    assert result.eligible_studies == ()
    assert [item.study_id for item in result.blocked_studies] == [
        "reader_validity",
        "substrate_authority",
    ]
    assert result.loop.demand_count == 1
    assert result.loop.open_demand_count == 0


def test_portfolio_seal_is_content_addressed_create_only_and_idempotent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = _portfolio(config)
    draft = json.loads(source.read_text(encoding="utf-8"))
    expected_id = draft.pop("portfolio_id")
    draft_path = _write_json(
        config.paths.repo_root / "research" / "portfolio_drafts" / "four_axes.json",
        draft,
    )

    first = seal_research_portfolio(config=config, draft_path=draft_path)
    second = seal_research_portfolio(config=config, draft_path=draft_path)

    assert first.portfolio_id == expected_id
    assert first.portfolio_path.name == expected_id.partition(":")[2] + ".json"
    assert first.reused is False
    assert second == type(second)(
        portfolio_id=first.portfolio_id,
        portfolio_path=first.portfolio_path,
        reused=True,
    )
    with pytest.raises(ResearchPortfolioError, match="below research/portfolios"):
        seal_research_portfolio(
            config=config,
            draft_path=draft_path,
            output_path=config.paths.repo_root / "outside.json",
        )


def test_portfolio_rejects_cycle_and_demand_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = _portfolio(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["studies"][0]["depends_on"] = ["substrate_authority"]
    payload["portfolio_id"] = _identity(
        "research-portfolio", payload, "portfolio_id"
    )
    _write_json(path, payload)
    with pytest.raises(ResearchPortfolioError, match="cycle"):
        validate_research_portfolio(config=config, portfolio_path=path)

    path = _portfolio(config)
    demand_path = config.paths.repo_root / "research/demands/reader_validity.json"
    demand_path.write_text(demand_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ResearchPortfolioError, match="SHA-256 drift"):
        validate_research_portfolio(config=config, portfolio_path=path)
