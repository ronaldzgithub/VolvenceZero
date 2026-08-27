"""Coding-lab Packet 3.5: interventional (RCT) junction calibration.

Runs episode chains in which, per episode, a seeded random draw assigns
either a forced protocol action at the first target-state junction or a
control annotation. The oracle settles every episode as usual, giving
intention-to-treat pass rates per (state key, assigned action) — the
causal replacement for the observational ``credit_expert_actions``
table, whose confounding is registered in ``docs/specs/coding-lab.md``
(§7.5: survivorship / difficulty confounds).

Evidence discipline: the runner itself is tier-agnostic; the CLI marks
smoke runs as development tier, and formal runs must carry a frozen
prereg whose SHA-256 is embedded in the report. No capability claim is
authorized by this module — its output is a calibration table for the
Packet 3.6 episode-outcome gate.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import statistics
import time
from dataclasses import dataclass, field
from random import Random
from typing import Any

from lifeform_domain_coding.lab.episode import EpisodeBudget, EpisodeResult, run_episode
from lifeform_domain_coding.lab.generation import EnvSpec
from lifeform_domain_coding.lab.hands import (
    APIHandConfig,
    ConstraintAwareScriptedHand,
    ForcedActionAssignment,
    ForcedActionHand,
    Hand,
    OpenAICompatHand,
)
from lifeform_domain_coding.lab.junctions import (
    ASSIGNMENT_ARM_CONTROL,
    ASSIGNMENT_ARM_INTERVENTION,
    JUNCTION_ACTIONS,
    InterventionalAssignmentRecord,
    build_action_outcome_table,
    build_interventional_action_outcome_table,
    collect_forced_assignments,
    collect_junctions,
    credit_expert_actions,
    interventional_expert_actions,
    wilson_interval,
)
from lifeform_domain_coding.lab.tasks import ChainTask, generate_task_chain
from lifeform_domain_coding.lab.workspace import ChainWorkspace

HAND_SCRIPTED = "scripted"
HAND_API = "api"


@dataclass(frozen=True)
class InterventionalConfig:
    """Frozen configuration of one Packet 3.5 run (smoke or formal)."""

    run_id: str
    output_root: pathlib.Path
    target_state_keys: tuple[str, ...]
    env_seed: int = 20260812
    chains: int = 4
    episodes_per_chain: int = 8
    hand_kind: str = HAND_SCRIPTED
    scripted_hand_seed: int = 11
    scripted_invariant_sabotage_rate: float = 0.25
    scripted_acceptance_sabotage_rate: float = 0.25
    api_hand_config: APIHandConfig | None = None
    convention_ids: tuple[str, ...] = ()
    assignment_seed: int = 20260827
    control_weight: float = 0.2
    budget: EpisodeBudget = field(default_factory=EpisodeBudget)
    min_free_disk_bytes: int = 2 * 1024**3
    resume: bool = False

    def __post_init__(self) -> None:
        if self.hand_kind not in (HAND_SCRIPTED, HAND_API):
            raise ValueError(f"hand_kind must be scripted|api, got {self.hand_kind!r}")
        if self.hand_kind == HAND_API and self.api_hand_config is None:
            raise ValueError("api hand requires api_hand_config")
        if not self.target_state_keys:
            raise ValueError("target_state_keys must be non-empty (prereg-frozen)")
        if not 0.0 <= self.control_weight < 1.0:
            raise ValueError("control_weight must be in [0, 1)")
        if self.chains < 1 or self.episodes_per_chain < 1:
            raise ValueError("chains and episodes_per_chain must be >= 1")


@dataclass(frozen=True)
class InterventionalEpisodeRow:
    """Per-episode accounting row (assignment fields mirror the trajectory)."""

    chain_index: int
    episode_index: int
    task_id: str
    category: str
    passed: bool
    assignment_drawn: str | None
    assignment_arm: str
    triggered: bool
    submitted: bool
    steps_used: int
    wall_seconds: float
    prompt_tokens: int
    completion_tokens: int
    trajectory_sha256: str


def draw_assignment(config: InterventionalConfig, chain_index: int, episode_index: int) -> str | None:
    """Seeded uniform draw: control with ``control_weight`` mass, else one action."""

    rng = Random(
        config.assignment_seed * 1_000_003 + chain_index * 10_007 + episode_index * 101
    )
    if rng.random() < config.control_weight:
        return None
    return JUNCTION_ACTIONS[rng.randrange(len(JUNCTION_ACTIONS))]


def _chain_spec(config: InterventionalConfig, chain_index: int) -> EnvSpec:
    return EnvSpec(
        env_seed=config.env_seed + chain_index * 13,
        convention_ids=config.convention_ids,
    )


def _build_inner_hand(
    config: InterventionalConfig, chain: tuple[ChainTask, ...], chain_index: int
) -> Hand:
    if config.hand_kind == HAND_API:
        assert config.api_hand_config is not None  # guarded in __post_init__
        return OpenAICompatHand(config.api_hand_config)
    return ConstraintAwareScriptedHand(
        tasks_by_id={task.task_id: task for task in chain},
        episode_index_by_task_id={task.task_id: index for index, task in enumerate(chain)},
        hand_seed=config.scripted_hand_seed + 101 * chain_index,
        invariant_sabotage_rate=config.scripted_invariant_sabotage_rate,
        acceptance_sabotage_rate=config.scripted_acceptance_sabotage_rate,
    )


def _chain_rows_path(run_dir: pathlib.Path, chain_index: int) -> pathlib.Path:
    return run_dir / "chains" / f"chain-{chain_index:02d}" / "rows.json"


def _load_chain_rows(path: pathlib.Path) -> list[InterventionalEpisodeRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [InterventionalEpisodeRow(**row) for row in payload]


async def _run_chain(
    config: InterventionalConfig,
    chain_index: int,
    run_dir: pathlib.Path,
) -> list[InterventionalEpisodeRow]:
    spec = _chain_spec(config, chain_index)
    chain_dir = run_dir / "chains" / f"chain-{chain_index:02d}"
    rows_path = _chain_rows_path(run_dir, chain_index)
    if config.resume and rows_path.is_file():
        return _load_chain_rows(rows_path)
    if config.resume and chain_dir.exists():
        shutil.rmtree(chain_dir)
    workspace = ChainWorkspace(spec=spec, chain_root=chain_dir)
    workspace.initialize()
    chain = generate_task_chain(spec, chain_seed=chain_index, length=config.episodes_per_chain)
    inner_hand = _build_inner_hand(config, chain, chain_index)
    rows: list[InterventionalEpisodeRow] = []
    for episode_index, task in enumerate(chain):
        assignment_drawn = draw_assignment(config, chain_index, episode_index)
        assignment = ForcedActionAssignment(
            target_state_keys=config.target_state_keys,
            assigned_action=assignment_drawn,
            assignment_id=(
                f"p35:{config.assignment_seed}:c{chain_index:02d}:e{episode_index:03d}"
            ),
        )
        hand = ForcedActionHand(
            inner=inner_hand,
            category=task.category,
            assignment=assignment,
        )
        handle = workspace.begin_episode(episode_index, task)
        result: EpisodeResult = await run_episode(
            spec=spec,
            task=task,
            episode_index=episode_index,
            worktree=handle.worktree,
            hand=hand,
            trajectory_path=chain_dir / "trajectories" / f"episode-{episode_index:03d}.jsonl",
            budget=config.budget,
        )
        workspace.finalize_episode(handle, passed=result.outcome.passed, task=task)
        rows.append(
            InterventionalEpisodeRow(
                chain_index=chain_index,
                episode_index=episode_index,
                task_id=task.task_id,
                category=task.category,
                passed=result.outcome.passed,
                assignment_drawn=assignment_drawn,
                assignment_arm=(
                    ASSIGNMENT_ARM_CONTROL
                    if assignment_drawn is None
                    else ASSIGNMENT_ARM_INTERVENTION
                ),
                triggered=hand.triggered,
                submitted=result.submitted,
                steps_used=result.steps_used,
                wall_seconds=result.wall_seconds,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                trajectory_sha256=result.trajectory.sha256,
            )
        )
    rows_path.write_text(
        json.dumps([dataclasses.asdict(row) for row in rows], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return rows


def _trajectory_paths(run_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return tuple(sorted(run_dir.glob("chains/chain-*/trajectories/episode-*.jsonl")))


def _stat_payload(stat: Any) -> dict[str, Any]:
    low, high = stat.wilson()
    return {
        "state_key": stat.state_key,
        "assigned_action": stat.assigned_action,
        "trials": stat.trials,
        "passes": stat.passes,
        "pass_rate": round(stat.pass_rate, 6),
        "compliant_trials": stat.compliant_trials,
        "compliance_rate": round(stat.compliance_rate, 6),
        "wilson95_low": round(low, 6),
        "wilson95_high": round(high, 6),
    }


def _assignment_summary(
    records: tuple[InterventionalAssignmentRecord, ...],
) -> dict[str, Any]:
    intervention = [r for r in records if r.arm == ASSIGNMENT_ARM_INTERVENTION]
    control = [r for r in records if r.arm == ASSIGNMENT_ARM_CONTROL]
    compliance_by_action: dict[str, list[bool]] = {}
    for record in intervention:
        assert record.assigned_action is not None
        compliance_by_action.setdefault(record.assigned_action, []).append(record.compliant)
    return {
        "triggered_total": len(records),
        "intervention_records": len(intervention),
        "control_records": len(control),
        "compliance_rate_by_action": {
            action: round(statistics.fmean(float(v) for v in values), 6)
            for action, values in sorted(compliance_by_action.items())
        },
        "distinct_state_keys_triggered": len({r.state_key for r in records}),
    }


def _preflight_disk(config: InterventionalConfig) -> dict[str, Any]:
    usage = shutil.disk_usage(config.output_root)
    if usage.free < config.min_free_disk_bytes:
        raise RuntimeError(
            f"preflight: only {usage.free} bytes free under {config.output_root!s}, "
            f"need >= {config.min_free_disk_bytes}"
        )
    return {"free_bytes": usage.free, "required_bytes": config.min_free_disk_bytes}


async def run_interventional_calibration(
    config: InterventionalConfig,
    *,
    evidence_tier: str = "development",
    prereg_sha256: str | None = None,
) -> dict[str, Any]:
    """Run all chains, build ITT tables from trajectories, write report artifacts."""

    if evidence_tier == "formal" and not prereg_sha256:
        raise ValueError("formal runs must carry the frozen prereg SHA-256")
    # Absolute run dir: chain worktrees are created by `git worktree add`
    # relative to the chain repo's cwd, so a relative output root would
    # silently land worktrees inside the generated repo.
    run_dir = (pathlib.Path(config.output_root) / config.run_id).resolve()
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"interventional run dir already exists: {run_dir!s}")
    run_dir.mkdir(parents=True, exist_ok=config.resume)
    disk = _preflight_disk(config)
    started = time.time()

    rows: list[InterventionalEpisodeRow] = []
    for chain_index in range(config.chains):
        rows.extend(await _run_chain(config, chain_index, run_dir))

    trajectory_paths = _trajectory_paths(run_dir)
    assignments = collect_forced_assignments(trajectory_paths)
    interventional_table = build_interventional_action_outcome_table(assignments)
    interventional_experts = interventional_expert_actions(assignments)

    # Observational comparison uses CONTROL trajectories only: intervention
    # episodes are contaminated downstream of the forced step.
    control_sha256 = {
        r.trajectory_sha256 for r in assignments if r.arm == ASSIGNMENT_ARM_CONTROL
    }
    row_by_sha = {row.trajectory_sha256: row for row in rows}
    control_paths = tuple(
        path
        for path in trajectory_paths
        if row_by_sha.get(_sha256_of(path)) is not None
        and _sha256_of(path) in control_sha256
    )
    control_junctions = collect_junctions(control_paths)
    observational_table = build_action_outcome_table(control_junctions)
    observational_experts = credit_expert_actions(control_junctions)

    disagreements = sorted(
        key
        for key in set(interventional_experts) & set(observational_experts)
        if interventional_experts[key] != observational_experts[key]
    )

    report: dict[str, Any] = {
        "packet": "coding-lab-packet-3.5",
        "run_id": config.run_id,
        "evidence_tier": evidence_tier,
        "prereg_sha256": prereg_sha256,
        "started_unix": int(started),
        "config": {
            "env_seed": config.env_seed,
            "chains": config.chains,
            "episodes_per_chain": config.episodes_per_chain,
            "hand_kind": config.hand_kind,
            "api_model": (
                config.api_hand_config.model if config.api_hand_config is not None else None
            ),
            "convention_ids": list(config.convention_ids),
            "target_state_keys": list(config.target_state_keys),
            "assignment_seed": config.assignment_seed,
            "control_weight": config.control_weight,
            "scripted_rates": {
                "invariant_sabotage": config.scripted_invariant_sabotage_rate,
                "acceptance_sabotage": config.scripted_acceptance_sabotage_rate,
            },
        },
        "preflight_disk": disk,
        "episodes": [dataclasses.asdict(row) for row in rows],
        "assignment_summary": _assignment_summary(assignments),
        "interventional_table": {
            state_key: [_stat_payload(stat) for stat in stats]
            for state_key, stats in sorted(interventional_table.items())
        },
        "interventional_expert_actions": dict(sorted(interventional_experts.items())),
        "observational_control_table": {
            state_key: [
                {
                    "action": stat.action,
                    "trials": stat.trials,
                    "passes": stat.passes,
                    "pass_rate": round(stat.pass_rate, 6),
                }
                for stat in stats
            ]
            for state_key, stats in sorted(observational_table.items())
        },
        "observational_control_expert_actions": dict(sorted(observational_experts.items())),
        "expert_disagreements": disagreements,
        "honest_boundaries": {
            "intention_to_treat": True,
            "randomized_within_reached_state": True,
            "state_reachability_not_randomized": True,
            "capability_claim_authorized": False,
            "note": (
                "ITT pass rates are causal for the assigned action GIVEN the state "
                "was reached; which states get reached remains observational. A "
                "disagreement between the interventional and observational expert "
                "maps is a finding about confounding, not a failure."
            ),
        },
        "wall_seconds": time.time() - started,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


_SHA_CACHE: dict[pathlib.Path, str] = {}


def _sha256_of(path: pathlib.Path) -> str:
    cached = _SHA_CACHE.get(path)
    if cached is None:
        import hashlib

        cached = hashlib.sha256(path.read_bytes()).hexdigest()
        _SHA_CACHE[path] = cached
    return cached


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["assignment_summary"]
    lines = [
        "# coding-lab Packet 3.5 interventional calibration",
        "",
        f"- run_id: `{report['run_id']}` (tier: {report['evidence_tier']})",
        f"- episodes: {len(report['episodes'])}, triggered assignments: {summary['triggered_total']}"
        f" (intervention {summary['intervention_records']} / control {summary['control_records']})",
        f"- compliance by action: {summary['compliance_rate_by_action']}",
        f"- interventional expert map: {report['interventional_expert_actions']}",
        f"- observational (control-only) expert map: {report['observational_control_expert_actions']}",
        f"- expert disagreements: {report['expert_disagreements']}",
        "",
        "ITT boundary: causal per assigned action given reached state; no capability claim.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "HAND_API",
    "HAND_SCRIPTED",
    "InterventionalConfig",
    "InterventionalEpisodeRow",
    "draw_assignment",
    "run_interventional_calibration",
    "wilson_interval",
]
