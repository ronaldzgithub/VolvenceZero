"""Coding-lab Packet 3.6: episode-outcome gate for intervention timing.

Lifts the "when to intervene" claim from proxy metrics onto the oracle's
episode pass rate. Four matched arms over identical chains:

* ``noop`` — the bare hand, never intervened;
* ``always_on`` — at every certified opportunity (the episode category has
  a causally certified expert cell from the Packet 3.5 interventional RCT),
  force the expert action once;
* ``random_gate`` — at each opportunity, steer with a pre-registered,
  rate-matched coin;
* ``table_gate`` — steer only where the interventional credit table says
  the expert action's ITT pass rate beats the natural (control) pass rate
  by at least a pre-registered margin.

Claim boundary (pre-registered): this measures ACTION-LEVEL intervention
timing driven by interventional credit. It does not lift the residual-level
Steerable claim (S3-E stays at NLL/mechanism tier), and with the current
two-cell certified surface ``table_gate`` and ``always_on`` are expected to
coincide except on zero-gain cells — their contrast is report-only. The
decisive comparisons are vs ``noop`` (does certified intervention help at
outcome level) and vs ``random_gate`` (does credit-driven placement beat
rate-matched chance).
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
    JUNCTION_ACTIONS,
    wilson_interval,
)
from lifeform_domain_coding.lab.tasks import ChainTask, generate_task_chain
from lifeform_domain_coding.lab.workspace import ChainWorkspace, remove_tree

HAND_SCRIPTED = "scripted"
HAND_API = "api"

ARM_NOOP = "noop"
ARM_ALWAYS_ON = "always_on"
ARM_RANDOM_GATE = "random_gate"
ARM_TABLE_GATE = "table_gate"
ALL_ARMS: tuple[str, ...] = (ARM_NOOP, ARM_ALWAYS_ON, ARM_RANDOM_GATE, ARM_TABLE_GATE)


@dataclass(frozen=True)
class CertifiedCell:
    """One causally certified (state key, expert action) steering cell."""

    state_key: str
    category: str
    expert_action: str
    expert_itt_pass_rate: float
    natural_control_pass_rate: float

    @property
    def credited_gain(self) -> float:
        return self.expert_itt_pass_rate - self.natural_control_pass_rate


def derive_certified_cells(
    calibration_report: dict[str, Any],
) -> tuple[CertifiedCell, ...]:
    """Certified steering surface from the Packet 3.5 formal report.

    Expert cells come from ``interventional_expert_actions`` (randomized,
    margin-cleared). The natural comparison point is the CONTROL arm's
    weighted pass rate at the same state key; a key without control
    coverage is dropped (no honest gain estimate).
    """

    experts = calibration_report["interventional_expert_actions"]
    interventional = calibration_report["interventional_table"]
    control = calibration_report["observational_control_table"]
    cells: list[CertifiedCell] = []
    for state_key in sorted(experts):
        expert_action = experts[state_key]
        if expert_action not in JUNCTION_ACTIONS:
            raise ValueError(f"expert action outside protocol surface: {expert_action!r}")
        stats = {row["assigned_action"]: row for row in interventional[state_key]}
        expert_row = stats[expert_action]
        control_rows = control.get(state_key, [])
        control_trials = sum(int(row["trials"]) for row in control_rows)
        if control_trials == 0:
            continue
        control_passes = sum(int(row["passes"]) for row in control_rows)
        cells.append(
            CertifiedCell(
                state_key=state_key,
                category=state_key.split("|", 1)[0],
                expert_action=expert_action,
                expert_itt_pass_rate=float(expert_row["pass_rate"]),
                natural_control_pass_rate=control_passes / control_trials,
            )
        )
    if not cells:
        raise ValueError("calibration report yields no certified steering cells")
    return tuple(cells)


@dataclass(frozen=True)
class Packet36Config:
    """Frozen configuration of one Packet 3.6 run (smoke or formal)."""

    run_id: str
    output_root: pathlib.Path
    certified_cells: tuple[CertifiedCell, ...]
    env_seed: int = 20260812
    chains: int = 4
    episodes_per_chain: int = 8
    hand_kind: str = HAND_SCRIPTED
    scripted_hand_seed: int = 11
    scripted_invariant_sabotage_rate: float = 0.25
    scripted_acceptance_sabotage_rate: float = 0.25
    api_hand_config: APIHandConfig | None = None
    convention_ids: tuple[str, ...] = ()
    gate_seed: int = 20260828
    random_gate_steer_probability: float = 0.5
    table_gate_min_gain: float = 0.05
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260828
    budget: EpisodeBudget = field(default_factory=EpisodeBudget)
    min_free_disk_bytes: int = 2 * 1024**3
    resume: bool = False

    def __post_init__(self) -> None:
        if self.hand_kind not in (HAND_SCRIPTED, HAND_API):
            raise ValueError(f"hand_kind must be scripted|api, got {self.hand_kind!r}")
        if self.hand_kind == HAND_API and self.api_hand_config is None:
            raise ValueError("api hand requires api_hand_config")
        if not self.certified_cells:
            raise ValueError("certified_cells must be non-empty")
        if not 0.0 < self.random_gate_steer_probability < 1.0:
            raise ValueError("random_gate_steer_probability must be in (0, 1)")
        if self.chains < 2 or self.episodes_per_chain < 1:
            raise ValueError("need >= 2 chains for paired statistics")


def arm_steers(
    config: Packet36Config,
    *,
    arm: str,
    cell: CertifiedCell | None,
    chain_index: int,
    episode_index: int,
) -> bool:
    """Frozen per-arm steering policy for one episode's opportunity."""

    if cell is None:
        return False
    if arm == ARM_NOOP:
        return False
    if arm == ARM_ALWAYS_ON:
        return True
    if arm == ARM_RANDOM_GATE:
        rng = Random(
            config.gate_seed * 1_000_003 + chain_index * 10_007 + episode_index * 101
        )
        return rng.random() < config.random_gate_steer_probability
    if arm == ARM_TABLE_GATE:
        return cell.credited_gain >= config.table_gate_min_gain
    raise ValueError(f"unknown arm: {arm}")


@dataclass(frozen=True)
class Packet36EpisodeRow:
    arm: str
    chain_index: int
    episode_index: int
    task_id: str
    category: str
    passed: bool
    opportunity: bool
    steer_decided: bool
    triggered: bool
    expert_action: str | None
    submitted: bool
    steps_used: int
    wall_seconds: float
    prompt_tokens: int
    completion_tokens: int
    trajectory_sha256: str


def _chain_spec(config: Packet36Config, chain_index: int) -> EnvSpec:
    return EnvSpec(
        env_seed=config.env_seed + chain_index * 13,
        convention_ids=config.convention_ids,
    )


def _build_inner_hand(
    config: Packet36Config, chain: tuple[ChainTask, ...], chain_index: int
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


def _cell_rows_path(run_dir: pathlib.Path, arm: str, chain_index: int) -> pathlib.Path:
    return run_dir / arm / f"chain-{chain_index:02d}" / "rows.json"


def _load_cell_rows(path: pathlib.Path) -> list[Packet36EpisodeRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Packet36EpisodeRow(**row) for row in payload]


async def _run_arm_chain(
    config: Packet36Config,
    *,
    arm: str,
    chain_index: int,
    run_dir: pathlib.Path,
) -> list[Packet36EpisodeRow]:
    cell_dir = run_dir / arm / f"chain-{chain_index:02d}"
    rows_path = _cell_rows_path(run_dir, arm, chain_index)
    if config.resume and rows_path.is_file():
        return _load_cell_rows(rows_path)
    if config.resume and cell_dir.exists():
        remove_tree(cell_dir)
    spec = _chain_spec(config, chain_index)
    workspace = ChainWorkspace(spec=spec, chain_root=cell_dir)
    workspace.initialize()
    chain = generate_task_chain(spec, chain_seed=chain_index, length=config.episodes_per_chain)
    inner_hand = _build_inner_hand(config, chain, chain_index)
    cell_by_category = {cell.category: cell for cell in config.certified_cells}
    rows: list[Packet36EpisodeRow] = []
    for episode_index, task in enumerate(chain):
        cell = cell_by_category.get(task.category)
        steer = arm_steers(
            config, arm=arm, cell=cell, chain_index=chain_index, episode_index=episode_index
        )
        hand: Hand
        if steer and cell is not None:
            hand = ForcedActionHand(
                inner=inner_hand,
                category=task.category,
                assignment=ForcedActionAssignment(
                    target_state_keys=(cell.state_key,),
                    assigned_action=cell.expert_action,
                    assignment_id=(
                        f"p36:{arm}:{config.gate_seed}:c{chain_index:02d}:e{episode_index:03d}"
                    ),
                ),
            )
        else:
            hand = inner_hand
        handle = workspace.begin_episode(episode_index, task)
        result: EpisodeResult = await run_episode(
            spec=spec,
            task=task,
            episode_index=episode_index,
            worktree=handle.worktree,
            hand=hand,
            trajectory_path=cell_dir / "trajectories" / f"episode-{episode_index:03d}.jsonl",
            budget=config.budget,
        )
        workspace.finalize_episode(handle, passed=result.outcome.passed, task=task)
        rows.append(
            Packet36EpisodeRow(
                arm=arm,
                chain_index=chain_index,
                episode_index=episode_index,
                task_id=task.task_id,
                category=task.category,
                passed=result.outcome.passed,
                opportunity=cell is not None,
                steer_decided=steer,
                triggered=isinstance(hand, ForcedActionHand) and hand.triggered,
                expert_action=cell.expert_action if (steer and cell is not None) else None,
                submitted=result.submitted,
                steps_used=result.steps_used,
                wall_seconds=result.wall_seconds,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                trajectory_sha256=result.trajectory.sha256,
            )
        )
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps([dataclasses.asdict(row) for row in rows], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return rows


def paired_gap_statistics(
    rows: list[Packet36EpisodeRow],
    *,
    arm_a: str,
    arm_b: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Chain-paired pass-rate gap (a − b) with a deterministic bootstrap."""

    chain_indices = sorted({row.chain_index for row in rows})
    gaps: list[float] = []
    for chain_index in chain_indices:
        rate = {}
        for arm in (arm_a, arm_b):
            arm_rows = [
                row for row in rows if row.arm == arm and row.chain_index == chain_index
            ]
            if not arm_rows:
                raise ValueError(f"missing rows for arm {arm} chain {chain_index}")
            rate[arm] = statistics.fmean(1.0 if row.passed else 0.0 for row in arm_rows)
        gaps.append(rate[arm_a] - rate[arm_b])
    rng = Random(seed)
    resampled_means: list[float] = []
    count = len(gaps)
    for _ in range(resamples):
        sample = [gaps[rng.randrange(count)] for _ in range(count)]
        resampled_means.append(statistics.fmean(sample))
    resampled_means.sort()
    lower_index = max(0, int(0.05 * resamples) - 1)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "mean_gap": statistics.fmean(gaps),
        "bootstrap_ci_lower_5pct": resampled_means[lower_index],
        "bootstrap_samples": resamples,
        "per_chain_gaps": gaps,
    }


def _preflight_disk(config: Packet36Config) -> dict[str, Any]:
    usage = shutil.disk_usage(config.output_root)
    if usage.free < config.min_free_disk_bytes:
        raise RuntimeError(
            f"preflight: only {usage.free} bytes free under {config.output_root!s}, "
            f"need >= {config.min_free_disk_bytes}"
        )
    return {"free_bytes": usage.free, "required_bytes": config.min_free_disk_bytes}


async def run_packet36(
    config: Packet36Config,
    *,
    evidence_tier: str = "development",
    prereg_sha256: str | None = None,
    calibration_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Run all four arms over matched chains and write report artifacts."""

    if evidence_tier == "formal" and not (prereg_sha256 and calibration_report_sha256):
        raise ValueError("formal runs must carry frozen prereg and calibration SHA-256s")
    run_dir = (pathlib.Path(config.output_root) / config.run_id).resolve()
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"packet36 run dir already exists: {run_dir!s}")
    run_dir.mkdir(parents=True, exist_ok=config.resume)
    disk = _preflight_disk(config)
    started = time.time()

    rows: list[Packet36EpisodeRow] = []
    for chain_index in range(config.chains):
        for arm in ALL_ARMS:
            rows.extend(
                await _run_arm_chain(config, arm=arm, chain_index=chain_index, run_dir=run_dir)
            )

    pass_rates = {
        arm: statistics.fmean(1.0 if row.passed else 0.0 for row in rows if row.arm == arm)
        for arm in ALL_ARMS
    }
    contrasts = {
        "timing_vs_noop": paired_gap_statistics(
            rows,
            arm_a=ARM_TABLE_GATE,
            arm_b=ARM_NOOP,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        ),
        "timing_vs_random": paired_gap_statistics(
            rows,
            arm_a=ARM_TABLE_GATE,
            arm_b=ARM_RANDOM_GATE,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed + 1,
        ),
        "always_vs_noop": paired_gap_statistics(
            rows,
            arm_a=ARM_ALWAYS_ON,
            arm_b=ARM_NOOP,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed + 2,
        ),
        "table_vs_always_report_only": paired_gap_statistics(
            rows,
            arm_a=ARM_TABLE_GATE,
            arm_b=ARM_ALWAYS_ON,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed + 3,
        ),
    }
    verdicts = {
        "outcome_timing_gate": contrasts["timing_vs_noop"]["bootstrap_ci_lower_5pct"] > 0.0,
        "placement_gate": contrasts["timing_vs_random"]["bootstrap_ci_lower_5pct"] > 0.0,
        "intervention_gate": contrasts["always_vs_noop"]["bootstrap_ci_lower_5pct"] > 0.0,
    }
    # Pre-declared power boundary: with a two-cell surface (one at zero gain)
    # the rate-matched random arm captures ~half the certified effect, so the
    # placement contrast is directional-report-only; verdict-binding gates are
    # the two below (frozen in the prereg before any formal outcome existed).
    binding_gates = ("outcome_timing_gate", "intervention_gate")

    opportunity_rows = [row for row in rows if row.opportunity]
    steered_rows = [row for row in rows if row.steer_decided]
    triggered_rows = [row for row in rows if row.triggered]
    steered_pass = sum(1 for row in steered_rows if row.passed)
    steer_wilson = (
        wilson_interval(steered_pass, len(steered_rows)) if steered_rows else (0.0, 0.0)
    )

    report: dict[str, Any] = {
        "packet": "coding-lab-packet-3.6",
        "run_id": config.run_id,
        "evidence_tier": evidence_tier,
        "prereg_sha256": prereg_sha256,
        "calibration_report_sha256": calibration_report_sha256,
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
            "gate_seed": config.gate_seed,
            "random_gate_steer_probability": config.random_gate_steer_probability,
            "table_gate_min_gain": config.table_gate_min_gain,
            "certified_cells": [dataclasses.asdict(cell) for cell in config.certified_cells],
        },
        "preflight_disk": disk,
        "pass_rates_by_arm": pass_rates,
        "contrasts": contrasts,
        "verdicts": verdicts,
        "binding_gates": list(binding_gates),
        "binding_gates_pass": all(verdicts[name] for name in binding_gates),
        "mechanism": {
            "episodes_total": len(rows),
            "opportunity_episodes": len(opportunity_rows),
            "steer_decided_episodes": len(steered_rows),
            "trigger_delivered_episodes": len(triggered_rows),
            "trigger_delivery_rate": (
                len(triggered_rows) / len(steered_rows) if steered_rows else None
            ),
            "steered_pass_rate_wilson95": [round(v, 6) for v in steer_wilson],
        },
        "honest_boundaries": {
            "claim_scope": "action-level intervention timing on oracle episode pass rate",
            "residual_steerable_lifted": False,
            "table_vs_always_expected_null": (
                "With a two-cell certified surface where one cell has zero credited "
                "gain, table_gate and always_on coincide except on zero-gain cells; "
                "their contrast is report-only by preregistration."
            ),
            "certified_surface_from": "packet35 interventional RCT (randomized, ITT)",
            "capability_claim_authorized": False,
        },
        "episodes": [dataclasses.asdict(row) for row in rows],
        "wall_seconds": time.time() - started,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    verdicts = report["verdicts"]
    contrasts = report["contrasts"]

    def line(name: str) -> str:
        c = contrasts[name]
        return (
            f"- {name}: mean {c['mean_gap']:+.4f}, 5% lower {c['bootstrap_ci_lower_5pct']:+.4f}"
        )

    lines = [
        "# coding-lab Packet 3.6 episode-outcome gate",
        "",
        f"- run_id: `{report['run_id']}` (tier: {report['evidence_tier']})",
        f"- pass rates: {report['pass_rates_by_arm']}",
        line("timing_vs_noop"),
        line("timing_vs_random"),
        line("always_vs_noop"),
        line("table_vs_always_report_only"),
        f"- verdicts: {verdicts}",
        f"- mechanism: {report['mechanism']}",
        "",
        "Claim boundary: action-level timing on oracle pass rate; residual Steerable not lifted.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ALL_ARMS",
    "ARM_ALWAYS_ON",
    "ARM_NOOP",
    "ARM_RANDOM_GATE",
    "ARM_TABLE_GATE",
    "CertifiedCell",
    "HAND_API",
    "HAND_SCRIPTED",
    "Packet36Config",
    "Packet36EpisodeRow",
    "arm_steers",
    "derive_certified_cells",
    "paired_gap_statistics",
    "run_packet36",
]
