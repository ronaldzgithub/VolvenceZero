"""Packet 0 calibration run: environment determinism + oracle teeth.

Produces the Packet 0 verdict artifacts:

* ``environment_deterministic`` — same spec twice => identical tree
  hash and identical task chain (bit-level, environment side only);
* ``oracle_band`` — under the configured hand, episode pass rate falls
  in the pre-registered [low, high] band and outcomes vary across
  episodes (the instrument has discrimination);
* cost accounting — wall seconds, workspace bytes, tokens per episode.

Honesty note carried in every report: a scripted-hand band verdict
calibrates the MACHINERY (oracle teeth, knobs, plumbing). The frozen
API hand must re-run this calibration before its band verdict counts
for Packet 2 prereg.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from lifeform_domain_coding.lab.episode import EpisodeBudget, EpisodeResult, run_episode
from lifeform_domain_coding.lab.generation import EnvSpec, generate_environment
from lifeform_domain_coding.lab.hands import APIHandConfig, Hand, OpenAICompatHand, ScriptedHand
from lifeform_domain_coding.lab.heldout import seal_heldout_variants
from lifeform_domain_coding.lab.tasks import ChainTask, generate_task_chain
from lifeform_domain_coding.lab.workspace import ChainWorkspace

HAND_SCRIPTED = "scripted"
HAND_API = "api"


@dataclass(frozen=True)
class CalibrationConfig:
    run_id: str
    output_root: pathlib.Path
    env_seed: int = 20260812
    chains: int = 4
    episodes_per_chain: int = 8
    hand_kind: str = HAND_SCRIPTED
    scripted_hand_seed: int = 11
    scripted_invariant_sabotage_rate: float = 0.25
    scripted_acceptance_sabotage_rate: float = 0.25
    api_hand_config: APIHandConfig | None = None
    band_low: float = 0.2
    band_high: float = 0.8
    #: Active house conventions (difficulty knob); threaded into every
    #: chain EnvSpec and recorded in the report config.
    convention_ids: tuple[str, ...] = ()
    heldout_variants: int = 2
    budget: EpisodeBudget = field(default_factory=EpisodeBudget)
    min_free_disk_bytes: int = 2 * 1024**3
    #: Chain-level checkpoint resume: chains with a committed rows.json
    #: are loaded as-is; a chain interrupted mid-flight is wiped and
    #: rerun whole (no mid-chain git-state reconstruction).
    resume: bool = False

    def __post_init__(self) -> None:
        if self.hand_kind not in (HAND_SCRIPTED, HAND_API):
            raise ValueError(f"hand_kind must be scripted|api, got {self.hand_kind!r}")
        if self.hand_kind == HAND_API and self.api_hand_config is None:
            raise ValueError("api hand requires api_hand_config")
        if not (0.0 <= self.band_low < self.band_high <= 1.0):
            raise ValueError("band bounds must satisfy 0 <= low < high <= 1")
        if self.chains < 1 or self.episodes_per_chain < 1:
            raise ValueError("chains and episodes_per_chain must be >= 1")


def _chain_spec(config: CalibrationConfig, chain_index: int) -> EnvSpec:
    return EnvSpec(
        env_seed=config.env_seed + chain_index * 13,
        convention_ids=config.convention_ids,
    )


def _serialize_chain(chain: tuple[ChainTask, ...]) -> str:
    return json.dumps(
        [dataclasses.asdict(task) for task in chain],
        ensure_ascii=False,
        sort_keys=True,
    )


def check_environment_determinism(config: CalibrationConfig) -> dict[str, Any]:
    """Same spec twice => identical tree; different seed => different tree."""

    spec = _chain_spec(config, 0)
    with tempfile.TemporaryDirectory(prefix="coding-lab-det-") as scratch:
        root = pathlib.Path(scratch)
        first = generate_environment(spec, root / "a")
        second = generate_environment(spec, root / "b")
        other = generate_environment(
            EnvSpec(env_seed=spec.env_seed + 1), root / "c"
        )
    chain_a = _serialize_chain(generate_task_chain(spec, chain_seed=0, length=config.episodes_per_chain))
    chain_b = _serialize_chain(generate_task_chain(spec, chain_seed=0, length=config.episodes_per_chain))
    same_tree = first.tree_hash == second.tree_hash
    same_chain = chain_a == chain_b
    distinct_seed_distinct_tree = first.tree_hash != other.tree_hash
    return {
        "tree_hash": first.tree_hash,
        "same_spec_same_tree": same_tree,
        "same_spec_same_chain": same_chain,
        "distinct_seed_distinct_tree": distinct_seed_distinct_tree,
        "environment_deterministic": same_tree and same_chain and distinct_seed_distinct_tree,
    }


def _build_hand(
    config: CalibrationConfig, chain: tuple[ChainTask, ...], chain_index: int
) -> Hand:
    if config.hand_kind == HAND_API:
        assert config.api_hand_config is not None  # guarded in __post_init__
        return OpenAICompatHand(config.api_hand_config)
    # Mix the chain index into the hand seed: chains must draw
    # independent error-mode sequences or the cross-chain variance
    # structure degenerates (all chains failing at identical positions).
    return ScriptedHand(
        tasks_by_id={task.task_id: task for task in chain},
        episode_index_by_task_id={task.task_id: index for index, task in enumerate(chain)},
        hand_seed=config.scripted_hand_seed + 101 * chain_index,
        invariant_sabotage_rate=config.scripted_invariant_sabotage_rate,
        acceptance_sabotage_rate=config.scripted_acceptance_sabotage_rate,
    )


@dataclass(frozen=True)
class EpisodeRow:
    chain_index: int
    episode_index: int
    task_id: str
    category: str
    passed: bool
    acceptance_passed: bool
    regression_passed: bool
    invariant_violations: tuple[str, ...]
    submitted: bool
    tests_tampered: bool
    steps_used: int
    wall_seconds: float
    workspace_bytes: int
    prompt_tokens: int
    completion_tokens: int
    trajectory_sha256: str


def _chain_rows_path(run_dir: pathlib.Path, chain_index: int) -> pathlib.Path:
    return run_dir / "chains" / f"chain-{chain_index:02d}" / "rows.json"


def _load_chain_rows(path: pathlib.Path) -> list[EpisodeRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EpisodeRow(
            **{
                **row,
                "invariant_violations": tuple(row["invariant_violations"]),
            }
        )
        for row in payload
    ]


async def _run_chain(
    config: CalibrationConfig,
    chain_index: int,
    run_dir: pathlib.Path,
) -> list[EpisodeRow]:
    spec = _chain_spec(config, chain_index)
    chain_dir = run_dir / "chains" / f"chain-{chain_index:02d}"
    rows_path = _chain_rows_path(run_dir, chain_index)
    if config.resume and rows_path.is_file():
        return _load_chain_rows(rows_path)
    if config.resume and chain_dir.exists():
        # Interrupted mid-chain: whole-chain rerun keeps the chain repo's
        # merge history deterministic instead of reconstructing git state.
        shutil.rmtree(chain_dir)
    workspace = ChainWorkspace(spec=spec, chain_root=chain_dir)
    workspace.initialize()
    chain = generate_task_chain(spec, chain_seed=chain_index, length=config.episodes_per_chain)
    hand = _build_hand(config, chain, chain_index)
    rows: list[EpisodeRow] = []
    for episode_index, task in enumerate(chain):
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
        tampered = workspace.tests_tampered(handle)
        workspace_bytes, _tree_hash = workspace.finalize_episode(
            handle, passed=result.outcome.passed, task=task
        )
        rows.append(
            EpisodeRow(
                chain_index=chain_index,
                episode_index=episode_index,
                task_id=task.task_id,
                category=task.category,
                passed=result.outcome.passed,
                acceptance_passed=result.outcome.acceptance_passed,
                regression_passed=result.outcome.regression_passed,
                invariant_violations=result.outcome.invariant_violations,
                submitted=result.submitted,
                tests_tampered=tampered,
                steps_used=result.steps_used,
                wall_seconds=result.wall_seconds,
                workspace_bytes=workspace_bytes,
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


def _band_metrics(config: CalibrationConfig, rows: list[EpisodeRow]) -> dict[str, Any]:
    outcomes = [1.0 if row.passed else 0.0 for row in rows]
    pass_rate = statistics.fmean(outcomes) if outcomes else 0.0
    chain_rates = []
    for chain_index in sorted({row.chain_index for row in rows}):
        chain_outcomes = [1.0 if row.passed else 0.0 for row in rows if row.chain_index == chain_index]
        chain_rates.append(statistics.fmean(chain_outcomes))
    variance_present = len(set(outcomes)) > 1
    in_band = config.band_low <= pass_rate <= config.band_high
    return {
        "pass_rate": pass_rate,
        "per_chain_pass_rates": chain_rates,
        "band_low": config.band_low,
        "band_high": config.band_high,
        "in_band": in_band,
        "outcome_variance_present": variance_present,
        "oracle_band_verdict": in_band and variance_present,
    }


def _cost_metrics(rows: list[EpisodeRow]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "episodes": len(rows),
        "mean_wall_seconds": statistics.fmean(row.wall_seconds for row in rows),
        "max_wall_seconds": max(row.wall_seconds for row in rows),
        "mean_workspace_bytes": int(statistics.fmean(row.workspace_bytes for row in rows)),
        "total_prompt_tokens": sum(row.prompt_tokens for row in rows),
        "total_completion_tokens": sum(row.completion_tokens for row in rows),
        "mean_steps_used": statistics.fmean(row.steps_used for row in rows),
        "tests_tampered_episodes": sum(1 for row in rows if row.tests_tampered),
    }


def _preflight_disk(config: CalibrationConfig) -> dict[str, Any]:
    usage = shutil.disk_usage(config.output_root)
    if usage.free < config.min_free_disk_bytes:
        raise RuntimeError(
            f"preflight: only {usage.free} bytes free under {config.output_root!s}, "
            f"need >= {config.min_free_disk_bytes} (C3 disk-death lesson: fail before running)"
        )
    return {"free_bytes": usage.free, "required_bytes": config.min_free_disk_bytes}


async def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    """Run the full Packet 0 calibration and write report artifacts."""

    run_dir = pathlib.Path(config.output_root) / config.run_id
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"calibration run dir already exists: {run_dir!s}")
    run_dir.mkdir(parents=True, exist_ok=config.resume)
    disk = _preflight_disk(config)
    started = time.time()
    determinism = check_environment_determinism(config)
    rows: list[EpisodeRow] = []
    for chain_index in range(config.chains):
        rows.extend(await _run_chain(config, chain_index, run_dir))
    band = _band_metrics(config, rows)
    heldout = seal_heldout_variants(
        base_spec=_chain_spec(config, 0),
        count=config.heldout_variants,
        manifest_path=run_dir / "heldout" / "sealed_variants.json",
    )
    hand_scope = (
        "machinery-only (scripted hand); frozen API hand must re-run before Packet 2 prereg"
        if config.hand_kind == HAND_SCRIPTED
        else "frozen-hand calibration"
    )
    report: dict[str, Any] = {
        "packet": "coding-lab-packet-0",
        "run_id": config.run_id,
        "started_unix": int(started),
        "hand_kind": config.hand_kind,
        "hand_scope": hand_scope,
        "config": {
            "env_seed": config.env_seed,
            "chains": config.chains,
            "episodes_per_chain": config.episodes_per_chain,
            "band": [config.band_low, config.band_high],
            "convention_ids": list(config.convention_ids),
            "scripted_rates": {
                "invariant_sabotage": config.scripted_invariant_sabotage_rate,
                "acceptance_sabotage": config.scripted_acceptance_sabotage_rate,
            },
            "api_model": (
                config.api_hand_config.model if config.api_hand_config is not None else None
            ),
        },
        "preflight_disk": disk,
        "determinism": determinism,
        "oracle_band": band,
        "cost": _cost_metrics(rows),
        "heldout_sealed": [variant.variant_id for variant in heldout],
        "episodes": [dataclasses.asdict(row) for row in rows],
        "verdicts": {
            "environment_deterministic": determinism["environment_deterministic"],
            "oracle_band": band["oracle_band_verdict"],
            "heldout_sealed": len(heldout) >= 1,
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


def _render_markdown(report: dict[str, Any]) -> str:
    verdicts = report["verdicts"]
    band = report["oracle_band"]
    lines = [
        "# coding-lab Packet 0 calibration",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- hand: `{report['hand_kind']}` — {report['hand_scope']}",
        f"- environment_deterministic: **{verdicts['environment_deterministic']}**",
        (
            f"- oracle_band: **{verdicts['oracle_band']}** "
            f"(pass_rate={band['pass_rate']:.3f} in [{band['band_low']}, {band['band_high']}], "
            f"variance_present={band['outcome_variance_present']})"
        ),
        f"- per-chain pass rates: {band['per_chain_pass_rates']}",
        f"- heldout sealed: {report['heldout_sealed']}",
        f"- episodes: {report['cost'].get('episodes', 0)}, "
        f"mean wall {report['cost'].get('mean_wall_seconds', 0.0):.2f}s, "
        f"mean bytes {report['cost'].get('mean_workspace_bytes', 0)}",
        "",
        "Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "HAND_API",
    "HAND_SCRIPTED",
    "CalibrationConfig",
    "EpisodeRow",
    "check_environment_determinism",
    "run_calibration",
]
