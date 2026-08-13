"""Coding-lab Packet 2 arms: brain digest vs long-context steelman.

Three arms run the SAME chains (same env seeds, same task sequences,
same hand construction) and differ ONLY in the context preamble handed
to the coder:

* ``brain``     — owner-generated digest assembled from the SHADOW
  observer brain's published snapshot descriptions and memory summaries
  (SSOT: the harness never rebuilds owner internals, it quotes the
  descriptions owners publish), bounded by a hard char budget;
* ``steelman``  — the FULL transcript of every previous episode in the
  chain (the "dump everything into context" baseline that the quality
  gate must beat with <= 10% of the tokens);
* ``stateless`` — empty preamble (qualification arm only, per the MSC
  precedent: beating stateless proves nothing by itself).

Dual gate (per the frozen plan):

* quality  — chain-paired slope difference (brain - steelman) of the
  episode pass indicator over episode index, chain-clustered bootstrap;
* scaling  — context budget asymmetry is EVIDENCE, not matched: brain
  context tokens / steelman context tokens must stay <= 0.10.
"""

from __future__ import annotations

import pathlib
import statistics
from dataclasses import dataclass
from random import Random
from typing import Any

from lifeform_domain_coding.lab.episode import EpisodeBudget, run_episode
from lifeform_domain_coding.lab.generation import EnvSpec
from lifeform_domain_coding.lab.hands import Hand
from lifeform_domain_coding.lab.tasks import ChainTask, generate_task_chain
from lifeform_domain_coding.lab.trajectory import read_trajectory
from lifeform_domain_coding.lab.workspace import ChainWorkspace

from lifeform_evolution.coding_lab_observer import CodingLabChainObserver

ARM_BRAIN = "brain"
ARM_STEELMAN = "steelman"
ARM_STATELESS = "stateless"
ALL_ARMS: tuple[str, ...] = (ARM_BRAIN, ARM_STEELMAN, ARM_STATELESS)

_DIGEST_SLOTS: tuple[str, ...] = (
    "memory",
    "prediction_error",
    "plan_intent",
    "open_loop",
    "belief_assumption",
    "execution_result",
    "user_model",
)


def approx_tokens(text: str) -> int:
    """Chars/4 approximation, used symmetrically across arms.

    The formal (API-hand) run replaces this with provider-reported
    usage; the approximation only ever feeds the smoke's ratio check.
    """

    return max(0, len(text) // 4)


def recall_for_task(
    observer: CodingLabChainObserver, *, task: ChainTask
) -> tuple[Any, ...]:
    """Recall persisted experience keyed on MINIMAL task metadata.

    The hint and facets deliberately contain only harness-known metadata
    (category + target files), never the task description: otherwise the
    upcoming task's own text would flow into the pack and any needle
    check would be self-referential (leak, not memory).

    Goes through ``CodingLabChainObserver.recall_experience`` (the memory
    owner's official ``retrieve`` contract, EPISODIC + DURABLE only) —
    see its docstring for why a conversational recall turn cannot work.
    """

    return observer.recall_experience(
        hint=(
            "Prepare relevant experience for the next task: "
            f"category={task.category} files={','.join(task.target_files)}"
        ),
        facets=("coding-lab", f"category:{task.category}"),
        limit=8,
    )


def build_brain_context_pack(
    observer: CodingLabChainObserver,
    *,
    recalled: tuple[Any, ...],
    char_budget: int,
) -> str:
    """Assemble the injection pack from owner-published descriptions.

    SSOT discipline: every line quotes an owner-generated surface —
    recalled entry ``content`` strings (owner-stored), snapshot
    ``description`` fields, and the memory owner's stratum summaries.
    The harness adds only labels, ordering, and truncation. Recalled
    experience leads the pack so budget truncation drops the generic
    owner digests first, never the task-relevant experience.
    """

    if char_budget < 200:
        raise ValueError("char_budget must be >= 200")
    session = observer.session
    snapshots = session.runner.upstream_snapshots
    parts: list[str] = []
    for entry in recalled[:12]:
        parts.append(f"[memory:entry:{entry.stratum}] {entry.content}")
    memory_snapshot = snapshots.get("memory")
    if memory_snapshot is not None:
        value = memory_snapshot.value
        parts.append(f"[memory] {value.description}")
        for label, summary in (
            ("durable", value.durable_summary),
            ("episodic", value.episodic_summary),
        ):
            if summary:
                parts.append(f"[memory:{label}] {summary}")
    for slot in _DIGEST_SLOTS:
        if slot == "memory":
            continue
        published = snapshots.get(slot)
        if published is None:
            continue
        description = str(published.value.description)
        if description:
            parts.append(f"[{slot}] {description}")
    pack = "\n".join(parts)
    if len(pack) > char_budget:
        pack = pack[:char_budget] + "\n... [digest truncated at budget]"
    return pack


def render_episode_transcript(trajectory_path: pathlib.Path) -> str:
    """Render one logged episode as plain text (steelman accumulation)."""

    lines: list[str] = []
    for event in read_trajectory(trajectory_path):
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "task_presented":
            lines.append(f"=== task {payload['task_id']} ({payload['category']}) ===")
            lines.append(str(payload["description"]))
        elif event_type == "hand_decision":
            if payload.get("kind") == "submit":
                lines.append(f"[hand] submit: {str(payload.get('note', ''))[:200]}")
            else:
                lines.append(
                    f"[hand] {payload.get('tool_name')} {payload.get('parameters')}"
                )
        elif event_type == "tool_result":
            result = payload.get("result", {})
            lines.append(
                f"[tool:{payload.get('tool_name')}] succeeded={payload.get('succeeded')} "
                f"{str(result)[:600]}"
            )
        elif event_type == "oracle_outcome":
            lines.append(
                f"[oracle] passed={payload['passed']} acceptance={payload['acceptance_passed']} "
                f"regression={payload['regression_passed']} "
                f"invariant_violations={payload.get('invariant_violations', [])}"
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class ArmEpisodeRow:
    arm: str
    chain_index: int
    episode_index: int
    task_id: str
    category: str
    passed: bool
    context_chars: int
    context_tokens_approx: int
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    invariant_violations: tuple[str, ...]


@dataclass(frozen=True)
class ArmChainConfig:
    env_seed: int
    chain_index: int
    episodes: int
    brain_digest_char_budget: int
    budget: EpisodeBudget


async def run_chain_arm(
    *,
    arm: str,
    config: ArmChainConfig,
    arm_root: pathlib.Path,
    hand_factory: Any,
) -> tuple[ArmEpisodeRow, ...]:
    """Run one arm over one chain.

    ``hand_factory(chain: tuple[ChainTask, ...], chain_index: int) -> Hand``
    must construct identical hands across arms (the hand is frozen; only
    the context differs).
    """

    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    spec = EnvSpec(env_seed=config.env_seed + config.chain_index * 13)
    chain = generate_task_chain(
        spec, chain_seed=config.chain_index, length=config.episodes
    )
    workspace = ChainWorkspace(spec=spec, chain_root=arm_root / "workspace")
    workspace.initialize()
    hand: Hand = hand_factory(chain, config.chain_index)
    observer: CodingLabChainObserver | None = None
    if arm == ARM_BRAIN:
        observer = CodingLabChainObserver(
            chain_id=f"arm-brain-{config.chain_index:02d}",
            brain_state_root=arm_root / "brain_state",
        )
    steelman_transcripts: list[str] = []
    rows: list[ArmEpisodeRow] = []
    for episode_index, task in enumerate(chain):
        if arm == ARM_BRAIN:
            assert observer is not None
            recalled = recall_for_task(observer, task=task)
            preamble = build_brain_context_pack(
                observer,
                recalled=recalled,
                char_budget=config.brain_digest_char_budget,
            )
        elif arm == ARM_STEELMAN:
            preamble = "\n\n".join(steelman_transcripts)
        else:
            preamble = ""
        handle = workspace.begin_episode(episode_index, task)
        trajectory_path = arm_root / "trajectories" / f"episode-{episode_index:03d}.jsonl"
        result = await run_episode(
            spec=spec,
            task=task,
            episode_index=episode_index,
            worktree=handle.worktree,
            hand=hand,
            trajectory_path=trajectory_path,
            budget=config.budget,
            context_preamble=preamble,
        )
        workspace.finalize_episode(handle, passed=result.outcome.passed, task=task)
        if arm == ARM_BRAIN:
            assert observer is not None
            await observer.observe_episode(
                episode_index=episode_index, trajectory_path=trajectory_path
            )
        elif arm == ARM_STEELMAN:
            steelman_transcripts.append(render_episode_transcript(trajectory_path))
        rows.append(
            ArmEpisodeRow(
                arm=arm,
                chain_index=config.chain_index,
                episode_index=episode_index,
                task_id=task.task_id,
                category=task.category,
                passed=result.outcome.passed,
                context_chars=len(preamble),
                context_tokens_approx=approx_tokens(preamble),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_seconds=result.wall_seconds,
                invariant_violations=result.outcome.invariant_violations,
            )
        )
    if observer is not None:
        observer.persist()
    return tuple(rows)


# ---------------------------------------------------------------------------
# Dual-gate statistics
# ---------------------------------------------------------------------------


def _ols_slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return numerator / denominator


def chain_slopes(rows: tuple[ArmEpisodeRow, ...]) -> dict[tuple[str, int], float]:
    """Per (arm, chain) OLS slope of the pass indicator over episode index."""

    grouped: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault((row.arm, row.chain_index), []).append(
            (float(row.episode_index), 1.0 if row.passed else 0.0)
        )
    return {key: _ols_slope(points) for key, points in grouped.items()}


def paired_slope_gap(
    rows: tuple[ArmEpisodeRow, ...],
    *,
    arm_a: str,
    arm_b: str,
    bootstrap_samples: int = 2_000,
    seed: int = 7,
) -> dict[str, Any]:
    """Chain-paired slope difference (arm_a - arm_b) with chain bootstrap."""

    slopes = chain_slopes(rows)
    chain_indexes = sorted({chain for arm, chain in slopes if arm == arm_a})
    gaps = []
    for chain_index in chain_indexes:
        if (arm_b, chain_index) not in slopes:
            raise ValueError(f"arm {arm_b!r} missing chain {chain_index} (paired design broken)")
        gaps.append(slopes[(arm_a, chain_index)] - slopes[(arm_b, chain_index)])
    if not gaps:
        raise ValueError("no paired chains found")
    rng = Random(seed)
    resampled_means = []
    for _ in range(bootstrap_samples):
        sample = [gaps[rng.randrange(len(gaps))] for _ in gaps]
        resampled_means.append(statistics.fmean(sample))
    resampled_means.sort()
    lower_index = max(0, int(0.05 * len(resampled_means)) - 1)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "per_chain_gaps": gaps,
        "mean_gap": statistics.fmean(gaps),
        "bootstrap_ci_lower_5pct": resampled_means[lower_index],
        "bootstrap_samples": bootstrap_samples,
    }


def scaling_gate(
    rows: tuple[ArmEpisodeRow, ...],
    *,
    max_token_ratio: float = 0.10,
) -> dict[str, Any]:
    """Context-budget asymmetry check: brain tokens << steelman tokens.

    Episode 0 is excluded on both sides — neither arm has accumulated
    context yet, and a shared zero would spuriously deflate the ratio.
    """

    brain_tokens = [
        row.context_tokens_approx
        for row in rows
        if row.arm == ARM_BRAIN and row.episode_index > 0
    ]
    steelman_tokens = [
        row.context_tokens_approx
        for row in rows
        if row.arm == ARM_STEELMAN and row.episode_index > 0
    ]
    if not brain_tokens or not steelman_tokens:
        raise ValueError("scaling gate requires post-first-episode rows in both arms")
    brain_mean = statistics.fmean(brain_tokens)
    steelman_mean = statistics.fmean(steelman_tokens)
    ratio = brain_mean / steelman_mean if steelman_mean > 0 else float("inf")
    return {
        "brain_mean_context_tokens": brain_mean,
        "steelman_mean_context_tokens": steelman_mean,
        "token_ratio": ratio,
        "max_token_ratio": max_token_ratio,
        "passed": ratio <= max_token_ratio,
    }


def scaling_structure(rows: tuple[ArmEpisodeRow, ...]) -> dict[str, Any]:
    """Structural scaling teeth for toy-scale smokes.

    The production threshold (mean token ratio <= 0.10) only separates
    at realistic chain lengths and transcript sizes; at smoke scale the
    honest assertions are directional: the brain digest stays bounded
    while the steelman context grows, so the per-episode ratio falls.
    """

    def _sizes(arm: str) -> list[tuple[int, int]]:
        return sorted(
            (row.episode_index, row.context_tokens_approx)
            for row in rows
            if row.arm == arm and row.episode_index > 0
        )

    brain_sizes = _sizes(ARM_BRAIN)
    steelman_sizes = _sizes(ARM_STEELMAN)
    if len(brain_sizes) < 2 or len(steelman_sizes) < 2:
        raise ValueError("scaling structure needs >= 2 post-first episodes per arm")
    brain_slope = _ols_slope([(float(x), float(y)) for x, y in brain_sizes])
    steelman_slope = _ols_slope([(float(x), float(y)) for x, y in steelman_sizes])
    first_ratio = brain_sizes[0][1] / max(1, steelman_sizes[0][1])
    last_ratio = brain_sizes[-1][1] / max(1, steelman_sizes[-1][1])
    return {
        "brain_context_slope_tokens_per_episode": brain_slope,
        "steelman_context_slope_tokens_per_episode": steelman_slope,
        "first_episode_ratio": first_ratio,
        "last_episode_ratio": last_ratio,
        "steelman_grows": steelman_slope > 0,
        "brain_bounded": abs(brain_slope) < 0.25 * max(steelman_slope, 1e-9),
        "ratio_decreasing": last_ratio < first_ratio,
        "passed": (
            steelman_slope > 0
            and abs(brain_slope) < 0.25 * max(steelman_slope, 1e-9)
            and last_ratio < first_ratio
        ),
    }


def wall_seconds_by_arm(rows: tuple[ArmEpisodeRow, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for arm in ALL_ARMS:
        arm_rows = [row.wall_seconds for row in rows if row.arm == arm]
        if arm_rows:
            result[arm] = statistics.fmean(arm_rows)
    return result


__all__ = [
    "ALL_ARMS",
    "ARM_BRAIN",
    "ARM_STATELESS",
    "ARM_STEELMAN",
    "ArmChainConfig",
    "ArmEpisodeRow",
    "approx_tokens",
    "build_brain_context_pack",
    "chain_slopes",
    "paired_slope_gap",
    "recall_for_task",
    "render_episode_transcript",
    "run_chain_arm",
    "scaling_gate",
    "scaling_structure",
    "wall_seconds_by_arm",
]
