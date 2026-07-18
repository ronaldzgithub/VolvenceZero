# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Batch generation pipeline: public scenarios -> labelled trajectories.

Two modes:

* ``fsm`` — fully deterministic, zero LLM cost. User turns come from the
  bench's :class:`DeterministicFakeUtteranceClient`; assistant turns come
  from the in-package deterministic fake SUT. Structure (FSM probe
  placement, gaps, labels) is exact ground truth; surface text is
  synthetic filler.
* ``llm`` — user turns from the bench's ``OpenAIUtteranceClient``;
  assistant turns from a real OpenAI-compatible SUT endpoint (same
  procurement conventions as a Companion Bench run).

Held-out exclusion is structural: scenarios are loaded exclusively via
:func:`companion_bench.spec.load_scenarios_dir` with
``include_held_out=False``, from the public scenarios directory only. This
package never imports ``companion_bench.heldout_loader`` (guarded by
``tests/contracts/test_companion_trajgen_boundaries.py``).

Train/val split is by whole scenario family: every family is assigned to
exactly one split, so paraphrases and sibling scenarios of a validation
family can never leak into training.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from importlib.resources import files
from typing import Mapping

from companion_bench.arc_runner import ArcRunConfig, run_arc
from companion_bench.spec import ScenarioSpec, load_scenarios_dir
from companion_bench.sut_client import SUTClient, SUTResponse
from companion_bench.user_simulator import (
    DeterministicFakeUtteranceClient,
    UtteranceClient,
)

from companion_standard import InteractionTrajectory, TrajectorySource
from companion_trajgen.exporter import (
    arc_record_to_trajectory,
    trajectory_manifest_entry,
    write_manifest,
    write_trajectory,
)

DEFAULT_VAL_FAMILIES: tuple[str, ...] = ("F5", "F6")


class DeterministicFakeSUTClient:
    """Hash-derived assistant replies; no network, byte-deterministic.

    Mirrors the bench's fake-utterance philosophy: enough surface shape to
    exercise the exporter / labeler / conformance pipeline without
    spending inference money. NOT a model — fsm-mode text is filler by
    design and documented as such in the dataset manifest.
    """

    def __init__(self, *, model_id: str = "trajgen-fake-sut") -> None:
        self._model_id = model_id

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> SUTResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        digest = hashlib.sha256(f"{session_id}|{last_user}".encode("utf-8")).hexdigest()
        openers = (
            "I hear you.",
            "That sounds like a lot to carry.",
            "Thanks for telling me.",
            "Let's stay with that for a moment.",
            "I remember you mentioned this before.",
            "That matters.",
        )
        opener = openers[int(digest[:6], 16) % len(openers)]
        return SUTResponse(
            text=f"{opener} (deterministic fsm-mode reply {digest[:8]})",
            model_id=self._model_id,
            response_headers={},
            usage_prompt_tokens=None,
            usage_completion_tokens=None,
            raw={},
        )


def public_scenarios_dir() -> pathlib.Path:
    """The bench's packaged public scenarios directory."""
    return pathlib.Path(str(files("companion_bench").joinpath("scenarios/public")))


def load_public_scenarios() -> tuple[ScenarioSpec, ...]:
    """Load the public scenario set. Held-out is excluded structurally."""
    specs = load_scenarios_dir(public_scenarios_dir(), include_held_out=False)
    if not specs:
        raise FileNotFoundError(
            f"no public scenarios found under {public_scenarios_dir()}"
        )
    return specs


def split_for_family(family: str, val_families: tuple[str, ...]) -> str:
    return "val" if family in val_families else "train"


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    trajectories: tuple[InteractionTrajectory, ...]
    manifest_path: pathlib.Path
    train_count: int
    val_count: int


def generate_dataset(
    *,
    out_dir: pathlib.Path | str,
    mode: str,
    seeds_per_scenario: int | None = None,
    val_families: tuple[str, ...] = DEFAULT_VAL_FAMILIES,
    scenarios: tuple[ScenarioSpec, ...] | None = None,
    sut_client: SUTClient | None = None,
    user_backend: UtteranceClient | None = None,
    submission_id: str = "companion-trajgen",
    user_simulator_model: str = "deterministic-fake",
) -> GenerationResult:
    """Run the batch and write trajectories + manifest under ``out_dir``.

    ``mode`` selects the trajectory ``source`` tag and, when no explicit
    clients are given, the default deterministic clients (fsm mode only —
    llm mode requires explicit clients from the CLI wiring).
    """
    if mode not in ("fsm", "llm"):
        raise ValueError(f"mode must be 'fsm' or 'llm', got {mode!r}")
    source = (
        TrajectorySource.SYNTHETIC_FSM if mode == "fsm" else TrajectorySource.SYNTHETIC_LLM
    )
    if mode == "fsm":
        sut_client = sut_client or DeterministicFakeSUTClient()
        user_backend = user_backend or DeterministicFakeUtteranceClient()
    else:
        if sut_client is None or user_backend is None:
            raise ValueError(
                "llm mode requires an explicit sut_client and user_backend "
                "(see companion_trajgen.cli for the OpenAI-compatible wiring)"
            )

    specs = scenarios if scenarios is not None else load_public_scenarios()
    out_directory = pathlib.Path(out_dir)

    trajectories: list[InteractionTrajectory] = []
    manifest: list[dict] = []
    train_count = 0
    val_count = 0
    for spec in specs:
        seed_count = (
            seeds_per_scenario
            if seeds_per_scenario is not None
            else spec.paraphrase_seed_count
        )
        seed_count = min(seed_count, spec.paraphrase_seed_count)
        for seed in range(seed_count):
            record = run_arc(
                spec=spec,
                paraphrase_seed=seed,
                sut_client=sut_client,
                user_backend=user_backend,
                config=ArcRunConfig(
                    submission_id=submission_id,
                    user_simulator_model=user_simulator_model,
                ),
            )
            trajectory = arc_record_to_trajectory(
                record=record, spec=spec, source=source
            )
            split = split_for_family(trajectory.family, val_families)
            write_trajectory(trajectory, out_directory / split)
            entry = trajectory_manifest_entry(trajectory)
            entry["split"] = split
            entry["mode"] = mode
            manifest.append(entry)
            trajectories.append(trajectory)
            if split == "train":
                train_count += 1
            else:
                val_count += 1

    manifest_path = write_manifest(manifest, out_directory)
    return GenerationResult(
        trajectories=tuple(trajectories),
        manifest_path=manifest_path,
        train_count=train_count,
        val_count=val_count,
    )


def summarize_result(result: GenerationResult) -> Mapping[str, int]:
    return {
        "total": len(result.trajectories),
        "train": result.train_count,
        "val": result.val_count,
    }


__all__ = [
    "DEFAULT_VAL_FAMILIES",
    "DeterministicFakeSUTClient",
    "GenerationResult",
    "generate_dataset",
    "load_public_scenarios",
    "public_scenarios_dir",
    "split_for_family",
    "summarize_result",
]
