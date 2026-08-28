"""Packet 1 observer mechanics tests.

Covers the SHADOW observer's contract obligations:

* replay drives the brain only through facade channels and produces one
  observation per logged episode;
* bet-then-settle ordering is enforced (pre-oracle bet strictly before
  outcome submission);
* the terminal outcome reaches ``ActualOutcome`` with provenance
  (``external_outcome_refs``) and moves settled task progress in the
  right direction;
* scoped persistence survives a simulated process restart.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

import pytest

from lifeform_domain_coding.lab.calibration import CalibrationConfig, run_calibration
from lifeform_evolution.coding_lab_observer import (
    _ORDERING_CLOCK_NAME,
    _ordering_clock,
    EpisodeObservation,
    ObserverBet,
    observe_calibration_chain,
    recovered_memory_entry_count,
)


@pytest.fixture(scope="module")
def calibration_run(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    output_root = tmp_path_factory.mktemp("coding-lab-observer") / "artifacts"
    import asyncio

    config = CalibrationConfig(
        run_id="observer-fixture",
        output_root=output_root,
        chains=1,
        episodes_per_chain=3,
        scripted_invariant_sabotage_rate=0.0,
        scripted_acceptance_sabotage_rate=1.0,
        heldout_variants=1,
        min_free_disk_bytes=1,
    )
    asyncio.run(run_calibration(config))
    return output_root / "observer-fixture"


async def test_observer_replays_all_episodes_and_settles(
    calibration_run: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    result = await observe_calibration_chain(
        chain_id="chain-00",
        trajectories_dir=calibration_run / "chains" / "chain-00" / "trajectories",
        brain_state_root=tmp_path / "brain",
    )
    assert len(result.observations) == 3
    for observation in result.observations:
        # Bet-then-settle ordering is a dataclass invariant; reaching here
        # means it held. Terminal outcome must be attributed.
        assert observation.external_outcome_refs, "episode outcome must reach ActualOutcome"
        assert observation.bet_pre_oracle.monotonic_seconds < observation.outcome_submitted_monotonic
    # This fixture uses acceptance_sabotage_rate=1.0: every episode fails,
    # and settled task progress must sit clearly in the failure band
    # (TASK_REGRESSED bias is -0.50 on the task axis).
    assert all(not observation.passed for observation in result.observations)
    assert all(observation.settled_task_progress < 0.4 for observation in result.observations)
    assert result.persisted
    assert result.memory_entry_count_before_restart > 0


async def test_observer_recovery_after_restart(
    calibration_run: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    root = tmp_path / "brain"
    result = await observe_calibration_chain(
        chain_id="chain-00",
        trajectories_dir=calibration_run / "chains" / "chain-00" / "trajectories",
        brain_state_root=root,
    )
    recovered = recovered_memory_entry_count(chain_id="chain-00", brain_state_root=root)
    assert recovered > 0
    assert result.memory_entry_count_before_restart > 0


def test_bet_then_settle_ordering_is_enforced() -> None:
    bet = ObserverBet(
        turn_index=1,
        predicted_task_progress=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        prediction_id="p-1",
        monotonic_seconds=100.0,
    )
    template = EpisodeObservation(
        chain_id="c",
        episode_index=0,
        task_id="t",
        category="add_helper",
        passed=True,
        acceptance_passed=True,
        regression_passed=True,
        invariant_violations=(),
        bet_at_task_presented=bet,
        bet_pre_oracle=bet,
        outcome_submitted_monotonic=101.0,
        settled_task_progress=0.9,
        settled_signed_reward=0.2,
        settled_task_error=0.1,
        settled_magnitude=0.1,
        external_outcome_refs=("e-1",),
        turns_used=3,
    )
    with pytest.raises(ValueError, match="bet-then-settle"):
        dataclasses.replace(template, outcome_submitted_monotonic=99.0)
    # A sub-millisecond gap is the realistic spacing between the two sites and
    # must be accepted; see the resolution test below for why that matters.
    dataclasses.replace(template, outcome_submitted_monotonic=100.0 + 1e-6)


def test_ordering_clock_is_monotonic_and_sub_millisecond() -> None:
    """The bet-then-settle guard needs a finer clock than ``time.monotonic()``.

    Recording the pre-oracle bet and submitting the outcome are separated only
    by in-memory snapshot propagation. ``time.monotonic()`` resolves to ~15.6 ms
    on Windows (GetTickCount64), so both stamps landed in the same tick whenever
    the brain session settled quickly and the strict-ordering guard rejected a
    perfectly well-ordered episode. Guard the clock contract, not the platform.
    """

    info = time.get_clock_info(_ORDERING_CLOCK_NAME)
    assert info.monotonic, f"{_ORDERING_CLOCK_NAME} must be monotonic"
    assert info.resolution < 1e-3, (
        f"{_ORDERING_CLOCK_NAME} resolves to {info.resolution}s, too coarse to order "
        "two adjacent in-memory operations"
    )
    assert _ordering_clock() <= _ordering_clock()
