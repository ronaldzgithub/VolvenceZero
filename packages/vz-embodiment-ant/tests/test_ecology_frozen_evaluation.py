"""P0-C frozen-evaluation owner audit tests (plan 05 s3.4).

The committed v1 artifact reported ``frozen_evaluation: passed`` while its own
payload recorded 24 per-tick learned-owner changes under
``learning_enabled=False``: the gate only compared two of the eight published
owners, and it compared them endpoint-to-endpoint instead of per tick.  These
tests lock the explicit allow-list, the per-tick comparison and the
first-difference BLOCK reason.
"""

from __future__ import annotations

import pytest

from volvence_ant.experiments.ecology_curriculum import (
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyDataSplit,
    EcologyTrainingTier,
    _session_config,
    _world,
)
from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS,
    ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES,
    ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS,
    ECOLOGY_AUDIT_GATED_LEARNING_OWNERS,
    ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR,
    EcologyMechanismAuditConfig,
    EcologyMechanismAuditError,
    _classify_owner_names,
    _curriculum_config,
    _frozen_evaluation_audit,
)
from volvence_ant.runtime import KernelColonyRunner


_KERNEL_PUBLISHED_LEARNING_OWNERS = (
    "credit",
    "dual-track-gate",
    "joint-loop/memory",
    "joint-loop/policy",
    "joint-loop/temporal-learning",
    "prediction",
    "reflection",
    "regime",
)


def _config() -> EcologyMechanismAuditConfig:
    return EcologyMechanismAuditConfig(
        n_ants=1,
        temporal_latent_dim=4,
        episode_rounds=1,
        episodes_per_stage=1,
        evaluation_rounds=3,
        seed=5,
    )


def test_every_gated_owner_carries_a_written_justification() -> None:
    names = tuple(name for name, _ in ECOLOGY_AUDIT_GATED_LEARNING_OWNERS)

    assert tuple(sorted(names)) == _KERNEL_PUBLISHED_LEARNING_OWNERS
    assert len(set(names)) == len(names)
    assert all(
        len(justification.strip()) > 40
        for _name, justification in ECOLOGY_AUDIT_GATED_LEARNING_OWNERS
    )
    # An allow-list entry is a licence for a learned owner to move during a
    # frozen evaluation. It stays empty, and any future entry must arrive with
    # its own justification string.
    assert ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS == ()


def test_owner_classification_rejects_drift_in_either_direction() -> None:
    published = set(_KERNEL_PUBLISHED_LEARNING_OWNERS)

    _classify_owner_names(published)

    with pytest.raises(EcologyMechanismAuditError) as unknown:
        _classify_owner_names(published | {"joint-loop/brand-new-owner"})
    assert "not classified" in str(unknown.value) or "has not" in str(
        unknown.value
    )

    with pytest.raises(EcologyMechanismAuditError) as missing:
        _classify_owner_names(published - {"credit"})
    assert "no longer publishes" in str(missing.value)


def test_minimal_repro_cases_are_the_prescribed_ones() -> None:
    """plan 05:186-188 -- butter_only/seed=307 and heat_forced_escape/101."""

    assert ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES == (
        (EcologyEvaluationScenario.BUTTER_ONLY, 307),
        (EcologyEvaluationScenario.HEAT_FORCED_ESCAPE, 101),
    )
    assert ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS == (101, 307)
    assert _curriculum_config(_config()).heldout_seeds == (101, 307)


def test_audit_seed_may_not_collide_with_a_frozen_repro_seed() -> None:
    with pytest.raises(ValueError) as excinfo:
        EcologyMechanismAuditConfig(seed=307 - 43)
    assert "held-out" in str(excinfo.value)


def test_replay_coverage_floor_is_frozen() -> None:
    assert ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR == 0.99


async def test_frozen_evaluation_compares_every_owner_per_tick() -> None:
    """A learned-owner change anywhere, on any tick, must BLOCK.

    This is the honest outcome on the current kernel: six of the eight
    published learning owners move while ``learning_enabled=False``.
    """

    config = _config()
    curriculum_config = _curriculum_config(config)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum_config,
            stage=EcologyStage.BUTTER,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum_config,
            seed=config.seed,
            session_id="test:p0c:shared-initial",
            optimize=True,
        ),
    )
    checkpoints = bootstrap.export_learning_checkpoints(
        checkpoint_prefix="test:p0c:shared-initial",
        include_runtime_replay=False,
    )
    scenario, seed = ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES[0]

    audit = await _frozen_evaluation_audit(
        audit_config=config,
        curriculum_config=curriculum_config,
        checkpoints=checkpoints,
        scenario=scenario,
        seed=seed,
    )

    assert audit.scenario == "butter_only"
    assert audit.seed == 307
    assert audit.gated_owner_names
    assert audit.allowed_owner_names == ()
    # policy and temporal-learning stay frozen -- that is the ONLY thing the
    # v1/v2 gate ever checked.
    assert audit.policy_stable
    assert audit.temporal_learning_stable
    # ...and the other published learning owners do not, which the new gate
    # now refuses to hide.
    assert audit.unstable_owner_names
    assert set(audit.unstable_owner_names) <= set(
        _KERNEL_PUBLISHED_LEARNING_OWNERS
    )
    assert not audit.passed
    assert audit.first_differences
    head = audit.first_differences[0]
    assert head.owner_name in audit.unstable_owner_names
    assert head.field_name == (
        f"learning_owner_fingerprints[{head.owner_name}]"
    )
    assert 0 <= head.tick < config.evaluation_rounds
    assert head.before_fingerprint != head.after_fingerprint
    assert head.owner_name in audit.block_reason
    assert f"tick={head.tick}" in audit.block_reason
