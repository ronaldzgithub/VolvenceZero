"""Production wiring of identity_seed through LifeformConfig (packet 1.3''').

Closes the production wiring path that ties packet 1.3'' end-to-end:

* ``LifeformConfig.with_identity_seed(seed)`` attaches a seed to the
  config.
* ``Lifeform`` forwards the seed to ``Brain``.
* ``Brain`` forwards to ``AgentSessionRunner``.
* ``AgentSessionRunner`` passes ``identity_seed=`` to
  ``run_final_wiring_turn`` each turn.
* ``DualTrackModule`` populates ``self_track.traits`` from the seed.
* The protocol identity gate's self-trait branch fires real subset /
  forbidden checks (instead of ``self_traits_populator_pending``).

Also asserts that the growth-advisor lifeform builder auto-wires
the identity seed by default — so users who call
``build_cheng_laoshi_lifeform()`` get the full self-trait machinery
without manual wiring.
"""

from __future__ import annotations

from lifeform_core import LifeformConfig
from volvence_zero.identity_seed import IdentitySeed


# ---------------------------------------------------------------------------
# LifeformConfig.with_identity_seed
# ---------------------------------------------------------------------------


def test_lifeform_config_default_has_no_identity_seed() -> None:
    config = LifeformConfig()
    assert config.identity_seed is None


def test_lifeform_config_with_identity_seed_attaches() -> None:
    seed = IdentitySeed(traits=("warm_peer_register",))
    config = LifeformConfig().with_identity_seed(seed)
    assert config.identity_seed is seed


def test_lifeform_config_with_identity_seed_is_immutable_replace() -> None:
    """``with_*`` returns a clone (frozen dataclass; ``dataclasses.replace``).
    Original config is unaffected.
    """

    seed = IdentitySeed(traits=("warm",))
    original = LifeformConfig()
    new = original.with_identity_seed(seed)
    assert original.identity_seed is None
    assert new.identity_seed is seed
    assert original is not new


def test_lifeform_config_with_identity_seed_none_drops_seed() -> None:
    seed = IdentitySeed(traits=("warm",))
    config = LifeformConfig().with_identity_seed(seed)
    config = config.with_identity_seed(None)
    assert config.identity_seed is None


# ---------------------------------------------------------------------------
# Lifeform → Brain forwarding
# ---------------------------------------------------------------------------


def test_lifeform_forwards_seed_to_brain() -> None:
    from lifeform_core import Lifeform

    seed = IdentitySeed(traits=("warm_peer_register", "long_horizon"))
    lifeform = Lifeform(LifeformConfig().with_identity_seed(seed))
    assert lifeform.brain.identity_seed is seed


def test_lifeform_without_seed_brain_seed_is_none() -> None:
    from lifeform_core import Lifeform

    lifeform = Lifeform(LifeformConfig())
    assert lifeform.brain.identity_seed is None


def test_brain_with_identity_seed_clone_helper() -> None:
    from volvence_zero.brain import Brain

    seed = IdentitySeed(traits=("warm",))
    brain = Brain().with_identity_seed(seed)
    assert brain.identity_seed is seed

    cleared = brain.with_identity_seed(None)
    assert cleared.identity_seed is None


# ---------------------------------------------------------------------------
# Growth-advisor builder auto-wires the seed
# ---------------------------------------------------------------------------


def test_build_cheng_laoshi_lifeform_default_auto_wires_identity_seed() -> None:
    from lifeform_domain_growth_advisor import build_cheng_laoshi_lifeform

    bundle = build_cheng_laoshi_lifeform()
    seed = bundle.lifeform.brain.identity_seed
    assert seed is not None
    assert "warm_peer_register" in seed.traits
    assert "long_horizon" in seed.traits


def test_build_cheng_laoshi_lifeform_with_identity_seed_disabled() -> None:
    """Ablation knob: set ``use_identity_seed=False`` to drop the
    seed for evaluation harnesses comparing baseline vs. identity-
    gated behaviour. Mirrors ``use_vitals_bootstrap`` knob.
    """

    from lifeform_domain_growth_advisor import build_cheng_laoshi_lifeform

    bundle = build_cheng_laoshi_lifeform(use_identity_seed=False)
    assert bundle.lifeform.brain.identity_seed is None


def test_build_growth_advisor_identity_seed_traits_match_protocol_assertion() -> None:
    """Sanity: the auto-wired seed's traits exactly match the
    cheng_laoshi BehaviorProtocol's IdentityAssertion. This is what
    makes the identity gate's self_traits_required_match branch
    fire end-to-end.
    """

    from lifeform_domain_growth_advisor import (
        build_cheng_laoshi_profile,
        build_growth_advisor_identity_seed,
        growth_advisor_profile_to_behavior_protocol,
    )

    profile = build_cheng_laoshi_profile()
    seed = build_growth_advisor_identity_seed(profile)
    bp = growth_advisor_profile_to_behavior_protocol(profile)

    required = set(bp.identity_assertion.requires_self_traits)
    present = set(seed.traits)
    assert required.issubset(present), (
        f"seed traits {present} do not cover the protocol's "
        f"requires_self_traits {required}; identity gate would "
        f"filter cheng_laoshi out of its own active mixture."
    )
