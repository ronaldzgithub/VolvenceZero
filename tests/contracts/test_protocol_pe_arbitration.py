"""PE-driven arbitration (packet 1.5c-i): replace lex tiebreak with pe_utility.

Asserts the contract for ``_resolve_co_activation_incompatibility``
and its integration into ``compute_active_mixture``:

* **Cold start**: every protocol's ``pe_utility = 0`` → lex
  tiebreak preserved (lower ``protocol_id`` wins). cheng_laoshi-only
  fixtures (single protocol, no incompatibility) are completely
  unaffected.
* **PE-driven win**: when one side has strictly higher
  ``pe_utility``, that side wins regardless of lex order.
* **PE-driven loss**: when one side has strictly lower
  ``pe_utility``, that side is dropped regardless of lex order.
* **Equal pe_utility, non-zero**: still falls to lex. (Determinism
  is required for snapshot stability.)
* **Asymmetric declaration**: only A declares ``B`` incompatible;
  arbitration still works (the resolver sees both sides).
* **Symmetric declaration**: A↔B both declare each other
  incompatible; result identical to asymmetric.
* **Three-way conflict**: A↔B and A↔C; PE history determines
  whether A or B/C survives.
* **Default behaviour**: omitting ``pe_utility_by_id`` (legacy
  callers / tests calling ``compute_active_mixture`` directly
  without owner) gives lex behaviour — backwards-compatible with
  packets 1.0–1.5b.

Tests use synthetic protocols with explicit ``co_activation_incompatible``
declarations. cheng_laoshi has no co-activation incompatibilities,
so its e2e behaviour is byte-equivalent.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
)
from volvence_zero.protocol_runtime import compute_active_mixture
from volvence_zero.protocol_runtime.activation import (
    _resolve_co_activation_incompatibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _retag(
    protocol: BehaviorProtocol,
    *,
    new_id: str,
    incompatible_with: tuple[str, ...] = (),
) -> BehaviorProtocol:
    """Return a clone with new id and explicit incompatibility list."""
    new_conditions = ActivationConditions(
        context_match_signals=protocol.activation_conditions.context_match_signals,
        co_activation_compatible=protocol.activation_conditions.co_activation_compatible,
        co_activation_incompatible=incompatible_with,
        minimum_weight_floor=protocol.activation_conditions.minimum_weight_floor,
    )
    return _replace(
        protocol,
        protocol_id=new_id,
        activation_conditions=new_conditions,
    )


# ---------------------------------------------------------------------------
# Direct unit tests on _resolve_co_activation_incompatibility
# ---------------------------------------------------------------------------


def test_no_incompatibility_passes_all_through() -> None:
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha")
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility((p_a, p_b))

    assert {p.protocol_id for p in survivors} == {"alpha", "beta"}


def test_lex_tiebreak_when_no_pe_history() -> None:
    """A↔B with empty pe_utility → lex wins (alpha < beta → drop beta)."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility((p_a, p_b))

    assert {p.protocol_id for p in survivors} == {"alpha"}


def test_lex_tiebreak_when_pe_utility_explicitly_zero() -> None:
    """All-zero pe_utility = cold start = lex behaviour."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": 0.0, "beta": 0.0},
    )

    assert {p.protocol_id for p in survivors} == {"alpha"}


# ---------------------------------------------------------------------------
# PE-driven win / loss
# ---------------------------------------------------------------------------


def test_higher_pe_utility_wins_against_lex() -> None:
    """beta has higher PE → beta wins despite lex losing.

    A↔B (declared by alpha). Alpha has lex priority but
    pe_utility(alpha) = 0.1, pe_utility(beta) = 0.5 → beta wins.
    """
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": 0.1, "beta": 0.5},
    )

    assert {p.protocol_id for p in survivors} == {"beta"}, [
        p.protocol_id for p in survivors
    ]


def test_lower_pe_utility_loses_with_lex_priority() -> None:
    """alpha has lower PE → alpha loses despite lex priority."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": -0.4, "beta": 0.6},
    )

    assert {p.protocol_id for p in survivors} == {"beta"}


def test_higher_pe_wins_when_declared_by_lex_loser() -> None:
    """beta declares alpha incompatible. PE drives drop, not declaration ownership."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha")
    p_b = _retag(base, new_id="beta", incompatible_with=("alpha",))

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": -0.2, "beta": 0.4},
    )

    assert {p.protocol_id for p in survivors} == {"beta"}


# ---------------------------------------------------------------------------
# Symmetric declaration
# ---------------------------------------------------------------------------


def test_symmetric_declaration_is_idempotent() -> None:
    """Both A and B declare each other → same outcome as asymmetric."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta", incompatible_with=("alpha",))

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": 0.1, "beta": 0.7},
    )

    assert {p.protocol_id for p in survivors} == {"beta"}


# ---------------------------------------------------------------------------
# Three-way conflict
# ---------------------------------------------------------------------------


def test_three_way_pe_arbitration_drops_two_losers() -> None:
    """alpha↔beta and alpha↔gamma. PE: alpha=0.8 highest → alpha wins both.

    Both beta and gamma get dropped. (PE wins each pairwise
    arbitration alpha conducts.)
    """
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta", "gamma"))
    p_b = _retag(base, new_id="beta")
    p_g = _retag(base, new_id="gamma")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b, p_g),
        pe_utility_by_id={"alpha": 0.8, "beta": 0.2, "gamma": 0.1},
    )

    assert {p.protocol_id for p in survivors} == {"alpha"}


def test_three_way_pe_arbitration_self_dropped_stops_iteration() -> None:
    """alpha declared incompatible with beta and gamma; alpha has lowest PE.

    alpha vs beta: alpha drops first. alpha is gone → no more
    iteration into alpha's incompatibility list (beta and gamma
    coexist if they don't conflict with each other).
    """
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta", "gamma"))
    p_b = _retag(base, new_id="beta")
    p_g = _retag(base, new_id="gamma")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b, p_g),
        pe_utility_by_id={"alpha": -0.5, "beta": 0.4, "gamma": 0.4},
    )

    assert {p.protocol_id for p in survivors} == {"beta", "gamma"}, [
        p.protocol_id for p in survivors
    ]


# ---------------------------------------------------------------------------
# Determinism with equal non-zero pe_utility
# ---------------------------------------------------------------------------


def test_equal_nonzero_pe_utility_falls_to_lex() -> None:
    """Equal PE (e.g. both at +0.5) → deterministic lex tiebreak."""
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    survivors = _resolve_co_activation_incompatibility(
        (p_a, p_b),
        pe_utility_by_id={"alpha": 0.5, "beta": 0.5},
    )

    # Lex: alpha < beta → keep alpha, drop beta
    assert {p.protocol_id for p in survivors} == {"alpha"}


# ---------------------------------------------------------------------------
# Integration via compute_active_mixture
# ---------------------------------------------------------------------------


def test_compute_active_mixture_threads_pe_utility_through_arbitration() -> None:
    """End-to-end: ``compute_active_mixture`` drops the right protocol.

    Without pe_utility (or with cold start) → alpha wins. With
    pe_utility favouring beta → beta wins. Same upstream, same
    protocols, only pe_utility_by_id differs.
    """
    base = _cheng_laoshi_protocol()
    p_a = _retag(base, new_id="alpha", incompatible_with=("beta",))
    p_b = _retag(base, new_id="beta")

    cold = compute_active_mixture(
        loaded_protocols=(p_a, p_b),
        upstream={},
    )
    assert {e.protocol_id for e in cold.active_protocols} == {"alpha"}

    pe_favours_beta = compute_active_mixture(
        loaded_protocols=(p_a, p_b),
        upstream={},
        pe_utility_by_id={"alpha": -0.3, "beta": 0.6},
    )
    assert {e.protocol_id for e in pe_favours_beta.active_protocols} == {"beta"}


# ---------------------------------------------------------------------------
# cheng_laoshi behaviour preservation
# ---------------------------------------------------------------------------


def test_cheng_laoshi_e2e_unaffected_by_packet_1_5c_i() -> None:
    """cheng_laoshi has no incompatibilities and runs alone in test fixtures.

    Therefore it never enters the arbitration code path. Pin
    behaviour-preservation across packet 1.5c-i.
    """
    p = _cheng_laoshi_protocol()
    assert p.activation_conditions.co_activation_incompatible == ()

    snapshot = compute_active_mixture(
        loaded_protocols=(p,),
        upstream={},
        pe_utility_by_id={p.protocol_id: 0.7},  # even with PE history
    )
    assert len(snapshot.active_protocols) == 1
    assert snapshot.active_protocols[0].protocol_id == p.protocol_id
    assert snapshot.active_protocols[0].activation_weight == 1.0
