"""Matched-control: protocol-driven vs vertical-driven boundary hints.

Closes spec ``docs/specs/protocol-runtime.md`` SHADOW → ACTIVE
checklist condition 5 ("at least 1 downstream consumer passes a
matched-control dual-run test") for the boundary path. The
vertical compile (existing
``DomainExperiencePackage.boundary_hints`` flow) and the protocol
compile (packet 1.2 ``compile_protocol_to_application_artifacts``)
must produce ``BoundaryPriorHint`` records that are byte-for-byte
identical *modulo* the lineage prefix on ``hint_id``.

Once that holds, the downstream ``boundary_policy`` execution is
provably identical because:

* ``BoundaryPolicyModule.process()`` reads only
  ``ApplicationRareHeavyState.boundary_prior_hints`` (filtered by
  ``regime_id`` / ``trigger_reasons``) and a fixed set of cognitive
  upstream snapshots.
* The cognitive upstream is independent of which path populated
  the hints.
* The filter keys (``regime_id`` / ``trigger_reasons``) are
  preserved bit-equivalently across paths.
* Per-field hint contents (``answer_depth_limit_hint`` /
  ``clarification_required`` / ``refer_out_required`` /
  ``blocked_topics`` / ``required_disclaimers`` / ``confidence`` /
  ``description``) are preserved bit-equivalently across paths.

The only delta is ``hint_id``:
* Vertical: ``rid-growth-advisor:boundary:bp-no-hard-sell``
* Protocol: ``protocol:growth_advisor:cheng-laoshi:boundary:bp-no-hard-sell``

This delta is intentional lineage information; downstream code
does not branch on ``hint_id`` (only on ``regime_id`` + content
fields). The matched-control test below normalises ``hint_id``
before comparison so the equivalence is asserted on what
*actually matters*.

If a future packet adds new fields to ``BoundaryContract`` /
``BoundaryPriorHint`` and forgets to thread them through both
compile paths, this test catches the drift immediately.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from lifeform_domain_growth_advisor.compiler import build_growth_advisor_package
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.types import BoundaryPriorHint
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_NORMALIZED_HINT_ID = "<lineage-normalized>"


def _populate_via_vertical_path() -> ApplicationRareHeavyState:
    """Populate ``boundary_prior_hints`` via the existing vertical compile.

    Mirrors what ``apply_domain_experience_packages`` does for the
    boundary slice — direct upsert from the vertical compiler's
    output.
    """

    state = ApplicationRareHeavyState()
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    state.upsert_boundary_prior_hints(package.boundary_hints)
    return state


def _populate_via_protocol_path() -> ApplicationRareHeavyState:
    """Populate ``boundary_prior_hints`` via the packet 1.2 protocol compile."""

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return state


def _normalize_hint(hint: BoundaryPriorHint) -> BoundaryPriorHint:
    """Strip the ``hint_id`` lineage prefix for cross-path comparison.

    ``hint_id`` deliberately differs across paths to encode lineage
    (vertical vs protocol). Every other field is supposed to be
    bit-equivalent. Replacing ``hint_id`` with a sentinel makes
    that property visible as plain dataclass equality.
    """

    return _replace(hint, hint_id=_NORMALIZED_HINT_ID)


def _sort_key(hint: BoundaryPriorHint) -> tuple[str | None, tuple[str, ...]]:
    """Stable sort key independent of hint_id (which differs by path).

    Uses ``(regime_id, trigger_reasons)`` — the merge key the
    upsert pipeline uses, so two equivalent hints sort to the
    same position even if they came in via different paths.
    """

    return (hint.regime_id, hint.trigger_reasons)


# ---------------------------------------------------------------------------
# Sanity: both paths produce 4 hints (cheng_laoshi has 4 boundaries)
# ---------------------------------------------------------------------------


def test_both_paths_produce_four_hints() -> None:
    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()
    assert len(state_v.boundary_prior_hints) == 4
    assert len(state_p.boundary_prior_hints) == 4


# ---------------------------------------------------------------------------
# Hint-set equivalence (modulo hint_id lineage prefix)
# ---------------------------------------------------------------------------


def test_protocol_path_produces_same_hints_as_vertical_path() -> None:
    """The matched-control gate: every ``BoundaryPriorHint`` field
    *except* ``hint_id`` must match across paths.

    If this fires, either (a) ``BoundaryContract`` schema has
    grown a field that the vertical-side
    ``GrowthAdvisorBoundaryPrior`` doesn't have (or vice versa),
    or (b) one of the two compile functions is dropping a field.
    Both are contract violations.
    """

    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()

    norm_v = sorted(
        (_normalize_hint(h) for h in state_v.boundary_prior_hints),
        key=_sort_key,
    )
    norm_p = sorted(
        (_normalize_hint(h) for h in state_p.boundary_prior_hints),
        key=_sort_key,
    )
    assert norm_v == norm_p


# ---------------------------------------------------------------------------
# Lineage prefix invariant on hint_id (the only path-dependent field)
# ---------------------------------------------------------------------------


def test_vertical_hint_ids_use_growth_advisor_prefix() -> None:
    state_v = _populate_via_vertical_path()
    for hint in state_v.boundary_prior_hints:
        assert hint.hint_id.startswith("rid-growth-advisor:boundary:"), hint.hint_id


def test_protocol_hint_ids_use_protocol_prefix() -> None:
    state_p = _populate_via_protocol_path()
    for hint in state_p.boundary_prior_hints:
        assert hint.hint_id.startswith("protocol:growth_advisor:cheng-laoshi:boundary:"), (
            hint.hint_id
        )


def test_each_boundary_id_appears_once_per_path() -> None:
    """Within one path, ``boundary_id`` (extracted from hint_id
    suffix) must be unique. Cross-path duplication is fine
    because the lineage prefix differs.
    """

    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()

    def boundary_ids(state: ApplicationRareHeavyState) -> set[str]:
        return {h.hint_id.rsplit(":", 1)[-1] for h in state.boundary_prior_hints}

    expected = {
        "bp-no-hard-sell",
        "bp-no-overclaim",
        "bp-no-flooding",
        "bp-no-judgmental",
    }
    assert boundary_ids(state_v) == expected
    assert boundary_ids(state_p) == expected


# ---------------------------------------------------------------------------
# Per-field passthrough (extra granularity for diagnosability)
# ---------------------------------------------------------------------------


def _hints_keyed_by_boundary_id(
    state: ApplicationRareHeavyState,
) -> dict[str, BoundaryPriorHint]:
    """Index hints by the ``boundary_id`` suffix of ``hint_id``.

    This is the cross-path identifier that survives the lineage
    prefix difference and is what real consumers (boundary_policy
    via regime_id+trigger_reasons match) effectively pivot on.
    """

    return {h.hint_id.rsplit(":", 1)[-1]: h for h in state.boundary_prior_hints}


def test_per_field_equivalence_across_paths() -> None:
    """Diagnostic-friendly per-field check.

    Failure mode: if the aggregate equality test above fires, this
    test pinpoints which field diverged — useful when ``BoundaryContract``
    or ``BoundaryPriorHint`` grows a new field.
    """

    by_id_v = _hints_keyed_by_boundary_id(_populate_via_vertical_path())
    by_id_p = _hints_keyed_by_boundary_id(_populate_via_protocol_path())

    assert set(by_id_v) == set(by_id_p)

    diverged_fields: dict[str, set[str]] = {}
    for boundary_id in by_id_v:
        h_v = by_id_v[boundary_id]
        h_p = by_id_p[boundary_id]
        for field_name in (
            "regime_id",
            "trigger_reasons",
            "answer_depth_limit_hint",
            "clarification_required",
            "refer_out_required",
            "blocked_topics",
            "required_disclaimers",
            "confidence",
            "description",
        ):
            if getattr(h_v, field_name) != getattr(h_p, field_name):
                diverged_fields.setdefault(boundary_id, set()).add(field_name)
    assert not diverged_fields, (
        f"Per-field divergence between vertical and protocol paths: "
        f"{diverged_fields}. Either BoundaryContract is missing a field "
        f"that GrowthAdvisorBoundaryPrior has, or one of the compilers "
        f"is dropping a field."
    )


# ---------------------------------------------------------------------------
# Apply order does not affect outcome (idempotency / commutativity)
# ---------------------------------------------------------------------------


def test_loading_protocol_after_vertical_does_not_break_state() -> None:
    """Defensive: simultaneous use of both paths in production
    (e.g. cheng_laoshi loaded both as DomainExperiencePackage AND
    as BehaviorProtocol) must not corrupt state.

    The merge key ``(regime_id, trigger_reasons)`` collapses the
    two paths' hints into a single entry per logical boundary
    (last-write wins by confidence; equal confidence ⇒ last
    overwrites). Both paths produce equal-confidence hints for
    cheng_laoshi, so the final state has exactly 4 hints.
    """

    state = ApplicationRareHeavyState()

    # Apply vertical first
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    state.upsert_boundary_prior_hints(package.boundary_hints)
    assert len(state.boundary_prior_hints) == 4

    # Then apply protocol on top of same state
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)

    # Still exactly 4 (the merge key collapsed; hint_id is whichever
    # path won the last-write tie; doesn't matter for execution)
    assert len(state.boundary_prior_hints) == 4
