"""Matched-control: protocol-driven vs vertical-driven playbook rules.

Mirror of ``test_protocol_boundary_matched_control.py`` for the
strategy / playbook compile path. Closes the strategy half of
spec ``docs/specs/protocol-runtime.md`` SHADOW → ACTIVE checklist
condition 7 ("BehaviorProtocol → application owners compile path").

The vertical compile (existing
``DomainExperiencePackage.playbook_rules`` flow producing
``PlaybookRule`` records via ``compiler._playbook_rule``) and the
protocol compile (packet 1.3b
``compile_protocol_to_application_artifacts`` producing
``PlaybookRule`` records via
``protocol_runtime.compiler._strategy_prior_to_playbook_rule``)
must produce records that are byte-for-byte identical *modulo* the
lineage prefix on ``rule_id``:

* Vertical: ``rid-growth-advisor:playbook:funnel-height``
* Protocol: ``protocol:growth_advisor:cheng-laoshi:playbook:funnel-height``

Once that holds, the downstream ``StrategyPlaybookModule``
execution is provably identical because:

* ``StrategyPlaybookModule.process()`` reads
  ``ApplicationRareHeavyState.distilled_playbook_rules`` and
  filters by ``problem_pattern`` (the merge key) — not by
  ``rule_id``.
* ``StrategyPlaybookSnapshot.matched_rules`` carries the full
  ``PlaybookRule`` objects; downstream consumers act on
  ``recommended_regime`` / ``recommended_ordering`` /
  ``recommended_pacing`` / ``avoid_patterns`` /
  ``knowledge_weight_hint`` / ``experience_weight_hint``, all of
  which the matched-control gate proves are bit-equivalent.

If a future packet adds a new field to ``StrategyPrior`` /
``PlaybookRule`` and forgets to thread it through both compile
paths, this test catches the drift.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from lifeform_domain_growth_advisor.compiler import build_growth_advisor_package
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.types import PlaybookRule
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_NORMALIZED_RULE_ID = "<lineage-normalized>"


def _populate_via_vertical_path() -> ApplicationRareHeavyState:
    """Populate ``distilled_playbook_rules`` via the existing vertical compile."""

    state = ApplicationRareHeavyState()
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    state.upsert_distilled_playbook_rules(package.playbook_rules)
    return state


def _populate_via_protocol_path() -> ApplicationRareHeavyState:
    """Populate ``distilled_playbook_rules`` via the packet 1.3b protocol compile."""

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return state


def _normalize_rule(rule: PlaybookRule) -> PlaybookRule:
    """Strip the ``rule_id`` lineage prefix for cross-path comparison."""

    return _replace(rule, rule_id=_NORMALIZED_RULE_ID)


def _sort_key(rule: PlaybookRule) -> str:
    """Stable sort key independent of rule_id (which differs by path).

    ``problem_pattern`` is the merge key used by
    ``upsert_distilled_playbook_rules``, so two equivalent rules
    sort to the same position regardless of compile path.
    """

    return rule.problem_pattern


# ---------------------------------------------------------------------------
# Sanity: both paths produce 16 rules (cheng_laoshi has 16 strategies)
# ---------------------------------------------------------------------------


def test_both_paths_produce_sixteen_rules() -> None:
    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()
    assert len(state_v.distilled_playbook_rules) == 16
    assert len(state_p.distilled_playbook_rules) == 16


# ---------------------------------------------------------------------------
# Rule-set equivalence (modulo rule_id lineage prefix)
# ---------------------------------------------------------------------------


def test_protocol_path_produces_same_rules_as_vertical_path() -> None:
    """The matched-control gate: every ``PlaybookRule`` field
    *except* ``rule_id`` must match across paths.

    If this fires, either (a) ``StrategyPrior`` schema has grown a
    field that the vertical-side ``GrowthAdvisorStrategyPrior``
    doesn't have (or vice versa), or (b) one of the two compile
    functions is dropping a field. Both are contract violations.
    """

    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()

    norm_v = sorted(
        (_normalize_rule(r) for r in state_v.distilled_playbook_rules),
        key=_sort_key,
    )
    norm_p = sorted(
        (_normalize_rule(r) for r in state_p.distilled_playbook_rules),
        key=_sort_key,
    )
    assert norm_v == norm_p


# ---------------------------------------------------------------------------
# Lineage prefix invariant on rule_id (the only path-dependent field)
# ---------------------------------------------------------------------------


def test_vertical_rule_ids_use_growth_advisor_prefix() -> None:
    state_v = _populate_via_vertical_path()
    for rule in state_v.distilled_playbook_rules:
        assert rule.rule_id.startswith("rid-growth-advisor:playbook:"), rule.rule_id


def test_protocol_rule_ids_use_protocol_prefix() -> None:
    state_p = _populate_via_protocol_path()
    for rule in state_p.distilled_playbook_rules:
        assert rule.rule_id.startswith(
            "protocol:growth_advisor:cheng-laoshi:playbook:"
        ), rule.rule_id


def test_each_strategy_id_appears_once_per_path() -> None:
    """Within one path, the strategy id (extracted from rule_id
    suffix) must be unique. Cross-path duplication is fine because
    the lineage prefix differs.
    """

    state_v = _populate_via_vertical_path()
    state_p = _populate_via_protocol_path()

    def strategy_ids(state: ApplicationRareHeavyState) -> set[str]:
        return {r.rule_id.rsplit(":", 1)[-1] for r in state.distilled_playbook_rules}

    expected_ids_v = strategy_ids(state_v)
    expected_ids_p = strategy_ids(state_p)
    assert expected_ids_v == expected_ids_p
    assert len(expected_ids_v) == 16


# ---------------------------------------------------------------------------
# Per-field passthrough (extra granularity for diagnosability)
# ---------------------------------------------------------------------------


def _rules_keyed_by_strategy_id(
    state: ApplicationRareHeavyState,
) -> dict[str, PlaybookRule]:
    return {
        r.rule_id.rsplit(":", 1)[-1]: r for r in state.distilled_playbook_rules
    }


def test_per_field_equivalence_across_paths() -> None:
    """Diagnostic-friendly per-field check.

    Failure mode: if the aggregate equality test above fires, this
    test pinpoints which field diverged — useful when ``StrategyPrior``
    or ``PlaybookRule`` grows a new field.
    """

    by_id_v = _rules_keyed_by_strategy_id(_populate_via_vertical_path())
    by_id_p = _rules_keyed_by_strategy_id(_populate_via_protocol_path())

    assert set(by_id_v) == set(by_id_p)

    diverged_fields: dict[str, set[str]] = {}
    for strategy_id in by_id_v:
        r_v = by_id_v[strategy_id]
        r_p = by_id_p[strategy_id]
        for field_name in (
            "problem_pattern",
            "recommended_regime",
            "recommended_ordering",
            "recommended_pacing",
            "avoid_patterns",
            "knowledge_weight_hint",
            "experience_weight_hint",
            "applicability_scope",
            "confidence",
            "description",
            "continuum_band_id",
            "mean_continuum_position",
        ):
            if getattr(r_v, field_name) != getattr(r_p, field_name):
                diverged_fields.setdefault(strategy_id, set()).add(field_name)
    assert not diverged_fields, (
        f"Per-field divergence between vertical and protocol paths: "
        f"{diverged_fields}. Either StrategyPrior is missing a field "
        f"that GrowthAdvisorStrategyPrior has, or one of the compilers "
        f"is dropping a field."
    )


# ---------------------------------------------------------------------------
# Apply order does not affect outcome (idempotency / commutativity)
# ---------------------------------------------------------------------------


def test_loading_protocol_after_vertical_does_not_break_state() -> None:
    """Defensive: simultaneous use of both paths in production
    (e.g. cheng_laoshi loaded both as DomainExperiencePackage AND
    as BehaviorProtocol) must not corrupt state.

    The merge key ``problem_pattern`` collapses the two paths' rules
    into a single entry per logical strategy (max-confidence wins;
    equal confidence ⇒ replace). Both paths produce equal-confidence
    rules for cheng_laoshi, so the final state has exactly 16 rules.
    """

    state = ApplicationRareHeavyState()

    # Apply vertical first
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    state.upsert_distilled_playbook_rules(package.playbook_rules)
    assert len(state.distilled_playbook_rules) == 16

    # Then apply protocol on top of same state
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)

    # Still exactly 16 (the merge key collapsed; rule_id is whichever
    # path won the last-write tie; doesn't matter for execution)
    assert len(state.distilled_playbook_rules) == 16
