"""Tests for the adopted-seed cultivation helpers.

The self-learning workshop lets an operator seed a cultivation from an
existing baked persona/template instead of an empty seed. These cover
the pure helper that fills seed defaults from the source template's
``persona_spec`` (operator-supplied fields still win) and that the
filled payload parses into a valid :class:`CultivationSeed`.
"""

from __future__ import annotations

from types import SimpleNamespace

from lifeform_cultivation import CultivationSeed

from dlaas_platform_api.cultivation import _seed_defaults_from_template


def _template(**overrides):
    base = dict(
        template_name="Einstein companion",
        domain="physics",
        persona_spec={
            "display_name": "爱因斯坦",
            "role_archetype": "理论物理学家",
            "domain": "物理",
            "focus": "相对论",
            "value_boundaries": ["不杜撰史料"],
            "single_school_objective": "形成自洽的物理直觉体系",
        },
        seed_config={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_defaults_filled_from_persona_spec() -> None:
    merged = _seed_defaults_from_template({"slug": "einstein"}, _template())
    assert merged["display_name"] == "爱因斯坦"
    assert merged["role_archetype"] == "理论物理学家"
    assert merged["domain"] == "物理"
    assert merged["focus"] == "相对论"
    assert merged["value_boundaries"] == ["不杜撰史料"]
    # Parses into a valid seed without error.
    seed = CultivationSeed.from_json(merged)
    assert seed.display_name == "爱因斯坦"
    assert seed.domain == "物理"


def test_operator_overrides_win_over_template() -> None:
    merged = _seed_defaults_from_template(
        {
            "slug": "einstein",
            "display_name": "操作员命名",
            "domain": "应用物理",
        },
        _template(),
    )
    assert merged["display_name"] == "操作员命名"
    assert merged["domain"] == "应用物理"
    # Unspecified fields still come from the template.
    assert merged["role_archetype"] == "理论物理学家"


def test_role_archetype_falls_back_to_domain_label() -> None:
    # A baked template whose persona_spec omits role_archetype: the
    # helper must still produce a parseable seed (domain-based label).
    tmpl = _template(
        persona_spec={"display_name": "无角色", "domain": "教育"}
    )
    merged = _seed_defaults_from_template({"slug": "x"}, tmpl)
    assert merged["role_archetype"] == "教育专家"
    seed = CultivationSeed.from_json(merged)
    assert seed.role_archetype == "教育专家"


def test_domain_falls_back_to_template_domain_field() -> None:
    tmpl = _template(
        domain="generic",
        persona_spec={"display_name": "N"},
    )
    merged = _seed_defaults_from_template({"slug": "x"}, tmpl)
    assert merged["domain"] == "generic"
    assert merged["role_archetype"] == "generic专家"
