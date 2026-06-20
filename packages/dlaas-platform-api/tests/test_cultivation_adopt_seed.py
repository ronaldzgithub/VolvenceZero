"""Tests for the adopted-seed cultivation helpers.

The self-learning workshop lets an operator seed a cultivation from an
existing baked persona/template instead of an empty seed. These cover
the pure helper that fills seed defaults from the source template's
``persona_spec`` (operator-supplied fields still win) and that the
filled payload parses into a valid :class:`CultivationSeed`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from aiohttp import web

from dlaas_platform_contracts import TemplateStatus
from dlaas_platform_registry import TemplateNotFound
from lifeform_cultivation import CultivationSeed

from dlaas_platform_api.cultivation import (
    _resolve_source_template,
    _seed_defaults_from_template,
    _source_provenance,
)


def _template(**overrides):
    base = dict(
        template_id="tpl_einstein",
        tenant_id="tnt_owner",
        status=TemplateStatus.PUBLISHED,
        figure_artifact_id="",
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


class _FakeTemplates:
    def __init__(self, template):
        self._t = template

    async def get(self, template_id):
        if self._t is None or template_id != self._t.template_id:
            raise TemplateNotFound(template_id)
        return self._t


class _FakeBundle:
    def __init__(self, template):
        self.templates = _FakeTemplates(template)


# --- Provenance inference -------------------------------------------------


def test_source_provenance_infers_kind_and_bundle_continuation():
    tmpl = _template(persona_spec={"bake_angle": "author", "domain": "物理"})
    prov = _source_provenance({}, tmpl, {"protocols": []})
    assert prov["source_kind"] == "author"
    assert prov["source_template_id"] == "tpl_einstein"
    assert prov["continuation_mode"] == "protocol_bundle"


def test_source_provenance_metadata_only_when_no_bundle():
    prov = _source_provenance({}, _template(), None)
    assert prov["continuation_mode"] == "metadata_only"


def test_source_provenance_figure_default_and_operator_override():
    figure = _template(persona_spec={}, figure_artifact_id="figure-bundle:x")
    assert _source_provenance({}, figure, None)["source_kind"] == "figure"
    overridden = _source_provenance(
        {"source_kind": "interpreter"}, _template(), None
    )
    assert overridden["source_kind"] == "interpreter"


def test_source_provenance_defaults_to_expert():
    plain = _template(persona_spec={"display_name": "x"})
    assert _source_provenance({}, plain, None)["source_kind"] == "expert"


# --- Access control -------------------------------------------------------


def test_resolve_source_template_owner_allowed():
    async def run():
        tmpl = _template(tenant_id="tnt_a", status=TemplateStatus.DRAFT)
        bundle = _FakeBundle(tmpl)
        tenant = SimpleNamespace(tenant_id="tnt_a")
        result = await _resolve_source_template(bundle, "tpl_einstein", tenant)
        assert not isinstance(result, web.Response)
        template, _payload = result
        assert template.template_id == "tpl_einstein"

    asyncio.run(run())


def test_resolve_source_template_published_allowed_cross_tenant():
    async def run():
        tmpl = _template(tenant_id="tnt_a", status=TemplateStatus.PUBLISHED)
        bundle = _FakeBundle(tmpl)
        tenant = SimpleNamespace(tenant_id="tnt_b")
        result = await _resolve_source_template(bundle, "tpl_einstein", tenant)
        assert not isinstance(result, web.Response)

    asyncio.run(run())


def test_resolve_source_template_cross_tenant_unpublished_forbidden():
    async def run():
        tmpl = _template(tenant_id="tnt_a", status=TemplateStatus.DRAFT)
        bundle = _FakeBundle(tmpl)
        tenant = SimpleNamespace(tenant_id="tnt_b")
        result = await _resolve_source_template(bundle, "tpl_einstein", tenant)
        assert isinstance(result, web.Response)
        assert result.status == 403

    asyncio.run(run())


def test_resolve_source_template_operator_allowed():
    async def run():
        tmpl = _template(tenant_id="tnt_a", status=TemplateStatus.DRAFT)
        bundle = _FakeBundle(tmpl)
        # operator (control-plane) acts cross-tenant: tenant is None.
        result = await _resolve_source_template(bundle, "tpl_einstein", None)
        assert not isinstance(result, web.Response)

    asyncio.run(run())


def test_resolve_source_template_not_found():
    async def run():
        bundle = _FakeBundle(None)
        result = await _resolve_source_template(bundle, "tpl_missing", None)
        assert isinstance(result, web.Response)
        assert result.status == 404

    asyncio.run(run())


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
