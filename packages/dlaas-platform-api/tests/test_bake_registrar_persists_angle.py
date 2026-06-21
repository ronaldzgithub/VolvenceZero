"""The bake registrar persists angle/source_ref/run_id as first-class fields.

A management console must categorize souls by angle (author/interpreter/
character) without parsing the template_id string. This asserts the
registrar writes `persona_spec.bake.angle` on the minted template and the
angle into the lifecycle notes.
"""

from __future__ import annotations

import pytest

from dlaas_platform_api.bake import (
    CompiledArtifact,
    RegistryBakeArtifactRegistrar,
)
from dlaas_platform_contracts import BakeAngleKind, BakeRun, BakeRunStatus
from dlaas_platform_registry import (
    PersonaLifecycleStore,
    Registry,
    TemplateStore,
    TenantStore,
)


def _compiled(kind: BakeAngleKind, slug: str) -> CompiledArtifact:
    is_figure = kind in (BakeAngleKind.AUTHOR, BakeAngleKind.INTERPRETER)
    prefix = "figure-bundle" if is_figure else "character-template"
    return CompiledArtifact(
        angle_kind=kind,
        angle_slug=slug,
        is_figure=is_figure,
        bundle_id=f"{prefix}:{slug}:abc123",
        figure_artifact_id=(f"{prefix}:{slug}:abc123" if is_figure else ""),
        integrity_hash="abc123",
        figure_id=slug,
        display_name=slug.title(),
        bundle_root="",  # skip figure-bundle store registration in this test
    )


@pytest.mark.asyncio
async def test_registrar_persists_angle_in_persona_spec() -> None:
    registry = Registry(db_path=":memory:")
    tenant = await TenantStore(registry).create(
        tenant_name="t", contact_email="t@example.test", tenant_id="tenant_x"
    )
    registrar = RegistryBakeArtifactRegistrar(registry)

    run = BakeRun(
        run_id="bake_run_1",
        source_ref="work:hlm",
        status=BakeRunStatus.RUNNING,
        tenant_id=tenant.tenant_id,
        app_id="myriad",
        runtime_template_id="myriad.figure.v0",
    )

    from dlaas_platform_contracts import BakeAngle

    for kind in (
        BakeAngleKind.AUTHOR,
        BakeAngleKind.INTERPRETER,
        BakeAngleKind.CHARACTER,
    ):
        angle = BakeAngle(kind=kind, slug=f"s_{kind.value}")
        compiled = _compiled(kind, angle.slug)
        registered = await registrar.register(run=run, angle=angle, compiled=compiled)
        assert registered.lifecycle_stage == "pretrained"

        template = await TemplateStore(registry).get(registered.template_id)
        bake_meta = template.persona_spec.get("bake")
        assert bake_meta is not None, "persona_spec.bake missing"
        assert bake_meta["angle"] == kind.value
        assert bake_meta["source_ref"] == "work:hlm"
        assert bake_meta["bake_run_id"] == "bake_run_1"
        # runtime_template_id carried through (previously an AttributeError).
        assert template.runtime_template_id == "myriad.figure.v0"

        lifecycle = await PersonaLifecycleStore(registry).get_by_template(
            registered.template_id
        )
        assert kind.value in lifecycle.notes
