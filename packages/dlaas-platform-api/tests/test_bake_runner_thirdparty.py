"""Unit tests for the third-party-LLM bake runner's 解析/吸收/register split.

These assert the boundary without the heavy VZ stack:
- 解析 (analysis) calls the third-party LLM once per angle.
- 吸收 (absorption) goes through the injected VZ compiler ONLY (no LLM).
- register goes through the injected registrar and supplies the result's
  template id + lifecycle stage.
"""

from __future__ import annotations

import pytest

from dlaas_platform_api.bake import (
    CompiledArtifact,
    RegisteredArtifact,
    ThirdPartyLlmBakeAngleRunner,
)
from dlaas_platform_contracts import (
    BakeAngle,
    BakeAngleKind,
    BakeRun,
    BakeRunStatus,
    RawMaterial,
    ThirdPartyLlmJsonResponse,
)
import dlaas_platform_api.bake as bake_mod


class _FakeCompiler:
    """Stand-in 吸收 stage; records calls, never touches an LLM."""

    def __init__(self) -> None:
        self.calls: list[BakeAngleKind] = []

    def compile(self, *, run, angle, profile_json, materials):  # noqa: ANN001
        self.calls.append(angle.kind)
        is_figure = angle.kind in (
            BakeAngleKind.AUTHOR,
            BakeAngleKind.INTERPRETER,
        )
        prefix = "figure-bundle" if is_figure else "character-template"
        return CompiledArtifact(
            angle_kind=angle.kind,
            angle_slug=angle.slug,
            is_figure=is_figure,
            bundle_id=f"{prefix}:{angle.slug}:deadbeef",
            figure_artifact_id=(f"{prefix}:{angle.slug}:deadbeef" if is_figure else ""),
            integrity_hash="deadbeef",
            figure_id=angle.slug,
            display_name=angle.display_name or angle.slug,
            bundle_root="",
        )


class _FakeRegistrar:
    def __init__(self) -> None:
        self.registered: list[str] = []

    async def register(self, *, run, angle, compiled):  # noqa: ANN001
        self.registered.append(compiled.bundle_id)
        return RegisteredArtifact(
            template_id=f"tpl_{angle.kind.value}_{angle.slug}",
            lifecycle_stage="pretrained",
        )


def _run() -> BakeRun:
    return BakeRun(
        run_id="bake_test",
        source_ref="work:test",
        status=BakeRunStatus.RUNNING,
        tenant_id="tenant_x",
    )


def _angle(kind: str, slug: str) -> BakeAngle:
    return BakeAngle(kind=BakeAngleKind(kind), slug=slug, display_name=slug.title())


@pytest.mark.asyncio
async def test_runner_splits_analysis_absorption_register(monkeypatch) -> None:
    llm_calls: list[str] = []

    async def _fake_complete_json(*, config, llm_request):  # noqa: ANN001
        # 解析: record that the LLM ran and return a minimal profile.
        llm_calls.append(llm_request.schema_name)
        return ThirdPartyLlmJsonResponse(
            content={
                "slug": "x",
                "display_name": "X",
                "description": "d",
                "displayName": "X",
                "brief": "b",
            },
            provider="fake",
            model="fake-model",
        )

    monkeypatch.setattr(bake_mod, "complete_json_with_config", _fake_complete_json)

    compiler = _FakeCompiler()
    registrar = _FakeRegistrar()
    runner = ThirdPartyLlmBakeAngleRunner(
        config=object(), compiler=compiler, registrar=registrar
    )

    materials = (RawMaterial(kind="text", text="hello"),)
    for kind, slug in (("author", "a"), ("interpreter", "i"), ("character", "c")):
        result = await runner.run(
            run=_run(),
            angle=_angle(kind, slug),
            materials=materials,
            shared_profile={},
        )
        assert result.angle_kind.value == kind
        assert result.template_id == f"tpl_{kind}_{slug}"
        assert result.lifecycle_stage == "pretrained"

    # 解析 ran once per angle (3), each with the angle-specific schema.
    assert len(llm_calls) == 3
    # 吸收 ran for every angle through the VZ compiler seam only.
    assert compiler.calls == [
        BakeAngleKind.AUTHOR,
        BakeAngleKind.INTERPRETER,
        BakeAngleKind.CHARACTER,
    ]
    # register ran for every angle.
    assert len(registrar.registered) == 3
    # author/interpreter route to figure family; character does not.
    assert registrar.registered[0].startswith("figure-bundle:")
    assert registrar.registered[2].startswith("character-template:")


@pytest.mark.asyncio
async def test_runner_requires_materials_or_profile() -> None:
    runner = ThirdPartyLlmBakeAngleRunner(
        config=object(), compiler=_FakeCompiler(), registrar=_FakeRegistrar()
    )
    with pytest.raises(RuntimeError):
        await runner.run(
            run=_run(), angle=_angle("author", "a"), materials=(), shared_profile={}
        )
