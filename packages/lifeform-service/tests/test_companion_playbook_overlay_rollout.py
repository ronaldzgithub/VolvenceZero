from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lifeform_domain_emogpt
import pytest

from lifeform_service import VerticalSpec
from lifeform_service import cli, verticals
from volvence_zero.runtime import WiringLevel


class _FakeLifeform:
    def with_thinking_adapter_factory(self, _factory):
        return self

    def ensure_affordance_registry(self) -> tuple[object, object]:
        return object(), object()


def _spec(name: str = "companion") -> VerticalSpec:
    return VerticalSpec(
        name=name,
        factory=lambda _runtime: _FakeLifeform(),  # type: ignore[arg-type]
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )


def test_shadow_overlay_is_validated_at_discovery_and_threaded_to_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overlay_path = tmp_path / "candidate.json"
    validation_calls: list[dict[str, object]] = []
    build_calls: list[dict[str, object]] = []

    def fake_resolve(**kwargs):
        validation_calls.append(kwargs)
        return SimpleNamespace(
            overlay=SimpleNamespace(
                overlay_id="candidate-v1",
                content_sha256="a" * 64,
            ),
            baseline_rules=(object(),),
            candidate_rules=(object(), object()),
        )

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return _FakeLifeform()

    monkeypatch.setattr(
        lifeform_domain_emogpt,
        "resolve_companion_package_overlay",
        fake_resolve,
    )
    monkeypatch.setattr(
        lifeform_domain_emogpt,
        "build_companion_lifeform",
        fake_build,
    )

    spec = verticals._try_companion(
        playbook_overlay_wiring=WiringLevel.SHADOW,
        playbook_overlay_path=overlay_path,
    )
    assert spec is not None
    spec.factory(None)

    assert validation_calls == [
        {
            "wiring_level": WiringLevel.SHADOW,
            "overlay_path": overlay_path,
        }
    ]
    assert build_calls[0]["playbook_overlay_wiring"] is WiringLevel.SHADOW
    assert build_calls[0]["playbook_overlay_path"] == overlay_path


def test_service_boundary_rejects_active_overlay() -> None:
    with pytest.raises(ValueError, match="ACTIVE requires a separate reviewed"):
        verticals._try_companion(
            playbook_overlay_wiring=WiringLevel.ACTIVE,
        )


def test_disabled_overlay_rejects_unused_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path requires SHADOW"):
        verticals._try_companion(
            playbook_overlay_path=tmp_path / "candidate.json",
        )


def test_cli_threads_shadow_rollout_to_vertical_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    overlay_path = tmp_path / "candidate.json"

    def fake_discover(**kwargs):
        captured.update(kwargs)
        return {"companion": _spec()}

    monkeypatch.setattr(cli, "discover_verticals", fake_discover)

    rc = cli.main(
        [
            "--list-verticals",
            "--companion-playbook-overlay-mode",
            "shadow",
            "--companion-playbook-overlay-path",
            str(overlay_path),
        ]
    )

    assert rc == 0
    assert captured["companion_playbook_overlay_wiring"] is WiringLevel.SHADOW
    assert captured["companion_playbook_overlay_path"] == overlay_path


def test_cli_rejects_overlay_path_while_disabled(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    rc = cli.main(
        [
            "--companion-playbook-overlay-path",
            str(tmp_path / "candidate.json"),
        ]
    )

    assert rc == 1
    assert "requires --companion-playbook-overlay-mode shadow" in capsys.readouterr().err


def test_cli_rejects_shadow_for_non_companion_vertical(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(
        [
            "--vertical",
            "coding",
            "--companion-playbook-overlay-mode",
            "shadow",
        ]
    )

    assert rc == 1
    assert "--vertical must be companion" in capsys.readouterr().err
