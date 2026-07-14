from __future__ import annotations

from lifeform_service import VerticalSpec
from lifeform_service import cli


def _spec(name: str) -> VerticalSpec:
    return VerticalSpec(
        name=name,
        factory=lambda _runtime: None,  # type: ignore[arg-type,return-value]
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )


def test_ablation_bundle_cli_uses_multi_vertical_app(monkeypatch) -> None:
    captured: dict[str, object] = {}
    verticals = {
        "companion": _spec("companion"),
        "companion-cold": _spec("companion-cold"),
    }
    runtime = object()

    monkeypatch.setattr(cli, "discover_companion_ablation_verticals", lambda: verticals)
    monkeypatch.setattr(cli, "_build_shared_substrate", lambda _args: runtime)

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.web, "run_app", lambda *_args, **_kwargs: None)

    rc = cli.main(["--ablation-bundle", "--substrate-mode", "hf-shared"])

    assert rc == 0
    assert captured["verticals"] is verticals
    assert captured["default_vertical"] == "companion"
    assert captured["substrate_runtime"] is runtime
    assert "vertical" not in captured


def test_ablation_bundle_cli_rejects_single_vertical() -> None:
    rc = cli.main(["--ablation-bundle", "--vertical", "companion"])
    assert rc == 1
