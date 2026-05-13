"""Contract test: judge family rotation on the OpenRouter smoke path.

`docs/specs/companion-bench.md` §5 requires that the arc judge come
from a different model family than the per-turn judge for any
"real" reference run. The wheel does not enforce this (judges are
injected); the orchestrator must guarantee it. For the smoke
runner ([`scripts/companion_bench/run_companion_bench_smoke.py`](../../scripts/companion_bench/run_companion_bench_smoke.py)),
this test:

* Asserts the `openrouter` provider's per-turn and arc models come
  from different vendor prefixes (OpenRouter `<vendor>/<model>` format).
* Verifies the `qwen` provider is explicitly labelled "weak proxy"
  (same family across per-turn + arc — pipeline-validation only,
  must not ship to leaderboard).
* Verifies the YAML rosters carry the matching weak/strong
  annotations so a future contributor cannot silently demote the
  smoke run by swapping yaml without updating the docstring.

Refs:
* `docs/specs/companion-bench.md` §5 Judge contract
* `docs/external/companion-bench-openrouter-setup.md` §"Judge 合格度档级"
* `docs/known-debts.md` #48
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SMOKE_RUNNER = _REPO_ROOT / "scripts" / "companion_bench" / "run_companion_bench_smoke.py"
_OPENROUTER_ROSTER = _REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.smoke.yaml"
_QWEN_ROSTER = _REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.smoke_qwen.yaml"


def _load_runner_module():
    """Import run_companion_bench_smoke as a module to read its PROVIDERS table."""
    module_name = "_companion_bench_smoke_runner_for_test"
    spec = importlib.util.spec_from_file_location(module_name, _SMOKE_RUNNER)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot import {_SMOKE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so dataclasses inside the
    # script can resolve their own __module__ during class-definition time.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_NATIVE_FAMILY_PREFIXES = (
    ("qwen", "qwen"),       # DashScope: qwen3-max, qwen-max-latest, qwen-plus, qwen-flash
    ("gpt", "openai"),      # raw: gpt-5, gpt-5-mini, gpt-4o
    ("claude", "anthropic"),  # raw: claude-3.7-sonnet, claude-opus-4.6
    ("gemini", "google"),
    ("deepseek", "deepseek"),
    ("llama", "meta"),
    ("mistral", "mistral"),
)


def _vendor_prefix(model_id: str) -> str:
    """Return the vendor / family prefix of a model identifier.

    Recognises both OpenRouter-style ``<vendor>/<model>`` ids
    (returns ``<vendor>``) and DashScope/native names lacking a
    slash (returns the model-family stem via ``_NATIVE_FAMILY_PREFIXES``).
    Falls back to the full id when no rule matches — equality of
    fall-backs still implies same model.
    """
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    lower = model_id.lower()
    for stem, family in _NATIVE_FAMILY_PREFIXES:
        if lower.startswith(stem):
            return family
    return model_id


def test_openrouter_provider_uses_cross_family_judges() -> None:
    """OpenRouter smoke path: per-turn and arc judges must differ in vendor prefix."""
    module = _load_runner_module()
    providers = module.PROVIDERS
    assert "openrouter" in providers, "PROVIDERS dict must include 'openrouter'"
    cfg = providers["openrouter"]
    perturn_vendor = _vendor_prefix(cfg.perturn_model)
    arc_vendor = _vendor_prefix(cfg.arc_model)
    assert perturn_vendor != arc_vendor, (
        f"OpenRouter judge family rotation broken: per-turn vendor "
        f"{perturn_vendor!r} == arc vendor {arc_vendor!r}. "
        f"Per-turn={cfg.perturn_model!r}, arc={cfg.arc_model!r}. "
        "Companion-bench spec §5 requires cross-family judge ensemble."
    )


def test_qwen_provider_is_acknowledged_weak_proxy() -> None:
    """Qwen smoke path: same-family is acceptable IF properly labelled."""
    module = _load_runner_module()
    providers = module.PROVIDERS
    assert "qwen" in providers
    cfg = providers["qwen"]
    perturn_vendor = _vendor_prefix(cfg.perturn_model)
    arc_vendor = _vendor_prefix(cfg.arc_model)
    # Qwen path intentionally same vendor (DashScope only ships Qwen).
    # The acknowledgement lives in the roster YAML's docstring header.
    assert perturn_vendor == arc_vendor, (
        "If you've changed the qwen path to use a non-Qwen judge, "
        "remove the WEAK PROXY warning from reference_systems.smoke_qwen.yaml."
    )
    roster_text = _QWEN_ROSTER.read_text(encoding="utf-8")
    assert "WEAK FAMILY PROXY" in roster_text, (
        f"{_QWEN_ROSTER.name} is missing the WEAK FAMILY PROXY header. "
        "Add it back so contributors know not to ship qwen-only smoke "
        "data to the official leaderboard."
    )


def test_setup_doc_describes_three_judge_tiers() -> None:
    """Operator guide must document the A/B/C qualification tiers."""
    setup_md = _REPO_ROOT / "docs" / "external" / "companion-bench-openrouter-setup.md"
    text = setup_md.read_text(encoding="utf-8")
    for token in ("Weak Proxy", "Family Rotation", "Judge 合格度档级"):
        assert token in text, (
            f"{setup_md.name} missing required section/term {token!r}; "
            "the three-tier judge ladder must be documented for ops handoff."
        )
